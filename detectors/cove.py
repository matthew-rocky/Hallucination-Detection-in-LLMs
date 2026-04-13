"""CoVe-style verification pipeline built on local retrieval."""

from statistics import mean
import time
from typing import Any

from detectors.base import make_result, make_unavailable
from utils.scoring_utils import score_with_conflicts
from retrieval.chunking import SourceDoc, ingest_docs
from retrieval.indexing import VectorIndex
from retrieval.search import classify_grounding
from utils.grounding_utils import SUPPORT_STATUSES
from utils.revision_utils import make_revision
from utils.text_utils import match_claim, extract_claims, extract_numbers, safe_text, split_into_sentences


GROUNDING_RISK = {
    "supported": 0.10,
    "abstractly_supported": 0.18,
    "weakly_supported": 0.28,
    "contradicted": 0.94,
    "insufficient evidence": 0.54,
}


def draft_answer_stage(answer: str) -> str:
    """Stage 1: the initial answer draft to be verified."""
    return safe_text(answer)


def make_questions_stage(draft_answer: str, question: str, max_questions: int = 6) -> list[dict[str, str]]:
    """Stage 2: derive verification questions from draft claims."""
    claims = extract_claims(draft_answer, max_claims=max_questions)
    prompts: list[dict[str, str]] = []
    for claim in claims:
        if extract_numbers(claim):
            verification_question = f"Which retrieved evidence verifies the numeric details in: {claim}"
        elif safe_text(question):
            verification_question = f"Using the evidence only, what supports or contradicts this answer claim: {claim}"
        else:
            verification_question = f"What evidence supports or contradicts this claim: {claim}"
        prompts.append({"claim": claim, "verification_question": verification_question})
    return prompts


def _first_evidence_sent(text: str) -> str:
    sentences = split_into_sentences(text)
    return sentences[0] if sentences else safe_text(text)


def _status_rank(status: str) -> int:
    return {"contradicted": 4, "supported": 3, "abstractly_supported": 3, "weakly_supported": 2, "insufficient evidence": 1}.get(status, 0)


def _best_grounding(claim: str, compare_claim: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best_grounding_result: dict[str, Any] | None = None
    best_query = ""
    best_hits: list[dict[str, Any]] = []

    for candidate in candidates:
        hits = candidate["hits"]
        # Rank verdicts.
        grounding = classify_grounding(claim, hits, compare_claim=compare_claim)
        score_key = (
            _status_rank(grounding["status"]),
            -float(grounding["score"]),
            float((grounding["best_hit"] or {}).get("score", 0.0)),
        )
        if best_grounding_result is None:
            best_grounding_result = grounding
            best_query = candidate["query"]
            best_hits = hits
            best_score_key = score_key
            continue
        if score_key > best_score_key:
            best_grounding_result = grounding
            best_query = candidate["query"]
            best_hits = hits
            best_score_key = score_key

    return {
        "query_used": best_query,
        "grounding": best_grounding_result or classify_grounding(claim, [], compare_claim=compare_claim),
        "hits": best_hits,
        "all_candidates": [
            {
                "query": candidate["query"],
                "hits": candidate["hits"],
            }
            for candidate in candidates
        ],
    }


def answer_qs_stage(
    prompts: list[dict[str, str]],
    *,
    question: str,
    index: VectorIndex,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Stage 3: answer verification questions from retrieved evidence only."""
    answered: list[dict[str, Any]] = []
    for item in prompts:
        claim_query = match_claim(item["claim"], question)
        candidates = [
            {"query": item["verification_question"], "hits": index.search(item["verification_question"], top_k=top_k)},
        ]
        # Search claim too.
        if safe_text(claim_query) != safe_text(item["verification_question"]):
            candidates.append({"query": claim_query, "hits": index.search(claim_query, top_k=top_k)})

        selected = _best_grounding(item["claim"], claim_query, candidates)
        grounding = selected["grounding"]
        best_hit = grounding["best_hit"]
        grounding_status = grounding["status"]
        if grounding_status in SUPPORT_STATUSES:
            status = "verified"
            independent_answer = (
                _first_evidence_sent(best_hit["text"]) if best_hit else "Retrieved evidence supports the claim."
            )
        elif grounding_status == "contradicted":
            status = "contradicted"
            independent_answer = (
                _first_evidence_sent(best_hit["text"]) if best_hit else "Retrieved evidence contradicts the claim."
            )
        else:
            status = "unresolved"
            independent_answer = "The indexed evidence does not resolve this question with enough direct support."

        answered.append(
            {
                **item,
                "status": status,
                "grounding_status": grounding_status,
                "independent_answer": independent_answer,
                "best_hit": best_hit,
                "retrieved_hits": grounding["assessed_hits"],
                "score": GROUNDING_RISK[grounding_status],
                "reason": grounding["reason"],
                "retrieval_query_used": selected["query_used"],
                "retrieval_candidates": selected["all_candidates"],
            }
        )
    return answered


def revise_stage(answered_questions: list[dict[str, Any]]) -> str:
    """Stage 4: synthesize a revised answer from supported findings only."""
    return make_revision(answered_questions, max_sentences=3)


def summary_stage(answered_questions: list[dict[str, Any]]) -> tuple[str, float | None, float]:
    """Stage 5: summarize verification outcomes and compute overall score."""
    if not answered_questions:
        return "No verification questions were generated.", None, 0.0
    risk_score = score_with_conflicts(
        [item["score"] * 100.0 for item in answered_questions],
        [item["status"] for item in answered_questions],
        contradicted_statuses=("contradicted",),
    )
    score = None if risk_score is None else risk_score / 100.0
    verified = sum(1 for item in answered_questions if item["status"] == "verified")
    contradicted = sum(1 for item in answered_questions if item["status"] == "contradicted")
    unresolved = sum(1 for item in answered_questions if item["status"] == "unresolved")
    summary = (
        f"CoVe-style verification checked {len(answered_questions)} question(s): "
        f"{verified} verified, {contradicted} contradicted, and {unresolved} unresolved."
    )
    confidence = min(0.95, 0.4 + (0.1 * min(len(answered_questions), 5)))
    return summary, score, confidence


def run_cove_detector(
    *,
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    extra_documents: list[SourceDoc | dict[str, Any]] | None = None,
    method_name: str = "CoVe-Style Verification",
    family: str = "cove-verification",
    top_k: int = 3,
    max_questions: int = 6,
    preferred_backend: str = "sentence-transformer",
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run the staged CoVe-style retrieval and revision pipeline."""
    start_time = time.perf_counter()
    documents = ingest_docs(
        source_text=source_text,
        evidence_text=evidence_text,
        extra_documents=extra_documents,
    )
    draft = draft_answer_stage(answer)
    if not draft:
        return make_unavailable(
            method_name=method_name,
            family=family,
            summary="The answer is empty, so there is no draft to verify.",
            explanation="The answer is empty, so there is no draft to verify.",
            evidence_used="No answer text was provided.",
            limitations="CoVe-style verification needs a draft answer and local evidence to inspect.",
            impl_status="unavailable",
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )
    if not documents:
        return make_unavailable(
            method_name=method_name,
            family=family,
            summary="No local evidence was available, so the CoVe-style pipeline could not verify the draft.",
            explanation="No local evidence was available, so the CoVe-style pipeline could not verify the draft.",
            evidence_used="No source text, evidence text, or uploaded documents were provided.",
            limitations="This CoVe-style prototype performs independent verification over a local evidence index.",
            impl_status="unavailable",
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )

    index = VectorIndex.from_documents(
        documents,
        preferred_backend=preferred_backend,
        model_name=model_name,
        max_sentences=1,
        max_chars=420,
        overlap=0,
    )
    vq_prompts = make_questions_stage(draft, question, max_questions=max_questions)
    independent_answers = answer_qs_stage(
        vq_prompts,
        question=question,
        index=index,
        top_k=top_k,
    )
    # Checked claims only.
    revised_answer = revise_stage(independent_answers)
    vq_summary, score, confidence = summary_stage(independent_answers)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    evidence = []
    citations = []
    claim_findings = []
    vq_rows = []
    # UI rows.
    for item in independent_answers:
        best_hit = item["best_hit"]
        if best_hit:
            evidence.append(
                {
                    "evidence_id": best_hit.get("chunk_id"),
                    "document_id": best_hit.get("document_id"),
                    "title": best_hit.get("title"),
                    "source_type": best_hit.get("source_type") or best_hit.get("source_label"),
                    "content": best_hit.get("text"),
                    "score": best_hit.get("score"),
                    "metadata": dict(best_hit.get("metadata") or {}),
                }
            )
            citations.append(
                {
                    "citation_id": best_hit.get("citation_id") or best_hit.get("chunk_id"),
                    "document_id": best_hit.get("document_id"),
                    "title": best_hit.get("title"),
                    "snippet": best_hit.get("text"),
                    "score": best_hit.get("score"),
                    "source_label": best_hit.get("source_label"),
                    "metadata": dict(best_hit.get("metadata") or {}),
                }
            )
        claim_findings.append(
            {
                "claim": item["claim"],
                "status": item["status"],
                "score": round(item["score"] * 100.0, 1),
                "best_match": best_hit.get("text", "No retrieved evidence.") if best_hit else "No retrieved evidence.",
                "reason": item["reason"],
            }
        )
        vq_rows.append(
            {
                "claim": item["claim"],
                "verification_question": item["verification_question"],
                "status": item["status"],
            }
        )

    intermediate_steps = [
        {"stage": "draft_answer", "output": draft},
        {"stage": "verification_question_generation", "output": vq_prompts},
        {
            "stage": "independent_answering",
            "output": [
                {
                    "claim": item["claim"],
                    "verification_question": item["verification_question"],
                    "retrieval_query_used": item["retrieval_query_used"],
                    "independent_answer": item["independent_answer"],
                    "status": item["status"],
                    "retrieval_candidates": [
                        {
                            "query": candidate["query"],
                            "hits": [
                                {
                                    "chunk_id": hit.get("chunk_id"),
                                    "title": hit.get("title"),
                                    "score": hit.get("score"),
                                    "text": hit.get("text"),
                                }
                                for hit in candidate["hits"]
                            ],
                        }
                        for candidate in item["retrieval_candidates"]
                    ],
                    "retrieved_hits": [
                        {
                            "chunk_id": hit.get("chunk_id"),
                            "title": hit.get("title"),
                            "score": hit.get("score"),
                            "grounding_status": hit.get("grounding_status"),
                            "text": hit.get("text"),
                        }
                        for hit in item["retrieved_hits"]
                    ],
                }
                for item in independent_answers
            ],
        },
        {"stage": "revised_answer", "output": revised_answer},
        {"stage": "final_summary", "output": vq_summary},
    ]

    explanation = (
        "Implemented CoVe-style pipeline with explicit stages: draft answer, verification-question generation, independent evidence answers, revision, and final summary. "
        "The independent answers are built from retrieved evidence rather than copied from the original draft."
    )
    limitations = (
        "This is a local CoVe-style prototype, not a paper-scale reproduction. It relies on local retrieval and extractive synthesis rather than multiple large-model calls with broad web access."
    )

    return make_result(
        method_name=method_name,
        family=family,
        score=score,
        confidence=confidence,
        summary=vq_summary,
        explanation=explanation,
        evidence_used=(
            f"{len(documents)} local document(s) indexed for claim-by-claim verification questions."
        ),
        evidence=evidence,
        citations=citations,
        intermediate_steps=intermediate_steps,
        claim_findings=claim_findings,
        revised_answer=revised_answer,
        limitations=limitations,
        impl_status="implemented",
        latency_ms=latency_ms,
        metadata={
            "retrieval_backend": index.embedder.backend_name,
            "document_count": len(documents),
            "chunk_count": len(index.chunks),
            "top_k": top_k,
            "max_questions": max_questions,
        },
        original_draft=draft,
        verification_questions=vq_rows,
        independent_answers=[
            {
                "claim": item["claim"],
                "verification_question": item["verification_question"],
                "retrieval_query_used": item["retrieval_query_used"],
                "independent_answer": item["independent_answer"],
                "status": item["status"],
            }
            for item in independent_answers
        ],
        verification_summary=vq_summary,
        suggested_revision=revised_answer,
        suggestion_label="Revised answer",
        reference_chunk_catalog=list(index.chunks),
    )

