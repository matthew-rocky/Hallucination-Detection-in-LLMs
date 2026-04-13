"""CRITIC-lite detector with real local tool routing and revision."""

from statistics import mean
import time
from typing import Any

from detectors.base import make_result, make_unavailable
from utils.scoring_utils import score_with_conflicts
from retrieval.chunking import SourceDoc, ingest_docs
from retrieval.indexing import VectorIndex
from retrieval.search import classify_grounding, ground_answer
from utils.grounding_utils import SUPPORT_STATUSES
from utils.revision_utils import make_revision
from utils.text_utils import match_claim, extract_claims, extract_numbers, safe_text, split_into_sentences


RISK_BY_STATUS = {
    "supported": 0.14,
    "abstractly_supported": 0.20,
    "weakly_supported": 0.30,
    "contradicted": 0.95,
    "insufficient evidence": 0.56,
}


def initial_answer_stage(answer: str) -> str:
    """Stage 1: accept the initial answer that will be critiqued."""
    return safe_text(answer)


def route_tools_stage(claims: list[str]) -> list[dict[str, Any]]:
    """Stage 2: choose which tools to run for each claim."""
    routes: list[dict[str, Any]] = []
    for claim in claims:
        tools = ["local_retrieval"]
        if extract_numbers(claim):
            tools.append("calculator_numeric_check")
        routes.append({"claim": claim, "tools": tools})
    return routes


def _first_sentence(text: str) -> str:
    sentences = split_into_sentences(text)
    return sentences[0] if sentences else safe_text(text)


def _numeric_check_tool(claim: str, evidence_text: str) -> dict[str, Any]:
    claim_numbers = sorted(extract_numbers(claim))
    evidence_numbers = sorted(extract_numbers(evidence_text))
    if not claim_numbers or not evidence_numbers:
        return {
            "tool_name": "calculator_numeric_check",
            "status": "not_applicable",
            "message": "No comparable numeric values were available.",
            "claim_numbers": claim_numbers,
            "evidence_numbers": evidence_numbers,
        }

    claim_values = [float(value) for value in claim_numbers]
    evidence_values = [float(value) for value in evidence_numbers]
    closest_deltas = [min(abs(claim_value - evidence_value) for evidence_value in evidence_values) for claim_value in claim_values]
    mismatch = any(delta > 0.0 for delta in closest_deltas)
    return {
        "tool_name": "calculator_numeric_check",
        "status": "mismatch" if mismatch else "matched",
        "message": (
            f"Numeric comparison found evidence numbers {evidence_numbers} for claim numbers {claim_numbers}."
        ),
        "claim_numbers": claim_numbers,
        "evidence_numbers": evidence_numbers,
        "closest_deltas": closest_deltas,
    }


def execute_tools_stage(
    routes: list[dict[str, Any]],
    *,
    question: str,
    index: VectorIndex,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Stage 3: execute the routed tools."""
    outputs: list[dict[str, Any]] = []
    for route in routes:
        claim = route["claim"]
        query = match_claim(claim, question)
        hits = index.search(query, top_k=top_k)
        grounding = classify_grounding(claim, hits, compare_claim=query)
        best_hit = grounding["best_hit"]
        tool_results = [
            {
                "tool_name": "local_retrieval",
                "status": grounding["status"],
                "message": grounding["reason"],
                "top_hits": [
                    {
                        "chunk_id": hit.get("chunk_id"),
                        "title": hit.get("title"),
                        "score": hit.get("score"),
                        "grounding_status": hit.get("grounding_status"),
                        "text": hit.get("text"),
                    }
                    for hit in grounding["assessed_hits"]
                ],
            }
        ]
        if "calculator_numeric_check" in route["tools"]:
            numeric_evidence_text = " ".join(hit.get("text", "") for hit in grounding.get("assessed_hits", [])[: min(len(grounding.get("assessed_hits", [])), 3)])
            tool_results.append(_numeric_check_tool(claim, numeric_evidence_text or (best_hit["text"] if best_hit else "")))

        outputs.append(
            {
                "claim": claim,
                "query": query,
                "route": route,
                "grounding": grounding,
                "best_hit": best_hit,
                "tool_results": tool_results,
            }
        )
    return outputs


def critique_stage(tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stage 4: critique claims using the tool outputs."""
    critiques: list[dict[str, Any]] = []
    for output in tool_outputs:
        best_hit = output["best_hit"]
        grounding = output["grounding"]
        retrieval_result = output["tool_results"][0]
        numeric_result = next(
            (item for item in output["tool_results"] if item["tool_name"] == "calculator_numeric_check"),
            None,
        )

        grounding_status = grounding["status"]
        final_status = grounding_status
        if grounding_status == "contradicted":
            critique = (
                f"Retrieved evidence contradicts the claim. Best evidence says: {_first_sentence(best_hit['text'])}" if best_hit
                else "Retrieved evidence contradicts the claim."
            )
        elif numeric_result and numeric_result["status"] == "mismatch":
            final_status = "contradicted"
            critique = (
                f"The numeric check disagrees with the claim. {numeric_result['message']}"
            )
        elif grounding_status in SUPPORT_STATUSES:
            critique = "The retrieved evidence supports this claim."
        else:
            critique = "The available tools did not verify this claim with enough direct evidence."

        critiques.append(
            {
                "claim": output["claim"],
                "status": final_status,
                "score": RISK_BY_STATUS[final_status],
                "critique": critique,
                "best_hit": best_hit,
                "tool_results": output["tool_results"],
                "retrieval_message": retrieval_result["message"],
            }
        )
    return critiques


def revise_answer_stage(critiques: list[dict[str, Any]]) -> str:
    """Stage 5: revise the answer from tool-backed grounded findings."""
    return make_revision(critiques, max_sentences=3)


def run_second_loop(revised_answer: str, *, question: str, index: VectorIndex, top_k: int) -> dict[str, Any] | None:
    """Stage 6: optionally re-check the revised answer when it changed materially."""
    if not safe_text(revised_answer):
        return None
    return ground_answer(question=question, answer=revised_answer, index=index, top_k=top_k)


def run_critic_detector(
    *,
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    extra_documents: list[SourceDoc | dict[str, Any]] | None = None,
    method_name: str = "CRITIC-lite Tool Check",
    family: str = "critic-lite",
    top_k: int = 3,
    preferred_backend: str = "sentence-transformer",
    model_name: str | None = None,
    second_loop: bool = True,
) -> dict[str, Any]:
    """Run the CRITIC-lite tool loop over local evidence."""
    start_time = time.perf_counter()
    documents = ingest_docs(
        source_text=source_text,
        evidence_text=evidence_text,
        extra_documents=extra_documents,
    )
    initial_answer = initial_answer_stage(answer)
    if not initial_answer:
        return make_unavailable(
            method_name=method_name,
            family=family,
            summary="The answer is empty, so there is nothing to critique with tools.",
            explanation="The answer is empty, so there is nothing to critique with tools.",
            evidence_used="No answer text was provided.",
            limitations="CRITIC-lite requires an answer draft plus local evidence tools.",
            impl_status="unavailable",
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )
    if not documents:
        return make_unavailable(
            method_name=method_name,
            family=family,
            summary="No local evidence was available, so the CRITIC-lite tool loop could not run.",
            explanation="No local evidence was available, so the CRITIC-lite tool loop could not run.",
            evidence_used="No source text, evidence text, or uploaded documents were provided.",
            limitations="This CRITIC-lite prototype routes claims to real local retrieval and numeric-check tools over a local evidence index.",
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
    claims = extract_claims(initial_answer)
    routes = route_tools_stage(claims)
    tool_outputs = execute_tools_stage(routes, question=question, index=index, top_k=top_k)
    critiques = critique_stage(tool_outputs)
    revised_answer = revise_answer_stage(critiques)

    second_pass = None
    if second_loop and safe_text(revised_answer) and safe_text(revised_answer) != safe_text(initial_answer):
        second_pass = run_second_loop(revised_answer, question=question, index=index, top_k=top_k)

    score = None if not critiques else score_with_conflicts([item["score"] * 100.0 for item in critiques], [item["status"] for item in critiques]) / 100.0
    confidence = min(0.95, 0.45 + (0.08 * min(len(critiques), 5)))
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    evidence = []
    citations = []
    claim_findings = []
    for critique in critiques:
        best_hit = critique["best_hit"]
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
                "claim": critique["claim"],
                "status": critique["status"],
                "score": round(critique["score"] * 100.0, 1),
                "best_match": best_hit.get("text", "No retrieved evidence.") if best_hit else "No retrieved evidence.",
                "reason": critique["critique"],
            }
        )

    intermediate_steps = [
        {"stage": "initial_answer", "output": initial_answer},
        {"stage": "tool_routing", "output": routes},
        {
            "stage": "tool_execution",
            "output": [
                {
                    "claim": item["claim"],
                    "query": item["query"],
                    "tool_results": item["tool_results"],
                }
                for item in tool_outputs
            ],
        },
        {"stage": "critique", "output": critiques},
        {"stage": "revision", "output": revised_answer},
    ]
    if second_pass is not None:
        intermediate_steps.append(
            {
                "stage": "second_loop_recheck",
                "output": {
                    "score": second_pass["score"],
                    "claim_results": second_pass["claim_results"],
                },
            }
        )

    contradicted = sum(1 for item in critiques if item["status"] == "contradicted")
    insufficient = sum(1 for item in critiques if item["status"] == "insufficient evidence")
    summary = (
        f"CRITIC-lite routed {len(claims)} claim(s) through local tools and found {contradicted} contradicted and {insufficient} unresolved claim(s) before revising the answer."
    )
    explanation = (
        "Implemented CRITIC-lite workflow with explicit tool routing, real local retrieval, a numeric comparison tool for number-bearing claims, tool-backed critique, answer revision, and an optional second re-check loop."
    )
    limitations = (
        "This is a local CRITIC-lite prototype. It does not call external web APIs, and its numeric tool only compares extracted numbers rather than performing broad symbolic reasoning."
    )

    return make_result(
        method_name=method_name,
        family=family,
        score=score,
        confidence=confidence,
        summary=summary,
        explanation=explanation,
        evidence_used=(
            f"{len(documents)} local document(s) indexed for retrieval and numeric checks over extracted claims."
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
            "second_loop_enabled": second_loop,
            "second_loop_ran": second_pass is not None,
            "second_loop_score": None if second_pass is None else second_pass["score"],
        },
        tool_outputs=[
            {
                "claim": item["claim"],
                "tool_results": item["tool_results"],
            }
            for item in tool_outputs
        ],
        proposed_external_checks=routes,
    )




