"""Verification-style detector family approximation with a staged local and live workflow."""

import re

from retrieval.search import classify_grounding
from utils.grounding_utils import SUPPORT_STATUSES, reliable_cues
from utils.live_web_retrieval import fetch_wiki_evidence
from utils.local_retrieval_utils import build_chunk_records, rank_local_chunks
from utils.revision_utils import make_revision
from utils.scoring_utils import score_with_conflicts, make_method_result, unavailable_result
from utils.text_utils import match_claim, extract_claims, extract_numbers, safe_text


GROUNDING_RISK = {
    "supported": 10.0,
    "abstractly_supported": 18.0,
    "weakly_supported": 28.0,
    "contradicted": 95.0,
    "insufficient evidence": 58.0,
}
CLAIM_STATUS_TO_RISK = {
    "verified": 14.0,
    "contradicted": 95.0,
    "unresolved": 58.0,
}
VERDICT_RISK = {
    "retrieval_failure_or_weak_evidence": 54.0,
    "unsupported_despite_evidence": 68.0,
}
CONFLICT_PENALTY = 8.0
ABSOLUTE_LANGUAGE = {"always", "never", "only", "guarantees", "definitely", "proves"}


def _make_question(claim: str) -> str:
    """Create a short verification question for one claim."""
    if extract_numbers(claim):
        return f"What do the available passages say about the numeric or date details in: {claim}"
    if re.search(r"\b[A-Z][a-z]+\b", claim):
        return f"Which named entities and relationships are supported for: {claim}"
    if any(term in claim.lower() for term in ABSOLUTE_LANGUAGE):
        return f"Do the available passages really justify the absolute wording in: {claim}"
    return f"What evidence supports or contradicts: {claim}"


def _make_ref_chunks(source_text: str, evidence_text: str) -> list[dict]:
    """Combine local source and evidence passages into one searchable set."""
    reference_chunks = []
    if safe_text(source_text):
        reference_chunks.extend(
            build_chunk_records(source_text, chunk_prefix="S", source_label="source", max_sentences=1, max_chars=360, overlap=0)
        )
    if safe_text(evidence_text):
        reference_chunks.extend(
            build_chunk_records(evidence_text, chunk_prefix="E", source_label="evidence", max_sentences=1, max_chars=340, overlap=0)
        )
    return reference_chunks


def _clean_ranked_chunks(ranked_chunks: list[dict]) -> list[dict]:
    """Convert ranked verification chunks into retrieval-style hit records."""
    normalized_hits = []
    for chunk in ranked_chunks:
        normalized_hits.append(
            {
                **chunk,
                "score": max(
                    float(chunk.get("retrieval_score", 0.0)),
                    float(chunk.get("tfidf_score", 0.0)),
                    float(chunk.get("semanticish_score", 0.0)),
                ),
            }
        )
    return normalized_hits


def _classify_verdict(contradiction_claim: str, ranked_chunks: list[dict]) -> tuple[str, float, dict, list[dict]]:
    """Choose the strongest verification decision from the ranked evidence."""
    grounding = classify_grounding(
        contradiction_claim,
        _clean_ranked_chunks(ranked_chunks),
        compare_claim=contradiction_claim,
    )
    assessed_chunks = []
    for chunk in grounding["assessed_hits"]:
        grounding_status = chunk.get("grounding_status", "insufficient evidence")
        if grounding_status in SUPPORT_STATUSES:
            verification_status = "verified"
        elif grounding_status == "contradicted":
            verification_status = "contradicted"
        else:
            verification_status = "unresolved"
        assessed_chunks.append({**chunk, "verification_status": verification_status})

    grounding_status = grounding["status"]
    if grounding_status in SUPPORT_STATUSES:
        return "verified", GROUNDING_RISK[grounding_status], grounding["best_hit"], assessed_chunks
    if grounding_status == "contradicted":
        return "contradicted", GROUNDING_RISK["contradicted"], grounding["best_hit"], assessed_chunks
    return "unresolved", GROUNDING_RISK["insufficient evidence"], grounding["best_hit"], assessed_chunks


def _internal_conflicts(claims: list[str]) -> dict[int, list[dict]]:
    """Flag contradictory claim pairs inside the answer itself."""
    conflicts: dict[int, list[dict]] = {}
    for left_index, left_claim in enumerate(claims):
        for right_index in range(left_index + 1, len(claims)):
            cues = reliable_cues(left_claim, claims[right_index])
            if not cues:
                continue
            conflicts.setdefault(left_index, []).append({"other_claim": claims[right_index], "cues": cues})
            conflicts.setdefault(right_index, []).append({"other_claim": left_claim, "cues": cues})
    return conflicts


def _claim_diagnostic(
    status: str,
    claim_score: float,
    best_chunk: dict,
    ranked_chunks: list[dict],
    internal_conflicts: list[dict],
) -> tuple[str, float]:
    """Separate weak retrieval from unsupported claims that had relevant evidence."""
    if status == "contradicted":
        final_score = min(95.0, claim_score + (CONFLICT_PENALTY if internal_conflicts else 0.0))
        return "contradicted_by_evidence", final_score
    if status == "verified":
        final_score = min(95.0, claim_score + (CONFLICT_PENALTY if internal_conflicts else 0.0))
        return "verified_by_evidence", final_score
    top_strength = max(
        float(best_chunk.get("retrieval_score", 0.0)),
        float(best_chunk.get("tfidf_score", 0.0)),
        float(best_chunk.get("semanticish_score", 0.0)),
        float(best_chunk.get("score", 0.0)),
        0.0,
    )
    base_score = (
        VERDICT_RISK["retrieval_failure_or_weak_evidence"]
        if not ranked_chunks or top_strength < 0.16
        else max(claim_score, VERDICT_RISK["unsupported_despite_evidence"])
    )
    if internal_conflicts:
        base_score = min(95.0, base_score + CONFLICT_PENALTY)
    diagnostic = "retrieval_failure_or_weak_evidence" if (not ranked_chunks or top_strength < 0.16) else "unsupported_despite_evidence"
    return diagnostic, base_score


def _describe_chunk(best_chunk: dict) -> str:
    """Describe a supporting chunk in plain language."""
    if best_chunk.get("source_label") == "web":
        title = safe_text(best_chunk.get("page_title"))
        if title:
            return f"live Wikipedia chunk {best_chunk['chunk_id']} ({title})"
        return f"live Wikipedia chunk {best_chunk['chunk_id']}"
    return f"local {best_chunk.get('source_label', 'reference')} chunk {best_chunk['chunk_id']}"


def _build_check_answer(status: str, best_chunk: dict, retrieval_diagnostic: str) -> str:
    """Answer the generated verification question in plain language."""
    chunk_description = _describe_chunk(best_chunk)
    if status == "verified":
        return f"{chunk_description} supports the core claim wording."
    if status == "contradicted":
        contradiction_cues = best_chunk.get("contradiction_cues") or []
        if contradiction_cues:
            return f"{chunk_description} points the other way and shows contradiction cues: {', '.join(contradiction_cues[:2])}."
        return f"{chunk_description} points the other way."
    if retrieval_diagnostic == "unsupported_despite_evidence":
        return f"{chunk_description} is relevant, but it still does not support the claim."
    return f"{chunk_description} was only weakly relevant, so the claim remains unresolved."

def _build_revision(claim_records: list[dict]) -> tuple[str | None, str | None]:
    """Create a conservative revised answer from the verification trace."""
    if not claim_records:
        return None, None
    revised_answer = make_revision(
        [
            {
                "claim": record["claim"],
                "status": record["status"],
                "best_hit": record["best_chunk"],
                "retrieved_hits": record.get("retrieved_hits", []),
            }
            for record in claim_records
        ],
        max_sentences=4,
    )
    if not safe_text(revised_answer) or revised_answer.startswith("Available evidence does not clearly support"):
        return None, None
    contradicted = sum(1 for record in claim_records if record["status"] == "contradicted")
    suggestion_label = "Suggested corrected answer" if contradicted >= max(1, len(claim_records) // 2) else "Suggested revised answer"
    return suggestion_label, revised_answer


def _reference_scope(has_source_text: bool, has_evidence_text: bool, used_web: bool) -> str:
    """Describe which sources were used for verification."""
    parts = []
    if has_source_text:
        parts.append("source text")
    if has_evidence_text:
        parts.append("evidence text")
    if used_web:
        parts.append("live Wikipedia passages")
    if not parts:
        return "available passages"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _build_summary(reference_scope: str, counts: dict[str, int], num_not_checked: int = 0) -> str:
    """Create a compact verification-family summary."""
    if counts["contradicted"]:
        summary = (
            f"The staged verification workflow checked claims against {reference_scope} and found "
            f"{counts['verified']} verified, {counts['contradicted']} contradicted, and {counts['unresolved']} unresolved claim(s)."
        )
    elif counts["unresolved"]:
        summary = f"The staged verification workflow found some support in {reference_scope}, but {counts['unresolved']} claim(s) remained unresolved."
    else:
        summary = f"The staged verification workflow found direct support for most extracted claims in {reference_scope}."
    if num_not_checked:
        summary += f" Live retrieval also failed before {num_not_checked} claim(s) could be checked."
    return summary


def _build_check_summary(reference_scope: str, counts: dict[str, int], num_not_checked: int = 0) -> str:
    """Create the longer workflow-oriented verification summary shown in detail view."""
    if counts["contradicted"]:
        summary = (
            f"The workflow decomposed the answer into claims, generated verification questions, searched {reference_scope}, and found explicit contradictions for part of the answer."
        )
    elif counts["unresolved"]:
        summary = f"The workflow found relevant passages in {reference_scope} for each claim, but some questions still could not be resolved cleanly."
    else:
        summary = f"The workflow's generated verification questions were mostly answered directly by {reference_scope}."
    if num_not_checked:
        summary += " Some live-retrieval checks still failed before usable passages were returned."
    return summary


def _format_best_match(best_chunk: dict) -> str:
    """Format the best matching chunk with optional page metadata."""
    page_title = safe_text(best_chunk.get("page_title"))
    if page_title:
        return f"[{best_chunk['chunk_id']}] {page_title}: {best_chunk['text']}"
    return f"[{best_chunk['chunk_id']}] {best_chunk['text']}"


def _dedupe_web_sources(web_sources: list[dict]) -> list[dict]:
    """Deduplicate retrieved web sources across claims."""
    deduped = []
    seen = set()
    for source in web_sources:
        key = (safe_text(source.get("page_title")), safe_text(source.get("source_url")), safe_text(source.get("search_query")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _dedupe_chunk_catalog(chunks: list[dict]) -> list[dict]:
    """Deduplicate local and live chunk records for UI display."""
    deduped = []
    seen = set()
    for chunk in chunks:
        key = (
            safe_text(chunk.get("source_label")),
            safe_text(chunk.get("page_title")),
            safe_text(chunk.get("source_url")),
            safe_text(chunk.get("text")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def _dedupe_records(records: list[dict], id_key: str, text_key: str) -> list[dict]:
    """Deduplicate evidence and citation records."""
    deduped = []
    seen = set()
    for record in records:
        key = (safe_text(record.get(id_key)), safe_text(record.get(text_key)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped

def run_verify(
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    sampled_answers_text: str = "",
    top_k: int = 3,
    allow_web: bool = False,
    live_web_max_pages: int = 3,
) -> dict:
    """Run a staged verification workflow over local and optional live evidence."""
    del sampled_answers_text

    claims = extract_claims(answer)
    matching_claims = [match_claim(claim, question) for claim in claims]
    internal_conflicts = _internal_conflicts(claims)
    if not claims:
        return unavailable_result(
            method_name="Verification-Based Workflow",
            family="verification-based",
            summary="The answer is empty, so there are no claims to verify.",
            evidence_used="No usable answer claims.",
            limitations="Verification requires at least one answer claim plus evidence or live retrieval.",
            details={"required_input": "source_text_or_evidence_text_or_live_web_retrieval"},
        )

    local_chunks = _make_ref_chunks(source_text, evidence_text)
    has_source_text = bool(safe_text(source_text))
    has_evidence_text = bool(safe_text(evidence_text))
    if not local_chunks and not allow_web:
        return unavailable_result(
            method_name="Verification-Based Workflow",
            family="verification-based",
            summary="No source text, evidence text, or live web retrieval was available, so verification could not be performed.",
            evidence_used="No source text, evidence text, or live retrieval available.",
            limitations=(
                "This method needs source text, evidence text, optional live Wikipedia retrieval, or some combination of them. It still does not run tool-interactive external verification or a full CoVe/CRITIC loop."
            ),
            details={
                "required_input": "source_text_or_evidence_text_or_live_web_retrieval",
                "num_claims": len(claims),
                "live_web_retrieval_enabled": False,
            },
            verification_steps=[],
        )

    claim_findings = []
    verification_steps = []
    vq_rows = []
    claim_records = []
    retrieval_backends = set()
    counts = {"verified": 0, "contradicted": 0, "unresolved": 0}
    diagnostic_counts = {
        "verified_by_evidence": 0,
        "contradicted_by_evidence": 0,
        "unsupported_despite_evidence": 0,
        "retrieval_failure_or_weak_evidence": 0,
    }
    web_sources = []
    live_web_queries = []
    live_web_errors = []
    chunk_catalog = list(local_chunks)
    used_web = False
    had_chunks = False
    num_not_checked = 0
    evidence = []
    citations = []

    for index, (claim, matching_claim) in enumerate(zip(claims, matching_claims), start=1):
        verification_question = _make_question(claim)
        claim_conflicts = internal_conflicts.get(index - 1, [])
        candidate_chunks = list(local_chunks)
        live_result = None

        # Local first.
        if allow_web:
            live_result = fetch_wiki_evidence(matching_claim, chunk_prefix=f"W{index}_", max_pages=live_web_max_pages)
            live_web_queries.append(
                {
                    "claim": claim,
                    "search_query": matching_claim,
                    "status": live_result["status"],
                    "message": live_result["message"],
                }
            )
            if live_result["status"] == "ok":
                used_web = True
                candidate_chunks.extend(live_result["chunks"])
                web_sources.extend(live_result["sources"])
                chunk_catalog.extend(live_result["chunks"])
            elif live_result["status"] == "error":
                live_web_errors.append(
                    {
                        "claim": claim,
                        "search_query": matching_claim,
                        "message": live_result["message"],
                    }
                )

        # Keep failures.
        if not candidate_chunks:
            if live_result and live_result["status"] == "error":
                num_not_checked += 1
                verification_answer = f"Live Wikipedia retrieval failed before any passage could be checked for this claim: {live_result['message']}"
                claim_findings.append(
                    {
                        "claim": claim,
                        "status": "not_checked",
                        "score": None,
                        "best_match": "No retrieved passage available.",
                        "reason": verification_answer,
                        "retrieval_diagnostic": "not_checked",
                        "internal_conflicts": claim_conflicts,
                    }
                )
                vq_rows.append(
                    {
                        "claim": claim,
                        "verification_question": verification_question,
                        "status": "not_checked",
                        "retrieval_diagnostic": "not_checked",
                    }
                )
                verification_steps.append(
                    {
                        "stage_1_draft_claim": claim,
                        "stage_2_verification_question": verification_question,
                        "stage_3_best_chunk_id": "",
                        "stage_3_source_label": "",
                        "stage_3_page_title": "",
                        "stage_3_source_url": "",
                        "stage_3_evidence_excerpt": "",
                        "stage_3_retrieval_diagnostic": "not_checked",
                        "stage_4_verification_result": "not_checked",
                        "stage_4_verification_answer": verification_answer,
                        "stage_4_internal_conflict_cues": [item["cues"] for item in claim_conflicts],
                        "retrieved_candidates": [],
                    }
                )
                continue

            counts["unresolved"] += 1
            diagnostic_counts["retrieval_failure_or_weak_evidence"] += 1
            unresolved_reason = (
                "No local or live passage could be retrieved for this claim."
                if allow_web
                else "No local source or evidence text was available for this claim."
            )
            if claim_conflicts:
                unresolved_reason += f" The claim also conflicts with another extracted claim via {', '.join(claim_conflicts[0]['cues'][:2])}."
            unresolved_score = min(95.0, CLAIM_STATUS_TO_RISK["unresolved"] + (CONFLICT_PENALTY if claim_conflicts else 0.0))
            claim_findings.append(
                {
                    "claim": claim,
                    "status": "unresolved",
                    "score": unresolved_score,
                    "best_match": "No retrieved passage available.",
                    "reason": unresolved_reason,
                    "retrieval_diagnostic": "retrieval_failure_or_weak_evidence",
                    "internal_conflicts": claim_conflicts,
                }
            )
            vq_rows.append(
                {
                    "claim": claim,
                    "verification_question": verification_question,
                    "status": "unresolved",
                    "retrieval_diagnostic": "retrieval_failure_or_weak_evidence",
                }
            )
            verification_steps.append(
                {
                    "stage_1_draft_claim": claim,
                    "stage_2_verification_question": verification_question,
                    "stage_3_best_chunk_id": "",
                    "stage_3_source_label": "",
                    "stage_3_page_title": "",
                    "stage_3_source_url": "",
                    "stage_3_evidence_excerpt": "",
                    "stage_3_retrieval_diagnostic": "retrieval_failure_or_weak_evidence",
                    "stage_4_verification_result": "unresolved",
                    "stage_4_verification_answer": unresolved_reason,
                    "stage_4_internal_conflict_cues": [item["cues"] for item in claim_conflicts],
                    "retrieved_candidates": [],
                }
            )
            continue

        had_chunks = True
        ranked_chunks, backend = rank_local_chunks(matching_claim, candidate_chunks, top_k=top_k)
        retrieval_backends.add(backend)
        status, claim_score, best_chunk, assessed_chunks = _classify_verdict(matching_claim, ranked_chunks)
        retrieval_diagnostic, final_claim_score = _claim_diagnostic(status, claim_score, best_chunk, ranked_chunks, claim_conflicts)
        verification_answer = _build_check_answer(status, best_chunk, retrieval_diagnostic)
        counts[status] += 1
        diagnostic_counts[retrieval_diagnostic] += 1

        if status == "verified":
            reason = f"The generated verification question was answered by {_describe_chunk(best_chunk)}."
            if claim_conflicts:
                reason += f" The claim also conflicts with another extracted claim via {', '.join(claim_conflicts[0]['cues'][:2])}."
        elif status == "contradicted":
            contradiction_text = ", ".join((best_chunk.get("contradiction_cues") or [])[:2]) or "contradiction cues"
            reason = f"{_describe_chunk(best_chunk)} answered the verification question with contradiction cues: {contradiction_text}."
        elif retrieval_diagnostic == "unsupported_despite_evidence":
            reason = f"{_describe_chunk(best_chunk)} was relevant to the verification question, but it still did not support the claim."
            if claim_conflicts:
                reason += f" The claim also conflicts with another extracted claim via {', '.join(claim_conflicts[0]['cues'][:2])}."
        else:
            reason = f"{_describe_chunk(best_chunk)} was only weakly relevant to the verification question, so the claim remained unresolved."
            if claim_conflicts:
                reason += f" The claim also conflicts with another extracted claim via {', '.join(claim_conflicts[0]['cues'][:2])}."

        claim_findings.append(
            {
                "claim": claim,
                "status": status,
                "score": final_claim_score,
                "best_match": _format_best_match(best_chunk),
                "reason": reason,
                "retrieval_diagnostic": retrieval_diagnostic,
                "internal_conflicts": claim_conflicts,
            }
        )
        vq_rows.append(
            {
                "claim": claim,
                "verification_question": verification_question,
                "status": status,
                "retrieval_diagnostic": retrieval_diagnostic,
                "best_chunk_id": best_chunk.get("chunk_id", ""),
            }
        )
        verification_steps.append(
            {
                "stage_1_draft_claim": claim,
                "stage_2_verification_question": verification_question,
                "stage_3_best_chunk_id": best_chunk["chunk_id"],
                "stage_3_source_label": best_chunk["source_label"],
                "stage_3_page_title": best_chunk.get("page_title", ""),
                "stage_3_source_url": best_chunk.get("source_url", ""),
                "stage_3_evidence_excerpt": best_chunk["text"],
                "stage_3_retrieval_score": round(
                    max(
                        float(best_chunk.get("retrieval_score", 0.0)),
                        float(best_chunk.get("tfidf_score", 0.0)),
                        float(best_chunk.get("semanticish_score", 0.0)),
                        float(best_chunk.get("score", 0.0)),
                    ),
                    4,
                ),
                "stage_3_retrieval_diagnostic": retrieval_diagnostic,
                "stage_4_verification_result": status,
                "stage_4_verification_answer": verification_answer,
                "stage_4_internal_conflict_cues": [item["cues"] for item in claim_conflicts],
                "retrieved_candidates": [
                    {
                        "chunk_id": item["chunk_id"],
                        "source_label": item["source_label"],
                        "page_title": item.get("page_title", ""),
                        "source_url": item.get("source_url", ""),
                        "retrieval_score": item.get("retrieval_score", item.get("score", 0.0)),
                        "assessment": item["verification_status"],
                        "chunk_text": item["text"],
                    }
                    for item in assessed_chunks
                ],
            }
        )
        # Checked claims only.
        claim_records.append({"claim": claim, "status": status, "best_chunk": best_chunk, "retrieved_hits": assessed_chunks})
        if safe_text(best_chunk.get("text")):
            evidence.append(
                {
                    "evidence_id": best_chunk.get("chunk_id"),
                    "title": best_chunk.get("page_title") or best_chunk.get("source_label", "reference"),
                    "source_type": best_chunk.get("source_label", "reference"),
                    "content": best_chunk.get("text", ""),
                    "score": round(float(best_chunk.get("score", 0.0)), 4),
                    "metadata": {"retrieval_diagnostic": retrieval_diagnostic, "source_url": best_chunk.get("source_url", "")},
                }
            )
            citations.append(
                {
                    "citation_id": best_chunk.get("chunk_id"),
                    "title": best_chunk.get("page_title") or best_chunk.get("source_label", "reference"),
                    "snippet": best_chunk.get("text", ""),
                    "score": round(float(best_chunk.get("score", 0.0)), 4),
                    "source_label": best_chunk.get("source_label", "reference"),
                    "metadata": {"retrieval_diagnostic": retrieval_diagnostic, "source_url": best_chunk.get("source_url", "")},
                }
            )

    if not had_chunks and live_web_errors and not local_chunks:
        return unavailable_result(
            method_name="Verification-Based Workflow",
            family="verification-based",
            summary="Live Wikipedia retrieval was enabled, but no passages could be retrieved from this runtime.",
            evidence_used="No usable local text or live web passages were available.",
            limitations=(
                "This method can use local source text, local evidence text, and optional live Wikipedia retrieval. In this run, live retrieval failed before any passages could be returned."
            ),
            details={
                "required_input": "source_text_or_evidence_text_or_live_web_retrieval",
                "num_claims": len(claims),
                "live_web_retrieval_enabled": True,
                "used_live_web_retrieval": False,
                "live_web_backend": "wikipedia-live",
                "live_web_errors": live_web_errors,
            },
            verification_steps=verification_steps,
            verification_questions=vq_rows,
            web_sources=_dedupe_web_sources(web_sources),
            live_web_queries=live_web_queries,
        )

    # Contradictions win.
    risk_score = score_with_conflicts(
        [finding["score"] for finding in claim_findings if finding.get("score") is not None],
        [finding["status"] for finding in claim_findings],
        contradicted_statuses=("contradicted",),
    )
    reference_scope = _reference_scope(has_source_text, has_evidence_text, used_web)
    conflict_count = sum(len(items) for items in internal_conflicts.values()) // 2
    summary = _build_summary(reference_scope, counts, num_not_checked=num_not_checked)
    if diagnostic_counts["unsupported_despite_evidence"]:
        summary += f" {diagnostic_counts['unsupported_despite_evidence']} claim(s) had relevant evidence retrieved but still stayed unsupported."
    if diagnostic_counts["retrieval_failure_or_weak_evidence"]:
        summary += f" {diagnostic_counts['retrieval_failure_or_weak_evidence']} claim(s) were limited mainly by weak retrieval."
    if conflict_count:
        summary += f" {conflict_count} internal claim conflict(s) were also flagged."
    vq_summary = _build_check_summary(reference_scope, counts, num_not_checked=num_not_checked)
    if diagnostic_counts["unsupported_despite_evidence"]:
        vq_summary += " Some claims remained unsupported even after relevant evidence was retrieved."
    if diagnostic_counts["retrieval_failure_or_weak_evidence"]:
        vq_summary += " Other claims were limited mainly by weak or missing retrieval support."
    if conflict_count:
        vq_summary += " The pipeline also detected internal inconsistency across extracted claims."
    explanation = (
        f"The workflow ran five local stages: claim extraction, verification-question generation, passage retrieval across {reference_scope}, "
        "claim-level verdict assignment, and final risk aggregation. It distinguishes contradiction, unsupported-after-retrieval, and weak-retrieval cases instead of collapsing them into one heuristic bucket."
    )
    suggestion_label, suggested_revision = _build_revision(claim_records)
    limitations = (
        "This remains a lightweight staged verification approximation. It can add live Wikipedia passages, but it still does not run tool-interactive external verification or reproduce CoVe/CRITIC-style systems faithfully."
    )

    if has_source_text or has_evidence_text:
        if used_web:
            evidence_used = (
                "Local source text and/or evidence text plus live Wikipedia passages. The workflow generates verification questions and searches all available passages without full tool orchestration."
            )
        else:
            evidence_used = (
                "Local source text and/or evidence text only. The workflow generates verification questions and searches the provided passages without live tool orchestration."
            )
    else:
        evidence_used = (
            "Live Wikipedia passages only. The workflow generates verification questions and searches retrieved lead extracts without full tool orchestration."
        )

    # Dedupe passages.
    deduped_catalog = _dedupe_chunk_catalog(chunk_catalog)
    deduped_web_sources = _dedupe_web_sources(web_sources)
    evidence = _dedupe_records(evidence, "evidence_id", "content")
    citations = _dedupe_records(citations, "citation_id", "snippet")
    # Detail trace.
    intermediate_steps = [
        {"stage": "claim_extraction", "output": claims},
        {
            "stage": "verification_question_generation",
            "output": [
                {"claim": item["claim"], "verification_question": item["verification_question"]}
                for item in vq_rows
            ],
        },
        {
            "stage": "evidence_retrieval",
            "output": {
                "num_reference_chunks": len(deduped_catalog),
                "num_local_reference_chunks": len(local_chunks),
                "used_live_web_retrieval": used_web,
                "retrieval_backend": ", ".join(sorted(retrieval_backends)) or "none",
                "diagnostic_counts": diagnostic_counts,
            },
        },
        {
            "stage": "claim_verdict_assignment",
            "output": [
                {
                    "claim": item["claim"],
                    "status": item["status"],
                    "retrieval_diagnostic": item.get("retrieval_diagnostic", ""),
                    "reason": item["reason"],
                }
                for item in claim_findings
            ],
        },
        {
            "stage": "final_aggregation",
            "output": {
                "risk_score": risk_score,
                "counts": counts,
                "diagnostic_counts": diagnostic_counts,
                "internal_conflict_count": conflict_count,
                "num_not_checked": num_not_checked,
            },
        },
    ]

    # Stable keys.
    return make_method_result(
        method_name="Verification-Based Workflow",
        family="verification-based",
        risk_score=risk_score,
        summary=summary,
        evidence_used=evidence_used,
        claim_findings=claim_findings,
        details={
            "retrieval_backend": ", ".join(sorted(retrieval_backends)) or "none",
            "num_claims": len(claims),
            "num_reference_chunks": len(deduped_catalog),
            "num_local_reference_chunks": len(local_chunks),
            "num_live_web_chunks": len([chunk for chunk in deduped_catalog if chunk.get("source_label") == "web"]),
            "reference_sources": sorted({record["source_label"] for record in deduped_catalog}),
            "top_k": top_k,
            "question_conditioned_matching": any(claim != matching_claim for claim, matching_claim in zip(claims, matching_claims)),
            "live_web_retrieval_enabled": allow_web,
            "used_live_web_retrieval": used_web,
            "live_web_backend": "wikipedia-live" if allow_web else "none",
            "live_web_error_count": len(live_web_errors),
            "live_web_errors": live_web_errors,
            "verification_diagnostic_counts": diagnostic_counts,
            "unsupported_with_evidence_count": diagnostic_counts["unsupported_despite_evidence"],
            "retrieval_failure_count": diagnostic_counts["retrieval_failure_or_weak_evidence"],
            "internal_conflict_count": conflict_count,
        },
        limitations=limitations,
        explanation=explanation,
        impl_status="approximate",
        intermediate_steps=intermediate_steps,
        verification_steps=verification_steps,
        verification_questions=vq_rows,
        verification_summary=vq_summary,
        suggestion_label=suggestion_label,
        suggested_revision=suggested_revision,
        web_sources=deduped_web_sources,
        live_web_queries=live_web_queries,
        reference_chunk_catalog=deduped_catalog,
        evidence=evidence,
        citations=citations,
    )


