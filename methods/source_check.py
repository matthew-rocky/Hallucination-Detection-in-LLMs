"""Source-grounded consistency method."""

from retrieval.search import classify_grounding
from utils.scoring_utils import (
    score_with_conflicts,
    make_method_result,
    unavailable_result,
)
from utils.text_utils import (
    chunk_text,
    sim_matrix,
    match_claim,
    extract_claims,
    normalize_text,
    safe_text,
    truncate_text,
)


SUPPORTED_RISK = 8.0
ABSTRACT_SUPPORT_RISK = 14.0
WEAK_SUPPORT_RISK = 32.0
UNCLEAR_RISK = 55.0
UNSUPPORTED_RISK = 84.0
CONTRADICTED_RISK = 97.0


def _build_summary(claim_findings: list[dict]) -> str:
    """Create a compact grounded-consistency summary."""
    contradicted = sum(1 for item in claim_findings if item["status"] == "contradicted")
    unsupported = sum(1 for item in claim_findings if item["status"] == "unsupported")
    weak = sum(1 for item in claim_findings if item["status"] == "weakly supported")
    unclear = sum(1 for item in claim_findings if item["status"] == "unclear")
    supported = sum(1 for item in claim_findings if item["status"] == "supported")
    abstract_supported = sum(
        1
        for item in claim_findings
        if item["status"] == "supported" and item.get("support_type") == "abstraction"
    )

    if contradicted or unsupported:
        return (
            f"{supported} grounded claim(s), {contradicted} contradicted claim(s), and "
            f"{unsupported} unsupported claim(s) were found when the answer was compared "
            "against the supplied source passage."
        )
    if weak or unclear:
        return (
            f"Most claims aligned with the source, but {weak} claim(s) were only partially supported "
            f"and {unclear} claim(s) remained unclear after claim-to-source matching."
        )
    if abstract_supported:
        return (
            f"Most claims aligned well with the supplied source passage, including "
            f"{abstract_supported} claim(s) supported by faithful abstraction or paraphrase."
        )
    return f"Most claims aligned well with the supplied source passage ({supported} supported claim(s))."


def _source_chunks(source_text: str) -> tuple[list[str], list[dict]]:
    """Build sentence-level source candidates for matching."""
    base_chunks = chunk_text(source_text, max_sentences=1, max_chars=420, overlap=0)
    candidates = []
    seen = set()
    for index, chunk in enumerate(base_chunks, start=1):
        normalized = normalize_text(chunk)
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            {
                "chunk_id": f"S{index}",
                "text": chunk,
                "title": "Source text",
                "source_label": "source",
                "source_type": "source",
                "candidate_type": "sentence",
                "metadata": {"chunk_numbers": [index]},
            }
        )
    return base_chunks, candidates


def _build_source_hits(candidate_chunks: list[dict], similarities) -> list[dict]:
    hits = []
    for chunk, similarity in zip(candidate_chunks, similarities):
        hits.append({**chunk, "score": float(similarity)})
    hits.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return hits


def _map_assessment(grounding: dict) -> dict:
    best_hit = grounding.get("best_hit") or {}
    status = grounding["status"]
    support_type = best_hit.get("support_type")
    if status == "supported":
        return {
            "status": "supported",
            "risk_score": SUPPORTED_RISK,
            "support_type": support_type or "direct",
            "reason": grounding["reason"],
        }
    if status == "abstractly_supported":
        return {
            "status": "supported",
            "risk_score": ABSTRACT_SUPPORT_RISK,
            "support_type": support_type or "abstraction",
            "reason": grounding["reason"],
        }
    if status == "weakly_supported":
        return {
            "status": "weakly supported",
            "risk_score": WEAK_SUPPORT_RISK,
            "support_type": support_type or "partial",
            "reason": grounding["reason"],
        }
    if status == "contradicted":
        return {
            "status": "contradicted",
            "risk_score": CONTRADICTED_RISK,
            "support_type": "contradicted",
            "reason": grounding["reason"],
        }

    combined_support = float(best_hit.get("combined_support", 0.0))
    lexical_score = float(best_hit.get("lexical_overlap", 0.0))
    if combined_support >= 0.24 or lexical_score >= 0.12:
        return {
            "status": "unclear",
            "risk_score": UNCLEAR_RISK,
            "support_type": "unclear",
            "reason": grounding["reason"],
        }
    return {
        "status": "unsupported",
        "risk_score": UNSUPPORTED_RISK,
        "support_type": "unsupported",
        "reason": grounding["reason"],
    }


def _describe_source_hit(hit: dict) -> str:
    chunk_id = safe_text(hit.get("chunk_id")) or "source"
    candidate_type = hit.get("candidate_type") or "sentence"
    if candidate_type == "adjacent_window":
        return f"source window {chunk_id}"
    return f"source snippet {chunk_id}"


def _format_claim_reason(assessment: dict, grounding: dict) -> str:
    best_hit = grounding.get("best_hit") or {}
    hit_description = _describe_source_hit(best_hit)
    reason = assessment["reason"]
    if assessment["status"] == "supported":
        return f"Supported by {hit_description}: {reason}"
    if assessment["status"] == "weakly supported":
        return f"Partially supported by {hit_description}: {reason}"
    if assessment["status"] == "contradicted":
        return f"Contradicted by {hit_description}: {reason}"
    if assessment["status"] == "unclear":
        return f"The closest match was {hit_description}, but it only gave partial grounding: {reason}"
    return f"No source snippet clearly supported this claim; the closest match was {hit_description}."


def _format_best_match(best_hit: dict) -> str:
    text = safe_text(best_hit.get("text")) or "No source chunk available."
    chunk_id = safe_text(best_hit.get("chunk_id"))
    prefix = f"[{chunk_id}] " if chunk_id else ""
    return prefix + truncate_text(text, 320)


def _dedupe_by_text(records: list[dict], text_key: str) -> list[dict]:
    deduped = []
    seen = set()
    for record in records:
        key = (safe_text(record.get("chunk_id") or record.get("citation_id") or record.get("evidence_id")), safe_text(record.get(text_key)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _build_explanation(claim_findings: list[dict], backend: str, num_candidates: int) -> str:
    contradicted = sum(1 for item in claim_findings if item["status"] == "contradicted")
    unsupported = sum(1 for item in claim_findings if item["status"] == "unsupported")
    supported = sum(1 for item in claim_findings if item["status"] == "supported")
    return (
        f"The answer was decomposed into {len(claim_findings)} claim(s), matched against {num_candidates} local source snippet(s) "
        f"with {backend} similarity, and then classified claim by claim. This run found {supported} supported, "
        f"{contradicted} contradicted, and {unsupported} unsupported claim(s), with contradiction weighted more heavily than missing support."
    )


def run_source(
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    sampled_answers_text: str = "",
) -> dict:
    """Check whether answer claims are supported by a supplied source passage."""
    del evidence_text
    del sampled_answers_text

    claims = extract_claims(answer)
    matching_claims = [match_claim(claim, question) for claim in claims]
    if not claims:
        return unavailable_result(
            method_name="Source-Grounded Consistency",
            family="source-grounded consistency",
            summary="The answer is empty, so there are no claims to compare against the source.",
            evidence_used="User-provided source text.",
            limitations="This method needs answer claims and a source passage.",
            details={"required_input": "source_text"},
        )

    if not safe_text(source_text):
        return unavailable_result(
            method_name="Source-Grounded Consistency",
            family="source-grounded consistency",
            summary="No source text was supplied, so grounded consistency could not be estimated.",
            evidence_used="No source text available.",
            limitations=(
                "This method requires a source passage. Without one, it cannot compare "
                "claims against source support."
            ),
            details={"required_input": "source_text", "num_claims": len(claims)},
        )

    source_chunks, candidate_chunks = _source_chunks(source_text)
    if not candidate_chunks:
        return unavailable_result(
            method_name="Source-Grounded Consistency",
            family="source-grounded consistency",
            summary="The supplied source text could not be chunked into usable passages.",
            evidence_used="User-provided source text.",
            limitations="The source text appears empty after preprocessing.",
            details={"required_input": "source_text", "num_claims": len(claims)},
        )

    similarity_matrix, backend = sim_matrix(
        matching_claims,
        [chunk["text"] for chunk in candidate_chunks],
    )
    claim_findings = []
    support_chunks = []
    evidence = []
    citations = []
    stage_rows = []

    for claim_index, claim in enumerate(claims):
        hits = _build_source_hits(candidate_chunks, similarity_matrix[claim_index])
        grounding = classify_grounding(
            claim,
            hits[: min(len(hits), 6)],
            compare_claim=matching_claims[claim_index],
        )
        assessment = _map_assessment(grounding)
        best_hit = grounding.get("best_hit") or {}
        best_chunk = _format_best_match(best_hit)
        claim_reason = _format_claim_reason(assessment, grounding)

        claim_findings.append(
            {
                "claim": claim,
                "status": assessment["status"],
                "score": assessment["risk_score"],
                "best_match": best_chunk,
                "reason": claim_reason,
                "support_type": assessment.get("support_type"),
                "source_chunk_id": best_hit.get("chunk_id", ""),
            }
        )
        support_chunks.append(
            {
                "claim": claim,
                "best_source_chunk": best_chunk,
                "source_chunk_id": best_hit.get("chunk_id", ""),
                "semantic_similarity": round(float(best_hit.get("score", 0.0)), 3),
                "lexical_overlap": round(float(best_hit.get("lexical_overlap", 0.0)), 3),
                "status": assessment["status"],
                "support_type": assessment.get("support_type"),
                "candidate_type": best_hit.get("candidate_type", "sentence"),
                "contradiction_cues": best_hit.get("contradiction_cues", []),
            }
        )
        supporting_hits = grounding.get("supporting_hits") or ([best_hit] if best_hit else [])
        for hit in supporting_hits:
            if not hit:
                continue
            evidence.append(
                {
                    "evidence_id": hit.get("chunk_id"),
                    "title": "Source text",
                    "source_type": "source",
                    "content": safe_text(hit.get("text")),
                    "score": round(float(hit.get("score", 0.0)), 4),
                    "metadata": dict(hit.get("metadata") or {}),
                }
            )
            citations.append(
                {
                    "citation_id": hit.get("chunk_id"),
                    "title": "Source text",
                    "snippet": safe_text(hit.get("text")),
                    "score": round(float(hit.get("score", 0.0)), 4),
                    "source_label": "source",
                    "metadata": dict(hit.get("metadata") or {}),
                }
            )
        stage_rows.append(
            {
                "claim": claim,
                "matching_claim": matching_claims[claim_index],
                "status": assessment["status"],
                "best_chunk_id": best_hit.get("chunk_id", ""),
                "best_chunk_score": round(float(best_hit.get("score", 0.0)), 4),
                "reason": claim_reason,
            }
        )

    risk_score = score_with_conflicts(
        [item["score"] for item in claim_findings],
        [item["status"] for item in claim_findings],
    )
    summary = _build_summary(claim_findings)
    explanation = _build_explanation(claim_findings, backend, len(candidate_chunks))
    supported_claim_count = sum(1 for item in claim_findings if item["status"] == "supported")
    abstract_count = sum(
        1
        for item in claim_findings
        if item["status"] == "supported" and item.get("support_type") == "abstraction"
    )
    contra_count = sum(1 for item in claim_findings if item["status"] == "contradicted")
    unsupported_count = sum(1 for item in claim_findings if item["status"] == "unsupported")
    weak_count = sum(1 for item in claim_findings if item["status"] == "weakly supported")
    unclear_claim_count = sum(1 for item in claim_findings if item["status"] == "unclear")
    claim_alignment_score = round(
        (
            supported_claim_count
            + (0.75 * weak_count)
            - contra_count
            - (0.5 * unsupported_count)
            - (0.25 * unclear_claim_count)
        )
        / max(1, len(claim_findings)),
        3,
    )
    limitations = (
        "This method uses local semantic matching plus structured contradiction and paraphrase heuristics rather than full natural-language inference. "
        "It is designed to separate supported, contradicted, and merely unsupported claims while staying lightweight and deterministic."
    )
    evidence = _dedupe_by_text(evidence, "content")
    citations = _dedupe_by_text(citations, "snippet")
    intermediate_steps = [
        {"stage": "claim_extraction", "output": claims},
        {
            "stage": "source_chunking",
            "output": {
                "num_source_chunks": len(source_chunks),
                "num_candidate_chunks": len(candidate_chunks),
                "candidate_chunk_ids": [chunk["chunk_id"] for chunk in candidate_chunks],
            },
        },
        {"stage": "claim_grounding", "output": stage_rows},
        {
            "stage": "final_aggregation",
            "output": {
                "risk_score": risk_score,
                "supported_claim_count": supported_claim_count,
                "contradicted_claim_count": contra_count,
                "unsupported_claim_count": unsupported_count,
                "weakly_supported_claim_count": weak_count,
                "unclear_claim_count": unclear_claim_count,
            },
        },
    ]

    return make_method_result(
        method_name="Source-Grounded Consistency",
        family="source-grounded consistency",
        risk_score=risk_score,
        summary=summary,
        explanation=explanation,
        evidence_used="User-provided source text, split into local source snippets and adjacent windows for claim-level support checking.",
        claim_findings=claim_findings,
        details={
            "similarity_backend": backend,
            "num_claims": len(claims),
            "num_source_chunks": len(source_chunks),
            "num_candidate_source_chunks": len(candidate_chunks),
            "question_conditioned_matching": any(
                claim != matching_claim for claim, matching_claim in zip(claims, matching_claims)
            ),
            "supported_claim_ratio": round(supported_claim_count / max(1, len(claim_findings)), 3),
            "contradicted_claim_count": contra_count,
            "unsupported_claim_count": unsupported_count,
            "weakly_supported_claim_count": weak_count,
            "unclear_claim_count": unclear_claim_count,
            "abstractly_supported_claim_count": abstract_count,
            "summary_tolerant_mode": True,
            "claim_alignment_score": claim_alignment_score,
        },
        limitations=limitations,
        impl_status="approximate",
        supporting_source_chunks=support_chunks,
        citations=citations,
        evidence=evidence,
        intermediate_steps=intermediate_steps,
    )


