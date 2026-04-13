"""Search and grounding helpers built on top of the local vector index."""

from typing import Any

from utils.grounding_utils import (
    SUPPORT_STATUSES,
    assess_claim_evidence,
    build_aggregate_hit,
    entity_overlap,
    numeric_alignment,
    phrase_overlap,
    token_coverage,
)
from utils.scoring_utils import score_with_conflicts
from utils.text_utils import (
    match_claim,
    extract_claims,
    lexical_overlap,
    safe_text,
)

from .indexing import VectorIndex


STATUS_TO_RISK = {
    "supported": 0.10,
    "abstractly_supported": 0.18,
    "weakly_supported": 0.32,
    "contradicted": 0.94,
    "insufficient evidence": 0.58,
}
COUNT_STATUS_MAP = {
    "supported": "supported",
    "abstractly_supported": "supported",
    "weakly_supported": "supported",
    "contradicted": "contradicted",
    "insufficient evidence": "insufficient evidence",
    "unsupported": "insufficient evidence",
}


def _citation_from_hit(hit: dict) -> dict[str, Any]:
    return {
        "citation_id": hit.get("citation_id") or hit.get("chunk_id"),
        "document_id": hit.get("document_id"),
        "title": hit.get("title") or hit.get("page_title") or hit.get("source_label", "evidence"),
        "snippet": hit.get("text", ""),
        "score": round(float(hit.get("score", 0.0)), 4),
        "source_label": hit.get("source_label", "evidence"),
        "metadata": dict(hit.get("metadata") or {}),
    }


def _evidence_from_hit(hit: dict) -> dict[str, Any]:
    return {
        "evidence_id": hit.get("chunk_id"),
        "document_id": hit.get("document_id"),
        "title": hit.get("title") or hit.get("page_title") or hit.get("source_label", "evidence"),
        "source_type": hit.get("source_type") or hit.get("source_label", "evidence"),
        "content": hit.get("text", ""),
        "score": round(float(hit.get("score", 0.0)), 4),
        "metadata": dict(hit.get("metadata") or {}),
    }


def _assess_hit(comparison_text: str, hit: dict[str, Any]) -> dict[str, Any]:
    lexical_score = lexical_overlap(comparison_text, hit["text"])
    assessment = assess_claim_evidence(
        comparison_text,
        hit["text"],
        semantic_score=float(hit.get("score", 0.0)),
        lexical_score=lexical_score,
        token_coverage_score=token_coverage(comparison_text, hit["text"]),
        phrase_overlap_score=phrase_overlap(comparison_text, hit["text"]),
        entity_overlap_score=entity_overlap(comparison_text, hit["text"]),
        number_score=numeric_alignment(comparison_text, hit["text"]),
    )
    return {
        **hit,
        "lexical_overlap": round(lexical_score, 3),
        "grounding_status": assessment["status"],
        "support_type": assessment["support_type"],
        "combined_support": assessment["support_strength"],
        "contradiction_cues": assessment["contradiction_cues"],
        "reason": assessment["reason"],
        "decisive_contradiction": bool(assessment.get("decisive_contradiction")),
    }


def _best_ranked_hit(hits: list[dict[str, Any]]) -> dict[str, Any]:
    return max(hits, key=lambda item: (bool(item.get("decisive_contradiction")), item["combined_support"], float(item.get("score", 0.0))))


def classify_grounding(
    claim: str,
    hits: list[dict],
    *,
    compare_claim: str | None = None,
) -> dict[str, Any]:
    """Assign a support verdict to one claim using retrieved chunks."""
    if not hits:
        return {
            "status": "insufficient evidence",
            "score": STATUS_TO_RISK["insufficient evidence"],
            "best_hit": None,
            "supporting_hits": [],
            "assessed_hits": [],
            "reason": "No retrieved evidence was available for this claim.",
        }

    comparison_text = safe_text(compare_claim) or claim
    assessed_hits = [_assess_hit(comparison_text, hit) for hit in hits]
    contradiction_hits = [item for item in assessed_hits if item["grounding_status"] == "contradicted"]
    strong_contras = [item for item in contradiction_hits if item.get("decisive_contradiction")]
    support_hits = [item for item in assessed_hits if item["grounding_status"] in SUPPORT_STATUSES]
    best_support_hit = None if not support_hits else max(
        support_hits,
        key=lambda item: (item["combined_support"], float(item.get("score", 0.0))),
    )

    if strong_contras:
        best_hit = _best_ranked_hit(strong_contras)
        if best_support_hit and best_support_hit["combined_support"] >= max(0.6, best_hit["combined_support"] + 0.18) and float(best_support_hit.get("score", 0.0)) >= float(best_hit.get("score", 0.0)) - 0.05:
            contradiction_hits = [item for item in contradiction_hits if item is not best_hit]
        else:
            cues = ", ".join(best_hit["contradiction_cues"][:2]) or "contradiction cues"
            return {
                "status": "contradicted",
                "score": STATUS_TO_RISK["contradicted"],
                "best_hit": best_hit,
                "supporting_hits": [best_hit],
                "assessed_hits": assessed_hits,
                "reason": f"Decisive evidence conflicts with the claim via {cues}.",
            }

    if contradiction_hits:
        best_hit = _best_ranked_hit(contradiction_hits)
        if not (best_support_hit and best_support_hit["combined_support"] >= max(0.58, best_hit["combined_support"] + 0.2) and float(best_support_hit.get("score", 0.0)) >= float(best_hit.get("score", 0.0)) - 0.05):
            cues = ", ".join(best_hit["contradiction_cues"][:2]) or "contradiction cues"
            return {
                "status": "contradicted",
                "score": STATUS_TO_RISK["contradicted"],
                "best_hit": best_hit,
                "supporting_hits": [best_hit],
                "assessed_hits": assessed_hits,
                "reason": f"Top evidence conflicts with the claim via {cues}.",
            }

    # Bundle weak hits.
    aggregate_hit = build_aggregate_hit(hits[: min(len(hits), 4)])
    aggregate_assessment = _assess_hit(comparison_text, aggregate_hit) if aggregate_hit is not None else None
    if aggregate_assessment and aggregate_assessment["grounding_status"] in SUPPORT_STATUSES:
        status = aggregate_assessment["grounding_status"]
        reason = aggregate_assessment["reason"]
        return {
            "status": status,
            "score": STATUS_TO_RISK[status],
            "best_hit": aggregate_assessment,
            "supporting_hits": hits[: min(len(hits), 3)],
            "assessed_hits": assessed_hits,
            "reason": reason,
        }

    if support_hits:
        best_hit = max(support_hits, key=lambda item: (item["combined_support"], float(item.get("score", 0.0))))
        status = best_hit["grounding_status"]
        return {
            "status": status,
            "score": STATUS_TO_RISK[status],
            "best_hit": best_hit,
            "supporting_hits": [best_hit],
            "assessed_hits": assessed_hits,
            "reason": best_hit["reason"],
        }

    best_hit = max(assessed_hits, key=lambda item: (item["combined_support"], float(item.get("score", 0.0))))
    return {
        "status": "insufficient evidence",
        "score": STATUS_TO_RISK["insufficient evidence"],
        "best_hit": best_hit,
        "supporting_hits": [best_hit],
        "assessed_hits": assessed_hits,
        "reason": "Retrieved evidence was relevant but did not verify the claim.",
    }


def ground_answer(
    *,
    question: str,
    answer: str,
    index: VectorIndex,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run claim-level grounding for an answer against an index."""
    claims = extract_claims(answer)
    claim_results: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    intermediate_steps: list[dict[str, Any]] = []
    counts = {"supported": 0, "contradicted": 0, "insufficient evidence": 0}

    for claim in claims:
        query = match_claim(claim, question)
        hits = index.search(query, top_k=top_k)
        grounding = classify_grounding(claim, hits, compare_claim=query)
        count_status = COUNT_STATUS_MAP.get(grounding["status"], "insufficient evidence")
        counts[count_status] += 1
        best_hit = grounding["best_hit"]
        supporting_hits = grounding.get("supporting_hits") or ([best_hit] if best_hit else [])
        best_match = "No retrieved evidence."
        if supporting_hits:
            best_match = "\n\n".join(safe_text(hit.get("text")) for hit in supporting_hits if safe_text(hit.get("text"))) or best_match
        claim_results.append(
            {
                "claim": claim,
                "status": grounding["status"],
                "score": round(float(grounding["score"]) * 100.0, 1),
                "best_match": best_match,
                "reason": grounding["reason"],
            }
        )
        for support_hit in supporting_hits:
            if support_hit:
                evidence.append(_evidence_from_hit(support_hit))
                citations.append(_citation_from_hit(support_hit))
        intermediate_steps.append(
            {
                "claim": claim,
                "retrieval_query": query,
                "retrieved_hits": [
                    {
                        "chunk_id": hit.get("chunk_id"),
                        "title": hit.get("title"),
                        "score": hit.get("score"),
                        "grounding_status": hit.get("grounding_status"),
                        "support_type": hit.get("support_type"),
                        "contradiction_cues": hit.get("contradiction_cues"),
                        "text": hit.get("text"),
                    }
                    for hit in grounding["assessed_hits"]
                ],
                "selected_status": grounding["status"],
            }
        )

    overall_risk_score = score_with_conflicts(
        [item["score"] for item in claim_results if item.get("score") is not None],
        [item["status"] for item in claim_results],
    )
    overall_score = None if overall_risk_score is None else overall_risk_score / 100.0

    # Dedupe repeats.
    deduped_evidence: list[dict[str, Any]] = []
    seen_evidence: set[str] = set()
    for item in evidence:
        key = f"{item.get('evidence_id')}|{item.get('content')}"
        if key in seen_evidence:
            continue
        seen_evidence.add(key)
        deduped_evidence.append(item)

    deduped_citations: list[dict[str, Any]] = []
    seen_citations: set[str] = set()
    for item in citations:
        key = f"{item.get('citation_id')}|{item.get('snippet')}"
        if key in seen_citations:
            continue
        seen_citations.add(key)
        deduped_citations.append(item)

    return {
        "claims": claims,
        "claim_results": claim_results,
        "counts": counts,
        "score": overall_score,
        "evidence": deduped_evidence,
        "citations": deduped_citations,
        "intermediate_steps": intermediate_steps,
        "chunk_catalog": list(index.chunks),
        "retrieval_backend": index.embedder.backend_name,
        "index_backend": index.index_backend,
        "question_conditioned_matching": any(
            safe_text(step["retrieval_query"]) != safe_text(step["claim"])
            for step in intermediate_steps
        ),
    }