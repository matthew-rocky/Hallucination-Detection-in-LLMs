"""Transparent scoring helpers for the prototype."""

from typing import Any, Iterable

from detectors.base import (
    make_result,
    make_unavailable,
    probability_to_label,
    risk_to_prob,
)


LOW_MAX = 33.0
MEDIUM_MAX = 66.0

STATUS_TO_RISK = {
    "supported": 8.0,
    "weakly supported": 32.0,
    "unclear": 55.0,
    "unsupported": 84.0,
    "contradicted": 97.0,
    "not_checked": None,
}


def clip_score(score: float | int | None) -> float | None:
    """Clip a score into the 0-100 range, preserving unavailable values."""
    if score is None:
        return None
    return float(max(0.0, min(100.0, float(score))))


def score_to_label(score: float | int | None) -> str:
    """Map a risk score to a qualitative label."""
    return probability_to_label(risk_to_prob(score))


def claim_scores(
    claim_scores: Iterable[float | int | None],
    default_score: float | None = None,
) -> float | None:
    """Average numeric claim-level scores into a method-level score."""
    values = [clip_score(score) for score in claim_scores if score is not None]
    if not values:
        return clip_score(default_score)
    return clip_score(sum(values) / len(values))


def score_with_conflicts(
    claim_scores: Iterable[float | int | None],
    claim_statuses: Iterable[str | None],
    *,
    contradicted_statuses: tuple[str, ...] = ("contradicted",),
    default_score: float | None = None,
) -> float | None:
    """Aggregate claim scores while letting decisive contradictions dominate."""
    paired = [
        (clip_score(score), (status or "").strip().lower())
        for score, status in zip(claim_scores, claim_statuses)
        if score is not None
    ]
    if not paired:
        return clip_score(default_score)

    values = [score for score, _status in paired if score is not None]
    mean_score = clip_score(sum(values) / len(values))
    contradiction_values = [
        score
        for score, status in paired
        if status in {item.lower() for item in contradicted_statuses}
    ]
    if contradiction_values:
        return clip_score(max([mean_score or 0.0, *contradiction_values]))
    return mean_score


def classify_support(
    semantic_similarity: float,
    lexical_overlap_score: float,
    contradiction_cues: list[str] | None = None,
) -> dict[str, float | str | None]:
    """Map local similarity features into a simple support judgment."""
    contradiction_cues = contradiction_cues or []
    combined_support = (0.7 * semantic_similarity) + (0.3 * lexical_overlap_score)

    if contradiction_cues and (
        semantic_similarity >= 0.45
        or lexical_overlap_score >= 0.20
        or combined_support >= 0.40
    ):
        status = "contradicted"
    elif combined_support >= 0.70 or (
        semantic_similarity >= 0.78 and lexical_overlap_score >= 0.18
    ) or (
        semantic_similarity >= 0.58 and lexical_overlap_score >= 0.60
    ):
        status = "supported"
    elif combined_support >= 0.50 or (
        semantic_similarity >= 0.62 and lexical_overlap_score >= 0.18
    ):
        status = "weakly supported"
    elif combined_support >= 0.28 or semantic_similarity >= 0.35:
        status = "unclear"
    else:
        status = "unsupported"

    return {
        "status": status,
        "risk_score": STATUS_TO_RISK[status],
        "combined_support": round(combined_support, 3),
    }


def make_method_result(
    method_name: str,
    family: str,
    risk_score: float | None,
    summary: str,
    evidence_used: str,
    claim_findings: list[dict[str, Any]] | None,
    details: dict[str, Any] | None,
    limitations: str,
    explanation: str | None = None,
    **extra_fields: Any,
) -> dict[str, Any]:
    """Create a consistent method result payload.

    This legacy helper still accepts ``risk_score`` on the historical ``0-100``
    scale so older method implementations keep working during the refactor.
    """
    score = risk_to_prob(clip_score(risk_score))
    confidence = None if score is None else max(0.05, min(0.95, 1.0 - abs(score - 0.5)))
    return make_result(
        method_name=method_name,
        family=family,
        score=score,
        confidence=confidence,
        summary=summary,
        explanation=explanation or summary,
        evidence_used=evidence_used,
        claim_findings=claim_findings,
        details=details,
        limitations=limitations,
        **extra_fields,
    )


def unavailable_result(
    method_name: str,
    family: str,
    summary: str,
    evidence_used: str,
    limitations: str,
    claim_findings: list[dict[str, Any]] | None = None,
    details: dict[str, Any] | None = None,
    **extra_fields: Any,
) -> dict[str, Any]:
    """Create a standard response when a method cannot run honestly."""
    unavailable_details = {"available": False, **(details or {})}
    return make_unavailable(
        method_name=method_name,
        family=family,
        summary=summary,
        explanation=summary,
        evidence_used=evidence_used,
        limitations=limitations,
        claim_findings=claim_findings,
        details=unavailable_details,
        **extra_fields,
    )
