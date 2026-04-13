"""Shared detector result schema used across the prototype.

The app keeps a few legacy fields such as ``risk_score`` for UI compatibility,
but the canonical cross-method schema is:

- ``method_name``
- ``family``
- ``score`` in ``[0, 1]`` where higher means higher hallucination risk
- ``label``
- ``confidence`` in ``[0, 1]``
- ``explanation``
- ``evidence``
- ``citations``
- ``intermediate_steps``
- ``revised_answer``
- ``latency_ms``
- ``metadata``
"""

from typing import Any


LOW_RISK_MAX = 0.33
MEDIUM_RISK_MAX = 0.66


def clip_probability(value: float | int | None) -> float | None:
    """Clamp a value to the inclusive ``[0, 1]`` interval."""
    if value is None:
        return None
    return float(max(0.0, min(1.0, float(value))))


def risk_to_prob(score: float | int | None) -> float | None:
    """Convert a legacy ``0-100`` risk score into the shared ``0-1`` scale."""
    if score is None:
        return None
    return clip_probability(float(score) / 100.0)


def prob_to_risk(score: float | int | None) -> float | None:
    """Convert a shared ``0-1`` risk score into the legacy ``0-100`` scale."""
    clipped = clip_probability(score)
    if clipped is None:
        return None
    return round(clipped * 100.0, 1)


def probability_to_label(score: float | int | None) -> str:
    """Map the shared ``0-1`` risk score to a qualitative label."""
    clipped = clip_probability(score)
    if clipped is None:
        return "Not Available"
    if clipped <= LOW_RISK_MAX:
        return "Low"
    if clipped <= MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


def _prepare_metadata(
    metadata: dict[str, Any] | None,
    details: dict[str, Any] | None,
    impl_status: str,
) -> dict[str, Any]:
    merged = {**(details or {}), **(metadata or {})}
    merged.setdefault("implementation_status", impl_status)
    merged.setdefault("schema_version", "v2")
    return merged


def make_result(
    *,
    method_name: str,
    family: str,
    score: float | None,
    confidence: float | None,
    summary: str,
    explanation: str,
    evidence_used: str,
    limitations: str,
    impl_status: str = "implemented",
    evidence: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    intermediate_steps: list[dict[str, Any]] | None = None,
    claim_findings: list[dict[str, Any]] | None = None,
    revised_answer: str | None = None,
    latency_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    **extra_fields: Any,
) -> dict[str, Any]:
    """Build a detector result following the shared schema."""
    norm_score = clip_probability(score)
    norm_conf = clip_probability(confidence)
    meta = _prepare_metadata(metadata, details, impl_status)
    label = probability_to_label(norm_score)

    result = {
        "method_name": method_name,
        "family": family,
        "score": norm_score,
        "label": label,
        "confidence": norm_conf,
        "summary": summary,
        "explanation": explanation,
        "evidence_used": evidence_used,
        "evidence": evidence or [],
        "citations": citations or [],
        "intermediate_steps": intermediate_steps or [],
        "claim_findings": claim_findings or [],
        "revised_answer": revised_answer,
        "latency_ms": None if latency_ms is None else round(float(latency_ms), 2),
        "metadata": meta,
        "details": meta,
        "limitations": limitations,
        "implementation_status": impl_status,
        "available": norm_score is not None,
        "runtime_status": "completed",
        "risk_score": prob_to_risk(norm_score),
        "risk_label": label,
    }
    result.update(extra_fields)
    return result


def make_unavailable(
    *,
    method_name: str,
    family: str,
    summary: str,
    explanation: str,
    evidence_used: str,
    limitations: str,
    impl_status: str = "unavailable",
    evidence: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    intermediate_steps: list[dict[str, Any]] | None = None,
    claim_findings: list[dict[str, Any]] | None = None,
    revised_answer: str | None = None,
    latency_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    **extra_fields: Any,
) -> dict[str, Any]:
    """Build a result when a detector cannot run honestly."""
    meta = _prepare_metadata(metadata, details, impl_status)
    meta.setdefault("available", False)

    result = {
        "method_name": method_name,
        "family": family,
        "score": None,
        "label": "Not Available",
        "confidence": None,
        "summary": summary,
        "explanation": explanation,
        "evidence_used": evidence_used,
        "evidence": evidence or [],
        "citations": citations or [],
        "intermediate_steps": intermediate_steps or [],
        "claim_findings": claim_findings or [],
        "revised_answer": revised_answer,
        "latency_ms": None if latency_ms is None else round(float(latency_ms), 2),
        "metadata": meta,
        "details": meta,
        "limitations": limitations,
        "implementation_status": impl_status,
        "available": False,
        "runtime_status": "unavailable",
        "risk_score": None,
        "risk_label": "Not Available",
    }
    result.update(extra_fields)
    return result
