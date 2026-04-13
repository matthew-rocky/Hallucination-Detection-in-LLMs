"""Canonical entry point for the RAG-style grounded checker."""

from detectors.base import probability_to_label, risk_to_prob
from detectors.retrieval_check import run_retrieval_check
from utils.scoring_utils import score_with_conflicts
from utils.text_utils import safe_text, truncate_text


METHOD_NAME = "RAG Grounded Check"
FAMILY = "retrieval-grounded / RAG-style"
SUPPORTED_STATUSES = {"supported", "abstractly_supported", "weakly_supported"}


def _classify_rag_claim(step: dict, finding: dict) -> dict:
    status = safe_text(finding.get("status")).lower()
    hits = step.get("retrieved_hits") or []
    best_hit = hits[0] if hits else {}
    best_text = safe_text(best_hit.get("text")) or safe_text(finding.get("best_match")) or "No retrieved evidence."
    best_chunk_id = safe_text(best_hit.get("chunk_id"))
    retrieval_strength = float(best_hit.get("score", 0.0) or 0.0)
    contradiction_cues = best_hit.get("contradiction_cues") or []
    chunk_label = f"retrieved chunk {best_chunk_id}" if best_chunk_id else "the top retrieved evidence"
    formatted_best_match = (f"[{best_chunk_id}] " if best_chunk_id else "") + truncate_text(best_text, 320)

    if status == "contradicted":
        cue_text = ", ".join(contradiction_cues[:2]) or "retrieved contradiction cues"
        return {
            "status": "contradicted",
            "score": 97.0,
            "best_match": formatted_best_match,
            "reason": f"{chunk_label} contradicted this claim after retrieval: {cue_text}.",
            "retrieval_diagnosis": "contradicted_by_retrieved_evidence",
            "retrieval_strength": round(retrieval_strength, 4),
        }

    if status in SUPPORTED_STATUSES:
        if status == "weakly_supported":
            return {
                "status": "weakly supported",
                "score": 30.0,
                "best_match": formatted_best_match,
                "reason": f"{chunk_label} was retrieved and stayed relevant, but the grounding remained partial rather than decisive.",
                "retrieval_diagnosis": "partially_supported_after_retrieval",
                "retrieval_strength": round(retrieval_strength, 4),
            }
        return {
            "status": "supported",
            "score": 10.0 if status == "supported" else 16.0,
            "best_match": formatted_best_match,
            "reason": f"{chunk_label} grounded this claim after retrieval.",
            "retrieval_diagnosis": "grounded_by_retrieved_evidence",
            "retrieval_strength": round(retrieval_strength, 4),
        }

    if not hits or retrieval_strength < 0.16:
        return {
            "status": "unsupported",
            "score": 58.0,
            "best_match": formatted_best_match,
            "reason": "Retrieval did not surface strong enough evidence for this claim, so the method treats it as weakly grounded due to retrieval failure rather than decisive contradiction.",
            "retrieval_diagnosis": "retrieval_failure_or_weak_evidence",
            "retrieval_strength": round(retrieval_strength, 4),
        }

    return {
        "status": "unsupported",
        "score": 72.0,
        "best_match": formatted_best_match,
        "reason": f"Relevant evidence was retrieved, but {chunk_label} still did not support the claim.",
        "retrieval_diagnosis": "unsupported_despite_retrieval",
        "retrieval_strength": round(retrieval_strength, 4),
    }


def _build_rag_summary(total_claims: int, counts: dict[str, int]) -> str:
    if counts["contradicted"]:
        return (
            f"RAG-style verification retrieved evidence first, then checked {total_claims} answer claim(s): "
            f"{counts['supported']} grounded, {counts['contradicted']} contradicted, "
            f"{counts['unsupported']} unsupported, and {counts['retrieval_failure']} retrieval-limited."
        )
    if counts["unsupported"] or counts["retrieval_failure"]:
        return (
            f"RAG-style verification retrieved supporting passages for the answer, but some claims remained ungrounded "
            f"({counts['unsupported']} unsupported after retrieval, {counts['retrieval_failure']} retrieval-limited)."
        )
    return f"RAG-style verification retrieved evidence and grounded the extracted claims with no contradictions detected."


def _rag_explanation(total_claims: int, counts: dict[str, int], backend: str) -> str:
    return (
        f"The method retrieved top evidence chunks with the local {backend} backend before checking each of the {total_claims} extracted claim(s). "
        f"It distinguishes contradictions from simple lack of support, and also separates unsupported claims from cases where retrieval itself stayed weak."
    )


def run_rag(
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    sampled_answers_text: str = "",
    top_k: int = 5,
    extra_documents: list | None = None,
) -> dict:
    """Run the RAG-framed grounded checker over the local retrieval stack."""
    del sampled_answers_text
    result = run_retrieval_check(
        question=question,
        answer=answer,
        source_text=source_text,
        evidence_text=evidence_text,
        extra_documents=extra_documents,
        method_name=METHOD_NAME,
        family=FAMILY,
        top_k=top_k,
        impl_status="approximate",
    )
    if not result.get("available"):
        return result

    original_findings = result.get("claim_findings") or []
    original_steps = result.get("intermediate_steps") or []
    rag_claim_findings = []
    counts = {
        "supported": 0,
        "contradicted": 0,
        "unsupported": 0,
        "retrieval_failure": 0,
    }

    for finding, step in zip(original_findings, original_steps):
        rag_finding = {
            "claim": finding.get("claim", ""),
            **_classify_rag_claim(step, finding),
        }
        diagnosis = rag_finding["retrieval_diagnosis"]
        if rag_finding["status"] == "contradicted":
            counts["contradicted"] += 1
        elif rag_finding["status"] in {"supported", "weakly supported"}:
            counts["supported"] += 1
        elif diagnosis == "retrieval_failure_or_weak_evidence":
            counts["retrieval_failure"] += 1
        else:
            counts["unsupported"] += 1
        rag_claim_findings.append(rag_finding)

    risk_score = score_with_conflicts(
        [item["score"] for item in rag_claim_findings],
        [item["status"] for item in rag_claim_findings],
    )
    result["claim_findings"] = rag_claim_findings
    result["summary"] = _build_rag_summary(len(rag_claim_findings), counts)
    result["explanation"] = _rag_explanation(
        len(rag_claim_findings),
        counts,
        safe_text((result.get("metadata") or {}).get("retrieval_backend")) or "retrieval",
    )
    result["score"] = risk_to_prob(risk_score)
    result["risk_score"] = risk_score
    result["label"] = probability_to_label(result["score"])
    result["risk_label"] = result["label"]
    result["metadata"] = {
        **(result.get("metadata") or {}),
        "rag_style_claim_verification": True,
        "supported_claim_count": counts["supported"],
        "contradicted_claim_count": counts["contradicted"],
        "unsupported_claim_count": counts["unsupported"],
        "retrieval_failure_claim_count": counts["retrieval_failure"],
    }
    result["details"] = result["metadata"]
    result["intermediate_steps"] = original_steps + [
        {
            "stage": "rag_claim_aggregation",
            "output": {
                "supported_claim_count": counts["supported"],
                "contradicted_claim_count": counts["contradicted"],
                "unsupported_claim_count": counts["unsupported"],
                "retrieval_failure_claim_count": counts["retrieval_failure"],
                "risk_score": risk_score,
            },
        }
    ]
    return result


