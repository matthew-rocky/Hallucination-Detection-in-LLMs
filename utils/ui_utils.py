"""UI helper utilities for the Streamlit hallucination-detector prototype."""

import json

import pandas as pd

from ui.method_descriptions import METHOD_PROFILES, PROFILE_FIELD_ORDER, get_method_caption, get_method_profile
from utils.text_utils import safe_text, truncate_text


def profile_items(method_name: str) -> list[tuple[str, str]]:
    """Return profile fields in a stable display order."""
    profile = get_method_profile(method_name)
    return [(label, profile[field_key]) for label, field_key in PROFILE_FIELD_ORDER]


def format_score(score: float | None, label: str) -> str:
    """Format a risk score for table or metric display."""
    if label == "Not Available" or score is None:
        return "Not Available"
    return f"{score:.1f} / 100"


def format_confidence(confidence: float | None) -> str:
    """Format a confidence value for the UI."""
    if confidence is None:
        return "N/A"
    return f"{confidence:.2f}"


def format_status(result: dict, default_status: str = "N/A") -> str:
    """Format implementation status using the app's small runtime vocabulary."""
    fallback_mode = bool(result.get("metadata", {}).get("fallback_mode"))
    available = result.get("available")
    profile_status = str(default_status or result.get("implementation_status") or "N/A").replace("_", " ").title()
    if available is False:
        return "Unavailable"
    if fallback_mode:
        if profile_status in {"Implemented", "Approximate"}:
            return f"{profile_status} (Fallback)"
        return "Approximate (Fallback)"
    if profile_status in {"Implemented", "Approximate"}:
        return profile_status
    runtime_status = str(result.get("implementation_status") or default_status or "N/A").replace("_", " ").title()
    return runtime_status


def format_origin(result: dict) -> str:
    """Format the runtime path used to produce a result."""
    metadata = result.get("metadata", {})
    explicit_label = safe_text(metadata.get("result_origin_label", ""))
    if explicit_label:
        return explicit_label
    if metadata.get("fallback_mode"):
        return "Deterministic fallback approximation"
    if metadata.get("probe_loaded"):
        return "SEP-lite probe path"
    if metadata.get("backend_available"):
        return "Full backend scoring"
    return ""


def format_backend(result: dict) -> str:
    """Format a compact backend status label for internal methods."""
    metadata = result.get("metadata", {})
    model_name = safe_text(metadata.get("backend_model_name") or metadata.get("hf_model_name") or metadata.get("model_name"))
    if not model_name:
        return ""
    if metadata.get("backend_available") is False:
        return f"HF backend unavailable - {model_name}"
    return f"HF backend - {model_name}"


def build_compare_table(results: list[dict]) -> pd.DataFrame:
    """Build the side-by-side comparison table shown after analysis."""
    rows = []
    for result in results:
        profile = get_method_profile(result["method_name"])
        rows.append(
            {
                "Method": result["method_name"],
                "Implementation": format_status(result, profile["implementation"]),
                "Run Origin": format_origin(result),
                "Risk Score": format_score(result.get("risk_score"), result.get("risk_label", "")),
                "Risk Level": result.get("risk_label", "Not Available"),
                "Confidence": format_confidence(result.get("confidence")),
                "Backend": format_backend(result),
                "Evidence Source": profile["evidence_source"],
                "Inference Stage": profile["inference_stage"],
                "Main Strength": profile["main_strength"],
                "Main Weakness": profile["main_weakness"],
            }
        )
    return pd.DataFrame(rows)


def build_claim_table(claim_findings: list[dict]) -> pd.DataFrame:
    """Create a compact table for claim-level findings."""
    rows = []
    for finding in claim_findings:
        score = finding.get("score")
        rows.append(
            {
                "Claim": truncate_text(finding.get("claim", ""), 110),
                "Status": finding.get("status", ""),
                "Claim-Level Risk": "Not checked" if score is None else round(float(score), 1),
                "Best Match": truncate_text(finding.get("best_match", ""), 110),
                "Reason": truncate_text(finding.get("reason", ""), 140),
            }
        )
    return pd.DataFrame(rows)


def build_signal_table(sub_signals: list[dict]) -> pd.DataFrame:
    """Create a compact table for internal-signal sub-signals."""
    rows = []
    for signal in sub_signals:
        rows.append(
            {
                "Sub-signal": signal.get("signal", ""),
                "Value": signal.get("value", ""),
                "Risk Contribution": signal.get("risk", ""),
                "Explanation": truncate_text(signal.get("explanation", ""), 160),
            }
        )
    return pd.DataFrame(rows)


def build_trace_table(top_evidence: list[dict]) -> pd.DataFrame:
    """Create a claim-level retrieval trace table."""
    rows = []
    for item in top_evidence:
        top_chunks = item.get("retrieved_chunks", [])
        top_chunk_ids = ", ".join(chunk.get("chunk_id", "") for chunk in top_chunks[:3])
        rows.append(
            {
                "Claim": truncate_text(item.get("claim", ""), 110),
                "Selected Chunk": item.get("selected_best_chunk_id", ""),
                "Selected Source": item.get("selected_best_source_label", ""),
                "Selected Page": truncate_text(item.get("selected_best_page_title", ""), 80),
                "Selected Status": item.get("selected_best_status", ""),
                "Top Retrieved Chunks": top_chunk_ids,
            }
        )
    return pd.DataFrame(rows)


def build_chunk_table(chunks: list[dict]) -> pd.DataFrame:
    """Create a compact table of available local chunks."""
    rows = []
    for chunk in chunks:
        rows.append(
            {
                "Chunk ID": chunk.get("chunk_id", ""),
                "Document": truncate_text(chunk.get("title", ""), 80),
                "Source": chunk.get("source_label", chunk.get("source_type", "")),
                "Text": truncate_text(chunk.get("text", ""), 180),
            }
        )
    return pd.DataFrame(rows)


def build_vq_table(questions: list[dict]) -> pd.DataFrame:
    """Create a compact table for generated verification questions."""
    rows = []
    for item in questions:
        rows.append(
            {
                "Claim": truncate_text(item.get("claim", ""), 110),
                "Verification Question": truncate_text(item.get("verification_question", ""), 160),
                "Status": item.get("status", ""),
            }
        )
    return pd.DataFrame(rows)


def build_answer_table(answers: list[dict]) -> pd.DataFrame:
    """Create a compact table for CoVe independent answers."""
    rows = []
    for item in answers:
        rows.append(
            {
                "Claim": truncate_text(item.get("claim", ""), 110),
                "Verification Question": truncate_text(item.get("verification_question", ""), 140),
                "Independent Answer": truncate_text(item.get("independent_answer", ""), 160),
                "Status": item.get("status", ""),
            }
        )
    return pd.DataFrame(rows)


def build_check_table(checks: list[dict]) -> pd.DataFrame:
    """Create a compact table for CRITIC-lite tool routes."""
    rows = []
    for item in checks:
        rows.append(
            {
                "Claim": truncate_text(item.get("claim", ""), 110),
                "Tools": ", ".join(item.get("tools", [])) if isinstance(item.get("tools"), list) else str(item.get("tools", "")),
            }
        )
    return pd.DataFrame(rows)


def build_evidence_table(evidence_items: list[dict]) -> pd.DataFrame:
    """Create a compact table for evidence objects."""
    rows = []
    for item in evidence_items:
        rows.append(
            {
                "Evidence ID": item.get("evidence_id", ""),
                "Document": truncate_text(item.get("title", ""), 80),
                "Source": item.get("source_type", ""),
                "Score": item.get("score", ""),
                "Content": truncate_text(item.get("content", ""), 180),
            }
        )
    return pd.DataFrame(rows)


def build_citation_table(citations: list[dict]) -> pd.DataFrame:
    """Create a compact table for citations."""
    rows = []
    for item in citations:
        rows.append(
            {
                "Citation": item.get("citation_id", ""),
                "Document": truncate_text(item.get("title", ""), 80),
                "Score": item.get("score", ""),
                "Snippet": truncate_text(item.get("snippet", ""), 180),
            }
        )
    return pd.DataFrame(rows)


def build_step_table(steps: list[dict]) -> pd.DataFrame:
    """Create a compact stage summary table."""
    rows = []
    for step in steps:
        output = step.get("output")
        if isinstance(output, str):
            preview = truncate_text(output, 140)
        else:
            preview = truncate_text(json.dumps(output, default=str), 140)
        rows.append({"Stage": step.get("stage", ""), "Preview": preview})
    return pd.DataFrame(rows)

