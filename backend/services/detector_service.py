"""Service wrapper around the existing detector method implementations."""

from __future__ import annotations

from typing import Any

from data.sample_cases import list_cases, sample_pairs_for
from detectors.base import make_unavailable
from methods.cove_check import run_cove
from methods.critic_check import run_critic
from methods.internal_check import run_internal
from methods.rag_check import run_rag
from methods.retrieval_check import run_retrieval
from methods.sep_check import run_sep
from methods.source_check import run_source
from methods.verify_flow import run_verify
from retrieval.chunking import load_doc_bytes
from ui.input_forms import FIELD_STATE_KEYS, validate_inputs
from ui.method_descriptions import (
    FIELD_SPECS,
    METHOD_ORDER,
    get_method_profile,
    method_meta,
    optional_fields_for,
    required_fields_for,
    visible_fields_for,
)
from utils.text_utils import has_text


METHOD_RUNNERS = {
    "Internal-Signal Baseline": run_internal,
    "SEP-Inspired Internal Signal": run_sep,
    "Source-Grounded Consistency": run_source,
    "Retrieval-Grounded Checker": run_retrieval,
    "RAG Grounded Check": run_rag,
    "Verification-Based Workflow": run_verify,
    "CoVe-Style Verification": run_cove,
    "CRITIC-lite Tool Check": run_critic,
}

METHOD_COLORS = {
    "Internal-Signal Baseline": "cyan",
    "SEP-Inspired Internal Signal": "purple",
    "Source-Grounded Consistency": "green",
    "Retrieval-Grounded Checker": "teal",
    "RAG Grounded Check": "amber",
    "Verification-Based Workflow": "orange",
    "CoVe-Style Verification": "indigo",
    "CRITIC-lite Tool Check": "rose",
}

DOC_METHODS = {
    method_name for method_name in METHOD_ORDER if method_meta(method_name).get("supports_uploads")
}


def get_methods() -> list[dict[str, Any]]:
    """Return frontend-friendly metadata for every supported method."""
    methods = []
    for method_name in METHOD_ORDER:
        metadata = method_meta(method_name)
        profile = get_method_profile(method_name)
        methods.append(
            {
                "id": _method_id(method_name),
                "name": method_name,
                "family": metadata.get("family", ""),
                "how_it_works": metadata.get("how_it_works", ""),
                "short_purpose": metadata.get("short_purpose", ""),
                "best_for": metadata.get("best_for", ""),
                "required_fields": metadata.get("required_fields", []),
                "optional_fields": metadata.get("optional_fields", []),
                "visible_fields": metadata.get("visible_fields", []),
                "ignored_fields": metadata.get("ignored_fields", []),
                "supports_uploads": bool(metadata.get("supports_uploads")),
                "implementation": metadata.get("implementation", ""),
                "caption": metadata.get("caption", ""),
                "input_requirements": {
                    "required": required_fields_for([method_name]),
                    "optional": optional_fields_for([method_name]),
                    "visible": visible_fields_for([method_name]),
                },
                "strengths": [metadata.get("main_strength", "")],
                "weaknesses": [metadata.get("main_weakness", "")],
                "recommended_use": metadata.get("best_for", ""),
                "color": METHOD_COLORS.get(method_name, "cyan"),
                "tone": METHOD_COLORS.get(method_name, "cyan"),
                "profile": profile,
            }
        )
    return methods


def _method_id(method_name: str) -> str:
    """Build a stable API id from a display method name."""
    return (
        method_name.lower()
        .replace(" / ", "-")
        .replace(" ", "-")
        .replace("/", "-")
        .replace("style", "style")
        .replace("--", "-")
    )


def get_field_specs() -> dict[str, dict[str, Any]]:
    """Expose input labels and placeholders to the dashboard."""
    return FIELD_SPECS


def get_samples() -> list[dict[str, Any]]:
    """Return curated sample cases."""
    samples = []
    for case in list_cases():
        question = case.get("question", "")
        answer = case.get("answer", "")
        evidence = case.get("evidence_text", "")
        source = case.get("source_text", "")
        samples.append(
            {
                **case,
                "recommended_methods": case.get("method_targets", []),
                "sampled_answers_text": case.get("answer_samples", ""),
                "question_preview": _preview(question),
                "answer_preview": _preview(answer),
                "evidence_preview": _preview(evidence or source),
                "available_inputs": [
                    key
                    for key in ("question", "answer", "source_text", "evidence_text", "sampled_answers_text")
                    if has_text(case.get("answer_samples" if key == "sampled_answers_text" else key, ""))
                ],
            }
        )
    return samples


def _preview(text: str, limit: int = 150) -> str:
    """Return a short single-line preview for frontend cards."""
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}..."


def get_sample_pairs() -> dict[str, dict[str, Any]]:
    """Return low/high curated sample pairs per method."""
    return {method_name: sample_pairs_for(method_name) for method_name in METHOD_ORDER}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact response summary for dashboards."""
    labels = [str(result.get("risk_label") or result.get("label") or "") for result in results]
    risks = [float(result["risk_score"]) for result in results if result.get("risk_score") is not None]
    return {
        "method_count": len(results),
        "claims_checked": sum(len(result.get("claim_findings") or []) for result in results),
        "low": sum(1 for label in labels if label == "Low"),
        "medium": sum(1 for label in labels if label == "Medium"),
        "high": sum(1 for label in labels if label == "High"),
        "avg_risk": round(sum(risks) / len(risks), 2) if risks else 0.0,
    }


def order_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort results using the stable method order used by the original UI."""
    order_lookup = {method_name: index for index, method_name in enumerate(METHOD_ORDER)}
    return sorted(results, key=lambda item: order_lookup.get(item.get("method_name", ""), len(order_lookup)))


def normalize_selected_methods(mode: str, selected_methods: list[str]) -> list[str]:
    """Resolve and validate method selection for quick and compare modes."""
    cleaned = [method for method in selected_methods if method in METHOD_RUNNERS]
    if not cleaned:
        cleaned = [METHOD_ORDER[0]]
    if mode == "quick":
        return cleaned[:1]
    return cleaned


def validate_analysis_payload(selected_methods: list[str], payload: dict[str, str]) -> list[str]:
    """Reuse the Streamlit validation rules against a plain dictionary."""
    state = {
        "question_input": payload.get("question", ""),
        "answer_input": payload.get("answer", ""),
        "source_input": payload.get("source_text", ""),
        "evidence_input": payload.get("evidence_text", ""),
        "sampled_answers_input": payload.get("sampled_answers_text", ""),
    }
    return validate_inputs(selected_methods, state)


def runner_args(method_name: str, payload: dict[str, str], uploaded_documents: list | None = None) -> dict[str, Any]:
    """Build clean per-method kwargs so hidden fields do not leak into a run."""
    metadata = method_meta(method_name)
    visible_fields = set(metadata.get("visible_fields", []))
    kwargs: dict[str, Any] = {
        "question": payload.get("question", "") if "question" in visible_fields else "",
        "answer": payload.get("answer", ""),
        "source_text": payload.get("source_text", "") if "source_text" in visible_fields else "",
        "evidence_text": payload.get("evidence_text", "") if "evidence_text" in visible_fields else "",
        "sampled_answers_text": payload.get("sampled_answers_text", "") if "sampled_answers" in visible_fields else "",
    }
    if method_name in DOC_METHODS:
        kwargs["extra_documents"] = uploaded_documents or []
    if method_name == "Verification-Based Workflow":
        kwargs["allow_web"] = False
    return kwargs


def normalize_result(method_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Ensure each runner returns the shared detector shape."""
    if not isinstance(result, dict):
        raise TypeError(f"{method_name} returned {type(result).__name__} instead of a result dictionary.")
    if "method_name" not in result:
        raise ValueError(f"{method_name} did not return the shared detector schema.")
    return result


def error_result(method_name: str, exc: Exception) -> dict[str, Any]:
    """Convert a method-level failure into a stable unavailable result."""
    metadata = method_meta(method_name)
    return make_unavailable(
        method_name=method_name,
        family=metadata.get("family", "unknown"),
        summary=f"{method_name} hit a runtime error and could not complete this run.",
        explanation=(
            "The API caught a method-level runtime error and returned an unavailable result instead of "
            f"failing the whole request. Error: {type(exc).__name__}: {exc}"
        ),
        evidence_used="No result trace was produced because execution stopped before the method finished.",
        limitations="Retry after the underlying input or dependency issue is fixed.",
        impl_status="unavailable",
        metadata={
            "api_runtime_guard": True,
            "runtime_error": str(exc),
            "runtime_error_type": type(exc).__name__,
            "backend_status": "error",
            "backend_status_label": "Runtime error",
            "result_origin": "api_runtime_guard",
            "result_origin_label": "API runtime guard",
        },
    )


def run_analysis(
    *,
    mode: str,
    selected_methods: list[str],
    payload: dict[str, str],
    uploaded_documents: list | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run selected methods and return ordered shared-schema results."""
    active_methods = normalize_selected_methods(mode, selected_methods)
    warnings = validate_analysis_payload(active_methods, payload)
    if warnings:
        return [], warnings

    documents = uploaded_documents or []
    if documents and not has_text(payload.get("evidence_text", "")):
        payload = {
            **payload,
            "evidence_text": "\n\n".join(f"[{doc.title}]\n{doc.text}" for doc in documents if has_text(doc.text)),
        }

    results = []
    for method_name in active_methods:
        try:
            runner = METHOD_RUNNERS[method_name]
            result = normalize_result(method_name, runner(**runner_args(method_name, payload, documents)))
        except Exception as exc:
            result = error_result(method_name, exc)
        results.append(result)
    return order_results(results), []


def visible_fields(selected_methods: list[str]) -> list[str]:
    """Return the dynamic input field union for a selected method set."""
    return visible_fields_for(selected_methods)


async def load_upload_documents(files: list[Any] | None) -> tuple[list[Any], list[str]]:
    """Read uploaded files into retrieval SourceDoc objects."""
    documents = []
    warnings = []
    for file in files or []:
        try:
            documents.append(load_doc_bytes(file.filename, await file.read()))
        except Exception as exc:
            warnings.append(f"{file.filename}: {exc}")
    return documents, warnings
