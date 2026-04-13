"""Streamlit app for the hallucination-detector demo."""

import os

# Headless safe.
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    st = None

    def get_script_run_ctx():
        return None

from data.sample_cases import case_by_id, list_cases
from detectors.base import make_unavailable
from detectors.signal import SignalConfig, get_signal_status
from methods.cove_check import run_cove
from methods.critic_check import run_critic
from methods.internal_check import run_internal
from methods.rag_check import run_rag
from methods.retrieval_check import run_retrieval
from methods.sep_check import run_sep
from methods.source_check import run_source
from methods.verify_flow import run_verify
from retrieval.chunking import load_doc_bytes
from ui.comparison_table import render_compare
from ui.input_forms import (
    apply_pending_sample,
    read_uploads as ui_uploads,
    init_session as init_ui,
    load_demo as load_ui_sample,
    render_input_fields,
    render_samples,
    validate_inputs,
    uploaded_text,
)
from ui.layout import inject_base_styles, render_hero, render_section_intro
from ui.method_descriptions import METHOD_ORDER, method_meta
from ui.method_selector import render_methods, render_mode_toggle
from ui.result_cards import render_details, render_overview
from utils.text_utils import has_text


if st is not None:
    st.set_page_config(page_title="Hallucination Detection Studio (DEMO)", layout="wide")


# Label -> runner.
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
# Upload methods.
DOC_METHODS = {
    method_name for method_name in METHOD_ORDER if method_meta(method_name).get("supports_uploads")
}
SAMPLE_CASES = list_cases()
SAMPLE_CASE_LOOKUP = {case["id"]: case for case in SAMPLE_CASES}
DEFAULT_CASE_ID = SAMPLE_CASES[0]["id"] if SAMPLE_CASES else ""
SIGNAL_METHODS = {"Internal-Signal Baseline", "SEP-Inspired Internal Signal"}


def _require_streamlit() -> None:
    if st is None:
        raise RuntimeError("streamlit is required to run the demo UI, but it is not installed in this environment.")


def init_session() -> None:
    """Backwards-compatible wrapper around the UI session-state initializer."""
    init_ui(st, DEFAULT_CASE_ID)


def load_demo(case_id: str, method_names: list[str] | None = None) -> None:
    """Backwards-compatible sample loader used by the app."""
    case = SAMPLE_CASE_LOOKUP.get(case_id) or case_by_id(case_id)
    if not case:
        return
    # Keep selection.
    active_methods = method_names or st.session_state.get("selected_methods") or [
        st.session_state.get("selected_method", METHOD_ORDER[0])
    ]
    target_method = active_methods[0] if active_methods else METHOD_ORDER[0]
    load_ui_sample(st, case, active_methods, method_name=target_method)


def get_ordered_results(results: list[dict]) -> list[dict]:
    """Sort results using the stable method-order used throughout the UI."""
    # Stable order.
    order_lookup = {method_name: index for index, method_name in enumerate(METHOD_ORDER)}
    return sorted(results, key=lambda item: order_lookup.get(item.get("method_name", ""), len(order_lookup)))


def read_uploads() -> tuple[list, list[str]]:
    """Load uploaded documents into the retrieval format used by the methods."""
    return ui_uploads(st, load_doc_bytes)


def _runner_args(method_name: str, uploaded_documents: list) -> dict:
    """Build clean per-method kwargs so hidden fields do not leak into a run."""
    metadata = method_meta(method_name)
    visible_fields = set(metadata.get("visible_fields", []))
    # Hide unused fields.
    kwargs = {
        "question": st.session_state.get("question_input", "") if "question" in visible_fields else "",
        "answer": st.session_state.get("answer_input", ""),
        "source_text": st.session_state.get("source_input", "") if "source_text" in visible_fields else "",
        "evidence_text": st.session_state.get("evidence_input", "") if "evidence_text" in visible_fields else "",
        "sampled_answers_text": st.session_state.get("sampled_answers_input", "") if "sampled_answers" in visible_fields else "",
    }
    if method_name in DOC_METHODS:
        kwargs["extra_documents"] = uploaded_documents
    # No live web here.
    if method_name == "Verification-Based Workflow":
        kwargs["allow_live_web_retrieval"] = False
    return kwargs


def _error_result(method_name: str, exc: Exception) -> dict:
    """Convert an app-level method failure into a stable unavailable result."""
    metadata = method_meta(method_name)
    return make_unavailable(
        method_name=method_name,
        family=metadata.get("family", "unknown"),
        summary=f"{method_name} hit a runtime error and could not complete this run.",
        explanation=(
            "The app caught a method-level runtime error and kept the Streamlit session alive instead of "
            f"showing a full app exception. Error: {type(exc).__name__}: {exc}"
        ),
        evidence_used="No result trace was produced because execution stopped before the method finished.",
        limitations=(
            "This is an app-level runtime guard. The method needs to be retried after the underlying input or dependency issue is fixed."
        ),
        impl_status="unavailable",
        metadata={
            "app_runtime_guard": True,
            "runtime_error": str(exc),
            "runtime_error_type": type(exc).__name__,
            "backend_status": "error",
            "backend_status_label": "Runtime error",
            "result_origin": "app_runtime_guard",
            "result_origin_label": "App runtime guard",
        },
    )


def _normalize_result(method_name: str, result: dict) -> dict:
    """Ensure each runner returns the shared detector shape."""
    if not isinstance(result, dict):
        raise TypeError(f"{method_name} returned {type(result).__name__} instead of a result dictionary.")
    if "method_name" not in result:
        raise ValueError(f"{method_name} did not return the shared detector schema.")
    return result


def _show_signal_status(selected_methods: list[str] | None, results: list[dict] | None = None) -> bool:
    selected = set(selected_methods or [])
    if SIGNAL_METHODS & selected:
        return True
    return any(result.get("method_name") in SIGNAL_METHODS for result in (results or []))


def _signal_mode_label(results: list[dict], backend_status: dict) -> str:
    internal_results = [result for result in results if result.get("method_name") in SIGNAL_METHODS]
    if not internal_results:
        return "Real HF mode ready" if backend_status.get("backend_available") else "Fallback only"
    runtime_modes = {
        "fallback" if result.get("metadata", {}).get("fallback_mode") else "real"
        for result in internal_results
        if result.get("available", True)
    }
    if not runtime_modes:
        return "Real HF mode ready" if backend_status.get("backend_available") else "Fallback only"
    if runtime_modes == {"real"}:
        return "Real HF mode"
    if runtime_modes == {"fallback"}:
        return "Fallback mode"
    return "Mixed runtime modes"


def render_signal_status(selected_methods: list[str] | None, results: list[dict] | None = None) -> None:
    if not _show_signal_status(selected_methods, results):
        return
    backend_status = get_signal_status(SignalConfig())
    runtime_mode = _signal_mode_label(results or [], backend_status)
    status_line = (
        f"Internal backend: {backend_status['backend_status_label']} | "
        f"{backend_status['backend_model_name']} | {runtime_mode} | {backend_status['device']}"
    )
    if backend_status.get("backend_available"):
        st.caption(status_line)
    else:
        st.warning(f"Internal backend unavailable, using fallback. Model: {backend_status['backend_model_name']}")
    details = [
        f"Python: {backend_status['python_executable']}",
        f"Model: {backend_status['backend_model_name']}",
        f"torch: {backend_status.get('torch_version') or 'not installed'}",
        f"transformers: {backend_status.get('transformers_version') or 'not installed'}",
        f"Device: {backend_status['device']}",
        f"Mode: {runtime_mode}",
        f"Status: {backend_status['backend_status_label']}",
    ]
    if backend_status.get("backend_error"):
        details.append(f"Error: {backend_status['backend_error']}")
    with st.expander("Internal runtime", expanded=False):
        st.code("\n".join(details), language=None)


def run_methods(selected_methods: list[str] | None = None) -> None:
    """Execute the selected detector methods and store the results in session state."""
    active_methods = selected_methods or st.session_state.get("selected_methods", [])
    if not active_methods:
        st.session_state["analysis_results"] = []
        return

    uploaded_documents, warnings = read_uploads()
    st.session_state["upload_warnings"] = warnings

    # Use uploads as evidence.
    if uploaded_documents and not has_text(st.session_state.get("evidence_input", "")):
        document_text = uploaded_text(uploaded_documents)
        if document_text:
            st.session_state["evidence_input"] = document_text

    results = []
    for method_name in active_methods:
        runner = METHOD_RUNNERS[method_name]
        runner_kwargs = _runner_args(method_name, uploaded_documents)
        try:
            result = _normalize_result(method_name, runner(**runner_kwargs))
        # Isolate failures.
        except Exception as exc:
            result = _error_result(method_name, exc)
        results.append(result)
    st.session_state["analysis_results"] = results


def render_results(results: list[dict], compare_mode: bool) -> None:
    """Render the results section for quick or compare mode."""
    render_section_intro(
        st,
        "Results",
        "Start with the main signal, then open the trace sections only when you want the evidence and method internals.",
        kicker="Results",
    )
    # Table first.
    if compare_mode and len(results) > 1:
        render_compare(st, results)
        st.markdown("### Method details")
        for result in results:
            with st.expander(result["method_name"], expanded=False):
                render_overview(st, result)
                render_details(st, result)
        return

    result = results[0]
    render_overview(st, result)
    render_details(st, result)


def main() -> None:
    _require_streamlit()
    init_session()
    inject_base_styles(st)
    render_hero(st)

    render_section_intro(
        st,
        "Choose your workflow",
        "Start with one method for a quick read, or switch to comparison mode when you want to inspect several detection strategies side by side.",
        kicker="Mode",
    )
    compare_mode = render_mode_toggle(st) == "Compare Methods"

    render_section_intro(
        st,
        "Choose a method",
        "Every method card explains what it is best for and which inputs it needs, so you only fill what matters for that workflow.",
        kicker="Method",
    )
    selected_methods = render_methods(st, compare_mode)
    # Apply queued demo.
    apply_pending_sample(st.session_state)

    render_section_intro(
        st,
        "Provide inputs",
        "The form only shows fields used by the current selection. Load a curated low-risk or high-risk demo on the right when you want a fast teaching example.",
        kicker="Inputs",
    )
    input_left, input_right = st.columns([1.7, 1.15], gap="large")
    with input_left:
        render_input_fields(st, selected_methods, compare_mode)
    with input_right:
        render_samples(st, selected_methods, compare_mode)
        uploaded_documents, upload_warnings = read_uploads()
        if uploaded_documents:
            st.caption("Ready to index: " + ", ".join(document.title for document in uploaded_documents))
        if upload_warnings:
            for warning in upload_warnings:
                st.warning(f"Upload skipped: {warning}")
        if compare_mode:
            st.caption("Compare mode shows the union of fields needed by the selected methods.")
        else:
            method_name = selected_methods[0] if selected_methods else METHOD_ORDER[0]
            st.caption(f"Quick Check keeps the form focused on {method_name} only.")

    render_section_intro(
        st,
        "Run analysis",
        "Run the selected method or methods on the current form state. Results stay explicit about whether they come from implemented backends, approximations, or fallback paths.",
        kicker="Action",
    )
    render_signal_status(selected_methods, st.session_state.get("analysis_results", []))
    if st.button("Run analysis", type="primary", use_container_width=True):
        errors = validate_inputs(selected_methods, st.session_state)
        if errors:
            for message in errors:
                st.warning(message)
        else:
            with st.spinner("Running analysis..."):
                run_methods(selected_methods)

    results = get_ordered_results(st.session_state.get("analysis_results", []))
    if results:
        if st.session_state.get("upload_warnings"):
            st.warning("Some uploaded documents could not be indexed. See the messages above for details.")
        render_results(results, compare_mode)


if __name__ == "__main__":
    _require_streamlit()
    # CLI or Streamlit.
    if get_script_run_ctx() is None:
        from streamlit.web.bootstrap import run

        run(os.path.abspath(__file__), False, [], {})
    else:
        main()
