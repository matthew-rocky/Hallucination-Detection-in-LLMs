"""Dynamic input and demo-browser helpers for the Streamlit UI."""

from html import escape

from data.sample_cases import sample_pairs_for
from ui.method_descriptions import (
    FIELD_SPECS,
    METHOD_ORDER,
    method_meta,
    methods_requiring,
    methods_using,
    visible_fields_for,
)
from utils.text_utils import has_text, safe_text

SESSION_DEFAULTS = {
    "ui_mode": "Quick Check",
    "selected_method": METHOD_ORDER[0],
    "selected_methods": [METHOD_ORDER[0]],
    "question_input": "",
    "answer_input": "",
    "sampled_answers_input": "",
    "source_input": "",
    "evidence_input": "",
    "uploaded_evidence_files_key_version": 0,
    "analysis_results": [],
    "selected_sample_case_id": "",
    "loaded_sample_case_id": "",
    "loaded_sample_title": "",
    "loaded_sample_method": "",
    "loaded_sample_risk_level": "",
    "upload_warnings": [],
    "pending_sample_case_state": None,
}

FIELD_STATE_KEYS = {
    "question": "question_input",
    "answer": "answer_input",
    "sampled_answers": "sampled_answers_input",
    "source_text": "source_input",
    "evidence_text": "evidence_input",
}
FORM_STATE_KEYS = list(FIELD_STATE_KEYS.values())
UPLOAD_WIDGET_KEY = "uploaded_evidence_files"
UPLOAD_KEY_VERSION = "uploaded_evidence_files_key_version"


def _upload_key(session_state: dict) -> str:
    """Return the active uploader key so uploads can be reset without direct writes."""
    version = int(session_state.get(UPLOAD_KEY_VERSION, 0) or 0)
    return f"{UPLOAD_WIDGET_KEY}_{version}"


def _clear_uploads(session_state: dict) -> None:
    """Rotate the uploader key to clear selected files safely."""
    # Rotate upload key.
    current_version = int(session_state.get(UPLOAD_KEY_VERSION, 0) or 0)
    session_state[UPLOAD_KEY_VERSION] = current_version + 1
    session_state.pop(UPLOAD_WIDGET_KEY, None)


def _current_uploads(session_state: dict) -> list:
    """Read uploaded files from the active uploader key with a legacy fallback."""
    files = session_state.get(_upload_key(session_state))
    if files is None:
        files = session_state.get(UPLOAD_WIDGET_KEY)
    return files or []


def init_session(st, default_case_id: str = "") -> None:
    """Populate the Streamlit session state with stable defaults."""
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    if default_case_id:
        st.session_state.setdefault("selected_sample_case_id", default_case_id)


def _merge_source(source_text: str, evidence_text: str) -> str:
    """Fold source text into evidence text when the UI hides the source field."""
    parts = []
    if has_text(source_text):
        parts.append(f"[Source Passage]\n{source_text.strip()}")
    if has_text(evidence_text):
        parts.append(f"[Supporting Evidence]\n{evidence_text.strip()}")
    return "\n\n".join(parts)


def normalize_sample(case: dict, method_names: list[str]) -> dict[str, str]:
    """Adapt one sample case to the visible field set used in the current UI mode."""
    visible_fields = set(visible_fields_for(method_names))
    source_text = safe_text(case.get("source_text", ""))
    evidence_text = safe_text(case.get("evidence_text", ""))
    # Keep hidden source.
    if "source_text" not in visible_fields and has_text(source_text):
        evidence_text = _merge_source(source_text, evidence_text)
        source_text = ""

    return {
        "question_input": safe_text(case.get("question", "")) if "question" in visible_fields else "",
        "answer_input": safe_text(case.get("answer", "")) if "answer" in visible_fields else "",
        "sampled_answers_input": safe_text(case.get("answer_samples", "")) if "sampled_answers" in visible_fields else "",
        "source_input": source_text if "source_text" in visible_fields else "",
        "evidence_input": evidence_text if "evidence_text" in visible_fields else "",
    }


def make_sample_payload(case: dict, method_names: list[str], method_name: str | None = None) -> dict:
    """Build the queued session-state payload for one sample case."""
    active_method = method_name or (method_names[0] if method_names else "")
    return {
        "field_values": normalize_sample(case, method_names),
        "case_id": safe_text(case.get("id", "")),
        "title": safe_text(case.get("title", "")),
        "method_name": active_method,
        "risk_level": safe_text(case.get("risk_level", "")),
    }


def reset_input_state(session_state: dict) -> None:
    """Clear form inputs, sample labels, pending sample state, and prior results."""
    for key in FORM_STATE_KEYS:
        session_state[key] = ""
    _clear_uploads(session_state)
    session_state["analysis_results"] = []
    session_state["upload_warnings"] = []
    session_state.pop("pending_sample_case_state", None)
    session_state["selected_sample_case_id"] = ""
    session_state["loaded_sample_case_id"] = ""
    session_state["loaded_sample_title"] = ""
    session_state["loaded_sample_method"] = ""
    session_state["loaded_sample_risk_level"] = ""


def apply_sample_state(session_state: dict, state_update: dict[str, str], payload: dict | None = None) -> None:
    """Apply one normalized sample-case state update to session state."""
    # Clear old results.
    reset_input_state(session_state)
    for key, value in state_update.items():
        session_state[key] = value
    if payload:
        session_state["selected_sample_case_id"] = payload.get("case_id", "")
        session_state["loaded_sample_case_id"] = payload.get("case_id", "")
        session_state["loaded_sample_title"] = payload.get("title", "")
        session_state["loaded_sample_method"] = payload.get("method_name", "")
        session_state["loaded_sample_risk_level"] = payload.get("risk_level", "")


def load_demo(st, case: dict, method_names: list[str], method_name: str | None = None) -> None:
    """Load a built-in sample case into the current UI state."""
    payload = make_sample_payload(case, method_names, method_name=method_name)
    apply_sample_state(st.session_state, payload["field_values"], payload)


def queue_sample(st, case: dict, method_names: list[str], method_name: str | None = None) -> None:
    """Queue a sample-case update so it can be applied before widgets render on rerun."""
    # Before widgets lock.
    st.session_state["pending_sample_case_state"] = make_sample_payload(case, method_names, method_name=method_name)


def apply_pending_sample(session_state: dict) -> bool:
    """Apply any queued sample-case update before widget-backed inputs are created."""
    pending_state = session_state.pop("pending_sample_case_state", None)
    if not pending_state:
        return False
    if "field_values" in pending_state:
        apply_sample_state(session_state, pending_state.get("field_values", {}), pending_state)
        return True
    apply_sample_state(session_state, pending_state)
    return True


def field_usage_caption(method_names: list[str], field_key: str) -> str:
    """Explain which selected methods use one field."""
    required_by = methods_requiring(method_names, field_key)
    used_by = methods_using(method_names, field_key)
    optional_by = [method_name for method_name in used_by if method_name not in required_by]
    parts = []
    if required_by:
        parts.append("Required for: " + ", ".join(required_by))
    if optional_by:
        parts.append("Optional for: " + ", ".join(optional_by))
    return " | ".join(parts) if parts else ""


def _risk_badge_label(case: dict) -> str:
    return "Low" if safe_text(case.get("risk_level", "")).lower() == "low" else "High"


def _render_demo_card(st, case: dict, method_name: str) -> None:
    del method_name
    risk_level = safe_text(case.get("risk_level", "")).lower() or "low"
    st.markdown(
        f"""
        <div class="demo-card {escape(risk_level)} demo-card-minimal">
            <div class="demo-card-top">
                <span class="demo-risk-pill {escape(risk_level)}">{escape(_risk_badge_label(case))} risk</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sample_pair(st, method_name: str) -> None:
    pair = sample_pairs_for(method_name)
    low_case = pair.get("low")
    high_case = pair.get("high")
    if not low_case or not high_case:
        st.info("This method does not yet have both demo cases wired up.")
        return

    low_column, high_column = st.columns(2, gap="medium")
    with low_column:
        _render_demo_card(st, low_case, method_name)
        if st.button("Load low-risk example", key=f"load_low_{method_name}", use_container_width=True, type="primary"):
            queue_sample(st, low_case, [method_name], method_name=method_name)
            st.rerun()
    with high_column:
        _render_demo_card(st, high_case, method_name)
        if st.button("Load high-risk example", key=f"load_high_{method_name}", use_container_width=True, type="primary"):
            queue_sample(st, high_case, [method_name], method_name=method_name)
            st.rerun()


def render_samples(st, selected_methods: list[str], compare_mode: bool) -> None:
    """Render the method-aware low-risk/high-risk demo browser."""
    st.markdown(
        """
        <div class="sample-browser-shell">
            <div class="section-kicker">Demos</div>
            <h3 class="sample-browser-title">Method-paired demo browser</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if compare_mode:
        active_methods = selected_methods or []
        if not active_methods:
            st.info("Select at least one method to browse demos.")
            return
        tabs = st.tabs(active_methods)
        for tab, method_name in zip(tabs, active_methods):
            with tab:
                _render_sample_pair(st, method_name)
    else:
        method_name = selected_methods[0] if selected_methods else METHOD_ORDER[0]
        _render_sample_pair(st, method_name)

    loaded_title = safe_text(st.session_state.get("loaded_sample_title", ""))
    loaded_method = safe_text(st.session_state.get("loaded_sample_method", ""))
    loaded_risk = safe_text(st.session_state.get("loaded_sample_risk_level", ""))
    if loaded_title:
        label = "Loaded demo"
        if loaded_method:
            label += f": {loaded_method}"
        if loaded_risk:
            label += f" - {loaded_risk.title()} risk"
        st.caption(label)
    else:
        st.caption("No demo loaded.")


def render_case_picker(st, samples_or_methods, selected_methods: list[str] | None = None, compare_mode: bool = False) -> None:
    """Backwards-compatible wrapper for the old sample picker entry point."""
    del samples_or_methods
    render_samples(st, selected_methods or [METHOD_ORDER[0]], compare_mode)


def render_input_fields(st, selected_methods: list[str], compare_mode: bool) -> None:
    """Render only the fields needed for the current method selection."""
    visible_fields = visible_fields_for(selected_methods)
    for field_key in visible_fields:
        if field_key == "uploaded_documents":
            st.file_uploader(
                FIELD_SPECS[field_key]["label"],
                key=_upload_key(st.session_state),
                accept_multiple_files=True,
                type=["txt", "md", "json", "jsonl", "pdf"],
                help=FIELD_SPECS[field_key]["helper"],
            )
            continue

        field_spec = FIELD_SPECS[field_key]
        st.text_area(
            field_spec["label"],
            key=FIELD_STATE_KEYS[field_key],
            height=field_spec["height"],
            placeholder=field_spec["placeholder"],
            help=field_spec["helper"],
        )
        if compare_mode:
            usage_caption = field_usage_caption(selected_methods, field_key)
            if usage_caption:
                st.markdown(f'<div class="field-usage">{usage_caption}</div>', unsafe_allow_html=True)
        elif field_key == "sampled_answers":
            st.caption("Provide multiple alternative sampled answers to help this method judge answer stability.")


def read_uploads(st, load_doc_bytes) -> tuple[list, list[str]]:
    """Load uploaded documents into the retrieval pipeline format."""
    uploaded_files = _current_uploads(st.session_state)
    documents = []
    warnings = []
    for file in uploaded_files:
        try:
            documents.append(load_doc_bytes(file.name, file.getvalue()))
        except Exception as exc:
            warnings.append(f"{file.name}: {exc}")
    return documents, warnings


def uploaded_text(documents: list) -> str:
    """Flatten uploaded documents into a text block when needed."""
    return "\n\n".join(f"[{doc.title}]\n{doc.text}" for doc in documents if safe_text(doc.text))


def validate_inputs(selected_methods: list[str], state: dict) -> list[str]:
    """Return user-facing validation errors for the current selection."""
    if not selected_methods:
        return ["Select at least one method before running the analysis."]

    errors = []
    if not has_text(state.get("answer_input", "")):
        errors.append("An LLM answer is required before the analysis can run.")

    visible_fields = set(visible_fields_for(selected_methods))
    if "source_text" in visible_fields and not has_text(state.get("source_input", "")):
        errors.append("Source text is required for the selected source-grounded method.")

    if "evidence_text" in visible_fields and not has_text(state.get("evidence_input", "")):
        evidence_methods = methods_requiring(selected_methods, "evidence_text")
        methods_needing_text = [
            method_name
            for method_name in evidence_methods
            if not method_meta(method_name).get("supports_uploads")
        ]
        if methods_needing_text or not _current_uploads(state):
            errors.append("Evidence text is required for the selected grounded or verification methods.")
    return errors

