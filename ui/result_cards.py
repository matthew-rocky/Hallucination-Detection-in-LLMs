"""Main result-card rendering for the Streamlit UI."""

from html import escape

from ui.method_descriptions import method_meta, get_method_profile
from utils.text_utils import safe_text, truncate_text
from utils.ui_utils import (
    build_chunk_table,
    build_citation_table,
    build_claim_table,
    build_evidence_table,
    build_check_table,
    build_answer_table,
    build_step_table,
    build_trace_table,
    build_signal_table,
    build_vq_table,
    format_backend,
    format_confidence,
    format_status,
    format_origin,
    format_score,
)


def _risk_css_class(risk_label: str) -> str:
    label = safe_text(risk_label).lower()
    if label == "low":
        return "low"
    if label == "medium":
        return "medium"
    if label == "high":
        return "high"
    return "na"


def _reason_bullets(result: dict) -> list[str]:
    metadata = result.get("metadata", {})
    sub_signals = result.get("sub_signals") or []
    bullets: list[str] = []

    origin_text = format_origin(result)
    if origin_text and metadata.get("backend_available") is not None:
        if metadata.get("fallback_mode"):
            bullets.append("This internal-signal score came from a deterministic local approximation, not factual verification from evidence.")
        else:
            bullets.append(f"Runtime path: {origin_text}.")

    if sub_signals:
        ranked = sorted(sub_signals, key=lambda item: float(item.get("risk", 0.0)), reverse=True)
        for signal in ranked[:2]:
            explanation = safe_text(signal.get("explanation", ""))
            if explanation:
                bullets.append(explanation)
        if bullets:
            return [truncate_text(item, 150) for item in bullets[:3] if item]

    claim_findings = result.get("claim_findings") or []
    if claim_findings:
        contradicted = sum(1 for finding in claim_findings if finding.get("status") == "contradicted")
        unsupported = sum(
            1
            for finding in claim_findings
            if finding.get("status") in {"unsupported", "insufficient evidence", "unresolved"}
        )
        supported = sum(
            1
            for finding in claim_findings
            if finding.get("status") in {"supported", "abstractly_supported", "weakly_supported", "verified"}
        )
        if contradicted:
            bullets.append(f"{contradicted} claim(s) are directly contradicted by the available evidence.")
        if unsupported:
            bullets.append(f"{unsupported} claim(s) remain unsupported or unresolved.")
        if supported:
            bullets.append(f"{supported} claim(s) were supported or partially supported.")
        if bullets:
            return bullets[:3]

    if result.get("retrieval_counts"):
        counts = result["retrieval_counts"]
        return [
            f"Supported claims: {counts.get('supported', 0)}.",
            f"Contradicted claims: {counts.get('contradicted', 0)}.",
            f"Insufficient evidence: {counts.get('insufficient evidence', 0)}.",
        ]

    summary = safe_text(result.get("summary", ""))
    explanation = safe_text(result.get("explanation", ""))
    if summary:
        bullets.append(summary)
    if explanation and explanation != summary:
        bullets.append(explanation)
    return [truncate_text(item, 150) for item in bullets[:3] if item]


def _tool_rows(tool_outputs: list[dict]) -> list[dict]:
    """Flatten tool outputs into a compact table representation."""
    rows = []
    for item in tool_outputs or []:
        claim = truncate_text(item.get("claim", ""), 90)
        for tool_result in item.get("tool_results", []):
            rows.append(
                {
                    "Claim": claim,
                    "Tool": tool_result.get("tool_name", ""),
                    "Status": tool_result.get("status", ""),
                    "Summary": truncate_text(tool_result.get("message", ""), 120),
                }
            )
    return rows


def _build_metadata_rows(metadata: dict) -> list[dict]:
    """Convert metadata into a compact key-value table."""
    rows = []
    for key, value in sorted((metadata or {}).items()):
        if isinstance(value, (list, dict)):
            display = truncate_text(str(value), 180)
        else:
            display = str(value)
        rows.append({"Key": key, "Value": display})
    return rows


def _overview_badges(result: dict, status_text: str, origin_text: str, backend_text: str) -> str:
    badges = [f'<span class="status-pill">{escape(result["method_name"])}</span>']
    if status_text:
        badges.append(f'<span class="status-pill">{escape(status_text)}</span>')
    if origin_text:
        badges.append(f'<span class="status-pill">{escape(origin_text)}</span>')
    if backend_text:
        badges.append(f'<span class="status-pill">{escape(backend_text)}</span>')
    return "".join(badges)


def render_overview(st, result: dict) -> None:
    """Render the primary result card used in quick mode."""
    profile = get_method_profile(result["method_name"])
    metadata = method_meta(result["method_name"])
    score_text = format_score(result.get("risk_score"), result.get("risk_label", ""))
    confidence_text = format_confidence(result.get("confidence"))
    status_text = format_status(result, profile.get("implementation", "N/A"))
    origin_text = format_origin(result)
    backend_text = format_backend(result)
    risk_label = result.get("risk_label", "Not Available")
    summary = safe_text(result.get("summary", "")) or safe_text(result.get("explanation", ""))
    bullets = _reason_bullets(result)
    bullet_html = "".join(f"<li>{escape(item)}</li>" for item in bullets if item)
    method_note = metadata.get("short_purpose", "")

    st.markdown(
        f"""
        <div class="result-overview">
            <div class="result-badge-row">{_overview_badges(result, status_text, origin_text, backend_text)}</div>
            <h3 style="margin-top:0.75rem;">Hallucination Risk</h3>
            <div class="result-grid">
                <div class="metric-card metric-card-primary">
                    <div class="metric-label">Risk score</div>
                    <span class="risk-pill {_risk_css_class(risk_label)}">{escape(risk_label)}</span>
                    <div class="metric-value">{escape(score_text)}</div>
                    <div class="section-copy" style="margin-top:0.42rem;">{escape(method_note)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value small">{escape(confidence_text)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Implementation</div>
                    <div class="metric-value small">{escape(status_text)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Run origin</div>
                    <div class="metric-value small">{escape(origin_text or 'Standard pipeline')}</div>
                </div>
            </div>
            <div class="result-summary">{escape(summary)}</div>
            <div class="result-why-label">Why this score</div>
            <ul class="reason-list">{bullet_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.get("metadata", {}).get("app_runtime_guard"):
        st.caption("The app caught a method-level runtime error and kept the session alive. Open the details below for the error summary.")
    elif result.get("metadata", {}).get("fallback_mode"):
        st.caption("Fallback mode stays explicit: this path estimates answer suspiciousness from local deterministic cues and does not verify factual correctness from wording alone.")
    elif result.get("available") is False:
        st.caption("This method could not produce a score for this run. Open the advanced details for the reason.")


def _render_detail_tables(st, result: dict) -> None:
    if result.get("sub_signals"):
        st.dataframe(build_signal_table(result["sub_signals"]), use_container_width=True, hide_index=True)
    if result.get("verification_questions"):
        st.dataframe(build_vq_table(result["verification_questions"]), use_container_width=True, hide_index=True)
    if result.get("independent_answers"):
        st.dataframe(build_answer_table(result["independent_answers"]), use_container_width=True, hide_index=True)
    if result.get("retrieval_counts"):
        counts = result["retrieval_counts"]
        columns = st.columns(3)
        columns[0].metric("Supported", counts.get("supported", 0))
        columns[1].metric("Contradicted", counts.get("contradicted", 0))
        columns[2].metric("Insufficient", counts.get("insufficient evidence", 0))
    if result.get("top_retrieved_evidence"):
        st.dataframe(build_trace_table(result["top_retrieved_evidence"]), use_container_width=True, hide_index=True)
    if result.get("chunk_catalog"):
        st.markdown("**Indexed chunks**")
        st.dataframe(build_chunk_table(result["chunk_catalog"]), use_container_width=True, hide_index=True)
    if result.get("reference_chunk_catalog"):
        st.markdown("**Reference chunks**")
        st.dataframe(build_chunk_table(result["reference_chunk_catalog"]), use_container_width=True, hide_index=True)
    if result.get("proposed_external_checks"):
        st.markdown("**Tool routing**")
        st.dataframe(build_check_table(result["proposed_external_checks"]), use_container_width=True, hide_index=True)
    if result.get("tool_outputs"):
        tool_rows = _tool_rows(result["tool_outputs"])
        if tool_rows:
            st.markdown("**Tool results**")
            st.dataframe(tool_rows, use_container_width=True, hide_index=True)
    if result.get("sampled_answers"):
        st.markdown("**Sampled answers**")
        for index, sample in enumerate(result["sampled_answers"], start=1):
            st.caption(f"Sample {index}")
            st.write(sample)
    if result.get("original_draft"):
        st.markdown("**Original draft**")
        st.write(result["original_draft"])
    if result.get("verification_summary"):
        st.markdown("**Verification summary**")
        st.write(result["verification_summary"])
    if result.get("revised_answer"):
        st.markdown("**Revised answer**")
        st.write(result["revised_answer"])


def render_details(st, result: dict) -> None:
    """Render advanced result sections behind expanders."""
    with st.expander("Why this score"):
        origin_text = format_origin(result)
        backend_text = format_backend(result)
        if origin_text:
            st.markdown(f"**Run origin**: {origin_text}")
        if backend_text:
            st.markdown(f"**Backend**: {backend_text}")
        st.write(result.get("summary", ""))
        if result.get("explanation") and result.get("explanation") != result.get("summary"):
            st.write(result.get("explanation"))
        if result.get("evidence_used"):
            st.markdown("**Evidence used**")
            st.write(result.get("evidence_used"))
        if result.get("limitations"):
            st.markdown("**Limitations**")
            st.write(result.get("limitations"))

    if result.get("claim_findings"):
        with st.expander("Claim findings"):
            st.dataframe(build_claim_table(result["claim_findings"]), use_container_width=True, hide_index=True)

    if result.get("sub_signals"):
        with st.expander("Sub-signals"):
            st.dataframe(build_signal_table(result["sub_signals"]), use_container_width=True, hide_index=True)

    if any(
        result.get(key)
        for key in (
            "evidence",
            "citations",
            "retrieval_counts",
            "top_retrieved_evidence",
            "chunk_catalog",
            "reference_chunk_catalog",
            "tool_outputs",
        )
    ):
        with st.expander("Evidence and grounding trace"):
            if result.get("evidence"):
                st.dataframe(build_evidence_table(result["evidence"]), use_container_width=True, hide_index=True)
            if result.get("citations"):
                st.dataframe(build_citation_table(result["citations"]), use_container_width=True, hide_index=True)
            _render_detail_tables(st, result)

    if result.get("intermediate_steps") or result.get("verification_questions") or result.get("independent_answers") or result.get("proposed_external_checks"):
        with st.expander("Method trace"):
            if result.get("intermediate_steps"):
                st.dataframe(build_step_table(result["intermediate_steps"]), use_container_width=True, hide_index=True)
            _render_detail_tables(
                st,
                {
                    key: result.get(key)
                    for key in (
                        "verification_questions",
                        "independent_answers",
                        "proposed_external_checks",
                        "tool_outputs",
                        "sampled_answers",
                        "original_draft",
                        "verification_summary",
                        "revised_answer",
                    )
                },
            )

    metadata_rows = _build_metadata_rows(result.get("metadata", {}))
    if metadata_rows:
        with st.expander("Metadata"):
            st.dataframe(metadata_rows, use_container_width=True, hide_index=True)

