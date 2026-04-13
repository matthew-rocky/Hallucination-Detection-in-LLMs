"""Layout and styling helpers for the Streamlit UI."""

from html import escape
from textwrap import dedent

from ui.method_descriptions import FIELD_SPECS, get_input_summary, method_meta


BASE_STYLES = """
<style>
:root {
    color-scheme: dark;
    --bg: #050505;
    --bg-elevated: #0b0b0c;
    --bg-panel: linear-gradient(180deg, rgba(15, 15, 16, 0.98) 0%, rgba(9, 9, 10, 0.98) 100%);
    --bg-panel-soft: linear-gradient(180deg, rgba(18, 18, 20, 0.96) 0%, rgba(11, 11, 12, 0.96) 100%);
    --border: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(230, 219, 199, 0.24);
    --text: #f7f4ee;
    --muted: rgba(247, 244, 238, 0.74);
    --soft: rgba(247, 244, 238, 0.54);
    --accent: #e6dbc7;
    --accent-strong: #c8a873;
    --accent-soft: rgba(230, 219, 199, 0.12);
    --shadow: 0 18px 50px rgba(0, 0, 0, 0.38);
    --low: rgba(52, 211, 153, 0.16);
    --med: rgba(245, 158, 11, 0.16);
    --high: rgba(248, 113, 113, 0.16);
}

html, body, [class*="css"] {
    font-family: "Aptos", "Segoe UI Variable", "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(230, 219, 199, 0.08), transparent 24%),
        radial-gradient(circle at top right, rgba(200, 168, 115, 0.06), transparent 26%),
        linear-gradient(180deg, #060606 0%, #090909 44%, #040404 100%);
    color: var(--text);
}

.block-container {
    max-width: 1180px;
    padding-top: 1rem;
    padding-bottom: 2.4rem;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
}

h1, h2, h3, h4, h5, h6, p, li, label, span, small {
    color: inherit;
}

.hero-card,
.surface-card,
.info-card,
.method-card,
.result-overview,
.demo-card,
.sample-browser-shell {
    position: relative;
    overflow: hidden;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
    animation: rise-in 260ms ease-out;
}

.hero-card::before,
.surface-card::before,
.info-card::before,
.method-card::before,
.result-overview::before,
.demo-card::before,
.sample-browser-shell::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(230, 219, 199, 0), rgba(230, 219, 199, 0.78), rgba(200, 168, 115, 0.58), rgba(230, 219, 199, 0));
    pointer-events: none;
}

.hero-card {
    padding: 1.2rem 1.3rem 1.15rem;
    background:
        radial-gradient(circle at top right, rgba(230, 219, 199, 0.08), transparent 28%),
        radial-gradient(circle at bottom left, rgba(200, 168, 115, 0.08), transparent 32%),
        linear-gradient(135deg, rgba(11, 11, 12, 0.99) 0%, rgba(16, 16, 18, 0.99) 55%, rgba(12, 12, 13, 0.99) 100%);
    border-color: rgba(230, 219, 199, 0.10);
}

.hero-kicker,
.section-kicker {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--accent-strong);
    font-weight: 700;
}

.hero-card h1 {
    margin: 0.2rem 0 0;
    font-size: 2.35rem;
    line-height: 0.98;
    letter-spacing: -0.05em;
}

.hero-card p {
    margin: 0.75rem 0 0;
    max-width: 860px;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.58;
}

.hero-meta,
.result-badge-row,
.method-meta,
.demo-card-top {
    display: flex;
    flex-wrap: wrap;
    gap: 0.42rem;
}

.hero-meta {
    margin-top: 0.95rem;
}

.chip,
.risk-pill,
.status-pill,
.demo-chip,
.demo-risk-pill,
.method-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    border-radius: 999px;
    padding: 0.28rem 0.68rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    border: 1px solid rgba(255, 255, 255, 0.10);
    color: var(--text);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.025) 100%);
}

.chip,
.status-pill,
.demo-chip,
.method-tag {
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.method-tag.best,
.status-pill.accent {
    background: linear-gradient(180deg, rgba(200, 168, 115, 0.18) 0%, rgba(200, 168, 115, 0.07) 100%);
    border-color: rgba(200, 168, 115, 0.24);
}

.surface-card,
.info-card,
.sample-browser-shell {
    margin-top: 0.16rem;
    padding: 0.9rem 1rem;
    background: var(--bg-panel-soft);
}

.section-title,
.sample-browser-title {
    margin: 0.16rem 0 0;
    font-size: 1.08rem;
    letter-spacing: -0.025em;
}

.section-copy,
.sample-browser-copy,
.info-card p {
    margin: 0.34rem 0 0;
    color: var(--muted);
    line-height: 1.55;
}

.method-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 0.68rem;
    margin-top: 0.72rem;
}

.method-card {
    padding: 0.85rem 0.92rem;
    background:
        radial-gradient(circle at top right, rgba(230, 219, 199, 0.05), transparent 32%),
        linear-gradient(180deg, rgba(17, 17, 18, 0.98) 0%, rgba(10, 10, 11, 0.98) 100%);
    min-height: 150px;
}

.method-card.selected {
    border-color: rgba(230, 219, 199, 0.18);
    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.34), inset 0 0 0 1px rgba(230, 219, 199, 0.04);
}

.method-card h4 {
    margin: 0 0 0.3rem;
    font-size: 1rem;
    letter-spacing: -0.02em;
}

.method-card p {
    margin: 0;
    color: var(--muted);
    line-height: 1.48;
}

.method-meta {
    margin-top: 0.72rem;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.72rem;
    margin-top: 0.8rem;
}

.info-label,
.metric-label,
.result-why-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    color: var(--accent-strong);
    font-weight: 700;
}

.info-value {
    margin-top: 0.16rem;
    color: var(--text);
    line-height: 1.48;
}

.result-overview {
    padding: 1rem 1.05rem;
    background:
        radial-gradient(circle at top right, rgba(230, 219, 199, 0.07), transparent 34%),
        linear-gradient(180deg, rgba(16, 16, 17, 0.99) 0%, rgba(9, 9, 10, 0.99) 100%);
}

.result-overview h3 {
    margin: 0.82rem 0 0;
    font-size: 1.12rem;
}

.result-grid {
    display: grid;
    grid-template-columns: minmax(240px, 1.25fr) repeat(3, minmax(130px, 0.7fr));
    gap: 0.78rem;
    margin-top: 0.82rem;
}

.metric-card {
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.07);
    background: linear-gradient(180deg, rgba(20, 20, 21, 0.98) 0%, rgba(11, 11, 12, 0.98) 100%);
    padding: 0.84rem 0.88rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
}

.metric-card-primary {
    background:
        radial-gradient(circle at top right, rgba(200, 168, 115, 0.10), transparent 38%),
        linear-gradient(180deg, rgba(20, 20, 21, 0.99) 0%, rgba(11, 11, 12, 0.99) 100%);
}

.metric-value {
    margin-top: 0.28rem;
    font-size: 1.95rem;
    line-height: 1.0;
    letter-spacing: -0.05em;
    font-weight: 800;
}

.metric-value.small {
    font-size: 1.02rem;
    line-height: 1.34;
    letter-spacing: -0.02em;
}

.result-summary {
    margin-top: 0.82rem;
    color: var(--muted);
    line-height: 1.58;
}

.result-why-label {
    margin-top: 0.86rem;
}

.reason-list {
    margin: 0.46rem 0 0;
    padding-left: 1.15rem;
}

.reason-list li {
    margin-bottom: 0.28rem;
    color: var(--text);
}

.risk-pill.low,
.demo-risk-pill.low {
    background: linear-gradient(180deg, rgba(52, 211, 153, 0.22) 0%, rgba(52, 211, 153, 0.10) 100%);
    border-color: rgba(52, 211, 153, 0.24);
}

.risk-pill.medium,
.demo-risk-pill.medium {
    background: linear-gradient(180deg, rgba(245, 158, 11, 0.20) 0%, rgba(245, 158, 11, 0.10) 100%);
    border-color: rgba(245, 158, 11, 0.22);
}

.risk-pill.high,
.demo-risk-pill.high {
    background: linear-gradient(180deg, rgba(248, 113, 113, 0.22) 0%, rgba(248, 113, 113, 0.10) 100%);
    border-color: rgba(248, 113, 113, 0.24);
}

.risk-pill.na,
.status-pill {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.03) 100%);
}

.demo-card {
    padding: 0.9rem;
    min-height: 214px;
    background:
        radial-gradient(circle at top right, rgba(230, 219, 199, 0.04), transparent 38%),
        linear-gradient(180deg, rgba(16, 16, 17, 0.98) 0%, rgba(10, 10, 11, 0.98) 100%);
}

.demo-card.low {
    border-color: rgba(52, 211, 153, 0.18);
}

.demo-card.high {
    border-color: rgba(248, 113, 113, 0.18);
}

.demo-card-minimal {
    min-height: auto;
    padding: 0.72rem 0.78rem;
}

.demo-card-minimal .demo-card-top {
    margin: 0;
}

.demo-card h4 {
    margin: 0;
    font-size: 1rem;
    letter-spacing: -0.02em;
}

.demo-card p {
    margin: 0.42rem 0 0;
    min-height: 3.2rem;
    color: var(--muted);
    line-height: 1.5;
}

.demo-fields,
.field-usage {
    margin-top: 0.7rem;
    color: var(--soft);
    font-size: 0.84rem;
}

.field-usage {
    margin-top: -0.22rem;
    margin-bottom: 0.46rem;
}

.stButton > button,
.stDownloadButton > button {
    min-height: 2.9rem;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    background: linear-gradient(180deg, rgba(28, 28, 29, 0.98) 0%, rgba(12, 12, 13, 0.98) 100%) !important;
    color: var(--text) !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.26), inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
    transition: transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(230, 219, 199, 0.28) !important;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.32), 0 0 0 1px rgba(230, 219, 199, 0.05) inset !important;
}

.stButton > button[kind="primary"] {
    border-color: rgba(230, 219, 199, 0.22) !important;
    background:
        radial-gradient(circle at top, rgba(230, 219, 199, 0.10), transparent 65%),
        linear-gradient(180deg, rgba(33, 33, 35, 0.99) 0%, rgba(12, 12, 13, 0.99) 100%) !important;
}

.stTextArea textarea,
.stTextInput input,
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div,
.stFileUploader section {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: linear-gradient(180deg, rgba(16, 16, 17, 0.98) 0%, rgba(10, 10, 11, 0.98) 100%) !important;
    color: var(--text) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02) !important;
}

.stFileUploader section {
    border-style: dashed !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: rgba(230, 219, 199, 0.22) !important;
    box-shadow: 0 0 0 1px rgba(230, 219, 199, 0.12), 0 0 0 4px rgba(230, 219, 199, 0.05) !important;
}

.stRadio [role="radiogroup"] {
    gap: 0.45rem;
}

.stRadio [data-baseweb="radio"] {
    padding: 0.32rem 0.44rem;
    border-radius: 14px;
    background: rgba(14, 14, 15, 0.96);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.36rem;
    padding: 0.28rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(13, 13, 14, 0.92);
}

.stTabs [role="tab"] {
    height: 2.45rem;
    border-radius: 12px !important;
    border: 1px solid transparent !important;
    color: var(--muted) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    background: linear-gradient(180deg, rgba(31, 31, 33, 0.98) 0%, rgba(16, 16, 17, 0.98) 100%) !important;
    border-color: rgba(230, 219, 199, 0.16) !important;
}

.stExpander {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    background: linear-gradient(180deg, rgba(17, 17, 18, 0.98) 0%, rgba(10, 10, 11, 0.98) 100%) !important;
    overflow: hidden;
}

.stExpander summary {
    background: rgba(255, 255, 255, 0.02) !important;
}

.stAlert {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: linear-gradient(180deg, rgba(18, 18, 20, 0.98) 0%, rgba(11, 11, 12, 0.98) 100%) !important;
}

.stAlert[data-baseweb="notification"] {
    box-shadow: none !important;
}

.stDataFrame,
div[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(17, 17, 18, 0.98) 0%, rgba(10, 10, 11, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 0.78rem 0.82rem;
    border-radius: 16px;
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 0;
}

@keyframes rise-in {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 960px) {
    .hero-card h1 {
        font-size: 1.95rem;
    }

    .result-grid {
        grid-template-columns: 1fr;
    }

    .demo-card {
        min-height: auto;
    }
}
</style>
"""


def inject_base_styles(st) -> None:
    """Load the shared Streamlit styles."""
    st.markdown(BASE_STYLES, unsafe_allow_html=True)


def _render_html_block(st, html: str) -> None:
    """Render HTML reliably across Streamlit versions without leaking raw tags."""
    cleaned_html = dedent(html).strip()
    if hasattr(st, "html"):
        st.html(cleaned_html)
        return
    st.markdown(cleaned_html, unsafe_allow_html=True)


def render_hero(st) -> None:
    """Render the page hero copy."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Local Research Prototype</div>
            <h1>Hallucination Detector Comparison</h1>
            <p>Run internal-signal, grounded, verification, and tool-check methods on the same answer, then inspect the score, evidence, and runtime path.</p>
            <div class="hero-meta">
                <span class="chip">Internal signals</span>
                <span class="chip">Grounded evidence</span>
                <span class="chip">Verification traces</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(st, title: str, copy: str, kicker: str = "") -> None:
    """Render a compact section intro block."""
    kicker_html = f'<div class="section-kicker">{escape(kicker)}</div>' if kicker else ""
    st.markdown(
        f"""
        <div class="surface-card">
            {kicker_html}
            <h2 class="section-title">{escape(title)}</h2>
            <p class="section-copy">{escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_method_grid(st, method_names: list[str]) -> None:
    """Render compact cards for the active method set only."""
    active_methods = method_names or []
    if not active_methods:
        return

    card_html = []
    for method_name in active_methods:
        metadata = method_meta(method_name)
        card_html.append(
            dedent(
                f"""
                <div class="method-card selected">
                    <h4>{escape(method_name)}</h4>
                    <p>{escape(metadata.get('short_purpose', ''))}</p>
                    <div class="method-meta">
                        <span class="method-tag best">Best for: {escape(metadata.get('best_for', ''))}</span>
                        <span class="method-tag">Inputs: {escape(get_input_summary(method_name))}</span>
                    </div>
                </div>
                """
            )
        )
    _render_html_block(st, f'<div class="method-grid">{"".join(card_html)}</div>')


def render_method_info(st, method_name: str) -> None:
    """Render the rich description card for one selected method."""
    metadata = method_meta(method_name)
    if not metadata:
        return

    def field_list(field_keys: list[str]) -> str:
        if not field_keys:
            return "None"
        return ", ".join(FIELD_SPECS[field_key]["short_label"] for field_key in field_keys)

    st.markdown(
        f"""
        <div class="info-card">
            <div class="section-kicker">Method profile</div>
            <h3>{escape(method_name)}</h3>
            <p class="section-copy" style="margin-top:0.42rem;">{escape(metadata.get('how_it_works', ''))}</p>
            <div class="info-grid">
                <div>
                    <span class="info-label">Best For</span>
                    <div class="info-value">{escape(metadata.get('best_for', ''))}</div>
                </div>
                <div>
                    <span class="info-label">Required</span>
                    <div class="info-value">{escape(field_list(metadata.get('required_fields', [])))}</div>
                </div>
                <div>
                    <span class="info-label">Optional</span>
                    <div class="info-value">{escape(field_list(metadata.get('optional_fields', [])))}</div>
                </div>
                <div>
                    <span class="info-label">Hidden</span>
                    <div class="info-value">{escape(field_list(metadata.get('ignored_fields', [])))}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compare_tabs(st, selected_methods: list[str]) -> None:
    """Render method descriptions for compare mode."""
    if not selected_methods:
        return
    if len(selected_methods) == 1:
        render_method_info(st, selected_methods[0])
        return
    tabs = st.tabs(selected_methods)
    for tab, method_name in zip(tabs, selected_methods):
        with tab:
            render_method_info(st, method_name)
