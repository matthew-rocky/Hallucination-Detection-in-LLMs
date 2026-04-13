"""Method selection helpers for the Streamlit UI."""

from ui.method_descriptions import METHOD_ORDER
from ui.layout import render_compare_tabs, render_method_grid, render_method_info

MODE_OPTIONS = ["Quick Check", "Compare Methods"]


def render_mode_toggle(st) -> str:
    """Render the top-level analysis mode toggle."""
    return st.radio(
        "Analysis mode",
        options=MODE_OPTIONS,
        index=MODE_OPTIONS.index(st.session_state.get("ui_mode", MODE_OPTIONS[0])) if st.session_state.get("ui_mode") in MODE_OPTIONS else 0,
        key="ui_mode",
        horizontal=True,
        help="Quick Check focuses on one method. Compare Methods enables side-by-side evaluation.",
    )


def render_methods(st, compare_mode: bool) -> list[str]:
    """Render the method selection controls and a compact active-method preview."""
    if compare_mode:
        default_selection = st.session_state.get("selected_methods") or [st.session_state.get("selected_method", METHOD_ORDER[0])]
        st.multiselect(
            "Choose methods to compare",
            options=METHOD_ORDER,
            default=default_selection,
            key="selected_methods",
            help="Select the methods you want to run side by side.",
        )
        selected_methods = st.session_state.get("selected_methods", [])
        if selected_methods:
            render_method_grid(st, selected_methods)
            render_compare_tabs(st, selected_methods)
        return selected_methods

    default_method = st.session_state.get("selected_method", METHOD_ORDER[0])
    selected_method = st.selectbox(
        "Choose a detection method",
        options=METHOD_ORDER,
        index=METHOD_ORDER.index(default_method) if default_method in METHOD_ORDER else 0,
        key="selected_method",
        help="Pick the single method you want to run.",
    )
    st.session_state["selected_methods"] = [selected_method]
    render_method_grid(st, [selected_method])
    render_method_info(st, selected_method)
    return [selected_method]
