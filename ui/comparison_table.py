"""Comparison-table helpers for compare mode."""

import pandas as pd

from ui.method_descriptions import get_method_profile
from utils.text_utils import truncate_text
from utils.ui_utils import (
    build_compare_table,
    format_confidence,
    format_status,
    format_score,
)


def compact_table(results: list[dict]) -> pd.DataFrame:
    """Build the concise comparison table used in compare mode."""
    rows = []
    for result in results:
        profile = get_method_profile(result["method_name"])
        rows.append(
            {
                "Method": result["method_name"],
                "Score": format_score(result.get("risk_score"), result.get("risk_label", "")),
                "Risk": result.get("risk_label", "Not Available"),
                "Confidence": format_confidence(result.get("confidence")),
                "Status": format_status(result, profile.get("implementation", "N/A")),
                "Short Reason": truncate_text(result.get("summary") or result.get("explanation", ""), 110),
            }
        )
    return pd.DataFrame(rows)


def render_compare(st, results: list[dict]) -> None:
    """Render the compact and advanced comparison views."""
    st.dataframe(compact_table(results), use_container_width=True, hide_index=True)
    with st.expander("Advanced comparison details"):
        st.dataframe(build_compare_table(results), use_container_width=True, hide_index=True)
