"""Canonical entry point for the internal-signal uncertainty baseline."""

from detectors.signal import SignalConfig, run_signal_detector


METHOD_NAME = "Internal-Signal Baseline"
FAMILY = "internal-signal"


def run_internal(
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    sampled_answers_text: str = "",
) -> dict:
    """Run the local internal-signal baseline with honest fallback behavior."""
    del source_text
    del evidence_text
    return run_signal_detector(
        question=question,
        answer=answer,
        method_name=METHOD_NAME,
        mode="uncertainty_baseline",
        family=FAMILY,
        sampled_answers_text=sampled_answers_text,
        config=SignalConfig(),
    )


