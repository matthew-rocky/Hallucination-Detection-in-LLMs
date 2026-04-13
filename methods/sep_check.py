"""Canonical entry point for the SEP-inspired internal-signal method."""

from detectors.signal import SignalConfig, run_signal_detector


METHOD_NAME = "SEP-Inspired Internal Signal"
FAMILY = "internal-signal / SEP-inspired"


def run_sep(
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    sampled_answers_text: str = "",
    token_logprobs: list[list[float]] | None = None,
) -> dict:
    """Run the SEP-inspired detector with optional sampled-answer support."""
    del source_text
    del evidence_text
    return run_signal_detector(
        question=question,
        answer=answer,
        method_name=METHOD_NAME,
        mode="sep_lite",
        family=FAMILY,
        sampled_answers_text=sampled_answers_text,
        config=SignalConfig(),
        token_logprobs=token_logprobs,
    )


