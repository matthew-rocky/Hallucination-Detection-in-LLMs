"""Canonical entry point for the CRITIC-lite tool-check workflow."""

from detectors.critic import run_critic_detector


METHOD_NAME = "CRITIC-lite Tool Check"
FAMILY = "tool-augmented verification"


def run_critic(
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    sampled_answers_text: str = "",
    enabled_tools: list[str] | None = None,
    extra_documents: list | None = None,
) -> dict:
    """Run the tool-backed CRITIC-lite critique loop."""
    del sampled_answers_text
    del enabled_tools
    return run_critic_detector(
        question=question,
        answer=answer,
        source_text=source_text,
        evidence_text=evidence_text,
        extra_documents=extra_documents,
        method_name=METHOD_NAME,
        family=FAMILY,
    )


