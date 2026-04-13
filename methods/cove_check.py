"""Canonical entry point for the CoVe-style verification pipeline."""

from detectors.cove import run_cove_detector


METHOD_NAME = "CoVe-Style Verification"
FAMILY = "chain-of-verification"


def run_cove(
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    sampled_answers_text: str = "",
    max_questions: int = 6,
    extra_documents: list | None = None,
) -> dict:
    """Run the explicit CoVe-style verify-and-revise pipeline."""
    del sampled_answers_text
    return run_cove_detector(
        question=question,
        answer=answer,
        source_text=source_text,
        evidence_text=evidence_text,
        extra_documents=extra_documents,
        method_name=METHOD_NAME,
        family=FAMILY,
        max_questions=max_questions,
    )


