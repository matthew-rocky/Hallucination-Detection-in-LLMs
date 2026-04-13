"""Canonical entry point for the retrieval-grounded checker."""

from detectors.retrieval_check import run_retrieval_check


METHOD_NAME = "Retrieval-Grounded Checker"
FAMILY = "retrieval-grounded"


def run_retrieval(
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    sampled_answers_text: str = "",
    top_k: int = 4,
    allow_web: bool = False,
    live_web_max_pages: int = 3,
    extra_documents: list | None = None,
) -> dict:
    """Run the local retrieval-grounded checker over the indexed evidence corpus."""
    del sampled_answers_text
    del allow_web
    del live_web_max_pages
    return run_retrieval_check(
        question=question,
        answer=answer,
        source_text=source_text,
        evidence_text=evidence_text,
        extra_documents=extra_documents,
        method_name=METHOD_NAME,
        family=FAMILY,
        top_k=top_k,
        impl_status="implemented",
    )


