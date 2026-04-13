"""Detector method entry points used by the Streamlit app."""

from .critic_check import run_critic
from .cove_check import run_cove
from .internal_check import run_internal
from .rag_check import run_rag
from .retrieval_check import run_retrieval
from .sep_check import run_sep
from .source_check import run_source
from .verify_flow import run_verify

__all__ = [
    "run_critic",
    "run_cove",
    "run_internal",
    "run_rag",
    "run_retrieval",
    "run_sep",
    "run_source",
    "run_verify",
]