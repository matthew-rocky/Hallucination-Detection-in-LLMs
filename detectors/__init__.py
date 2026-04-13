"""Detector package exposing the shared result schema and detector runners."""

from .base import make_result, make_unavailable

__all__ = [
    "make_result",
    "make_unavailable",
]
