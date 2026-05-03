"""API schemas for the full-stack hallucination detection dashboard."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class MethodInfo(BaseModel):
    id: str
    name: str
    family: str
    how_it_works: str = ""
    short_purpose: str = ""
    best_for: str = ""
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)
    visible_fields: list[str] = Field(default_factory=list)
    ignored_fields: list[str] = Field(default_factory=list)
    supports_uploads: bool = False
    implementation: str = ""
    caption: str = ""
    input_requirements: dict[str, list[str]] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommended_use: str = ""
    color: str = ""
    tone: str = ""
    profile: dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    mode: Literal["quick", "compare"] = "quick"
    selected_methods: list[str] = Field(default_factory=list)
    question: str = ""
    answer: str = ""
    source_text: str = ""
    evidence_text: str = ""
    sampled_answers_text: str = ""


class AnalyzeResponse(BaseModel):
    ok: bool = True
    mode: Literal["quick", "compare"]
    selected_methods: list[str]
    results: list[dict[str, Any]]
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    ok: bool = True
    status: str = "online"
    message: str = "FastAPI backend is running"
    method_count: int = 8
    sample_count: int = 16
