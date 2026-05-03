"""FastAPI entry point for the full-stack hallucination detection dashboard."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, MethodInfo
from backend.services.detector_service import (
    get_field_specs,
    get_methods,
    get_sample_pairs,
    get_samples,
    load_upload_documents,
    normalize_selected_methods,
    run_analysis,
    summarize_results,
)


app = FastAPI(
    title="Hallucination Detection Dashboard API",
    version="1.0.0",
    description="FastAPI wrapper around the existing local hallucination detector methods.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return backend health and method count."""
    return HealthResponse(method_count=len(get_methods()), sample_count=len(get_samples()))


@app.get("/api/methods", response_model=list[MethodInfo])
def methods() -> list[dict]:
    """Return detector method metadata used by the frontend."""
    return get_methods()


@app.get("/api/fields")
def fields() -> dict:
    """Return input field metadata."""
    return get_field_specs()


@app.get("/api/samples")
def samples() -> list[dict]:
    """Return curated sample cases."""
    return get_samples()


@app.get("/api/sample-pairs")
def sample_pairs() -> dict:
    """Return low/high sample pairs for every detector method."""
    return get_sample_pairs()


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run selected detector methods."""
    selected = normalize_selected_methods(request.mode, request.selected_methods)
    payload = request.model_dump(exclude={"mode", "selected_methods"}) if hasattr(request, "model_dump") else request.dict(exclude={"mode", "selected_methods"})
    results, warnings = run_analysis(
        mode=request.mode,
        selected_methods=selected,
        payload=payload,
    )
    return AnalyzeResponse(
        ok=not warnings,
        mode=request.mode,
        selected_methods=selected,
        results=results,
        summary=summarize_results(results),
        warnings=warnings,
    )


@app.post("/api/upload-analyze", response_model=AnalyzeResponse)
async def upload_analyze(
    mode: Annotated[str, Form()] = "quick",
    selected_methods: Annotated[str, Form()] = "[]",
    question: Annotated[str, Form()] = "",
    answer: Annotated[str, Form()] = "",
    source_text: Annotated[str, Form()] = "",
    evidence_text: Annotated[str, Form()] = "",
    sampled_answers_text: Annotated[str, Form()] = "",
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> AnalyzeResponse:
    """Run analysis with uploaded local evidence documents."""
    try:
        parsed_methods = json.loads(selected_methods)
        if not isinstance(parsed_methods, list):
            parsed_methods = []
    except json.JSONDecodeError:
        parsed_methods = []
    resolved_mode = "compare" if mode == "compare" else "quick"
    selected = normalize_selected_methods(resolved_mode, [str(item) for item in parsed_methods])
    documents, upload_warnings = await load_upload_documents(files)
    results, validation_warnings = run_analysis(
        mode=resolved_mode,
        selected_methods=selected,
        payload={
            "question": question,
            "answer": answer,
            "source_text": source_text,
            "evidence_text": evidence_text,
            "sampled_answers_text": sampled_answers_text,
        },
        uploaded_documents=documents,
    )
    return AnalyzeResponse(
        ok=not validation_warnings,
        mode=resolved_mode,
        selected_methods=selected,
        results=results,
        summary=summarize_results(results),
        warnings=upload_warnings + validation_warnings,
    )
