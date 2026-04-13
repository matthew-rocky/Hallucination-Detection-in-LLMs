"""Retrieval-grounded hallucination checking backed by a local vector index."""

import time
from typing import Any

from detectors.base import make_result, make_unavailable
from retrieval.chunking import SourceDoc, ingest_docs
from retrieval.indexing import VectorIndex
from retrieval.search import ground_answer
from utils.text_utils import safe_text


def _trace_from_steps(intermediate_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for step in intermediate_steps:
        hits = step.get("retrieved_hits", [])
        selected_status = step.get("selected_status", "")
        traces.append(
            {
                "claim": step.get("claim", ""),
                "selected_best_chunk_id": hits[0].get("chunk_id", "") if hits else "",
                "selected_best_status": selected_status,
                "selected_best_source_label": hits[0].get("source_label", "") if hits else "",
                "selected_best_page_title": hits[0].get("title", "") if hits else "",
                "live_web_status": "not_requested",
                "retrieved_chunks": [
                    {
                        "chunk_id": hit.get("chunk_id", ""),
                        "source_label": hit.get("source_label", ""),
                        "page_title": hit.get("title", ""),
                        "source_url": hit.get("metadata", {}).get("url", ""),
                        "retrieval_score": hit.get("score", 0.0),
                        "assessment": hit.get("grounding_status", ""),
                        "chunk_text": hit.get("text", ""),
                    }
                    for hit in hits
                ],
            }
        )
    return traces


def _build_summary(counts: dict[str, int], document_count: int, backend_name: str) -> str:
    if counts["contradicted"]:
        return (
            f"Indexed {document_count} document(s) and grounded answer claims with {backend_name} embeddings, "
            f"finding {counts['supported']} supported, {counts['contradicted']} contradicted, and "
            f"{counts['insufficient evidence']} insufficient-evidence claim(s)."
        )
    if counts["insufficient evidence"]:
        return (
            f"Indexed {document_count} document(s) and found partial support, but "
            f"{counts['insufficient evidence']} claim(s) still lacked enough evidence."
        )
    return (
        f"Indexed {document_count} document(s) and found direct support for the extracted claims using {backend_name} retrieval."
    )


def run_retrieval_check(
    *,
    question: str,
    answer: str,
    source_text: str = "",
    evidence_text: str = "",
    extra_documents: list[SourceDoc | dict[str, Any]] | None = None,
    method_name: str = "Retrieval-Grounded Checker",
    family: str = "retrieval-grounded",
    top_k: int = 4,
    preferred_backend: str = "sentence-transformer",
    model_name: str | None = None,
    impl_status: str = "implemented",
) -> dict[str, Any]:
    """Ground an answer against local evidence using a real vector retrieval pipeline."""
    start_time = time.perf_counter()
    documents = ingest_docs(
        source_text=source_text,
        evidence_text=evidence_text,
        extra_documents=extra_documents,
    )
    if not safe_text(answer):
        return make_unavailable(
            method_name=method_name,
            family=family,
            summary="The answer is empty, so there are no claims to ground against retrieved evidence.",
            explanation="The answer is empty, so there are no claims to ground against retrieved evidence.",
            evidence_used="No answer text was provided.",
            limitations="Retrieval-grounded checking requires answer text plus local evidence documents.",
            impl_status="unavailable",
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )
    if not documents:
        return make_unavailable(
            method_name=method_name,
            family=family,
            summary="No local source text, evidence text, or uploaded documents were available to index.",
            explanation="No local source text, evidence text, or uploaded documents were available to index.",
            evidence_used="No evidence corpus was supplied.",
            limitations=(
                "This retrieval-grounded checker builds a local vector index from user-provided evidence. "
                "At least one source, evidence block, or uploaded document is required."
            ),
            impl_status="unavailable",
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )

    try:
        index = VectorIndex.from_documents(
            documents,
            preferred_backend=preferred_backend,
            model_name=model_name,
            max_sentences=1,
            max_chars=420,
            overlap=0,
        )
    except Exception as exc:
        return make_unavailable(
            method_name=method_name,
            family=family,
            summary="The local retrieval index could not be built from the supplied documents.",
            explanation=f"Index construction failed: {exc}",
            evidence_used="Local source, evidence, or uploaded documents.",
            limitations=(
                "This method depends on local chunking, embeddings, and vector indexing. "
                "If the embedding backend cannot initialize, the retrieval pipeline cannot run honestly."
            ),
            impl_status="unavailable",
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )

    grounded = ground_answer(
        question=question,
        answer=answer,
        index=index,
        top_k=top_k,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    score = grounded["score"]
    confidence = 0.35 + (min(len(grounded["citations"]), 5) * 0.1)
    confidence = min(0.95, confidence)

    explanation = (
        "Implemented local retrieval-grounded checker with document ingestion, chunking, embeddings, an in-memory vector index, top-k retrieval, "
        "and claim-level support or contradiction classification with citations."
    )
    limitations = (
        "This prototype grounds claims against the locally indexed corpus only. It does not reason over source trust, long-range multi-hop evidence, or full answer regeneration."
    )

    top_evidence = _trace_from_steps(grounded["intermediate_steps"])
    return make_result(
        method_name=method_name,
        family=family,
        score=score,
        confidence=confidence,
        summary=_build_summary(grounded["counts"], len(documents), grounded["retrieval_backend"]),
        explanation=explanation,
        evidence_used=(
            f"{len(documents)} locally indexed document(s) chunked and embedded with the {grounded['retrieval_backend']} backend."
        ),
        evidence=grounded["evidence"],
        citations=grounded["citations"],
        intermediate_steps=grounded["intermediate_steps"],
        claim_findings=grounded["claim_results"],
        limitations=limitations,
        impl_status=impl_status,
        latency_ms=latency_ms,
        metadata={
            "retrieval_backend": grounded["retrieval_backend"],
            "index_backend": grounded["index_backend"],
            "document_count": len(documents),
            "chunk_count": len(grounded["chunk_catalog"]),
            "question_conditioned_matching": grounded["question_conditioned_matching"],
            "top_k": top_k,
        },
        retrieval_counts=grounded["counts"],
        top_retrieved_evidence=top_evidence,
        chunk_catalog=grounded["chunk_catalog"],
    )

