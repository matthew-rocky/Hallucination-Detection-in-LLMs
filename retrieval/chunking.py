"""Document ingestion and chunking utilities for local retrieval."""

from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from typing import Any
import json

from utils.text_utils import chunk_text, safe_text


@dataclass(slots=True)
class SourceDoc:
    """A locally available document that can be indexed for retrieval."""

    document_id: str
    title: str
    text: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkRecord:
    """A chunked document segment that can be embedded and cited."""

    chunk_id: str
    document_id: str
    title: str
    source_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


_TEXT_SUFFIXES = {".txt", ".md", ".rst", ".log", ".csv"}
_JSON_SUFFIXES = {".json", ".jsonl"}


def _stable_id(prefix: str, title: str, text: str) -> str:
    digest = md5(f"{title}\n{text}".encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def make_text_doc(
    *,
    title: str,
    text: str,
    source_type: str,
    metadata: dict[str, Any] | None = None,
    document_id: str | None = None,
) -> SourceDoc:
    """Build a source document from raw text."""
    cleaned_text = safe_text(text)
    resolved_title = safe_text(title) or "Untitled document"
    resolved_source_type = safe_text(source_type) or "evidence"
    return SourceDoc(
        document_id=document_id or _stable_id(resolved_source_type, resolved_title, cleaned_text),
        title=resolved_title,
        text=cleaned_text,
        source_type=resolved_source_type,
        metadata=metadata or {},
    )


def _bytes_to_text(raw_bytes: bytes, suffix: str, source_hint: str) -> str:
    if suffix in _TEXT_SUFFIXES:
        return raw_bytes.decode("utf-8", errors="ignore")
    if suffix in _JSON_SUFFIXES:
        return _json_file_to_text(raw_bytes.decode("utf-8", errors="ignore"), suffix)
    if suffix == ".pdf":
        # Optional PDF.
        try:
            from io import BytesIO
            from pypdf import PdfReader
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PDF ingestion requires the optional 'pypdf' dependency.") from exc
        reader = PdfReader(BytesIO(raw_bytes)) if source_hint == "upload" else PdfReader(str(source_hint))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported document type: {suffix}")


def load_doc_path(path: str | Path) -> SourceDoc:
    """Load a UTF-8 text-like document from disk for indexing."""
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    text = _bytes_to_text(resolved.read_bytes(), suffix, str(resolved))
    return make_text_doc(
        title=resolved.name,
        text=text,
        source_type="uploaded_document",
        metadata={"path": str(resolved.resolve())},
    )


def load_doc_bytes(filename: str, raw_bytes: bytes) -> SourceDoc:
    """Load an uploaded document directly from bytes."""
    suffix = Path(filename).suffix.lower()
    text = _bytes_to_text(raw_bytes, suffix, "upload")
    return make_text_doc(
        title=Path(filename).name,
        text=text,
        source_type="uploaded_document",
        metadata={"filename": Path(filename).name},
    )


def _json_file_to_text(raw_text: str, suffix: str) -> str:
    """Convert a simple JSON or JSONL file into text for retrieval."""
    if suffix == ".jsonl":
        lines = [line for line in raw_text.splitlines() if safe_text(line)]
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(line)
        return "\n".join(_flatten_json_record(record) for record in records)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text
    return _flatten_json_record(parsed)


def _flatten_json_record(record: Any) -> str:
    if isinstance(record, dict):
        return " ".join(f"{key}: {_flatten_json_record(value)}" for key, value in record.items())
    if isinstance(record, list):
        return " ".join(_flatten_json_record(item) for item in record)
    return safe_text(str(record))


def ingest_docs(
    *,
    source_text: str = "",
    evidence_text: str = "",
    extra_documents: list[SourceDoc | dict[str, Any]] | None = None,
) -> list[SourceDoc]:
    """Normalize user-provided local evidence into indexable documents."""
    documents: list[SourceDoc] = []

    if safe_text(source_text):
        documents.append(
            make_text_doc(
                title="Source text",
                text=source_text,
                source_type="source_text",
            )
        )
    if safe_text(evidence_text):
        documents.append(
            make_text_doc(
                title="Evidence text",
                text=evidence_text,
                source_type="evidence_text",
            )
        )

    for item in extra_documents or []:
        if isinstance(item, SourceDoc):
            documents.append(item)
            continue
        documents.append(
            make_text_doc(
                title=str(item.get("title") or item.get("name") or "Uploaded document"),
                text=str(item.get("text") or ""),
                source_type=str(item.get("source_type") or "uploaded_document"),
                metadata=dict(item.get("metadata") or {}),
                document_id=item.get("document_id"),
            )
        )

    deduped: list[SourceDoc] = []
    seen: set[str] = set()
    for document in documents:
        key = f"{document.title}|{document.source_type}|{document.text}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped


def chunk_documents(
    documents: list[SourceDoc],
    *,
    max_sentences: int = 3,
    max_chars: int = 420,
    overlap: int = 1,
) -> list[dict[str, Any]]:
    """Chunk documents while preserving citation-friendly metadata."""
    chunk_records: list[dict[str, Any]] = []
    for document in documents:
        chunks = chunk_text(
            document.text,
            max_sentences=max_sentences,
            max_chars=max_chars,
            overlap=overlap,
        )
        for index, chunk in enumerate(chunks, start=1):
            chunk_records.append(
                {
                    "chunk_id": f"{document.document_id}-c{index}",
                    "document_id": document.document_id,
                    "title": document.title,
                    "source_label": document.source_type,
                    "source_type": document.source_type,
                    "text": chunk,
                    "citation_id": f"{document.document_id}#{index}",
                    "metadata": {**document.metadata, "chunk_number": index},
                }
            )
    return chunk_records

