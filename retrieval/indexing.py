"""Vector indexing for local retrieval-backed detectors."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from .chunking import SourceDoc, chunk_documents
from .embeddings import TextEmbedder, EmbedderState, make_embedder, cosine_similarity, restore_embedder


@dataclass(slots=True)
class SearchHit:
    """A retrieved chunk with similarity metadata."""

    chunk: dict
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {**self.chunk, "score": round(float(self.score), 4), "rank": self.rank}


class VectorIndex:
    """A simple in-memory vector index for local evidence retrieval."""

    def __init__(
        self,
        *,
        chunks: list[dict],
        embeddings: np.ndarray,
        embedder: TextEmbedder,
        index_backend: str = "numpy-cosine",
    ) -> None:
        self.chunks = chunks
        self.embeddings = np.asarray(embeddings, dtype=float)
        self.embedder = embedder
        self.index_backend = index_backend

    @classmethod
    def from_documents(
        cls,
        documents: list[SourceDoc],
        *,
        embedder: TextEmbedder | None = None,
        preferred_backend: str = "sentence-transformer",
        model_name: str | None = None,
        max_sentences: int = 3,
        max_chars: int = 420,
        overlap: int = 1,
    ) -> "VectorIndex":
        resolved_embedder = embedder or make_embedder(
            preferred_backend=preferred_backend,
            model_name=model_name,
        )
        chunks = chunk_documents(
            documents,
            max_sentences=max_sentences,
            max_chars=max_chars,
            overlap=overlap,
        )
        texts = [chunk["text"] for chunk in chunks]
        resolved_embedder.fit(texts)
        embeddings = resolved_embedder.encode(texts)
        return cls(
            chunks=chunks,
            embeddings=embeddings,
            embedder=resolved_embedder,
            index_backend="numpy-cosine",
        )

    def is_empty(self) -> bool:
        return not self.chunks or self.embeddings.size == 0

    def search(self, query: str, *, top_k: int = 3) -> list[dict]:
        """Retrieve the top-k chunks for a query."""
        if self.is_empty():
            return []

        query_vector = self.embedder.encode([query])
        scores = cosine_similarity(query_vector, self.embeddings)[0]
        ranked_indices = np.argsort(scores)[::-1][: max(1, top_k)]
        hits: list[dict] = []
        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[int(index)]
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=float(max(0.0, scores[int(index)])),
                    rank=rank,
                ).to_dict()
            )
        return hits

    def save(self, path: str | Path) -> None:
        """Persist the index to disk for reuse."""
        payload = {
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "embedder_state": self.embedder.get_state(),
            "index_backend": self.index_backend,
        }
        with open(Path(path), "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        """Load a previously persisted index."""
        with open(Path(path), "rb") as handle:
            payload = pickle.load(handle)
        embedder_state: EmbedderState = payload["embedder_state"]
        embedder = restore_embedder(embedder_state)
        return cls(
            chunks=list(payload["chunks"]),
            embeddings=np.asarray(payload["embeddings"], dtype=float),
            embedder=embedder,
            index_backend=str(payload.get("index_backend") or "numpy-cosine"),
        )
