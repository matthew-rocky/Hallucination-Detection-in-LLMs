"""Embedding backends for the local retrieval pipeline."""

from collections import Counter
from dataclasses import dataclass
import pickle

import numpy as np

from utils.text_utils import tokenize

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    TfidfVectorizer = None
    SKLEARN_AVAILABLE = False


@dataclass(slots=True)
class EmbedderState:
    backend_name: str
    model_name: str
    payload: bytes | None = None


class TextEmbedder:
    """Minimal embedding backend interface."""

    backend_name = "base"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def fit(self, texts: list[str]) -> "TextEmbedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def get_state(self) -> EmbedderState:
        return EmbedderState(self.backend_name, self.model_name, None)


class SentenceEmbedder(TextEmbedder):
    """Sentence-transformer embeddings with a local fallback path."""

    backend_name = "sentence-transformer"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)
        return np.asarray(
            self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )


class TfidfEmbedder(TextEmbedder):
    """TF-IDF embeddings with an internal NumPy fallback when sklearn is absent."""

    backend_name = "tfidf"

    def __init__(self, model_name: str = "tfidf") -> None:
        super().__init__(model_name)
        self._mode = "sklearn" if SKLEARN_AVAILABLE and TfidfVectorizer is not None else "numpy"
        self.vectorizer = (
            TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            if self._mode == "sklearn"
            else None
        )
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray = np.zeros(0, dtype=float)
        self._is_fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        if self._mode == "sklearn":
            self.vectorizer.fit(texts or [""])
        else:
            self._fit_numpy_fallback(texts or [""])
        self._is_fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("The TF-IDF embedder must be fitted before encoding queries.")
        feature_count = self._feature_count()
        if not texts:
            return np.zeros((0, feature_count), dtype=float)
        if self._mode == "sklearn":
            return self.vectorizer.transform(texts).toarray().astype(float)
        return self._numpy_fallback(texts)

    def _feature_count(self) -> int:
        if self._mode == "sklearn":
            return len(getattr(self.vectorizer, "vocabulary_", {}) or {})
        return len(self.vocabulary_)

    def _fit_numpy_fallback(self, texts: list[str]) -> None:
        feature_counts = [Counter(self._sim_features(text)) for text in texts]
        vocabulary = sorted({feature for counts in feature_counts for feature in counts})
        self.vocabulary_ = {feature: index for index, feature in enumerate(vocabulary)}
        if not vocabulary:
            self.idf_ = np.zeros(0, dtype=float)
            return

        document_frequency = np.zeros(len(vocabulary), dtype=float)
        for counts in feature_counts:
            for feature in counts:
                document_frequency[self.vocabulary_[feature]] += 1.0

        total_documents = max(1, len(texts))
        self.idf_ = np.log((1.0 + total_documents) / (1.0 + document_frequency)) + 1.0

    def _numpy_fallback(self, texts: list[str]) -> np.ndarray:
        feature_count = len(self.vocabulary_)
        matrix = np.zeros((len(texts), feature_count), dtype=float)
        if feature_count == 0:
            return matrix

        for row_index, text in enumerate(texts):
            counts = Counter(self._sim_features(text))
            total_terms = sum(counts.values()) or 1
            for feature, count in counts.items():
                column_index = self.vocabulary_.get(feature)
                if column_index is None:
                    continue
                matrix[row_index, column_index] = (count / total_terms) * self.idf_[column_index]
        return matrix

    @staticmethod
    def _sim_features(text: str) -> list[str]:
        tokens = tokenize(text)
        bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def get_state(self) -> EmbedderState:
        if self._mode == "sklearn":
            payload = {
                "mode": "sklearn",
                "vectorizer": self.vectorizer,
            }
        else:
            payload = {
                "mode": "numpy",
                "vocabulary": self.vocabulary_,
                "idf": self.idf_,
            }
        return EmbedderState(
            backend_name=self.backend_name,
            model_name=self.model_name,
            payload=pickle.dumps(payload),
        )

    @classmethod
    def from_state(cls, state: EmbedderState) -> "TfidfEmbedder":
        payload = pickle.loads(state.payload or b"")
        instance = cls.__new__(cls)
        TextEmbedder.__init__(instance, state.model_name)
        instance._is_fitted = True
        instance.vocabulary_ = {}
        instance.idf_ = np.zeros(0, dtype=float)

        payload_mode = payload.get("mode")
        if payload_mode == "sklearn" and SKLEARN_AVAILABLE and TfidfVectorizer is not None:
            instance._mode = "sklearn"
            instance.vectorizer = payload["vectorizer"]
            return instance

        if payload_mode == "sklearn":
            vectorizer = payload["vectorizer"]
            instance._mode = "numpy"
            instance.vectorizer = None
            instance.vocabulary_ = dict(getattr(vectorizer, "vocabulary_", {}) or {})
            idf = getattr(vectorizer, "idf_", None)
            instance.idf_ = np.asarray(idf if idf is not None else np.zeros(len(instance.vocabulary_)), dtype=float)
            return instance

        instance._mode = "numpy"
        instance.vectorizer = None
        instance.vocabulary_ = dict(payload.get("vocabulary") or {})
        stored_idf = payload.get("idf")
        if stored_idf is None:
            instance.idf_ = np.zeros(len(instance.vocabulary_), dtype=float)
        else:
            instance.idf_ = np.asarray(stored_idf, dtype=float)
        return instance


def make_embedder(
    *,
    preferred_backend: str = "sentence-transformer",
    model_name: str | None = None,
) -> TextEmbedder:
    """Create a retrieval embedder with an automatic fallback."""
    preferred = preferred_backend.lower()
    resolved_model_name = model_name or (
        "all-MiniLM-L6-v2" if preferred == "sentence-transformer" else "tfidf"
    )

    if preferred in {"sentence-transformer", "dense"}:
        try:
            return SentenceEmbedder(resolved_model_name)
        except Exception:
            return TfidfEmbedder("tfidf")

    return TfidfEmbedder(resolved_model_name)


def restore_embedder(state: EmbedderState) -> TextEmbedder:
    """Rebuild an embedder from serialized state."""
    if state.backend_name == "tfidf":
        return TfidfEmbedder.from_state(state)
    return make_embedder(preferred_backend=state.backend_name, model_name=state.model_name)


def cosine_similarity(left_vectors: np.ndarray, right_vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity without requiring scikit-learn."""
    left = np.asarray(left_vectors, dtype=float)
    right = np.asarray(right_vectors, dtype=float)

    if left.ndim == 1:
        left = left.reshape(1, -1)
    if right.ndim == 1:
        right = right.reshape(1, -1)
    if left.size == 0 or right.size == 0:
        return np.zeros((len(left), len(right)), dtype=float)

    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    normalized_left = np.divide(left, np.clip(left_norm, 1e-12, None))
    normalized_right = np.divide(right, np.clip(right_norm, 1e-12, None))
    return normalized_left @ normalized_right.T
