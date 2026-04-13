"""Shared local retrieval helpers for the qualitative prototype."""

from collections import Counter
from typing import Sequence
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
except Exception:
    TfidfVectorizer = None
    sk_cosine = None

from utils.text_utils import chunk_text, extract_numbers, lexical_overlap, normalize_text, safe_text, tokenize


CAP_TERM_RE = re.compile(r"\b[A-Z][a-z]+\b")


def build_chunk_records(
    text: str,
    chunk_prefix: str,
    source_label: str,
    max_sentences: int = 2,
    max_chars: int = 360,
    overlap: int = 1,
) -> list[dict]:
    """Chunk text locally while preserving stable chunk IDs."""
    chunks = chunk_text(
        text,
        max_sentences=max_sentences,
        max_chars=max_chars,
        overlap=overlap,
    )
    return [
        {
            "chunk_id": f"{chunk_prefix}{index}",
            "source_label": source_label,
            "text": chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def _sim_features(text: str) -> list[str]:
    """Build lightweight unigram and bigram features for ranking."""
    tokens = tokenize(text)
    bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _compute_tfidf_scores(query: str, documents: Sequence[str]) -> tuple[list[float], str]:
    """Compute local TF-IDF scores when optional ML packages are available."""
    cleaned_query = safe_text(query)
    cleaned_documents = [safe_text(document) for document in documents]
    if not cleaned_query or not cleaned_documents:
        return [0.0 for _ in cleaned_documents], "none"

    if TfidfVectorizer is not None and sk_cosine is not None:
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            matrix = vectorizer.fit_transform([cleaned_query] + cleaned_documents)
            scores = sk_cosine(matrix[:1], matrix[1:])[0]
            return [float(score) for score in scores], "tfidf"
        except Exception:
            pass

    query_features = Counter(_sim_features(cleaned_query))
    document_features = [Counter(_sim_features(document)) for document in cleaned_documents]
    if not query_features:
        return [0.0 for _ in cleaned_documents], "token-overlap-fallback"

    query_total = sum(query_features.values()) or 1
    scores = []
    for features in document_features:
        overlap = sum(min(count, features.get(feature, 0)) for feature, count in query_features.items())
        doc_total = sum(features.values()) or 1
        scores.append(float(overlap) / ((query_total + doc_total) / 2.0))
    return scores, "token-overlap-fallback"


def _phrase_overlap(query: str, text: str) -> float:
    """Measure lightweight phrase overlap using normalized bigrams."""
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0

    query_bigrams = {f"{left} {right}" for left, right in zip(query_tokens, query_tokens[1:])}
    text_bigrams = {f"{left} {right}" for left, right in zip(text_tokens, text_tokens[1:])}
    if query_bigrams and text_bigrams:
        return len(query_bigrams & text_bigrams) / len(query_bigrams | text_bigrams)

    query_set = set(query_tokens)
    text_set = set(text_tokens)
    return len(query_set & text_set) / len(query_set | text_set)


def _token_coverage(query: str, text: str) -> float:
    """Estimate how much of the query vocabulary appears in the chunk."""
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _capital_terms(text: str) -> set[str]:
    """Extract simple capitalized terms for overlap checks."""
    return {term.lower() for term in CAP_TERM_RE.findall(safe_text(text)) if len(term) > 2}


def _entity_overlap(query: str, text: str) -> float:
    """Measure lightweight named-entity overlap using capitalized terms."""
    query_entities = _capital_terms(query)
    text_entities = _capital_terms(text)
    if not query_entities:
        return 0.0
    return len(query_entities & text_entities) / len(query_entities)


def _numeric_alignment(query: str, text: str) -> float:
    """Measure numeric agreement between a claim and a chunk."""
    query_numbers = extract_numbers(query)
    text_numbers = extract_numbers(text)
    if not query_numbers:
        return 0.0
    return len(query_numbers & text_numbers) / len(query_numbers)


def rank_local_chunks(
    query: str,
    chunk_records: Sequence[dict],
    top_k: int = 3,
) -> tuple[list[dict], str]:
    """Rank local chunks using TF-IDF plus lightweight overlap features."""
    documents = [safe_text(record.get("text")) for record in chunk_records]
    tfidf_scores, backend = _compute_tfidf_scores(query, documents)

    ranked_records = []
    for record, tfidf_score in zip(chunk_records, tfidf_scores):
        chunk_text = safe_text(record.get("text"))
        lexical_score = lexical_overlap(query, chunk_text)
        phrase_score = _phrase_overlap(query, chunk_text)
        coverage_score = _token_coverage(query, chunk_text)
        entity_score = _entity_overlap(query, chunk_text)
        numeric_score = _numeric_alignment(query, chunk_text)
        semanticish_score = (0.6 * coverage_score) + (0.4 * phrase_score)
        retrieval_score = (
            (0.45 * float(tfidf_score))
            + (0.20 * lexical_score)
            + (0.20 * semanticish_score)
            + (0.10 * entity_score)
            + (0.05 * numeric_score)
        )

        ranked_records.append(
            {
                **record,
                "tfidf_score": round(float(tfidf_score), 3),
                "lexical_overlap": round(lexical_score, 3),
                "phrase_overlap": round(phrase_score, 3),
                "token_coverage": round(coverage_score, 3),
                "entity_overlap": round(entity_score, 3),
                "numeric_alignment": round(numeric_score, 3),
                "semanticish_score": round(semanticish_score, 3),
                "retrieval_score": round(retrieval_score, 3),
                "normalized_text": normalize_text(chunk_text),
            }
        )

    ranked_records.sort(key=lambda item: item["retrieval_score"], reverse=True)
    return ranked_records[: max(1, top_k)], backend