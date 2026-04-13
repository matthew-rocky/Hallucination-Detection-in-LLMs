"""Deterministic helpers for short grounded answer revision."""

import re
from typing import Any

from utils.text_utils import lexical_overlap, normalize_text, safe_text, split_into_sentences, tokenize


NOTE_PREFIX_PATTERN = re.compile(
    r"^(?:[A-Za-z]+(?:\s+[A-Za-z]+){0,2}\s+note(?:\s+\d+)?|Meeting\s+note|Planning\s+note|"
    r"Evidence\s+note\s+\d+|Dossier\s+note\s+\d+|Regulatory\s+note|Supplier\s+note|Briefing\s+note):\s*",
    flags=re.IGNORECASE,
)

SOURCE_PRIORITY = {
    "source": 0,
    "source_text": 0,
    "reference": 0,
    "evidence": 1,
    "evidence_text": 1,
    "uploaded_document": 1,
    "uploaded": 1,
    "web": 2,
}

REVISION_OK_STATUSES = {
    "verified",
    "supported",
    "abstractly_supported",
    "weakly_supported",
    "contradicted",
}


def clean_sentence(text: str | None) -> str:
    """Strip note prefixes and normalize one grounded sentence for a short revision."""
    cleaned = safe_text(text)
    if not cleaned:
        return ""
    cleaned = NOTE_PREFIX_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -\t")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _sentence_rank(record: dict[str, Any], hit: dict[str, Any], sentence: str, original_index: int) -> tuple[float, float, float, int]:
    source_label = safe_text(hit.get("source_label") or hit.get("source_type") or "evidence").lower()
    source_priority = SOURCE_PRIORITY.get(source_label, 1)
    record_status = safe_text(record.get("status")).lower()
    hit_status = safe_text(hit.get("grounding_status") or record_status).lower()
    record_status_bonus = 0.0 if record_status in {"verified", "supported", "abstractly_supported", "weakly_supported"} else 0.04
    hit_status_bonus = 0.0 if hit_status in {"supported", "abstractly_supported", "weakly_supported", "verified"} else 0.03
    claim_overlap = lexical_overlap(record.get("claim", ""), sentence)
    numeric_bonus = 0.02 if any(char.isdigit() for char in record.get("claim", "")) and any(char.isdigit() for char in sentence) else 0.0
    return (source_priority + record_status_bonus + hit_status_bonus - numeric_bonus, -claim_overlap, -float(hit.get("score", 0.0)), original_index)


def _dup_sentence(candidate: str, existing: list[str]) -> bool:
    normalized_candidate = normalize_text(candidate)
    candidate_tokens = set(tokenize(candidate, remove_stopwords=False))
    for sentence in existing:
        normalized_existing = normalize_text(sentence)
        if not normalized_existing:
            continue
        if normalized_candidate == normalized_existing:
            return True
        if normalized_candidate in normalized_existing or normalized_existing in normalized_candidate:
            return True
        existing_tokens = set(tokenize(sentence, remove_stopwords=False))
        if candidate_tokens and existing_tokens:
            overlap = len(candidate_tokens & existing_tokens) / max(len(candidate_tokens | existing_tokens), 1)
            if overlap >= 0.82:
                return True
    return False


def _candidate_hits(record: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    best_hit = record.get("best_hit") or {}
    if safe_text(best_hit.get("text")):
        hits.append(best_hit)
    for hit in record.get("retrieved_hits") or []:
        if not safe_text(hit.get("text")):
            continue
        hits.append(hit)
    independent_answer = safe_text(record.get("independent_answer"))
    if independent_answer:
        hits.append({"text": independent_answer, "source_label": "evidence", "score": 0.0})
    return hits


def make_revision(records: list[dict[str, Any]], max_sentences: int = 3) -> str:
    """Create a short natural revision from grounded records."""
    candidates: list[tuple[tuple[float, float, float, int], str]] = []
    unresolved_count = 0

    for index, record in enumerate(records):
        status = safe_text(record.get("status")).lower()
        if status not in REVISION_OK_STATUSES:
            unresolved_count += 1
            continue

        sentence_candidates: list[tuple[tuple[float, float, float, int], str]] = []
        for hit_index, hit in enumerate(_candidate_hits(record)):
            for raw_sentence in split_into_sentences(hit.get("text") or ""):
                sentence = clean_sentence(raw_sentence)
                if not sentence:
                    continue
                if len(tokenize(sentence, remove_stopwords=False)) < 4:
                    continue
                sentence_candidates.append((_sentence_rank(record, hit, sentence, hit_index), sentence))

        if not sentence_candidates:
            unresolved_count += 1
            continue

        best_rank, best_sentence = sorted(sentence_candidates, key=lambda item: item[0])[0]
        candidates.append((best_rank, best_sentence))

    ranked_sentences = [sentence for _rank, sentence in sorted(candidates, key=lambda item: item[0])]

    selected: list[str] = []
    for sentence in ranked_sentences:
        if _dup_sentence(sentence, selected):
            continue
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    if not selected:
        return "Available evidence does not clearly support a reliable revised answer."

    if unresolved_count and len(selected) < max_sentences:
        suffix = "detail" if unresolved_count == 1 else "details"
        selected.append(f"Available evidence does not resolve the remaining {suffix}.")

    return " ".join(selected[:max_sentences])