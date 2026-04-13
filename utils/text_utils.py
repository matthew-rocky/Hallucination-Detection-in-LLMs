"""Lightweight text processing utilities for the prototype."""

from collections import Counter
import re
from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
except Exception:
    TfidfVectorizer = None
    sk_cosine = None


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

NEGATION_CUES = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "cannot",
    "can't",
    "isn't",
    "wasn't",
    "didn't",
    "won't",
}

CONTRAST_PAIRS = [
    ("always", "never"),
    ("increase", "decrease"),
    ("higher", "lower"),
    ("more", "less"),
    ("before", "after"),
    ("accepted", "rejected"),
]

PREDICATE_HINTS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "became",
    "becomes",
    "remain",
    "remains",
    "remained",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "may",
    "might",
    "run",
    "ran",
    "provide",
    "provided",
    "provides",
    "increase",
    "increased",
    "increases",
    "reduce",
    "reduced",
    "reduces",
    "lead",
    "led",
    "leads",
    "result",
    "resulted",
    "results",
    "cause",
    "caused",
    "causes",
    "show",
    "showed",
    "shows",
    "report",
    "reported",
    "reports",
    "say",
    "expects",
    "expected",
    "expect",
    "charges",
    "charge",
    "saw",
    "keeps",
    "kept",
    "keep",
    "said",
    "says",
    "approve",
    "approved",
    "approves",
    "extend",
    "extended",
    "extends",
    "return",
    "returned",
    "returns",
    "land",
    "landed",
    "lands",
    "send",
    "sent",
    "sends",
    "move",
    "moved",
    "moves",
    "win",
    "won",
    "wins",
    "discover",
    "discovered",
    "discovers",
    "work",
    "worked",
    "works",
    "locate",
    "located",
    "locates",
    "vote",
    "voted",
    "votes",
}

SUBJECT_STARTERS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "they",
    "he",
    "she",
    "we",
    "i",
}

PREDICATE_REGEX = "|".join(sorted(re.escape(term) for term in PREDICATE_HINTS))
CLAUSE_START_REGEX = (
    rf"(?:and\s+)?(?:it\b|they\b|he\b|she\b|the\b|this\b|that\b|these\b|those\b|"
    rf"[A-Z][a-z]+\b|{PREDICATE_REGEX}|[a-z]+ed\b|[a-z]+ing\b)"
)


def safe_text(text: str | None) -> str:
    """Return a trimmed string, even if the input is None."""
    return (text or "").strip()


def has_text(text: str | None) -> bool:
    """Check whether a string contains non-whitespace text."""
    return bool(safe_text(text))


def normalize_text(text: str | None) -> str:
    """Lower-case and collapse whitespace for robust matching."""
    normalized = safe_text(text).lower()
    return re.sub(r"\s+", " ", normalized)


def tokenize(text: str | None, remove_stopwords: bool = True) -> List[str]:
    """Tokenize a string into lowercase word tokens."""
    tokens = re.findall(r"[a-zA-Z0-9']+", normalize_text(text))
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def split_into_sentences(text: str | None) -> List[str]:
    """Split text into simple sentence-like units."""
    cleaned = safe_text(text)
    if not cleaned:
        return []

    lines = [line.strip() for line in re.split(r"[\r\n]+", cleaned) if line.strip()]
    sentences: List[str] = []
    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            sentence = part.strip(" -\t")
            if sentence:
                sentences.append(sentence)
    return sentences


def _is_predicate_like(token: str) -> bool:
    """Heuristic check for tokens that look like predicates or verb heads."""
    lowered = token.lower().strip(".,;:!?")
    return lowered in PREDICATE_HINTS or lowered.endswith(("ed", "ing"))


def _subject_prefix(sentence: str) -> str:
    """Extract a short subject prefix that can be reused for split subclaims."""
    words = re.findall(r"[A-Za-z0-9']+", sentence)
    prefix = []
    for word in words:
        if _is_predicate_like(word):
            break
        prefix.append(word)
        if len(prefix) >= 8:
            break

    if 1 < len(prefix) <= 8:
        return " ".join(prefix)
    return ""


def _needs_subject_prefix(fragment: str) -> bool:
    """Decide whether a split fragment needs the sentence subject restored."""
    words = re.findall(r"[A-Za-z0-9']+", fragment)
    if not words:
        return False

    first = words[0]
    first_lower = first.lower()
    if first_lower in SUBJECT_STARTERS:
        return False
    if len(words) >= 2 and _is_predicate_like(words[1]):
        return False
    if _is_predicate_like(first_lower):
        return True
    if fragment[:1].islower():
        return True
    return False


def _normalize_fragment(fragment: str, subject_prefix: str) -> str:
    """Clean a candidate claim fragment and restore the subject when needed."""
    cleaned = fragment.strip(" ,;:")
    skip_subject_prefix = bool(re.match(r"^with\s+", cleaned, flags=re.IGNORECASE))
    cleaned = re.sub(r"^(and|but|while|with)\s+", "", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return ""

    if subject_prefix and not skip_subject_prefix and _needs_subject_prefix(cleaned):
        cleaned = f"{subject_prefix} {cleaned[:1].lower() + cleaned[1:]}"
    elif cleaned[:1].islower():
        cleaned = cleaned[:1].upper() + cleaned[1:]

    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _split_long_sentence(sentence: str) -> List[str]:
    """Conservatively decompose one long sentence into smaller factual claims."""
    cleaned = safe_text(sentence)
    if len(tokenize(cleaned, remove_stopwords=False)) < 12:
        return [cleaned]

    subject_prefix = _subject_prefix(cleaned)
    fragments = [cleaned]
    clause_head = rf"(?:{PREDICATE_REGEX}|[a-z]+(?:ed|ing|s)\b)"
    split_patterns = [
        (r";\s+", re.IGNORECASE),
        (r"\s+but\s+", re.IGNORECASE),
        (r"\s+while\s+", re.IGNORECASE),
        (rf",\s+(?=(?:[A-Za-z][A-Za-z0-9'\-]+\s+){{0,4}}{clause_head})", re.IGNORECASE),
        (rf"\s+and\s+(?=(?:[A-Za-z][A-Za-z0-9'\-]+\s+){{0,4}}{clause_head})", re.IGNORECASE),
        (r",\s+(?=with\s+(?:[A-Za-z][A-Za-z0-9'\-]+\s+){0,3}(?:expected|planned|scheduled|pending|discounts?|pricing|price|cost|costs|admission|entry|remain|remains|stays|staying|open|closed|available|located|under|through|during))", re.IGNORECASE),
    ]

    for pattern, flags in split_patterns:
        new_fragments = []
        for fragment in fragments:
            parts = [part for part in re.split(pattern, fragment, flags=flags) if part]
            if len(parts) > 1:
                new_fragments.extend(parts)
            else:
                new_fragments.append(fragment)
        fragments = new_fragments

    claims = []
    seen = set()
    for fragment in fragments:
        normalized_fragment = _normalize_fragment(fragment, subject_prefix)
        if not normalized_fragment:
            continue
        if len(tokenize(normalized_fragment, remove_stopwords=False)) < 3:
            continue
        normalized_key = normalize_text(normalized_fragment)
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        claims.append(normalized_fragment)

    return claims or [cleaned]


def _question_claim_match(question: str, claim: str) -> str:
    """Rewrite common short factoid questions into a declarative matching target."""
    cleaned_question = safe_text(question).rstrip(" ?")
    if not cleaned_question:
        return ""

    copular_match = re.match(r"(?i)^(who|what|where)\s+(is|are|was|were)\s+(.+)$", cleaned_question)
    if copular_match:
        _, verb, subject = copular_match.groups()
        declarative = f"{subject} {verb.lower()} {claim}"
        return declarative if declarative[-1] in ".!?" else declarative + "."

    temporal_match = re.match(r"(?i)^when\s+(was|were)\s+(.+)$", cleaned_question)
    if temporal_match:
        verb, subject = temporal_match.groups()
        declarative = f"{subject} {verb.lower()} {claim}"
        return declarative if declarative[-1] in ".!?" else declarative + "."

    return ""


def match_claim(claim: str | None, question: str | None = "") -> str:
    """Add question context when a claim is too short to match reliably on its own."""
    cleaned_claim = safe_text(claim)
    if not cleaned_claim:
        return ""

    if len(tokenize(cleaned_claim, remove_stopwords=False)) >= 3:
        return cleaned_claim

    cleaned_question = safe_text(question).rstrip(" ?")
    if not cleaned_question:
        return cleaned_claim

    declarative_match = _question_claim_match(cleaned_question, cleaned_claim)
    if declarative_match:
        return declarative_match

    contextualized = f"{cleaned_question} {cleaned_claim}"
    if contextualized[-1] not in ".!?":
        contextualized += "."
    return contextualized


def extract_claims(text: str | None, max_claims: int | None = 12) -> List[str]:
    """Extract lightweight claims by splitting sentences and simple compound clauses."""
    sentences = split_into_sentences(text)
    claims: List[str] = []
    for sentence in sentences:
        for claim in _split_long_sentence(sentence):
            if len(tokenize(claim, remove_stopwords=False)) >= 3:
                claims.append(claim)

    if not claims and has_text(text):
        claims = [safe_text(text)]
    if max_claims is not None:
        claims = claims[:max_claims]
    return claims


def chunk_text(
    text: str | None,
    max_sentences: int = 3,
    max_chars: int = 450,
    overlap: int = 1,
) -> List[str]:
    """Chunk text into short overlapping passages for local matching."""
    sentences = split_into_sentences(text)
    if sentences:
        expanded_sentences = []
        for sentence in sentences:
            if max_sentences == 1 and len(tokenize(sentence, remove_stopwords=False)) >= 12:
                expanded_sentences.extend(_split_long_sentence(sentence))
            else:
                expanded_sentences.append(sentence)
        sentences = expanded_sentences
    if not sentences:
        raw = safe_text(text)
        if not raw:
            return []
        words = raw.split()
        if not words:
            return []
        chunk_size = max(60, max_chars // 6)
        chunks = []
        start = 0
        step = max(1, chunk_size - 15)
        while start < len(words):
            chunks.append(" ".join(words[start : start + chunk_size]))
            start += step
        return chunks

    chunks = []
    start_index = 0
    step = max(1, max_sentences - overlap)
    while start_index < len(sentences):
        current_chunk = []
        current_length = 0
        index = start_index
        while index < len(sentences) and len(current_chunk) < max_sentences:
            sentence = sentences[index]
            projected_length = current_length + len(sentence) + (1 if current_chunk else 0)
            if current_chunk and projected_length > max_chars:
                break
            current_chunk.append(sentence)
            current_length = projected_length
            index += 1

        if not current_chunk:
            current_chunk = [sentences[start_index]]

        chunks.append(" ".join(current_chunk))
        start_index += step

    deduplicated = []
    seen = set()
    for chunk in chunks:
        normalized = normalize_text(chunk)
        if normalized not in seen:
            seen.add(normalized)
            deduplicated.append(chunk)
    return deduplicated


def lexical_overlap(text_a: str | None, text_b: str | None) -> float:
    """Compute a small Jaccard-style lexical overlap score."""
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def extract_numbers(text: str | None) -> set[str]:
    """Extract simple numeric tokens from text."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", safe_text(text)))


def contains_phrase(text: str | None, phrase: str) -> bool:
    """Check for a phrase using word boundaries when possible."""
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    pattern = r"\b" + re.escape(normalized_phrase) + r"\b"
    return bool(re.search(pattern, normalized_text))


def find_phrase_hits(text: str | None, phrases: Iterable[str]) -> List[str]:
    """Return phrases that appear in the text."""
    return [phrase for phrase in phrases if contains_phrase(text, phrase)]


def contains_negation(text: str | None) -> bool:
    """Detect simple negation cues."""
    return any(token in NEGATION_CUES for token in tokenize(text, remove_stopwords=False))


def estimate_specificity(text: str | None) -> float:
    """Estimate how detail-heavy a sentence looks using surface cues."""
    raw = safe_text(text)
    if not raw:
        return 0.0

    total_tokens = max(1, len(re.findall(r"\b\w+\b", raw)))
    numeric_count = len(extract_numbers(raw))
    capitalized_words = re.findall(r"\b[A-Z][a-z]+\b", raw)
    acronyms = re.findall(r"\b[A-Z]{2,}\b", raw)
    specificity_points = (
        (1.5 * numeric_count)
        + (0.5 * len(capitalized_words))
        + (0.75 * len(acronyms))
    )
    return specificity_points / total_tokens


def truncate_text(text: str | None, max_chars: int = 160) -> str:
    """Truncate text for compact UI display."""
    cleaned = safe_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _capital_terms(text: str | None) -> set[str]:
    """Extract simple capitalized terms for lightweight entity mismatch checks."""
    return {
        term.lower()
        for term in re.findall(r"\b[A-Z][a-z]+\b", safe_text(text))
        if len(term) > 2
    }


def _copular_tail(text: str | None) -> set[str]:
    """Extract content tokens after simple copular verbs."""
    match = re.search(
        r"\b(?:is|are|was|were|became|becomes|remains|remained)\b\s+(.+)",
        normalize_text(text),
    )
    if not match:
        return set()
    return {token for token in tokenize(match.group(1)) if len(token) > 2}


def find_cues(claim: str, reference: str) -> List[str]:
    """Approximate contradiction cues without full NLI."""
    cues = []
    overlap = lexical_overlap(claim, reference)
    shared_terms = set(tokenize(claim)) & set(tokenize(reference))
    if overlap < 0.10 and len(shared_terms) < 2:
        return cues

    claim_has_negation = contains_negation(claim)
    ref_has_negation = contains_negation(reference)
    if claim_has_negation != ref_has_negation:
        cues.append("negation mismatch")

    claim_numbers = extract_numbers(claim)
    reference_numbers = extract_numbers(reference)
    if claim_numbers and reference_numbers:
        claim_only = claim_numbers - reference_numbers
        reference_only = reference_numbers - claim_numbers
        if claim_numbers.isdisjoint(reference_numbers):
            cues.append("different numeric details")
        elif len(claim_numbers) == 1 and len(reference_numbers) == 1 and claim_only and reference_only:
            cues.append("different numeric details")

    normalized_claim = normalize_text(claim)
    normalized_reference = normalize_text(reference)
    claim_tokens = set(tokenize(claim))
    reference_tokens = set(tokenize(reference))

    if overlap >= 0.85 and claim_tokens == reference_tokens:
        return cues

    claim_entities = _capital_terms(claim)
    reference_entities = _capital_terms(reference)
    claim_tail_tokens = _copular_tail(claim)
    reference_tail_tokens = _copular_tail(reference)

    if (
        len(shared_terms) >= 1
        and claim_entities
        and reference_entities
        and (claim_entities - reference_entities)
        and (reference_entities - claim_entities)
    ):
        cues.append("different named entity details")
    elif (
        overlap >= 0.18
        and claim_tail_tokens
        and reference_tail_tokens
        and (claim_tail_tokens - reference_tail_tokens)
        and (reference_tail_tokens - claim_tail_tokens)
    ):
        cues.append("different predicate details")

    for left_term, right_term in CONTRAST_PAIRS:
        claim_has_pair = contains_phrase(normalized_claim, left_term) or contains_phrase(
            normalized_claim, right_term
        )
        reference_has_pair = contains_phrase(
            normalized_reference, left_term
        ) or contains_phrase(normalized_reference, right_term)
        if not (claim_has_pair and reference_has_pair):
            continue
        if (
            contains_phrase(normalized_claim, left_term)
            and contains_phrase(normalized_reference, right_term)
        ) or (
            contains_phrase(normalized_claim, right_term)
            and contains_phrase(normalized_reference, left_term)
        ):
            cues.append(f"opposing wording around '{left_term}/{right_term}'")
            break

    return cues


@lru_cache(maxsize=1)
def get_st_model():
    """Load a compact embedding model when available.

    If model loading fails, the app falls back to TF-IDF similarity so the demo
    still runs in offline or restricted environments.
    """

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    try:
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception:
        return None


def _dense_cosine(left_vectors: np.ndarray, right_vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity without relying on scikit-learn."""
    left = np.asarray(left_vectors, dtype=float)
    right = np.asarray(right_vectors, dtype=float)

    if left.ndim == 1:
        left = left.reshape(1, -1)
    if right.ndim == 1:
        right = right.reshape(1, -1)

    if left.size == 0 or right.size == 0:
        return np.zeros((len(left), len(right)))

    left_norms = np.linalg.norm(left, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right, axis=1, keepdims=True)

    normalized_left = np.divide(left, np.clip(left_norms, 1e-12, None))
    normalized_right = np.divide(right, np.clip(right_norms, 1e-12, None))
    return normalized_left @ normalized_right.T


def _sim_features(text: str) -> list[str]:
    """Build simple unigram and bigram features for local TF-IDF fallback."""
    tokens = tokenize(text)
    bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _numpy_tfidf_sim(
    cleaned_queries: Sequence[str],
    cleaned_candidates: Sequence[str],
) -> tuple[np.ndarray, str]:
    """Approximate TF-IDF similarity when optional ML packages are unavailable."""
    documents = list(cleaned_queries) + list(cleaned_candidates)
    feature_counts = [Counter(_sim_features(document)) for document in documents]

    vocabulary = sorted({feature for counts in feature_counts for feature in counts})
    if not vocabulary:
        return np.zeros((len(cleaned_queries), len(cleaned_candidates))), "numpy-tfidf-fallback"

    feature_index = {feature: index for index, feature in enumerate(vocabulary)}
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=float)

    for row_index, counts in enumerate(feature_counts):
        total_terms = sum(counts.values())
        if not total_terms:
            continue
        for feature, count in counts.items():
            matrix[row_index, feature_index[feature]] = count / total_terms

    document_frequency = np.zeros(len(vocabulary), dtype=float)
    for counts in feature_counts:
        for feature in counts:
            document_frequency[feature_index[feature]] += 1.0

    idf = np.log((1.0 + len(documents)) / (1.0 + document_frequency)) + 1.0
    matrix *= idf
    return _dense_cosine(
        matrix[: len(cleaned_queries)],
        matrix[len(cleaned_queries) :],
    ), "numpy-tfidf-fallback"


def sim_matrix(
    query_texts: Sequence[str],
    candidate_texts: Sequence[str],
) -> tuple[np.ndarray, str]:
    """Compute semantic similarity between claims and candidate passages."""
    cleaned_queries = [safe_text(text) for text in query_texts if safe_text(text)]
    cleaned_candidates = [safe_text(text) for text in candidate_texts if safe_text(text)]

    if not cleaned_queries or not cleaned_candidates:
        return np.zeros((len(cleaned_queries), len(cleaned_candidates))), "none"

    model = get_st_model()
    if model is not None:
        query_embeddings = model.encode(
            cleaned_queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        candidate_embeddings = model.encode(
            cleaned_candidates,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _dense_cosine(query_embeddings, candidate_embeddings), "sentence-transformer"

    if TfidfVectorizer is not None and sk_cosine is not None:
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            matrix = vectorizer.fit_transform(cleaned_queries + cleaned_candidates)
            query_matrix = matrix[: len(cleaned_queries)]
            candidate_matrix = matrix[len(cleaned_queries) :]
            return sk_cosine(query_matrix, candidate_matrix), "tfidf-fallback"
        except Exception:
            pass

    return _numpy_tfidf_sim(cleaned_queries, cleaned_candidates)
