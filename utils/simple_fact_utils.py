"""Small explicit fact bank for deterministic local sanity checks."""

from collections import Counter
from dataclasses import dataclass
import re

from utils.text_utils import normalize_text, safe_text, tokenize


@dataclass(frozen=True, slots=True)
class CuratedSimpleFact:
    fact_id: str
    question_patterns: tuple[str, ...]
    canonical_answer: str
    aliases: tuple[str, ...]
    extraction_mode: str
    role_terms: tuple[str, ...] = ()
    action_verbs: tuple[str, ...] = ()
    subject_aliases: tuple[str, ...] = ()


CURATED_SIMPLE_FACTS = (
    CuratedSimpleFact(
        fact_id="capital_canada",
        question_patterns=(
            r"\bcapital of canada\b",
            r"\bcanada'?s capital\b",
            r"\bcapital city of canada\b",
            r"\b(?:what|where)\s+(?:city\s+is\s+)?(?:the\s+)?canada'?s capital(?:\s+city)?\b",
            r"\b(?:what|where)\s+(?:city\s+is\s+)?(?:the\s+)?capital(?:\s+city)?\s+of\s+canada\b",
            r"\bname\s+(?:the\s+)?canada'?s capital(?:\s+city)?\b",
            r"\bname\s+(?:the\s+)?capital(?:\s+city)?\s+of\s+canada\b",
        ),
        canonical_answer="Ottawa",
        aliases=("ottawa",),
        extraction_mode="capital",
        role_terms=("capital",),
        subject_aliases=("canada", "canada's"),
    ),
    CuratedSimpleFact(
        fact_id="capital_japan",
        question_patterns=(
            r"\bcapital of japan\b",
            r"\bjapan'?s capital\b",
            r"\bcapital city of japan\b",
            r"\b(?:what|where)\s+(?:city\s+is\s+)?(?:the\s+)?japan'?s capital(?:\s+city)?\b",
            r"\b(?:what|where)\s+(?:city\s+is\s+)?(?:the\s+)?capital(?:\s+city)?\s+of\s+japan\b",
            r"\bname\s+(?:the\s+)?japan'?s capital(?:\s+city)?\b",
            r"\bname\s+(?:the\s+)?capital(?:\s+city)?\s+of\s+japan\b",
        ),
        canonical_answer="Tokyo",
        aliases=("tokyo",),
        extraction_mode="capital",
        role_terms=("capital",),
        subject_aliases=("japan", "japan's"),
    ),
    CuratedSimpleFact(
        fact_id="largest_planet",
        question_patterns=(
            r"\b(?:largest|biggest) planet(?: in the solar system)?\b",
            r"\bwhich planet is (?:the )?(?:largest|biggest)(?: in the solar system)?\b",
            r"\bwhat is the (?:largest|biggest) planet(?: in the solar system)?\b",
        ),
        canonical_answer="Jupiter",
        aliases=("jupiter",),
        extraction_mode="largest_planet",
        role_terms=("planet",),
        subject_aliases=("solar system",),
    ),
    CuratedSimpleFact(
        fact_id="pride_and_prejudice_author",
        question_patterns=(
            r"\bwho\s+(?:wrote|authored)\s+(?:the\s+novel\s+)?pride and prejudice\b",
            r"\b(?:who is|what is)\s+(?:the\s+)?author of pride and prejudice\b",
            r"\bname\s+(?:the\s+)?author of pride and prejudice\b",
            r"\bauthor of pride and prejudice\b",
            r"\bwriter of pride and prejudice\b",
        ),
        canonical_answer="Jane Austen",
        aliases=("jane austen", "austen"),
        extraction_mode="creator",
        role_terms=("author", "writer"),
        action_verbs=("wrote", "authored", "created"),
        subject_aliases=("pride and prejudice",),
    ),
    CuratedSimpleFact(
        fact_id="persistence_of_memory_painter",
        question_patterns=(
            r"\bwho\s+(?:painted|created)\s+(?:the\s+artwork\s+)?(?:the\s+)?persistence of memory\b",
            r"\b(?:who is|what is)\s+(?:the\s+)?painter of the persistence of memory\b",
            r"\bname\s+(?:the\s+)?painter of the persistence of memory\b",
            r"\bpainter of the persistence of memory\b",
            r"\bartist of the persistence of memory\b",
        ),
        canonical_answer="Salvador Dali",
        aliases=("salvador dali", "dali"),
        extraction_mode="creator",
        role_terms=("painter", "artist"),
        action_verbs=("painted", "created"),
        subject_aliases=("the persistence of memory", "persistence of memory"),
    ),
)

_ARTICLE_PREFIX = re.compile(r"^(?:the|a|an)\s+", flags=re.IGNORECASE)
_GENERIC_COPULA = r"(?:is|are|was|were|becomes|became|remains|remained)"
_SHELL_RE = re.compile(
    rf"^(?:the answer|answer|it|this|that)\s+{_GENERIC_COPULA}\s+(?P<candidate>[^.?!,;]+)",
    flags=re.IGNORECASE,
)
_TITLE_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2}\b")
_BOILERPLATE = {
    "answer",
    "the answer",
    "author",
    "the author",
    "writer",
    "the writer",
    "painter",
    "the painter",
    "artist",
    "the artist",
    "capital",
    "the capital",
    "planet",
    "the planet",
    "it",
    "this",
    "that",
}


def _normalize_fact_text(text: str) -> str:
    cleaned = safe_text(text).replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    return normalize_text(cleaned)


def _strip_articles(text: str) -> str:
    return _ARTICLE_PREFIX.sub("", safe_text(text)).strip()


def _canon_answer(fact: CuratedSimpleFact) -> str:
    return _normalize_fact_text(fact.canonical_answer)


def _alias_pattern(alias: str) -> str:
    return r"\b" + re.escape(alias) + r"\b"


def _canon_candidate(fact: CuratedSimpleFact, candidate: str) -> str:
    normalized_candidate = _normalize_fact_text(candidate)
    if not normalized_candidate:
        return ""

    canonical_key = _canon_answer(fact)
    if normalized_candidate == canonical_key:
        return canonical_key

    candidate_words = tokenize(normalized_candidate, remove_stopwords=False)
    for alias in fact.aliases:
        alias_key = _normalize_fact_text(alias)
        if normalized_candidate == alias_key:
            return canonical_key
        if re.search(_alias_pattern(alias_key), normalized_candidate) and len(candidate_words) <= len(tokenize(alias_key, remove_stopwords=False)) + 2:
            return canonical_key

    return normalized_candidate


def _clean_candidate(candidate: str, fact: CuratedSimpleFact | None = None) -> str:
    cleaned = safe_text(candidate).strip(" \t,;:.!?\"'")
    cleaned = re.sub(r"^(?:the answer|answer|it|this|that)\s+(?:is|are|was|were)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:the )?(?:author|writer|painter|artist|capital)\s+(?:is|are|was|were)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = _strip_articles(cleaned)
    normalized = _normalize_fact_text(cleaned)
    if not normalized or normalized in _BOILERPLATE:
        return ""
    if fact is not None:
        return _canon_candidate(fact, cleaned)
    return normalized


def _creator_patterns(fact: CuratedSimpleFact) -> tuple[str, ...]:
    verbs = "|".join(re.escape(verb) for verb in fact.action_verbs)
    roles = "|".join(re.escape(role) for role in fact.role_terms)
    subject = "|".join(re.escape(alias) for alias in fact.subject_aliases)
    candidate = r"(?P<candidate>[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2})"
    return (
        rf"{candidate}\s+(?:{verbs})\s+(?:{subject})\b",
        rf"\b(?:the\s+)?(?:{roles})(?:\s+of\s+(?:{subject}))?\s+{_GENERIC_COPULA}\s+(?P<candidate>[^.?!,;]+)",
        rf"{candidate}\s+{_GENERIC_COPULA}\s+(?:the\s+)?(?:{roles})\b",
    )


def _capital_patterns(fact: CuratedSimpleFact) -> tuple[str, ...]:
    countries = "|".join(re.escape(alias) for alias in fact.subject_aliases)
    candidate = r"(?P<candidate>[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2})"
    return (
        rf"{candidate}\s+{_GENERIC_COPULA}\s+[^.?!]*\b(?:{countries})\s+capital\b",
        rf"{candidate}\s+{_GENERIC_COPULA}\s+[^.?!]*\bcapital(?:\s+city)?\s+of\s+(?:{countries})\b",
        rf"\b(?:{countries})\s+capital(?:\s+city)?\s+{_GENERIC_COPULA}\s+(?P<candidate>[^.?!,;]+)",
        rf"\b(?:the\s+)?capital(?:\s+city)?\s+of\s+(?:{countries})\s+{_GENERIC_COPULA}\s+(?P<candidate>[^.?!,;]+)",
    )


def _planet_patterns() -> tuple[str, ...]:
    candidate = r"(?P<candidate>[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2})"
    return (
        rf"{candidate}\s+{_GENERIC_COPULA}\s+[^.?!]*\b(?:the\s+)?(?:largest|biggest)\s+planet(?:\s+in\s+the\s+solar\s+system)?\b",
        rf"\b(?:the\s+)?(?:largest|biggest)\s+planet(?:\s+in\s+the\s+solar\s+system)?\s+{_GENERIC_COPULA}\s+(?P<candidate>[^.?!,;]+)",
    )


def _answer_patterns(fact: CuratedSimpleFact) -> tuple[str, ...]:
    if fact.extraction_mode == "creator":
        return _creator_patterns(fact)
    if fact.extraction_mode == "capital":
        return _capital_patterns(fact)
    if fact.extraction_mode == "largest_planet":
        return _planet_patterns()
    return ()


def _fact_candidate(fact: CuratedSimpleFact, answer: str) -> str:
    for pattern in _answer_patterns(fact):
        match = re.search(pattern, safe_text(answer), flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_candidate(match.group("candidate"), fact)
        if candidate:
            return candidate

    shell_match = _SHELL_RE.search(safe_text(answer))
    if shell_match:
        candidate = _clean_candidate(shell_match.group("candidate"), fact)
        if candidate:
            return candidate

    return ""


def _titlecase_candidate(answer: str, fact: CuratedSimpleFact | None) -> str:
    spans = []
    for span in _TITLE_ENTITY_RE.findall(safe_text(answer)):
        normalized_span = _normalize_fact_text(span)
        if not normalized_span or normalized_span in _BOILERPLATE:
            continue
        spans.append((span, normalized_span))

    if not spans:
        return ""

    excluded_tokens: set[str] = set()
    if fact is not None:
        for subject in fact.subject_aliases:
            excluded_tokens.update(tokenize(subject, remove_stopwords=False))
        for role in fact.role_terms:
            excluded_tokens.update(tokenize(role, remove_stopwords=False))

    for raw_span, normalized_span in spans:
        span_tokens = set(tokenize(normalized_span, remove_stopwords=False))
        if span_tokens and span_tokens <= excluded_tokens:
            continue
        candidate = _clean_candidate(raw_span, fact)
        if candidate:
            return candidate

    return ""


def find_simple_fact(question: str) -> CuratedSimpleFact | None:
    normalized_question = _normalize_fact_text(question)
    for fact in CURATED_SIMPLE_FACTS:
        if any(re.search(pattern, normalized_question, flags=re.IGNORECASE) for pattern in fact.question_patterns):
            return fact
    return None


def simple_fact_answer(question: str, answer: str) -> str:
    """Extract the short factoid answer span for plurality and sanity checks."""
    cleaned_answer = safe_text(answer).replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    if not cleaned_answer:
        return ""

    fact = find_simple_fact(question)
    if fact is not None:
        candidate = _fact_candidate(fact, cleaned_answer)
        if candidate:
            return candidate

        normalized_answer = _normalize_fact_text(cleaned_answer)
        for alias in fact.aliases:
            alias_key = _normalize_fact_text(alias)
            if re.search(_alias_pattern(alias_key), normalized_answer):
                return _canon_answer(fact)

        candidate = _titlecase_candidate(cleaned_answer, fact)
        if candidate:
            return candidate

    shell_match = _SHELL_RE.search(cleaned_answer)
    if shell_match:
        candidate = _clean_candidate(shell_match.group("candidate"), fact)
        if candidate:
            return candidate

    titlecase_candidate = _titlecase_candidate(cleaned_answer, fact)
    if titlecase_candidate:
        return titlecase_candidate

    words = tokenize(cleaned_answer, remove_stopwords=False)
    if len(words) <= 8:
        candidate = _clean_candidate(cleaned_answer, fact)
        if candidate:
            return candidate
    return _normalize_fact_text(cleaned_answer)


def check_simple_fact(question: str, answer: str) -> dict | None:
    """Evaluate a short factoid answer against the explicit local fact bank."""
    fact = find_simple_fact(question)
    if fact is None:
        return None

    raw_candidate = simple_fact_answer(question, answer)
    if not raw_candidate:
        return None

    candidate = _canon_candidate(fact, raw_candidate)
    canonical_key = _canon_answer(fact)
    verdict = "correct" if candidate == canonical_key else "incorrect"
    return {
        "fact_id": fact.fact_id,
        "canonical_answer": fact.canonical_answer,
        "canonical_aliases": list(fact.aliases),
        "candidate": candidate,
        "raw_candidate": raw_candidate,
        "verdict": verdict,
    }


def simple_fact_plurality(question: str, answers: list[str]) -> dict | None:
    """Summarize short-answer plurality for simple factoid sample sets."""
    if not answers:
        return None

    fact = find_simple_fact(question)
    extracted_values = [simple_fact_answer(question, answer) for answer in answers]
    if fact is not None:
        extracted_values = [_canon_candidate(fact, value) if value else "" for value in extracted_values]
    main_value = extracted_values[0] if extracted_values else ""
    sample_pool = extracted_values[1:] if len(extracted_values) > 1 else extracted_values
    populated_values = [value for value in sample_pool if value]
    if not populated_values:
        return None

    counts = Counter(populated_values)
    plurality_value, plurality_count = counts.most_common(1)[0]
    plurality_share = plurality_count / max(1, len(populated_values))
    majority_differs = bool(main_value and plurality_value and main_value != plurality_value and plurality_share > 0.5)

    canonical_key = None if fact is None else _canon_answer(fact)
    vote_matches_fact = bool(canonical_key is not None and plurality_value == canonical_key)
    main_matches_fact = bool(canonical_key is not None and main_value == canonical_key)

    return {
        "main_value": main_value,
        "sample_values": extracted_values[1:],
        "all_values": extracted_values,
        "plurality_value": plurality_value,
        "plurality_count": plurality_count,
        "plurality_share": plurality_share,
        "majority_disagrees_with_main": majority_differs,
        "plurality_matches_canonical": vote_matches_fact,
        "main_matches_canonical": main_matches_fact,
        "fact_id": None if fact is None else fact.fact_id,
    }
