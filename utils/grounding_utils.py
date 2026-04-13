"""Shared lightweight claim-to-evidence grounding helpers."""

import re
from typing import Any

from utils.text_utils import (
    contains_phrase,
    find_cues,
    extract_numbers,
    lexical_overlap,
    normalize_text,
    safe_text,
    tokenize,
)


SUPPORT_STATUSES = {"supported", "abstractly_supported", "weakly_supported"}

# Narrow contradiction rules.
PRICING_TERMS = {
    "admission",
    "entry",
    "ticket",
    "tickets",
    "price",
    "prices",
    "pricing",
    "cost",
    "costs",
    "fee",
    "fees",
    "paid",
    "charge",
    "charges",
    "charged",
}
FREE_ACCESS_TERMS = {
    "free",
    "free admission",
    "entry is free",
    "admission is free",
    "no charge",
    "without charge",
}
DISCOUNT_TERMS = {"discount", "discounts", "reduced", "student", "students"}

ONGOING_TERMS = {"phase", "ongoing", "remain", "remains", "under treatment", "still"}
ABSOLUTE_CLAIM_TERMS = {"all", "every", "entire", "completely", "fully", "immediately"}
PARTIAL_STATUS_TERMS = {"some", "part", "partial", "modest", "modestly", "remain", "remains", "still", "temporary"}

APPROVAL_YES_TERMS = {"approved", "certified", "authorized", "received approval", "faa-approved"}
APPROVAL_NO_TERMS = {
    "not approved",
    "not certified",
    "not received approval",
    "has not received",
    "pending",
    "awaiting",
    "deferred",
    "withdrawn",
    "no permanent rollout",
    "not certified for in-flight use",
}

DEMOLITION_TERMS = {"demolition", "demolish", "demolished", "tear down", "torn down"}
REPAIR_TERMS = {"repair", "replacement", "replace", "renovation", "renovate", "hvac", "roof"}
CARE_SETTING_TERMS = {
    "outpatient": {"outpatient"},
    "inpatient": {"inpatient"},
}
EMERGENCY_TERMS = {"emergency department", "emergency wing", "ed"}

DECREASE_TERMS = {"decrease", "decreased", "down", "fell", "drop", "dropped", "reduced"}
INCREASE_TERMS = {"increase", "increased", "up", "rose", "growth"}
EXTENSION_TERMS = {"extended", "extend", "continued", "another quarter", "additional semester", "evaluation only"}
MODEST_TERMS = {"modest", "modestly", "some", "slight", "slightly"}
STRONG_CONTRA_CUES = {
    "negation mismatch",
    "different numeric details",
    "different year/date details",
    "different time-range details",
    "open/closed mismatch",
    "approval-status mismatch",
    "free/paid mismatch",
    "finished/ongoing mismatch",
    "absolute/partial mismatch",
    "increase/decrease mismatch",
    "demolition/repair mismatch",
    "inpatient/outpatient mismatch",
    "emergency-department mismatch",
    "manufacturer/source attribution mismatch",
    "different named entity details",
}
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_YEAR_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:19|20)\d{2}\b",
    flags=re.IGNORECASE,
)
SEASON_YEAR_PATTERN = re.compile(r"\b(?:spring|summer|fall|autumn|winter|early|late)\s+(?:19|20)\d{2}\b", flags=re.IGNORECASE)
TIME_RANGE_PATTERN = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*(?:to|-)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
    flags=re.IGNORECASE,
)
CAP_TERM_RE = re.compile(
    r"\b(?:[A-Z]{2,}|[A-Z][A-Za-z&'-]+)(?:\s+(?:[A-Z]{2,}|[A-Z][A-Za-z&'-]+)){0,3}\b"
)
ORG_PATTERN = r"(?:[A-Z]{2,}|[A-Z][A-Za-z&'-]+)(?:\s+(?:[A-Z]{2,}|[A-Z][A-Za-z&'-]+)){0,3}"
ROLE_PATTERNS = (
    (re.compile(rf"(?:manufactured|made|built)(?:\s+entirely)?\s+by\s+(?P<entity>{ORG_PATTERN})", flags=re.IGNORECASE), "manufacturer"),
    (re.compile(rf"hardware\s+(?:is\s+)?(?:by|from)\s+(?P<entity>{ORG_PATTERN})", flags=re.IGNORECASE), "hardware"),
    (re.compile(rf"software\s+(?:is\s+)?(?:by|from)\s+(?P<entity>{ORG_PATTERN})", flags=re.IGNORECASE), "software"),
    (re.compile(rf"(?P<entity>{ORG_PATTERN})\s+supplies\s+(?:the\s+)?(?:control\s+)?software", flags=re.IGNORECASE), "software"),
)


def phrase_overlap(text_a: str | None, text_b: str | None) -> float:
    """Measure normalized bigram overlap between two texts."""
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    bigrams_a = {f"{left} {right}" for left, right in zip(tokens_a, tokens_a[1:])}
    bigrams_b = {f"{left} {right}" for left, right in zip(tokens_b, tokens_b[1:])}
    if not bigrams_a or not bigrams_b:
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        return len(set_a & set_b) / max(1, len(set_a | set_b))
    return len(bigrams_a & bigrams_b) / max(1, len(bigrams_a | bigrams_b))


def token_coverage(text_a: str | None, text_b: str | None) -> float:
    """Estimate how much of text_a's vocabulary appears in text_b."""
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def entity_overlap(text_a: str | None, text_b: str | None) -> float:
    """Measure simple capitalized-term overlap."""
    entities_a = {term.lower() for term in CAP_TERM_RE.findall(safe_text(text_a)) if len(term) > 2}
    entities_b = {term.lower() for term in CAP_TERM_RE.findall(safe_text(text_b)) if len(term) > 2}
    if not entities_a:
        return 0.0
    return len(entities_a & entities_b) / len(entities_a)


def numeric_alignment(text_a: str | None, text_b: str | None) -> float:
    """Measure how many numeric details from text_a appear in text_b."""
    numbers_a = extract_numbers(text_a)
    numbers_b = extract_numbers(text_b)
    if not numbers_a:
        return 0.0
    return len(numbers_a & numbers_b) / len(numbers_a)


def _contains_any_phrase(text: str, phrases: set[str]) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)


def _has_any_token(text: str, terms: set[str]) -> bool:
    tokens = set(tokenize(text, remove_stopwords=False))
    return bool(tokens & terms)


def _time_markers(text: str) -> set[str]:
    cleaned = safe_text(text)
    return {
        *{match.group(0).lower() for match in MONTH_YEAR_PATTERN.finditer(cleaned)},
        *{match.group(0).lower() for match in SEASON_YEAR_PATTERN.finditer(cleaned)},
    }


def _extract_time_ranges(text: str) -> set[str]:
    return {
        re.sub(r"\s+", " ", match.group(0).replace("-", " to ")).strip().lower()
        for match in TIME_RANGE_PATTERN.finditer(safe_text(text))
    }


def _named_entities(text: str) -> set[str]:
    return {
        term.lower()
        for term in CAP_TERM_RE.findall(safe_text(text))
        if len(term) > 2
    }


def _role_entities(text: str) -> dict[str, str]:
    roles: dict[str, str] = {}
    cleaned = safe_text(text)
    for pattern, role in ROLE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        entity = normalize_text(match.group("entity"))
        entity = re.sub(r"^(?:while|and)\s+", "", entity)
        roles[role] = entity
    if re.search(rf"entirely\s+by\s+(?P<entity>{ORG_PATTERN})", cleaned, flags=re.IGNORECASE):
        roles.setdefault("manufacturer_exclusive", roles.get("manufacturer", ""))
    return {key: value for key, value in roles.items() if value}


def _place_entities(text: str) -> set[str]:
    return {
        normalize_text(match.group("entity"))
        for match in re.finditer(rf"\b(?:in|at|near|outside|inside|within)\s+(?P<entity>{ORG_PATTERN})", safe_text(text))
    }


def _shared_context(claim: str, reference: str) -> bool:
    shared_tokens = set(tokenize(claim)) & set(tokenize(reference))
    shared_entities = _named_entities(claim) & _named_entities(reference)
    return bool(
        len(shared_tokens) >= 2
        or shared_entities
        or lexical_overlap(claim, reference) >= 0.08
        or phrase_overlap(claim, reference) >= 0.08
    )

TOPIC_IGNORE_TERMS = {
    "approve",
    "approved",
    "approval",
    "authorize",
    "authorized",
    "certified",
    "certify",
    "received",
    "receive",
    "proposal",
    "plan",
    "project",
    "board",
    "phase",
    "note",
}
GENERIC_ENTITIES = {
    "the",
    "planning",
    "meeting",
    "note",
    "officials",
    "administrators",
    "briefing",
    "regulatory",
    "supplier",
    "evidence",
    "board",
}


def _has_topical_overlap(claim: str, reference: str) -> bool:
    shared_tokens = (set(tokenize(claim)) & set(tokenize(reference))) - TOPIC_IGNORE_TERMS
    shared_entities = (
        (_named_entities(claim) & _named_entities(reference)) - GENERIC_ENTITIES
    )
    return bool(shared_tokens or shared_entities)


def _has_no_approval(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(
        _contains_any_phrase(normalized, APPROVAL_NO_TERMS)
        or re.search(r"\b(?:not|no|without|pending|awaiting|deferred|withdrawn)\b[^.]{0,24}\b(?:approval|approved|certified|authorize|authorized|rollout)\b", normalized)
    )


def _has_yes_approval(text: str) -> bool:
    normalized = normalize_text(text)
    if _has_no_approval(normalized):
        return False
    return bool(
        _contains_any_phrase(normalized, APPROVAL_YES_TERMS)
        or re.search(r"\b(?:received|secured|gained|has)\b[^.]{0,16}\b(?:approval|approved|certified|authorization)\b", normalized)
    )

def _has_schedule_limit(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(
        re.search(r"\b(?:no earlier than|earliest|pending|awaiting|deferred|withdrawn|late|spring|summer|fall|winter)\b", normalized)
        or ("after permit" in normalized)
        or ("after permits" in normalized)
        or ("permit review" in normalized)
    )


_RESTRICTOR_RE = re.compile(
    r"\b(?:only|limited to|restricted to|exclusive(?:ly)? to|reserved for|members only)\b",
    flags=re.IGNORECASE,
)
_NO_ACCESS_RE = re.compile(
    r"\b(?:not allowed|not permitted|do not permit|does not permit|not available|unavailable|prohibited|forbidden|banned|excluded|closed to|not open to)\b",
    flags=re.IGNORECASE,
)
RESTRICTION_AXES = (
    {
        "allowed_patterns": (r"\boff[- ]?peak\b",),
        "disallowed_patterns": (r"\bpeak\b",),
        "reason": "an off-peak-only summary is supported by evidence that peak periods prohibit the activity",
    },
    {
        "allowed_patterns": (r"\bweekdays?\b",),
        "disallowed_patterns": (r"\bweekends?\b",),
        "reason": "a weekdays-only summary is supported by evidence that weekends are excluded",
    },
    {
        "allowed_patterns": (r"\bmembers?\b",),
        "disallowed_patterns": (r"\bnon[- ]?members?\b",),
        "reason": "a members-only summary is supported by evidence that non-members are excluded",
    },
)


def _matches_patterns(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _positive_restriction(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = normalize_text(text)
    return _matches_patterns(normalized, patterns) and bool(_RESTRICTOR_RE.search(normalized))


def _negative_restriction(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = normalize_text(text)
    if not _matches_patterns(normalized, patterns):
        return False

    joined_patterns = "(?:" + "|".join(patterns) + ")"
    return bool(
        _NO_ACCESS_RE.search(normalized)
        or re.search(rf"\b(?:not|no|without|excluding|except)\b[^.]{{0,28}}{joined_patterns}", normalized, flags=re.IGNORECASE)
        or re.search(rf"{joined_patterns}[^.]{{0,28}}\b(?:not|unavailable|closed|forbidden|banned|prohibited)\b", normalized, flags=re.IGNORECASE)
    )


def extra_support_reasons(claim: str, reference: str) -> list[str]:
    """Recognize summary restrictions implied by complementary exclusion evidence."""
    if not _shared_context(claim, reference):
        return []

    normalized_claim = normalize_text(claim)
    normalized_reference = normalize_text(reference)
    reasons: list[str] = []

    for axis in RESTRICTION_AXES:
        claim_positive = _positive_restriction(normalized_claim, axis["allowed_patterns"])
        claim_negative = _negative_restriction(normalized_claim, axis["disallowed_patterns"])
        reference_positive = _positive_restriction(normalized_reference, axis["allowed_patterns"])
        reference_negative = _negative_restriction(normalized_reference, axis["disallowed_patterns"])
        if (claim_positive and reference_negative) or (claim_negative and reference_positive):
            reasons.append(axis["reason"])

    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        deduped.append(reason)
    return deduped


def reliable_cues(claim: str, reference: str) -> list[str]:
    """Return only contradiction cues that are stable enough for local heuristics."""
    normalized_claim = normalize_text(claim)
    normalized_reference = normalize_text(reference)
    restrictions = extra_support_reasons(claim, reference)
    cues: list[str] = []

    for cue in find_cues(claim, reference):
        if cue == "different numeric details":
            cues.append(cue)
        elif cue == "negation mismatch":
            same_negative_status = (
                _has_no_approval(normalized_claim) and _has_no_approval(normalized_reference)
            )
            same_schedule_limit = (
                _has_schedule_limit(normalized_claim) and _has_schedule_limit(normalized_reference)
            )
            # Limits can support.
            if (
                not same_negative_status
                and not same_schedule_limit
                and not restrictions
                and "rather than" not in normalized_claim
                and "rather than" not in normalized_reference
                and (lexical_overlap(claim, reference) >= 0.22 or token_coverage(claim, reference) >= 0.34)
            ):
                cues.append(cue)
        elif cue.startswith("opposing wording around"):
            cues.append(cue)

    enough_shared_context = _shared_context(claim, reference)

    # Need same topic.
    if enough_shared_context:
        claim_temporal = _time_markers(claim)
        reference_temporal = _time_markers(reference)
        if claim_temporal and reference_temporal and claim_temporal.isdisjoint(reference_temporal):
            cues.append("different year/date details")
        elif YEAR_PATTERN.findall(claim) and YEAR_PATTERN.findall(reference):
            if set(YEAR_PATTERN.findall(claim)).isdisjoint(set(YEAR_PATTERN.findall(reference))):
                cues.append("different year/date details")

        claim_times = _extract_time_ranges(claim)
        reference_times = _extract_time_ranges(reference)
        if claim_times and reference_times and claim_times.isdisjoint(reference_times):
            cues.append("different time-range details")

        claim_locations = _place_entities(claim)
        reference_locations = _place_entities(reference)
        if claim_locations and reference_locations and claim_locations.isdisjoint(reference_locations):
            cues.append("different named entity details")

        claim_has_open = _has_any_token(normalized_claim, {"open", "reopen", "reopened", "daily"})
        reference_has_open = _has_any_token(normalized_reference, {"open", "reopen", "reopened", "daily"})
        claim_has_closed = _has_any_token(normalized_claim, {"closed", "close", "shut"})
        reference_has_closed = _has_any_token(normalized_reference, {"closed", "close", "shut"})
        if (claim_has_open and reference_has_closed) or (claim_has_closed and reference_has_open):
            cues.append("open/closed mismatch")

        claim_finished = _has_any_token(normalized_claim, {"finished", "completed", "done"})
        reference_ongoing = _contains_any_phrase(normalized_reference, ONGOING_TERMS)
        if claim_finished and reference_ongoing:
            cues.append("finished/ongoing mismatch")

        claim_absolute = _has_any_token(normalized_claim, ABSOLUTE_CLAIM_TERMS)
        reference_partial = _contains_any_phrase(normalized_reference, PARTIAL_STATUS_TERMS)
        if claim_absolute and reference_partial:
            cues.append("absolute/partial mismatch")

        claim_free = _contains_any_phrase(normalized_claim, FREE_ACCESS_TERMS)
        claim_admission = _contains_any_phrase(normalized_claim, PRICING_TERMS)
        reference_paid = _contains_any_phrase(normalized_reference, PRICING_TERMS) and bool(extract_numbers(reference))
        if claim_free and claim_admission and reference_paid:
            cues.append("free/paid mismatch")
        elif _contains_any_phrase(normalized_claim, {"paid", "charges admission"}) and _contains_any_phrase(normalized_reference, FREE_ACCESS_TERMS):
            cues.append("free/paid mismatch")

        claim_yes_approval = _has_yes_approval(normalized_claim)
        ref_no_approval = _has_no_approval(normalized_reference)
        claim_no_approval = _has_no_approval(normalized_claim)
        ref_yes_approval = _has_yes_approval(normalized_reference)
        if (
            ((claim_yes_approval and ref_no_approval) or (claim_no_approval and ref_yes_approval))
            and _has_topical_overlap(claim, reference)
        ):
            cues.append("approval-status mismatch")

        claim_increase = _has_any_token(normalized_claim, INCREASE_TERMS)
        reference_decrease = _has_any_token(normalized_reference, DECREASE_TERMS)
        claim_decrease = _has_any_token(normalized_claim, DECREASE_TERMS)
        reference_increase = _has_any_token(normalized_reference, INCREASE_TERMS)
        if (claim_increase and reference_decrease) or (claim_decrease and reference_increase):
            cues.append("increase/decrease mismatch")

        claim_demolition = _contains_any_phrase(normalized_claim, DEMOLITION_TERMS)
        reference_repair = _contains_any_phrase(normalized_reference, REPAIR_TERMS)
        claim_repair = _contains_any_phrase(normalized_claim, REPAIR_TERMS)
        reference_demolition = _contains_any_phrase(normalized_reference, DEMOLITION_TERMS)
        if (claim_demolition and not claim_repair and reference_repair and not reference_demolition) or (
            claim_repair and not claim_demolition and reference_demolition and not reference_repair
        ):
            cues.append("demolition/repair mismatch")

        if (_contains_any_phrase(normalized_claim, CARE_SETTING_TERMS["outpatient"]) and _contains_any_phrase(normalized_reference, CARE_SETTING_TERMS["inpatient"])) or (
            _contains_any_phrase(normalized_claim, CARE_SETTING_TERMS["inpatient"]) and _contains_any_phrase(normalized_reference, CARE_SETTING_TERMS["outpatient"])
        ):
            cues.append("inpatient/outpatient mismatch")

        claim_emergency = _contains_any_phrase(normalized_claim, EMERGENCY_TERMS)
        ref_no_er = bool(re.search(r"(?:no|not|without|did not approve|does not include)[^.]{0,24}emergency", normalized_reference))
        reference_emergency = _contains_any_phrase(normalized_reference, EMERGENCY_TERMS)
        claim_no_er = bool(re.search(r"(?:no|not|without|did not approve|does not include)[^.]{0,24}emergency", normalized_claim))
        if (claim_emergency and ref_no_er and not claim_no_er) or (
            claim_no_er and reference_emergency and not ref_no_er
        ):
            cues.append("emergency-department mismatch")

        claim_roles = _role_entities(claim)
        reference_roles = _role_entities(reference)
        if claim_roles and reference_roles:
            if claim_roles.get("manufacturer") and reference_roles.get("manufacturer") and claim_roles["manufacturer"] != reference_roles["manufacturer"]:
                cues.append("manufacturer/source attribution mismatch")
            elif claim_roles.get("manufacturer_exclusive") and reference_roles.get("manufacturer") and claim_roles.get("manufacturer_exclusive") != reference_roles.get("manufacturer"):
                cues.append("manufacturer/source attribution mismatch")
            elif claim_roles.get("manufacturer_exclusive") and reference_roles.get("software") and claim_roles.get("manufacturer_exclusive") == reference_roles.get("software") and reference_roles.get("manufacturer"):
                cues.append("manufacturer/source attribution mismatch")
            elif claim_roles.get("software") and reference_roles.get("software") and claim_roles["software"] != reference_roles["software"]:
                cues.append("manufacturer/source attribution mismatch")


    # Dedupe cues.
    deduped = []
    seen = set()
    for cue in cues:
        if cue in seen:
            continue
        seen.add(cue)
        deduped.append(cue)
    return deduped


def _pricing_support(claim: str, reference: str) -> list[str]:
    """Detect generic pricing summaries supported by specific pricing details."""
    normalized_claim = normalize_text(claim)
    normalized_reference = normalize_text(reference)
    claim_has_admission = _contains_any_phrase(normalized_claim, PRICING_TERMS)
    claim_has_free = _contains_any_phrase(normalized_claim, FREE_ACCESS_TERMS)
    reference_has_pricing = _contains_any_phrase(normalized_reference, PRICING_TERMS) or bool(extract_numbers(reference))
    reasons: list[str] = []

    if claim_has_admission and not claim_has_free and reference_has_pricing and not extract_numbers(claim):
        reasons.append("generic admission wording is supported by specific pricing details")

    claim_has_discount = _contains_any_phrase(normalized_claim, DISCOUNT_TERMS)
    reference_has_student = bool(re.search(r"\bstudent(?:s)?\b", normalized_reference))
    reference_has_adult = bool(re.search(r"\badult(?:s)?\b", normalized_reference))
    if claim_has_discount and reference_has_student and (reference_has_adult or len(extract_numbers(reference)) >= 2):
        reasons.append("discount wording is supported by differentiated pricing in the evidence")

    return reasons


def _generic_support(claim: str, reference: str) -> list[str]:
    """Detect faithful summary-style support without requiring near-verbatim overlap."""
    normalized_claim = normalize_text(claim)
    normalized_reference = normalize_text(reference)
    reasons: list[str] = []

    claim_decrease = _has_any_token(normalized_claim, DECREASE_TERMS)
    reference_decrease = _has_any_token(normalized_reference, DECREASE_TERMS)
    claim_increase = _has_any_token(normalized_claim, INCREASE_TERMS)
    reference_increase = _has_any_token(normalized_reference, INCREASE_TERMS)
    claim_modest = _contains_any_phrase(normalized_claim, MODEST_TERMS)
    if claim_modest and ((claim_decrease and reference_decrease) or (claim_increase and reference_increase)) and bool(extract_numbers(reference)):
        reasons.append("the claim compresses a quantified directional change into a faithful summary")

    claim_extended = _contains_any_phrase(normalized_claim, EXTENSION_TERMS) or "rather than" in normalized_claim
    reference_extended = _contains_any_phrase(normalized_reference, EXTENSION_TERMS)
    ref_not_approved = _contains_any_phrase(normalized_reference, APPROVAL_NO_TERMS) or "no permanent rollout" in normalized_reference
    if claim_extended and reference_extended and ref_not_approved:
        reasons.append("extension wording matches evidence that the project continued without permanent approval")

    claim_no_approval = _contains_any_phrase(normalized_claim, APPROVAL_NO_TERMS)
    if claim_no_approval and ref_not_approved:
        reasons.append("the claim faithfully summarizes a deferred, pending, or unapproved status")

    claim_temporal = _time_markers(claim)
    reference_temporal = _time_markers(reference)
    if claim_temporal and reference_temporal and claim_temporal <= reference_temporal and ("earliest" in normalized_reference or "no earlier than" in normalized_reference or "pending" in normalized_reference):
        reasons.append("the claim preserves the source's earliest-possible timing rather than overstating the schedule")

    return reasons


def assess_claim_evidence(
    claim: str,
    reference: str,
    *,
    semantic_score: float = 0.0,
    lexical_score: float | None = None,
    token_coverage_score: float | None = None,
    phrase_overlap_score: float | None = None,
    entity_overlap_score: float | None = None,
    number_score: float | None = None,
) -> dict[str, Any]:
    """Assess whether a reference text supports, contradicts, or misses a claim."""
    lexical_score = lexical_overlap(claim, reference) if lexical_score is None else float(lexical_score)
    token_coverage_score = token_coverage(claim, reference) if token_coverage_score is None else float(token_coverage_score)
    phrase_overlap_score = phrase_overlap(claim, reference) if phrase_overlap_score is None else float(phrase_overlap_score)
    entity_overlap_score = entity_overlap(claim, reference) if entity_overlap_score is None else float(entity_overlap_score)
    number_score = numeric_alignment(claim, reference) if number_score is None else float(number_score)
    contradiction_cues = reliable_cues(claim, reference)
    abstraction_reasons = (
        _pricing_support(claim, reference)
        + _generic_support(claim, reference)
        + extra_support_reasons(claim, reference)
    )
    support_strength = (
        (0.28 * float(semantic_score))
        + (0.24 * token_coverage_score)
        + (0.18 * phrase_overlap_score)
        + (0.12 * lexical_score)
        + (0.08 * entity_overlap_score)
        + (0.10 * number_score)
    )
    strong_contra = any(
        cue in STRONG_CONTRA_CUES or cue.startswith("opposing wording around")
        for cue in contradiction_cues
    )
    relatedness = max(
        float(semantic_score),
        lexical_score,
        token_coverage_score,
        phrase_overlap_score,
        entity_overlap_score,
        number_score,
    )

    if contradiction_cues and (strong_contra or relatedness >= 0.16 or _shared_context(claim, reference)):
        return {
            "status": "contradicted",
            "support_type": "contradicted",
            "contradiction_cues": contradiction_cues,
            "support_strength": round(support_strength, 3),
            "reason": f"Evidence points the other way: {', '.join(contradiction_cues[:2])}.",
            "decisive_contradiction": strong_contra,
        }

    if abstraction_reasons:
        return {
            "status": "abstractly_supported",
            "support_type": "abstraction",
            "contradiction_cues": [],
            "support_strength": round(max(support_strength, 0.5), 3),
            "reason": "Supported by faithful abstraction: " + "; ".join(abstraction_reasons[:2]) + ".",
            "decisive_contradiction": False,
        }

    if (
        support_strength >= 0.56
        or (float(semantic_score) >= 0.56 and token_coverage_score >= 0.32)
        or (token_coverage_score >= 0.7 and (float(semantic_score) >= 0.2 or lexical_score >= 0.14))
        or (number_score >= 0.66 and token_coverage_score >= 0.22)
        or (entity_overlap_score >= 0.66 and token_coverage_score >= 0.28)
    ):
        return {
            "status": "supported",
            "support_type": "direct",
            "contradiction_cues": [],
            "support_strength": round(support_strength, 3),
            "reason": "Retrieved evidence directly supports the claim.",
            "decisive_contradiction": False,
        }

    if (
        support_strength >= 0.4
        or (float(semantic_score) >= 0.42 and token_coverage_score >= 0.22)
        or (phrase_overlap_score >= 0.18 and token_coverage_score >= 0.32)
        or (entity_overlap_score >= 0.5 and token_coverage_score >= 0.22)
        or (number_score >= 0.5 and float(semantic_score) >= 0.12)
    ):
        return {
            "status": "abstractly_supported",
            "support_type": "abstraction",
            "contradiction_cues": [],
            "support_strength": round(support_strength, 3),
            "reason": "Evidence supports the claim through paraphrase or compressed detail.",
            "decisive_contradiction": False,
        }

    if support_strength >= 0.28 or (float(semantic_score) >= 0.28 and token_coverage_score >= 0.16):
        return {
            "status": "weakly_supported",
            "support_type": "partial",
            "contradiction_cues": [],
            "support_strength": round(support_strength, 3),
            "reason": "Evidence is related to the claim, but support remains partial.",
            "decisive_contradiction": False,
        }

    return {
        "status": "unsupported",
        "support_type": "unsupported",
        "contradiction_cues": [],
        "support_strength": round(support_strength, 3),
        "reason": "The available evidence does not clearly support this claim.",
        "decisive_contradiction": False,
    }


def build_aggregate_hit(hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Create a synthetic evidence record from multiple retrieved hits."""
    usable_hits = [hit for hit in hits if safe_text(hit.get("text"))]
    if not usable_hits:
        return None

    combined_ids = "+".join(str(hit.get("chunk_id", "")) for hit in usable_hits)
    combined_text = " ".join(safe_text(hit.get("text")) for hit in usable_hits)
    best_score = max(float(hit.get("score", 0.0)) for hit in usable_hits)
    base_hit = usable_hits[0]
    return {
        **base_hit,
        "chunk_id": combined_ids,
        "citation_id": combined_ids,
        "title": base_hit.get("title") or "Aggregated evidence",
        "text": combined_text,
        "score": best_score,
        "metadata": {**dict(base_hit.get("metadata") or {}), "aggregated_from": [hit.get("chunk_id") for hit in usable_hits]},
    }