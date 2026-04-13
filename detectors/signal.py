"""Internal-signal hallucination detectors backed by a local Hugging Face model.

Two local paths live here:
- ``uncertainty_baseline``: token-uncertainty scoring from one answer
- ``sep_lite``: SEP-inspired hidden-state features with optional probes

The SEP path is a compact approximation, not a full paper reproduction.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Any

import numpy as np

from detectors.base import make_result, make_unavailable
from utils.simple_fact_utils import (
    check_simple_fact,
    simple_fact_answer,
    find_simple_fact,
    simple_fact_plurality,
)
from utils.text_utils import (
    contains_negation,
    match_claim,
    find_cues,
    extract_claims,
    lexical_overlap,
    normalize_text,
    safe_text,
    truncate_text,
)


BLOCK_SPLIT_RE = re.compile(r"(?:^|\n)\s*(?:-{3,}|={3,}|\*{3,})\s*(?:\n|$)")
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
YEAR_PATTERN = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b")
TITLE_SPAN_RE = re.compile(r"\b[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,3}\b", re.UNICODE)
MONTH_TOKEN_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
MONTH_YEAR_PATTERN = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?P<year>(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2}))\b",
    re.IGNORECASE,
)
DEFAULT_MODEL_NAME = os.getenv("HD_INTERNAL_MODEL", "distilgpt2")
DEFAULT_LAYER_SPEC = os.getenv("HD_INTERNAL_LAYERS", "-1,-3,-5")
DEFAULT_PROBE_PATH = os.getenv("HD_INTERNAL_PROBE_PATH", "")
DEFAULT_LOCAL_ONLY = os.getenv("HD_HF_LOCAL_ONLY", "0") != "0"
PROBE_FEATURE_ORDER = [
    "mean_negative_log_prob",
    "token_log_prob_std",
    "entropy_mean",
    "entropy_std",
    "mean_token_probability",
    "top2_margin_mean",
    "hidden_norm_mean",
    "hidden_norm_var",
    "hidden_drift_mean",
    "layer_centroid_dispersion",
    "feature_sample_variance",
    "num_answer_tokens",
    "num_samples",
]

MONTH_ALIASES = {
    "jan": "january",
    "january": "january",
    "feb": "february",
    "february": "february",
    "mar": "march",
    "march": "march",
    "apr": "april",
    "april": "april",
    "may": "may",
    "jun": "june",
    "june": "june",
    "jul": "july",
    "july": "july",
    "aug": "august",
    "august": "august",
    "sep": "september",
    "sept": "september",
    "september": "september",
    "oct": "october",
    "october": "october",
    "nov": "november",
    "november": "november",
    "dec": "december",
    "december": "december",
}


@dataclass(slots=True)
class SignalConfig:
    model_name: str = DEFAULT_MODEL_NAME
    layer_spec: str = DEFAULT_LAYER_SPEC
    local_files_only: bool = DEFAULT_LOCAL_ONLY
    max_total_tokens: int = 320
    max_answer_tokens: int = 160
    probe_path: str = DEFAULT_PROBE_PATH


@dataclass(slots=True)
class SignalBackend:
    tokenizer: Any | None
    model: Any | None
    device: str
    python_executable: str
    torch_installed: bool
    torch_version: str | None
    has_transformers: bool
    transformers_version: str | None
    backend_available: bool
    backend_status: str
    backend_status_label: str
    backend_error: str | None
    model_name: str
    local_files_only: bool


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _import_torch() -> Any:
    import torch

    return torch


def _import_transformers() -> tuple[Any, Any, Any]:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return transformers, AutoModelForCausalLM, AutoTokenizer


# Cache HF load.
@lru_cache(maxsize=4)
def _init_hf(model_name: str, local_files_only: bool) -> SignalBackend:
    backend = SignalBackend(
        tokenizer=None,
        model=None,
        device="cpu",
        python_executable=sys.executable,
        torch_installed=_package_version("torch") is not None,
        torch_version=_package_version("torch"),
        has_transformers=_package_version("transformers") is not None,
        transformers_version=_package_version("transformers"),
        backend_available=False,
        backend_status="unavailable",
        backend_status_label="HF backend unavailable",
        backend_error=None,
        model_name=model_name,
        local_files_only=local_files_only,
    )
    try:
        torch = _import_torch()
    except Exception as exc:
        backend.backend_error = f"torch import failed: {exc}"
        return backend
    try:
        transformers_module, AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    except Exception as exc:
        backend.backend_error = f"transformers import failed: {exc}"
        return backend

    backend.torch_installed = True
    backend.has_transformers = True
    if backend.torch_version is None:
        backend.torch_version = getattr(torch, "__version__", None)
    if backend.transformers_version is None:
        backend.transformers_version = getattr(transformers_module, "__version__", None)

    tokenizer = None
    model = None
    preferred_device = "cuda" if bool(getattr(torch.cuda, "is_available", lambda: False)()) else "cpu"
    backend.device = preferred_device
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if preferred_device == "cuda":
            try:
                model.to("cuda")
            except Exception:
                backend.device = "cpu"
                model.to("cpu")
        else:
            model.to("cpu")
        model.eval()
    except Exception as exc:
        backend.backend_error = f"model/tokenizer load failed: {exc}"
        return backend

    backend.tokenizer = tokenizer
    backend.model = model
    backend.backend_available = True
    backend.backend_status = "available"
    backend.backend_status_label = "HF backend active"
    return backend


def get_signal_status(config: SignalConfig | None = None) -> dict[str, Any]:
    resolved_config = config or SignalConfig()
    backend = _init_hf(resolved_config.model_name, resolved_config.local_files_only)
    return {
        "python_executable": backend.python_executable,
        "torch_installed": backend.torch_installed,
        "torch_version": backend.torch_version,
        "transformers_installed": backend.has_transformers,
        "transformers_version": backend.transformers_version,
        "device": backend.device,
        "backend_available": backend.backend_available,
        "backend_status": backend.backend_status,
        "backend_status_label": backend.backend_status_label,
        "backend_error": backend.backend_error,
        "backend_model_name": backend.model_name,
        "local_files_only": backend.local_files_only,
    }


def _safe_sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _clip_probability(value: float) -> float:
    return float(max(0.0, min(1.0, value)))



def _normalize_logprobs(token_logprobs: list[list[float]] | None) -> list[np.ndarray]:
    if not token_logprobs:
        return []
    normalized: list[np.ndarray] = []
    for item in token_logprobs:
        if item is None:
            continue
        values = np.asarray([float(value) for value in item], dtype=float)
        if values.size:
            normalized.append(values)
    return normalized


def _logprob_bundle(token_logprobs: list[list[float]] | None) -> dict[str, float] | None:
    arrays = _normalize_logprobs(token_logprobs)
    if not arrays:
        return None
    primary = arrays[0]
    token_probabilities = np.exp(primary)
    mean_nll = float((-primary).mean())
    token_log_prob_std = float(primary.std())
    mean_token_prob = float(token_probabilities.mean())
    sample_nll_std = float(np.std([float((-item).mean()) for item in arrays])) if len(arrays) > 1 else 0.0
    uncertainty_score = _clip_probability(
        (0.45 * _clip_probability(mean_nll / 5.0))
        + (0.20 * _clip_probability(token_log_prob_std / 2.5))
        + (0.20 * _clip_probability(1.0 - mean_token_prob))
        + (0.15 * _clip_probability(sample_nll_std / 1.2))
    )
    return {
        "token_count": float(primary.size),
        "mean_negative_log_prob": mean_nll,
        "token_log_prob_std": token_log_prob_std,
        "mean_token_probability": mean_token_prob,
        "sample_nll_std": sample_nll_std,
        "uncertainty_score": uncertainty_score,
    }

def _split_sample_blocks(sampled_answers_text: str) -> list[str]:
    cleaned = safe_text(sampled_answers_text)
    if not cleaned:
        return []
    if BLOCK_SPLIT_RE.search(cleaned):
        return [block.strip() for block in BLOCK_SPLIT_RE.split(cleaned) if safe_text(block)]
    return [block.strip() for block in re.split(r"\n\s*\n+", cleaned) if safe_text(block)]


def _main_claim(answer_text: str, question: str) -> str:
    claims = extract_claims(answer_text, max_claims=1)
    base_claim = claims[0] if claims else safe_text(answer_text)
    return match_claim(base_claim, question)


def _compare_samples(left_claim: str, right_claim: str) -> dict[str, Any]:
    normalized_left = normalize_text(left_claim)
    normalized_right = normalize_text(right_claim)
    overlap = lexical_overlap(left_claim, right_claim)
    contradiction_cues = find_cues(left_claim, right_claim)

    if not normalized_left or not normalized_right:
        verdict = "mixed"
        weight = 0.5
    elif normalized_left == normalized_right or normalized_left in normalized_right or normalized_right in normalized_left:
        verdict = "supports"
        weight = 0.0
    elif contradiction_cues:
        verdict = "contradicts"
        weight = 1.0
    elif overlap >= 0.72:
        verdict = "supports"
        weight = 0.0
    elif overlap >= 0.35:
        verdict = "mixed"
        weight = 0.45
    else:
        verdict = "different"
        weight = 0.7

    return {
        "verdict": verdict,
        "weight": weight,
        "overlap": float(overlap),
        "contradiction_cues": contradiction_cues,
    }


def _sample_instability(claims: list[str]) -> float:
    if len(claims) < 2:
        return 0.0
    weights: list[float] = []
    for left_index in range(len(claims)):
        for right_index in range(left_index + 1, len(claims)):
            weights.append(_compare_samples(claims[left_index], claims[right_index])["weight"])
    return float(np.mean(weights)) if weights else 0.0


ENTITY_EXCLUSIONS = {
    "A",
    "An",
    "He",
    "I",
    "It",
    "She",
    "That",
    "The",
    "Their",
    "There",
    "These",
    "They",
    "This",
    "Those",
    "We",
}
ATTRIBUTION_TERMS = {
    "according to",
    "archive",
    "archives",
    "data",
    "evidence",
    "historian",
    "historians",
    "recorded",
    "records",
    "reported",
    "reportedly",
    "research",
    "source",
    "sources",
    "studies",
    "study",
}
HEDGE_TERMS = {
    "allegedly",
    "appears",
    "could",
    "likely",
    "may",
    "might",
    "perhaps",
    "possibly",
    "reportedly",
    "seems",
    "suggests",
    "unclear",
}
CERTAINTY_TERMS = {
    "certainly",
    "clearly",
    "created",
    "definitely",
    "ended",
    "established",
    "later became",
    "officially",
    "proved",
    "served as",
    "undeniably",
    "went on to",
}
# Narrow fallback cues.
EVENT_TREATY_TERMS = {
    "accord",
    "agreement",
    "peace accord",
    "peace treaty",
    "treaty",
}
EVENT_CONFLICT_TERMS = {
    "conflict",
    "demilitarized",
    "naval",
    "port",
    "trade port",
}
EVENT_STYLE_TERMS = EVENT_TREATY_TERMS | EVENT_CONFLICT_TERMS

SIMPLE_FACT_Q_RE = re.compile(r"^\s*(?:(?:who|what|where|when)\b(?:\s+\w+){0,2}\s+(?:is|are|was|were|did|does|do)|which\b(?:\s+\w+){0,3}\s+(?:is|are|was|were)|name\b)\b", flags=re.IGNORECASE)
TIME_Q_RE = re.compile(
    r"\b(?:"
    r"when|what year|which year|in what year|what date|which date|on what date|what month|which month|"
    r"founded|established|started|began|ended|launched|released|published|signed|opened|closed|"
    r"born|died|formed|created"
    r")\b",
    flags=re.IGNORECASE,
)
TIME_TOKEN_RE = re.compile(
    r"\b(?:"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"spring|summer|autumn|fall|winter|century|decade|era|timeline"
    r")\b",
    flags=re.IGNORECASE,
)
DIRECT_ANSWER_RE = re.compile(
    r"^\s*(?:the answer|answer|it|this|that)\s+(?:is|are|was|were)\b",
    flags=re.IGNORECASE,
)
BARE_TIME_PREFIXES = {"about", "around", "by", "during", "in", "on", "since", "through"}
BARE_TEMPORAL_TOKENS = {
    "ad", "apr", "april", "aug", "august", "autumn", "bce", "bc", "ce", "dec", "december",
    "fall", "feb", "february", "jan", "january", "jul", "july", "jun", "june", "mar", "march",
    "may", "nov", "november", "oct", "october", "sep", "september", "spring", "summer", "winter",
}
BASELINE_SPLIT_RE = re.compile(
    r"\s*(?:;|\b(?:although|but|however|instead|yet|though|while|whereas)\b)\s*",
    flags=re.IGNORECASE,
)


def _count_phrase_matches(text: str, phrases: set[str]) -> int:
    total = 0
    for phrase in phrases:
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
        total += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return total


def _title_spans(text: str) -> list[str]:
    spans: list[str] = []
    seen: set[str] = set()
    for span in TITLE_SPAN_RE.findall(text):
        cleaned = span.strip()
        if cleaned in ENTITY_EXCLUSIONS:
            continue
        normalized = normalize_text(cleaned)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        spans.append(cleaned)
    return spans


def _offline_error(backend_error: Exception | str | None) -> str:
    detail = safe_text("" if backend_error is None else str(backend_error))
    if not detail:
        return "backend offline"
    if "backend offline" in detail.lower():
        return detail
    return f"backend offline: {detail}"


def _bare_time_answer(answer_text: str, *, answer_years: set[str], features: dict[str, Any]) -> bool:
    if features["claim_count"] > 2 or features["word_count"] > 6 or features["number_count"] > 3:
        return False
    if not answer_years and not TIME_TOKEN_RE.search(answer_text):
        return False
    tokens = [token.lower() for token in WORD_PATTERN.findall(safe_text(answer_text))]
    if not tokens:
        return False
    saw_temporal_token = bool(answer_years)
    for token in tokens:
        if token in BARE_TIME_PREFIXES:
            continue
        if token in BARE_TEMPORAL_TOKENS:
            saw_temporal_token = True
            continue
        if YEAR_PATTERN.fullmatch(token):
            saw_temporal_token = True
            continue
        if token.isdigit() and 1 <= int(token) <= 31:
            continue
        return False
    return saw_temporal_token


def _base_frag_claims(answer_text: str, question: str) -> list[str]:
    del question
    fragments: list[str] = []
    for fragment in BASELINE_SPLIT_RE.split(safe_text(answer_text)):
        token_count = len(WORD_PATTERN.findall(fragment))
        cleaned = safe_text(fragment).strip()
        if not cleaned:
            continue
        if cleaned[-1] not in ".!?":
            cleaned += "."
        if token_count >= 3:
            fragments.append(cleaned)
            continue
        if token_count >= 2 and (
            NUMBER_PATTERN.search(cleaned)
            or YEAR_PATTERN.search(cleaned)
            or re.search(r"\b(?:no|not|never|none)\b", cleaned, flags=re.IGNORECASE)
        ):
            fragments.append(cleaned)
    return fragments[:6]


def _text_features(question: str, answer: str) -> dict[str, Any]:
    cleaned_question = safe_text(question)
    cleaned_answer = safe_text(answer)
    words = WORD_PATTERN.findall(cleaned_answer)
    claims = extract_claims(cleaned_answer)
    claim_count = len(claims) or 1
    word_count = len(words)
    question_normalized = normalize_text(cleaned_question)
    question_tokens = WORD_PATTERN.findall(cleaned_question)
    simple_fact_q = bool(SIMPLE_FACT_Q_RE.search(cleaned_question)) and len(question_tokens) <= 10
    # Demo fact bank.
    simple_fact_sanity = check_simple_fact(cleaned_question, cleaned_answer)

    unique_entities = _title_spans(cleaned_answer)
    new_entities = [span for span in unique_entities if normalize_text(span) not in question_normalized]

    number_count = len(NUMBER_PATTERN.findall(cleaned_answer))
    temporal_count = len(YEAR_PATTERN.findall(cleaned_answer))
    duration_count = len(
        re.findall(
            r"\b\d+\s*(?:year|years|month|months|day|days|week|weeks|century|centuries)\b|"
            r"\b\d+-(?:year|month|day|week|century)\b",
            cleaned_answer,
            flags=re.IGNORECASE,
        )
    )
    event_style_count = _count_phrase_matches(cleaned_answer, EVENT_STYLE_TERMS)
    detail_count = number_count + temporal_count + duration_count + event_style_count
    attribution_count = _count_phrase_matches(cleaned_answer, ATTRIBUTION_TERMS)
    hedge_count = _count_phrase_matches(cleaned_answer, HEDGE_TERMS)
    certainty_count = _count_phrase_matches(cleaned_answer, CERTAINTY_TERMS)

    named_entity_risk = _clip_probability(
        (0.68 * min(max(len(new_entities) - 1, 0) / 3.0, 1.0))
        + (0.32 * min(max(len(unique_entities) - 1, 0) / 4.0, 1.0))
    )
    specificity_risk = _clip_probability(
        (0.25 * min(number_count / 2.0, 1.0))
        + (0.15 * min(temporal_count / 2.0, 1.0))
        + (0.20 * min(duration_count / 1.0, 1.0))
        + (0.40 * min(event_style_count / 4.0, 1.0))
    )
    multi_claim_risk = _clip_probability(max(claim_count - 1, 0) / 2.0)
    claim_density_risk = _clip_probability(max(claim_count - 1, 0) / max(word_count / 18.0, 1.0))
    low_hedging = 1.0 if hedge_count == 0 else _clip_probability(1.0 - hedge_count / 3.0)
    attribution_absence = 1.0 if attribution_count == 0 else _clip_probability(1.0 - attribution_count / 3.0)
    confident_tone_risk = _clip_probability(
        (0.60 * min((certainty_count + max(claim_count - 1, 0)) / 3.0, 1.0))
        + (0.25 * attribution_absence)
        + (0.15 * min(word_count / 24.0, 1.0))
    ) * low_hedging
    if simple_fact_q and claim_count == 1 and detail_count <= 1 and word_count <= 10:
        confident_tone_risk *= 0.45

    unsupported_risk = _clip_probability(
        (
            (0.45 * specificity_risk)
            + (0.35 * named_entity_risk)
            + (0.20 * confident_tone_risk)
        )
        * ((0.65 * attribution_absence) + (0.35 * low_hedging))
    )
    obscure_event_risk = 0.0
    if not simple_fact_q:
        obscure_event_risk = _clip_probability(
            (0.28 * min(event_style_count / 2.0, 1.0))
            + (0.18 * min(temporal_count / 1.0, 1.0))
            + (0.18 * min(duration_count / 1.0, 1.0))
            + (0.18 * named_entity_risk)
            + (0.18 * unsupported_risk)
        )
    fact_boost = _clip_probability(
        (1.0 if simple_fact_q else 0.72)
        * (1.0 - specificity_risk)
        * (1.0 - multi_claim_risk)
        * (1.0 - named_entity_risk)
        * (1.0 - claim_density_risk)
        * min(10.0 / max(word_count, 1), 1.0)
    )

    return {
        "question_text": cleaned_question,
        "answer_text": cleaned_answer,
        "question_years": sorted(set(YEAR_PATTERN.findall(cleaned_question))),
        "answer_years": sorted(set(YEAR_PATTERN.findall(cleaned_answer))),
        "word_count": word_count,
        "claim_count": claim_count,
        "detail_count": detail_count,
        "number_count": number_count,
        "temporal_count": temporal_count,
        "duration_count": duration_count,
        "event_style_count": event_style_count,
        "attribution_count": attribution_count,
        "hedge_count": hedge_count,
        "certainty_count": certainty_count,
        "question_is_simple_factoid": simple_fact_q,
        "simple_fact_reassurance": fact_boost,
        "simple_fact_sanity": simple_fact_sanity,
        "unique_entity_spans": unique_entities,
        "new_entity_spans": new_entities,
        "unique_entity_count": len(unique_entities),
        "new_entity_count": len(new_entities),
        "named_entity_risk": named_entity_risk,
        "specificity_risk": specificity_risk,
        "multi_claim_risk": multi_claim_risk,
        "claim_density_risk": claim_density_risk,
        "confident_tone_risk": confident_tone_risk,
        "unsupported_specificity_risk": unsupported_risk,
        "obscure_event_risk": obscure_event_risk,
        "attribution_absence": attribution_absence,
        "low_hedging": low_hedging,
    }


def _sample_metrics(question: str, answer: str, sampled_answers_text: str) -> dict[str, Any] | None:
    raw_samples = _split_sample_blocks(sampled_answers_text)
    if not raw_samples:
        return None

    sample_answers = [safe_text(answer)] + [sample for sample in raw_samples if safe_text(sample)]
    primary_claim = _main_claim(answer, question)
    comparison_rows: list[dict[str, Any]] = []
    comparison_weights: list[float] = []
    sample_claims = [primary_claim]

    for index, sample in enumerate(sample_answers[1:], start=1):
        sample_claim = _main_claim(sample, question)
        sample_claims.append(sample_claim)
        relationship = _compare_samples(primary_claim, sample_claim)
        comparison_weights.append(float(relationship["weight"]))
        comparison_rows.append(
            {
                "sample_index": index,
                "sample_answer": truncate_text(sample, 180),
                "comparison_claim": sample_claim,
                "verdict": relationship["verdict"],
                "lexical_overlap": round(float(relationship["overlap"]), 4),
                "contradiction_cues": relationship["contradiction_cues"],
            }
        )

    if not comparison_rows:
        return None

    support_ratio = sum(1 for row in comparison_rows if row["verdict"] == "supports") / len(comparison_rows)
    contradiction_ratio = sum(1 for row in comparison_rows if row["verdict"] == "contradicts") / len(comparison_rows)
    mean_disagreement = float(np.mean(comparison_weights)) if comparison_weights else 0.0
    pairwise_instability = _sample_instability(sample_claims)
    unique_claim_ratio = len({normalize_text(claim) for claim in sample_claims if normalize_text(claim)}) / max(len(sample_claims), 1)

    return {
        "sample_answers": sample_answers,
        "primary_claim": primary_claim,
        "comparisons": comparison_rows,
        "support_ratio": support_ratio,
        "contradiction_ratio": contradiction_ratio,
        "mean_disagreement": mean_disagreement,
        "pairwise_instability": pairwise_instability,
        "unique_claim_ratio": unique_claim_ratio,
    }


def _fact_vote_bundle(question: str, sample_answers: list[str]) -> dict[str, Any] | None:
    summary = simple_fact_plurality(question, sample_answers)
    if summary is None:
        return None

    disagreement_score = 0.0
    if summary["majority_disagrees_with_main"]:
        disagreement_score = _clip_probability(
            (0.55 * summary["plurality_share"])
            + (0.25 if summary["plurality_matches_canonical"] and not summary["main_matches_canonical"] else 0.0)
            + (0.12 if summary["fact_id"] else 0.0)
        )

    return {
        **summary,
        "disagreement_score": disagreement_score,
    }


def _sample_proxy_score(metrics: dict[str, Any], mode: str) -> tuple[float, list[dict[str, Any]]]:
    support_ratio = float(metrics["support_ratio"])
    contradiction_ratio = float(metrics["contradiction_ratio"])
    mean_disagreement = float(metrics["mean_disagreement"])
    pairwise_instability = float(metrics["pairwise_instability"])
    unique_claim_ratio = float(metrics["unique_claim_ratio"])

    if mode == "uncertainty_baseline":
        score = _clip_probability(
            (0.05 * unique_claim_ratio)
            + (0.45 * contradiction_ratio)
            + (0.25 * (1.0 - support_ratio))
            + (0.25 * pairwise_instability)
        )
        sub_signals = [
            {
                "signal": "Primary-to-sample contradiction ratio",
                "value": round(contradiction_ratio, 4),
                "risk": round(contradiction_ratio * 100.0, 1),
                "explanation": "More sampled answers contradicting the primary answer increases fallback risk.",
            },
            {
                "signal": "Primary-to-sample support ratio",
                "value": round(support_ratio, 4),
                "risk": round((1.0 - support_ratio) * 100.0, 1),
                "explanation": "When few sampled answers agree with the primary answer, fallback risk rises.",
            },
            {
                "signal": "Pairwise sample instability",
                "value": round(pairwise_instability, 4),
                "risk": round(pairwise_instability * 100.0, 1),
                "explanation": "Higher disagreement across the sampled answers signals unstable model behavior.",
            },
        ]
    else:
        score = _clip_probability(
            (0.10 * unique_claim_ratio)
            + (0.35 * contradiction_ratio)
            + (0.20 * (1.0 - support_ratio))
            + (0.20 * pairwise_instability)
            + (0.15 * mean_disagreement)
        )
        sub_signals = [
            {
                "signal": "Primary-to-sample contradiction ratio",
                "value": round(contradiction_ratio, 4),
                "risk": round(contradiction_ratio * 100.0, 1),
                "explanation": "Contradictory sampled answers act as a rough instability cue when hidden states are unavailable.",
            },
            {
                "signal": "Pairwise sample instability",
                "value": round(pairwise_instability, 4),
                "risk": round(pairwise_instability * 100.0, 1),
                "explanation": "Disagreement across sampled answers substitutes for hidden-state instability only in fallback mode.",
            },
            {
                "signal": "Unique claim ratio",
                "value": round(unique_claim_ratio, 4),
                "risk": round(unique_claim_ratio * 100.0, 1),
                "explanation": "A higher share of distinct sampled claims suggests weaker answer self-consistency.",
            },
            {
                "signal": "Mean disagreement weight",
                "value": round(mean_disagreement, 4),
                "risk": round(mean_disagreement * 100.0, 1),
                "explanation": "This summarizes how strongly the sampled answers diverge from the primary answer.",
            },
        ]
    return score, sub_signals


def _base_time_check(
    *,
    question: str,
    answer_text: str,
    features: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    normalized_question = normalize_text(question)
    question_years = set(features.get("question_years") or YEAR_PATTERN.findall(question))
    answer_years = set(features.get("answer_years") or YEAR_PATTERN.findall(answer_text))
    bare_temporal_answer = _bare_time_answer(answer_text, answer_years=answer_years, features=features)
    temporal_question = bool(
        TIME_Q_RE.search(question)
        or TIME_TOKEN_RE.search(question)
        or question_years
        or bare_temporal_answer
        or " year " in f" {normalized_question} "
        or " date " in f" {normalized_question} "
        or " month " in f" {normalized_question} "
    )
    answer_has_time = bool(answer_years) or bool(TIME_TOKEN_RE.search(answer_text))
    year_mismatch = bool(question_years and answer_years and question_years.isdisjoint(answer_years))
    question_year_drift = bool(question_years and any(year not in question_years for year in answer_years))
    time_conflict = bool(re.search(r"\b(?:not|never|no|however|but|instead|yet)\b", normalize_text(answer_text)))
    multi_year_answer = len(answer_years) >= 2
    date_only_answer = bool(
        temporal_question and len(answer_years) == 1 and features["claim_count"] <= 2 and features["word_count"] <= 18 and features["number_count"] <= 2
    )

    time_mismatch = 0.0
    if temporal_question:
        time_mismatch = _clip_probability(
            (0.60 if year_mismatch else 0.0)
            + (0.48 if question_year_drift else 0.0)
            + (0.30 * min(max(len(answer_years) - 1, 0) / 2.0, 1.0))
            + (0.20 if not answer_has_time else 0.0)
            + (0.22 if multi_year_answer and time_conflict else 0.0)
            + (0.20 if bare_temporal_answer and not year_mismatch else 0.0)
            + (0.20 if profile["date_inconsistency_risk"] >= 0.18 else 0.0)
            + (0.18 if multi_year_answer and profile["contradiction_risk"] >= 0.18 else 0.0)
        )
    elif question_years and answer_years and question_years.isdisjoint(answer_years):
        time_mismatch = 0.55

    date_only_risk = 0.0
    if temporal_question and date_only_answer and not year_mismatch:
        date_only_risk = _clip_probability(
            0.18
            + (0.38 * features["attribution_absence"])
            + (0.18 * features["low_hedging"])
            + (0.14 if features["certainty_count"] > 0 else 0.0)
            + (0.12 if profile["weak_support_risk"] >= 0.18 else 0.0)
        )

    return {
        "temporal_question": temporal_question,
        "question_years": sorted(question_years),
        "answer_years": sorted(answer_years),
        "answer_has_temporal_token": answer_has_time,
        "explicit_year_mismatch": year_mismatch,
        "question_year_drift": question_year_drift,
        "temporal_conflict_marker": time_conflict,
        "multi_year_answer": multi_year_answer,
        "bare_temporal_answer": bare_temporal_answer,
        "date_only_answer": date_only_answer,
        "temporal_mismatch_risk": time_mismatch,
        "date_only_uncertainty_risk": date_only_risk,
    }


def _direct_answer_boost(
    *,
    features: dict[str, Any],
    profile: dict[str, Any],
    temporal_bundle: dict[str, Any],
) -> float:
    if not features["question_is_simple_factoid"]:
        return 0.0
    if temporal_bundle["bare_temporal_answer"]:
        return 0.0
    if temporal_bundle["temporal_question"]:
        return 0.0
    if features["claim_count"] > 2 or features["word_count"] > 24:
        return 0.0

    instability_peak = max(
        profile["contradiction_risk"],
        profile["numeric_inconsistency_risk"],
        profile["date_inconsistency_risk"],
        profile["entity_inconsistency_risk"],
        temporal_bundle["temporal_mismatch_risk"],
    )
    if instability_peak >= 0.18:
        return 0.0

    brevity_signal = min(1.0, 14.0 / max(features["word_count"], 1))
    low_detail_signal = 1.0 - _clip_probability(
        (0.45 * profile["suspicious_specificity_score"])
        + (0.30 * profile["weak_support_risk"])
        + (0.25 * profile["unsupported_detail_cluster"])
    )
    shell_bonus = 0.08 if DIRECT_ANSWER_RE.search(features.get("answer_text", "")) else 0.0
    reassurance = _clip_probability((0.58 * brevity_signal) + (0.42 * low_detail_signal) + shell_bonus)
    if features["detail_count"] > 2:
        reassurance *= 0.75
    return _clip_probability(reassurance)


def _top_signal_line(sub_signals: list[dict[str, Any]], max_items: int = 3) -> str:
    ranked = sorted(
        [signal for signal in sub_signals if float(signal.get("risk", 0.0) or 0.0) >= 10.0],
        key=lambda item: float(item.get("risk", 0.0) or 0.0),
        reverse=True,
    )
    if not ranked:
        return "internal instability stayed limited"
    labels = [safe_text(item.get("signal", "")).lower() for item in ranked[:max_items] if safe_text(item.get("signal", ""))]
    if not labels:
        return "internal instability stayed limited"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + ", and " + labels[-1]


def _text_base_score(
    features: dict[str, Any],
    *,
    return_diagnostics: bool = False,
) -> tuple[float, list[dict[str, Any]]] | tuple[float, list[dict[str, Any]], dict[str, Any]]:
    simple_fact_sanity = features.get("simple_fact_sanity")
    question_text = features.get("question_text", "")
    answer_text = features.get("answer_text", "")

    profile = _sep_text_profile(
        features,
        answer_text,
        question=question_text,
        dense_fallback=True,
    )
    temporal_bundle = _base_time_check(
        question=question_text,
        answer_text=answer_text,
        features=features,
        profile=profile,
    )
    direct_reassurance = _direct_answer_boost(
        features=features,
        profile=profile,
        temporal_bundle=temporal_bundle,
    )
    temporal_reassurance = features["simple_fact_reassurance"] * (0.30 if temporal_bundle["date_only_answer"] else 1.0)
    compound_pressure = _clip_probability(
        (0.58 * min(profile["contradiction_risk"], profile["date_inconsistency_risk"]))
        + (0.52 * min(profile["contradiction_risk"], profile["numeric_inconsistency_risk"]))
        + (0.44 * min(profile["contradiction_risk"], profile["entity_inconsistency_risk"]))
    )

    structure_risk = _clip_probability(
        (0.38 * profile["contradiction_risk"])
        + (0.30 * profile["date_inconsistency_risk"])
        + (0.16 * profile["numeric_inconsistency_risk"])
        + (0.16 * profile["entity_inconsistency_risk"])
        + (0.12 * compound_pressure)
    )
    weak_support_pressure = _clip_probability(
        (0.34 * profile["weak_support_risk"])
        + (0.26 * profile["unsupported_detail_cluster"])
        + (0.22 * profile["suspicious_specificity_score"])
        + (0.18 * features["confident_tone_risk"])
    )
    context_risk = profile["uncertainty_language_risk"]
    if structure_risk < 0.18 and weak_support_pressure < 0.22:
        context_risk *= 0.35

    base_score = _clip_probability(
        (0.44 * structure_risk)
        + (0.28 * temporal_bundle["temporal_mismatch_risk"])
        + (0.18 * weak_support_pressure)
        + (0.06 * context_risk)
        + (0.06 * features["obscure_event_risk"])
        - (0.16 * profile["simplicity_signal"])
        - (0.10 * temporal_reassurance)
        - (0.16 * direct_reassurance)
    )

    score = base_score
    if temporal_bundle["date_only_uncertainty_risk"] > 0.0 and simple_fact_sanity is None:
        score = _clip_probability(score + (0.24 * temporal_bundle["date_only_uncertainty_risk"]))

    if profile["contradiction_risk"] >= 0.40 and profile["internal_claim_count"] >= 2:
        score = max(score, 0.74)
    if profile["contradiction_risk"] >= 0.28 and max(
        profile["date_inconsistency_risk"],
        profile["numeric_inconsistency_risk"],
        profile["entity_inconsistency_risk"],
    ) >= 0.24:
        score = max(score, 0.68)
    if profile["date_inconsistency_risk"] >= 0.40 or temporal_bundle["explicit_year_mismatch"]:
        score = max(score, 0.72)
    if temporal_bundle["question_year_drift"]:
        score = max(score, 0.70)
    if temporal_bundle["multi_year_answer"] and temporal_bundle["temporal_conflict_marker"]:
        score = max(score, 0.68)
    if temporal_bundle["date_only_answer"] and simple_fact_sanity is None:
        score = max(score, 0.44)
    if temporal_bundle["date_only_uncertainty_risk"] >= 0.42 and simple_fact_sanity is None:
        score = max(score, 0.54)
    if (
        not features["question_is_simple_factoid"]
        and features["word_count"] >= 16
        and (
            (weak_support_pressure >= 0.66 and profile["suspicious_specificity_score"] >= 0.50)
            or (profile["unsupported_detail_cluster"] >= 0.62 and features["new_entity_count"] >= 2)
        )
    ):
        score = max(score, 0.70)

    if simple_fact_sanity is not None:
        if simple_fact_sanity["verdict"] == "incorrect":
            score = max(score, 0.82)
        else:
            score = min(score, 0.16)

    if (
        simple_fact_sanity is None
        and not temporal_bundle["temporal_question"]
        and direct_reassurance >= 0.55
        and structure_risk <= 0.16
        and temporal_bundle["temporal_mismatch_risk"] <= 0.28
    ):
        score = min(score, 0.29 if features["word_count"] <= 10 else 0.32)

    sub_signals = [
        {
            "signal": "Internal contradiction cues",
            "value": profile["contradiction_count"],
            "risk": round(profile["contradiction_risk"] * 100.0, 1),
            "explanation": "Conflicting statements inside one answer are a primary hallucination risk signal.",
        },
        {
            "signal": "Date inconsistency pressure",
            "value": profile["date_conflict_count"],
            "risk": round(profile["date_inconsistency_risk"] * 100.0, 1),
            "explanation": "Conflicting year/date details across claims are weighted strongly in baseline scoring.",
        },
        {
            "signal": "Numeric inconsistency pressure",
            "value": profile["numeric_conflict_count"],
            "risk": round(profile["numeric_inconsistency_risk"] * 100.0, 1),
            "explanation": "Conflicting quantitative details across related claims increase baseline risk.",
        },
        {
            "signal": "Entity inconsistency pressure",
            "value": profile["entity_conflict_count"],
            "risk": round(profile["entity_inconsistency_risk"] * 100.0, 1),
            "explanation": "Switching central named entities across related claims signals internal instability.",
        },
        {
            "signal": "Compound inconsistency pressure",
            "value": round(compound_pressure, 4),
            "risk": round(compound_pressure * 100.0, 1),
            "explanation": "Contradiction cues become riskier when they co-occur with date, numeric, or entity inconsistencies in the same answer.",
        },
        {
            "signal": "Temporal question-answer mismatch",
            "value": ", ".join(temporal_bundle["answer_years"]) if temporal_bundle["answer_years"] else "none",
            "risk": round(temporal_bundle["temporal_mismatch_risk"] * 100.0, 1),
            "explanation": "Temporal questions are checked for year mismatches and unstable date usage, including bare year/date answers.",
        },
        {
            "signal": "Bare temporal answer pressure",
            "value": "yes" if temporal_bundle["bare_temporal_answer"] else "no",
            "risk": round((temporal_bundle["date_only_uncertainty_risk"] if temporal_bundle["bare_temporal_answer"] else 0.0) * 100.0, 1),
            "explanation": "Short year/date-only answers to temporal questions receive less automatic reassurance unless a strong supporting signal is present.",
        },
        {
            "signal": "Weak-support specificity pressure",
            "value": features["detail_count"],
            "risk": round(weak_support_pressure * 100.0, 1),
            "explanation": "Specific unsupported detail contributes risk, but less than contradiction or numeric/date/entity inconsistency.",
        },
        {
            "signal": "Contextual uncertainty language",
            "value": round(profile["uncertainty_language_risk"], 4),
            "risk": round(context_risk * 100.0, 1),
            "explanation": "Hedging only contributes materially when stronger inconsistency or weak-support signals are also present.",
        },
        {
            "signal": "Direct-answer reassurance",
            "value": round(direct_reassurance, 4),
            "risk": round((1.0 - direct_reassurance) * 100.0, 1),
            "explanation": "Short coherent direct/paraphrase answers are normalized downward when no strong inconsistency cues are present.",
        },
        {
            "signal": "Simple common-fact reassurance",
            "value": round(temporal_reassurance, 4),
            "risk": round((1.0 - temporal_reassurance) * 100.0, 1),
            "explanation": "Simple factoid responses remain low-risk when contradiction and mismatch signals stay limited, but short date-focused answers receive less automatic reassurance.",
        },
    ]
    if simple_fact_sanity is not None:
        sub_signals.append(
            {
                "signal": "Curated simple-fact sanity",
                "value": simple_fact_sanity["candidate"],
                "risk": 92.0 if simple_fact_sanity["verdict"] == "incorrect" else 8.0,
                "explanation": f"A small explicit local fact bank matched this question and expected '{simple_fact_sanity['canonical_answer']}'.",
            }
        )

    diagnostics = {
        "profile": profile,
        "temporal_bundle": temporal_bundle,
        "direct_answer_reassurance": direct_reassurance,
        "temporal_reassurance": temporal_reassurance,
        "compound_inconsistency_pressure": compound_pressure,
        "structural_inconsistency": structure_risk,
        "weak_support_pressure": weak_support_pressure,
        "contextual_uncertainty_risk": context_risk,
    }

    if return_diagnostics:
        return score, sub_signals, diagnostics
    return score, sub_signals


def _fallback_floor(
    *,
    text_features: dict[str, Any],
    sample_metrics: dict[str, Any] | None,
    mode: str,
) -> float:
    if sample_metrics is not None:
        return 0.0

    floor = 0.08 if mode == "uncertainty_baseline" else 0.10
    if text_features["question_is_simple_factoid"]:
        floor = max(floor, 0.14 if mode == "uncertainty_baseline" else 0.16)
        if text_features["claim_count"] == 1 and text_features["detail_count"] == 0 and text_features["word_count"] <= 4:
            floor = max(floor, 0.18 if mode == "uncertainty_baseline" else 0.20)
    return float(floor)


SEP_PROXY_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "during",
    "for",
    "from",
    "had",
    "has",
    "have",
    "happened",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "of",
    "on",
    "or",
    "tell",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
SEP_INST_TERMS = {
    "accord",
    "agreement",
    "authority",
    "countries",
    "global",
    "guidance",
    "international",
    "nations",
    "policy",
    "signed",
    "summit",
    "treaty",
}
SEP_MED_TERMS = {
    "clinical",
    "framework",
    "medical",
    "protocol",
    "regulation",
    "robotic",
    "standards",
    "surgery",
    "surgical",
    "universal",
}
SEP_OBSCURE_TERMS = SEP_INST_TERMS | SEP_MED_TERMS
SEP_CONTRAST_TERMS = {
    "although",
    "but",
    "despite",
    "however",
    "instead",
    "yet",
}
SEP_PLACE_RE = re.compile(
    r"\b(?:in|at|from|to|near|outside|inside|across|through|between)\s+"
    r"([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})\b",
    re.UNICODE,
)


def _norm_sep_token(token: str) -> str:
    cleaned = safe_text(token).lower().strip("'\"")
    if cleaned.endswith("ies") and len(cleaned) > 4:
        cleaned = cleaned[:-3] + "y"
    elif cleaned.endswith("s") and len(cleaned) > 4 and not cleaned.endswith("ss"):
        cleaned = cleaned[:-1]
    return cleaned


def _sep_tokens(text: str) -> set[str]:
    return {
        token
        for token in (_norm_sep_token(raw) for raw in WORD_PATTERN.findall(safe_text(text)))
        if len(token) > 2 and token not in SEP_PROXY_STOPWORDS
    }


def _sep_anchor_tokens(question: str) -> set[str]:
    return _sep_tokens(question)


def _sep_entity_set(text: str) -> set[str]:
    return {_norm_sep_token(span) for span in _title_spans(text)}


def _location_spans(text: str) -> list[str]:
    spans: list[str] = []
    seen: set[str] = set()
    for span in SEP_PLACE_RE.findall(safe_text(text)):
        cleaned = span.strip()
        normalized = normalize_text(cleaned)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        spans.append(cleaned)
    return spans


def _sep_numeric_set(text: str) -> set[str]:
    cleaned = safe_text(text).lower()
    values = set(NUMBER_PATTERN.findall(cleaned))
    if "dozens" in cleaned:
        values.add("dozens")
    if "dozen" in cleaned:
        values.add("dozen")
    return values


def _pairwise_set_overlap(sets: list[set[str]], *, empty_score: float = 1.0) -> float:
    if len(sets) < 2:
        return empty_score
    overlaps: list[float] = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            left = sets[left_index]
            right = sets[right_index]
            if not left and not right:
                overlaps.append(empty_score)
                continue
            if not left or not right:
                overlaps.append(0.0)
                continue
            overlaps.append(len(left & right) / len(left | right))
    return float(np.mean(overlaps)) if overlaps else empty_score


def _presence_score(sets: list[set[str]]) -> float:
    if not sets:
        return 1.0
    counts: dict[str, int] = {}
    sample_count = len(sets)
    for values in sets:
        for item in values:
            counts[item] = counts.get(item, 0) + 1
    if not counts:
        return 1.0
    dominant = sorted((count / sample_count) for count in counts.values() if count >= 2)
    if dominant:
        top_values = dominant[-3:]
        return float(np.mean(top_values))
    return max(counts.values()) / sample_count


def _canonical_month_name(token: str) -> str:
    return MONTH_ALIASES.get(normalize_text(token), normalize_text(token))


def _set_similarity_score(
    left: set[str],
    right: set[str],
    *,
    empty_score: float = 1.0,
    missing_score: float = 0.72,
) -> float:
    if not left and not right:
        return empty_score
    if not left or not right:
        return missing_score
    if left <= right or right <= left:
        return 0.9
    overlap = len(left & right) / len(left | right)
    if overlap > 0.0:
        return max(overlap, 0.55)
    return 0.0


def _sep_time_slots(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    years = set(YEAR_PATTERN.findall(normalized))
    months = {_canonical_month_name(match.group(0)) for match in MONTH_TOKEN_PATTERN.finditer(normalized)}
    month_year_pairs = {
        f"{_canonical_month_name(match.group('month'))}:{match.group('year')}"
        for match in MONTH_YEAR_PATTERN.finditer(normalized)
    }
    return {
        "years": years,
        "months": months,
        "month_year_pairs": month_year_pairs,
    }


def _sep_entity_groups(text: str) -> list[set[str]]:
    groups: list[set[str]] = []
    seen: set[frozenset[str]] = set()
    for span in _title_spans(text):
        tokens = frozenset(
            token
            for token in (_norm_sep_token(raw) for raw in WORD_PATTERN.findall(span))
            if len(token) > 2 and token not in SEP_PROXY_STOPWORDS
        )
        if not tokens or tokens in seen:
            continue
        seen.add(tokens)
        groups.append(set(tokens))
    return groups


def _sep_fact_slot_bundle(question: str, answer_text: str) -> dict[str, Any]:
    primary_claim = _main_claim(answer_text, question)
    question_anchors = _sep_anchor_tokens(question)
    claim_tokens = _sep_tokens(primary_claim)
    core_claim_tokens = claim_tokens - question_anchors
    fact_candidate = ""
    if find_simple_fact(question) is not None:
        candidate = normalize_text(simple_fact_answer(question, answer_text))
        if 0 < len(candidate.split()) <= 4:
            fact_candidate = candidate

    temporal_bundle = _sep_time_slots(answer_text)
    numeric_values = {value for value in _sep_numeric_set(answer_text) if value not in temporal_bundle["years"]}
    entity_groups = _sep_entity_groups(answer_text)
    entity_keys = {" ".join(sorted(group)) for group in entity_groups}
    if fact_candidate:
        fact_tokens = _sep_tokens(fact_candidate)
        if fact_tokens:
            entity_groups = entity_groups + [set(fact_tokens)]
            entity_keys.add(fact_candidate)
            core_claim_tokens = fact_tokens if not core_claim_tokens else core_claim_tokens | fact_tokens

    return {
        "primary_claim": primary_claim,
        "claim_tokens": claim_tokens,
        "core_claim_tokens": core_claim_tokens,
        "entity_groups": entity_groups,
        "entity_keys": entity_keys,
        "fact_candidate": fact_candidate,
        "has_negation": contains_negation(primary_claim),
        "numeric_values": numeric_values,
        **temporal_bundle,
    }


def _sep_entity_score(left_slots: dict[str, Any], right_slots: dict[str, Any]) -> float:
    if left_slots["fact_candidate"] and right_slots["fact_candidate"]:
        return 1.0 if left_slots["fact_candidate"] == right_slots["fact_candidate"] else 0.0

    left_groups = left_slots["entity_groups"]
    right_groups = right_slots["entity_groups"]
    if not left_groups and not right_groups:
        return 1.0
    if not left_groups or not right_groups:
        return 0.72

    best_overlap = 0.0
    for left_group in left_groups:
        for right_group in right_groups:
            if left_group <= right_group or right_group <= left_group:
                return 1.0
            best_overlap = max(best_overlap, len(left_group & right_group) / len(left_group | right_group))
    if best_overlap > 0.0:
        return max(best_overlap, 0.55)
    return 0.15


def _sep_time_score(left_slots: dict[str, Any], right_slots: dict[str, Any]) -> float:
    left_years = left_slots["years"]
    right_years = right_slots["years"]
    left_months = left_slots["months"]
    right_months = right_slots["months"]
    left_pairs = left_slots["month_year_pairs"]
    right_pairs = right_slots["month_year_pairs"]

    if not (left_years or right_years or left_months or right_months or left_pairs or right_pairs):
        return 1.0
    if left_years and right_years:
        if left_years.isdisjoint(right_years):
            return 0.0
        if left_pairs and right_pairs:
            if left_pairs & right_pairs:
                return 1.0
            shared_years = {pair.split(":", 1)[1] for pair in left_pairs} & {pair.split(":", 1)[1] for pair in right_pairs}
            if shared_years:
                return 0.55 if left_months and right_months and left_months.isdisjoint(right_months) else 0.82
        if left_pairs or right_pairs:
            return 0.92
        return 1.0
    if left_pairs or right_pairs:
        return 0.62 if left_months & right_months else 0.35
    if left_months and right_months:
        return 0.82 if left_months & right_months else 0.45
    return 0.72


def _sep_num_score(left_slots: dict[str, Any], right_slots: dict[str, Any]) -> float:
    left_values = left_slots["numeric_values"]
    right_values = right_slots["numeric_values"]
    if ("dozen" in left_values or "dozens" in left_values or "dozen" in right_values or "dozens" in right_values) and left_values and right_values:
        overlap = left_values & right_values
        if overlap:
            return 0.9
        return 0.55
    return _set_similarity_score(left_values, right_values, empty_score=1.0, missing_score=0.74)


def _sep_core_score(left_slots: dict[str, Any], right_slots: dict[str, Any]) -> float:
    if left_slots["fact_candidate"] and right_slots["fact_candidate"]:
        return 1.0 if left_slots["fact_candidate"] == right_slots["fact_candidate"] else 0.0
    return _set_similarity_score(
        left_slots["core_claim_tokens"],
        right_slots["core_claim_tokens"],
        empty_score=1.0,
        missing_score=0.65,
    )


def _sep_polarity_score(left_slots: dict[str, Any], right_slots: dict[str, Any], relatedness: float) -> float:
    if left_slots["has_negation"] == right_slots["has_negation"]:
        return 1.0
    return 0.65 if relatedness < 0.2 else 0.0


def _sep_semantic_score(left_slots: dict[str, Any], right_slots: dict[str, Any], core_score: float) -> float:
    lexical = lexical_overlap(left_slots["primary_claim"], right_slots["primary_claim"])
    claim_overlap = _set_similarity_score(
        left_slots["claim_tokens"],
        right_slots["claim_tokens"],
        empty_score=1.0,
        missing_score=0.55,
    )
    return _clip_probability((0.40 * claim_overlap) + (0.35 * core_score) + (0.25 * lexical))


def _sep_pair_compare(
    question: str,
    left_text: str,
    right_text: str,
    *,
    left_slots: dict[str, Any] | None = None,
    right_slots: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left_slots = left_slots or _sep_fact_slot_bundle(question, left_text)
    right_slots = right_slots or _sep_fact_slot_bundle(question, right_text)
    core_score = _sep_core_score(left_slots, right_slots)
    claim_overlap = _set_similarity_score(
        left_slots["claim_tokens"],
        right_slots["claim_tokens"],
        empty_score=1.0,
        missing_score=0.55,
    )
    entity_score = _sep_entity_score(left_slots, right_slots)
    temporal_score = _sep_time_score(left_slots, right_slots)
    numeric_score = _sep_num_score(left_slots, right_slots)
    if temporal_score >= 0.82 and not left_slots["numeric_values"] and not right_slots["numeric_values"]:
        core_score = max(core_score, 0.88)
    relatedness = max(claim_overlap, core_score, entity_score, lexical_overlap(left_slots["primary_claim"], right_slots["primary_claim"]))
    polarity_score = _sep_polarity_score(left_slots, right_slots, relatedness)
    semantic_score = _sep_semantic_score(left_slots, right_slots, core_score)

    contradiction_cues = list(dict.fromkeys(find_cues(left_slots["primary_claim"], right_slots["primary_claim"])))
    if left_slots["fact_candidate"] and right_slots["fact_candidate"] and left_slots["fact_candidate"] != right_slots["fact_candidate"]:
        contradiction_cues.append("different short-answer target")
    if temporal_score <= 0.15 and relatedness >= 0.18:
        contradiction_cues.append("different temporal details")
    if numeric_score <= 0.15 and left_slots["numeric_values"] and right_slots["numeric_values"] and relatedness >= 0.18:
        contradiction_cues.append("different numeric details")
    if entity_score <= 0.15 and left_slots["entity_keys"] and right_slots["entity_keys"] and relatedness >= 0.18:
        contradiction_cues.append("different named entity details")
    if polarity_score == 0.0 and relatedness >= 0.18:
        contradiction_cues.append("negation mismatch")
    contradiction_cues = list(dict.fromkeys(contradiction_cues))
    if (
        core_score >= 0.82
        and temporal_score >= 0.82
        and numeric_score >= 0.82
        and entity_score >= 0.82
        and polarity_score >= 0.82
    ):
        contradiction_cues = []

    return {
        "explicit_contradiction": bool(contradiction_cues),
        "contradiction_cues": contradiction_cues,
        "temporal_consistency_score": temporal_score,
        "numeric_consistency_score": numeric_score,
        "entity_consistency_score": entity_score,
        "polarity_consistency_score": polarity_score,
        "core_claim_consistency_score": core_score,
        "semantic_consensus_score": semantic_score,
    }


def _sep_slot_bundle(question: str, sample_answers: list[str]) -> dict[str, Any]:
    slot_bundles = [_sep_fact_slot_bundle(question, sample_answer) for sample_answer in sample_answers]
    comparisons: list[dict[str, Any]] = []
    if len(slot_bundles) < 2:
        return {
            "comparisons": comparisons,
            "explicit_contradiction_score": 0.0,
            "temporal_consistency_score": 1.0,
            "temporal_inconsistency_score": 0.0,
            "numeric_consistency_score": 1.0,
            "numeric_inconsistency_score": 0.0,
            "entity_consistency_score": 1.0,
            "entity_inconsistency_score": 0.0,
            "polarity_consistency_score": 1.0,
            "core_claim_consistency_score": 1.0,
            "factual_slot_consistency_score": 1.0,
            "factual_slot_inconsistency_score": 0.0,
            "semantic_consensus_score": 1.0,
            "dominant_sample_risk_source": "stable paraphrase consensus",
        }

    for left_index in range(len(slot_bundles)):
        for right_index in range(left_index + 1, len(slot_bundles)):
            comparison = _sep_pair_compare(
                question,
                sample_answers[left_index],
                sample_answers[right_index],
                left_slots=slot_bundles[left_index],
                right_slots=slot_bundles[right_index],
            )
            comparisons.append(comparison)

    contradiction_values = [1.0 if comparison["explicit_contradiction"] else 0.0 for comparison in comparisons]
    contra_score = _clip_probability(
        (0.90 * float(np.mean(contradiction_values))) + (0.10 * max(contradiction_values))
    )
    time_score = float(np.mean([comparison["temporal_consistency_score"] for comparison in comparisons]))
    num_score = float(np.mean([comparison["numeric_consistency_score"] for comparison in comparisons]))
    entity_score = float(np.mean([comparison["entity_consistency_score"] for comparison in comparisons]))
    polarity_score = float(np.mean([comparison["polarity_consistency_score"] for comparison in comparisons]))
    core_score = float(np.mean([comparison["core_claim_consistency_score"] for comparison in comparisons]))
    consensus_score = float(np.mean([comparison["semantic_consensus_score"] for comparison in comparisons]))
    slot_score = _clip_probability(
        (0.34 * time_score)
        + (0.20 * num_score)
        + (0.18 * entity_score)
        + (0.10 * polarity_score)
        + (0.18 * core_score)
    )
    risk_sources = {
        "contradiction": contra_score * 0.85,
        "temporal inconsistency": 1.0 - time_score,
        "numeric inconsistency": 1.0 - num_score,
        "entity inconsistency": 1.0 - entity_score,
        "low semantic consensus": 1.0 - consensus_score,
    }
    top_sample_risk, dominant_value = max(risk_sources.items(), key=lambda item: (item[1], item[0]))
    if dominant_value < 0.18:
        top_sample_risk = "stable paraphrase consensus"

    return {
        "comparisons": comparisons[:6],
        "explicit_contradiction_score": contra_score,
        "temporal_consistency_score": time_score,
        "temporal_inconsistency_score": 1.0 - time_score,
        "numeric_consistency_score": num_score,
        "numeric_inconsistency_score": 1.0 - num_score,
        "entity_consistency_score": entity_score,
        "entity_inconsistency_score": 1.0 - entity_score,
        "polarity_consistency_score": polarity_score,
        "core_claim_consistency_score": core_score,
        "factual_slot_consistency_score": slot_score,
        "factual_slot_inconsistency_score": 1.0 - slot_score,
        "semantic_consensus_score": consensus_score,
        "dominant_sample_risk_source": top_sample_risk,
    }


def _sep_obscure_risk(answer_text: str) -> float:
    tokens = _sep_tokens(answer_text)
    if not tokens:
        return 0.0
    hits = len(tokens & SEP_OBSCURE_TERMS)
    return _clip_probability((0.65 * min(hits / 4.0, 1.0)) + (0.35 * min(hits / 3.0, 1.0)))


def _sep_conflict_bundle(
    question: str,
    answer_text: str,
    *,
    dense_fallback: bool = False,
) -> dict[str, Any]:
    claims = [match_claim(claim, question) for claim in extract_claims(answer_text, max_claims=6)]
    contrast_marker_count = _count_phrase_matches(answer_text, SEP_CONTRAST_TERMS)
    time_summary_q = bool(re.search(r"\b(?:timeline|chronology|history|sequence)\b", normalize_text(question)))
    fragment_split_used = False
    if dense_fallback:
        if len(claims) < 2 or (len(claims) < 3 and contrast_marker_count):
            fragments = _base_frag_claims(answer_text, question)
            if len(fragments) > len(claims):
                claims = fragments[:6]
                fragment_split_used = True
    elif len(claims) < 2 and contrast_marker_count:
        fragments = [
            match_claim(fragment, question)
            for fragment in re.split(r"\b(?:although|but|however|instead|yet)\b", safe_text(answer_text), flags=re.IGNORECASE)
            if len(WORD_PATTERN.findall(fragment)) >= 3
        ]
        if len(fragments) > len(claims):
            claims = fragments[:6]

    contradictions: list[dict[str, Any]] = []
    unique_cues: set[str] = set()
    num_conflicts = 0
    date_conflict_count = 0
    entity_conflict_count = 0

    for left_index in range(len(claims)):
        for right_index in range(left_index + 1, len(claims)):
            left_claim = claims[left_index]
            right_claim = claims[right_index]
            contradiction_cues = find_cues(left_claim, right_claim)
            left_entities = _sep_entity_set(left_claim)
            right_entities = _sep_entity_set(right_claim)
            left_years = set(YEAR_PATTERN.findall(left_claim))
            right_years = set(YEAR_PATTERN.findall(right_claim))
            left_numbers = _sep_numeric_set(left_claim)
            right_numbers = _sep_numeric_set(right_claim)
            overlap = lexical_overlap(left_claim, right_claim)
            related_pair = overlap >= 0.18 or bool(left_entities & right_entities)
            if dense_fallback and len(claims) <= 4:
                if contrast_marker_count and left_numbers and right_numbers:
                    related_pair = True
                if (contrast_marker_count or time_summary_q) and left_years and right_years:
                    related_pair = True

            numeric_conflict = False
            date_conflict = False
            entity_conflict = False

            if related_pair:
                left_non_year_numbers = left_numbers - left_years
                right_other_nums = right_numbers - right_years
                if left_non_year_numbers and right_other_nums and left_non_year_numbers.isdisjoint(right_other_nums):
                    numeric_conflict = True
                if left_years and right_years and left_years.isdisjoint(right_years):
                    date_conflict = True
                if left_entities and right_entities and left_entities.isdisjoint(right_entities):
                    entity_conflict = True

            if "different numeric details" in contradiction_cues:
                numeric_conflict = True
            if "different named entity details" in contradiction_cues:
                entity_conflict = True
            if "different predicate details" in contradiction_cues and left_years and right_years and left_years != right_years:
                date_conflict = True

            if numeric_conflict:
                num_conflicts += 1
            if date_conflict:
                date_conflict_count += 1
            if entity_conflict:
                entity_conflict_count += 1

            if not contradiction_cues:
                continue
            unique_cues.update(contradiction_cues)
            contradictions.append({"left_claim": left_claim, "right_claim": right_claim, "cues": contradiction_cues})

    comparison_count = (len(claims) * (len(claims) - 1)) // 2
    contradiction_ratio = len(contradictions) / comparison_count if comparison_count else 0.0
    num_conflict_rate = num_conflicts / comparison_count if comparison_count else 0.0
    date_conflict_ratio = date_conflict_count / comparison_count if comparison_count else 0.0
    entity_conflict_ratio = entity_conflict_count / comparison_count if comparison_count else 0.0

    contradiction_risk = _clip_probability(
        (0.72 * contradiction_ratio)
        + (0.10 * min(len(unique_cues) / 2.0, 1.0))
        + (0.09 * min(contrast_marker_count / 2.0, 1.0) if contradictions else 0.0)
        + (0.09 * max(date_conflict_ratio, num_conflict_rate, entity_conflict_ratio))
    )
    num_risk = _clip_probability((0.70 * num_conflict_rate) + (0.30 * min(num_conflicts / 2.0, 1.0)))
    date_risk = _clip_probability((0.72 * date_conflict_ratio) + (0.28 * min(date_conflict_count / 2.0, 1.0)))
    entity_risk = _clip_probability((0.68 * entity_conflict_ratio) + (0.32 * min(entity_conflict_count / 2.0, 1.0)))
    return {
        "claim_count": len(claims),
        "comparison_count": comparison_count,
        "contradiction_count": len(contradictions),
        "contradiction_ratio": contradiction_ratio,
        "contrast_marker_count": contrast_marker_count,
        "fragment_split_used": fragment_split_used,
        "contradiction_risk": contradiction_risk,
        "numeric_conflict_count": num_conflicts,
        "date_conflict_count": date_conflict_count,
        "entity_conflict_count": entity_conflict_count,
        "numeric_inconsistency_risk": num_risk,
        "date_inconsistency_risk": date_risk,
        "entity_inconsistency_risk": entity_risk,
        "contradiction_examples": contradictions[:2],
    }


def _sep_detail_bundle(question: str, answer_text: str, features: dict[str, Any]) -> dict[str, Any]:
    normalized_question = normalize_text(question)
    answer_tokens = _sep_tokens(answer_text)
    question_tokens = _sep_anchor_tokens(question)
    novel_detail_tokens = sorted(token for token in (answer_tokens - question_tokens) if len(token) > 4)
    novel_detail_pressure = _clip_probability(max(len(novel_detail_tokens) - 2, 0) / 5.0)
    location_spans = _location_spans(answer_text)
    new_location_spans = [span for span in location_spans if normalize_text(span) not in normalized_question]
    unsupported_cluster = _clip_probability(
        (0.24 * min(max(features["new_entity_count"] - 1, 0) / 3.0, 1.0))
        + (0.18 * min(features["number_count"] / 2.0, 1.0))
        + (0.14 * min(features["temporal_count"] / 1.0, 1.0))
        + (0.14 * min(len(new_location_spans) / 2.0, 1.0))
        + (0.16 * min(features["event_style_count"] / 3.0, 1.0))
        + (0.14 * novel_detail_pressure)
    )
    weak_support_risk = _clip_probability(
        unsupported_cluster
        * (
            (0.42 * features["attribution_absence"])
            + (0.33 * features["confident_tone_risk"])
            + (0.25 * features["low_hedging"])
        )
    )
    hedge_risk = _clip_probability(
        min(features["hedge_count"] / 3.0, 1.0)
        * (
            (0.55 * min(features["claim_count"] / 3.0, 1.0))
            + (0.45 * min(max(features["detail_count"], 1) / 4.0, 1.0))
        )
    )
    if features["question_is_simple_factoid"] and features["word_count"] <= 6 and features["detail_count"] == 0:
        hedge_risk *= 0.35
    return {
        "location_spans": location_spans,
        "new_location_spans": new_location_spans,
        "novel_detail_tokens": novel_detail_tokens[:8],
        "unsupported_detail_cluster": unsupported_cluster,
        "weak_support_risk": weak_support_risk,
        "uncertainty_language_risk": hedge_risk,
    }


def _sep_text_profile(
    features: dict[str, Any],
    answer_text: str,
    *,
    question: str = "",
    dense_fallback: bool = False,
) -> dict[str, Any]:
    obscure_event_risk = _sep_obscure_risk(answer_text)
    contradiction_bundle = _sep_conflict_bundle(
        question,
        answer_text,
        dense_fallback=dense_fallback,
    )
    detail_bundle = _sep_detail_bundle(question, answer_text, features)
    suspect_detail_score = _clip_probability(
        (0.20 * features["named_entity_risk"])
        + (0.14 * features["specificity_risk"])
        + (0.18 * features["unsupported_specificity_risk"])
        + (0.08 * features["claim_density_risk"])
        + (0.10 * obscure_event_risk)
        + (0.16 * detail_bundle["unsupported_detail_cluster"])
        + (0.14 * detail_bundle["weak_support_risk"])
    )
    simplicity_signal = _clip_probability(
        (1.0 - obscure_event_risk)
        * (1.0 - detail_bundle["unsupported_detail_cluster"])
        * (1.0 - contradiction_bundle["contradiction_risk"])
        * (1.0 - features["multi_claim_risk"])
        * min(12.0 / max(features["word_count"], 1), 1.0)
    )
    text_score = _clip_probability(
        (0.30 * suspect_detail_score)
        + (0.18 * detail_bundle["weak_support_risk"])
        + (0.16 * contradiction_bundle["contradiction_risk"])
        + (0.12 * features["confident_tone_risk"])
        + (0.12 * detail_bundle["uncertainty_language_risk"])
        + (0.12 * obscure_event_risk)
        - (0.18 * simplicity_signal)
    )
    if (
        features["question_is_simple_factoid"]
        and features["claim_count"] == 1
        and features["detail_count"] == 0
        and detail_bundle["weak_support_risk"] <= 0.10
        and contradiction_bundle["contradiction_risk"] == 0.0
    ):
        text_score = min(text_score, 0.18)
    return {
        "obscure_event_risk": obscure_event_risk,
        "simplicity_signal": simplicity_signal,
        "suspicious_specificity_score": suspect_detail_score,
        "text_score": text_score,
        "unsupported_detail_cluster": detail_bundle["unsupported_detail_cluster"],
        "weak_support_risk": detail_bundle["weak_support_risk"],
        "uncertainty_language_risk": detail_bundle["uncertainty_language_risk"],
        "contradiction_risk": contradiction_bundle["contradiction_risk"],
        "contradiction_count": contradiction_bundle["contradiction_count"],
        "contradiction_ratio": contradiction_bundle["contradiction_ratio"],
        "contrast_marker_count": contradiction_bundle["contrast_marker_count"],
        "internal_claim_count": contradiction_bundle["claim_count"],
        "fragment_split_used": contradiction_bundle["fragment_split_used"],
        "numeric_conflict_count": contradiction_bundle["numeric_conflict_count"],
        "date_conflict_count": contradiction_bundle["date_conflict_count"],
        "entity_conflict_count": contradiction_bundle["entity_conflict_count"],
        "numeric_inconsistency_risk": contradiction_bundle["numeric_inconsistency_risk"],
        "date_inconsistency_risk": contradiction_bundle["date_inconsistency_risk"],
        "entity_inconsistency_risk": contradiction_bundle["entity_inconsistency_risk"],
        "contradiction_examples": contradiction_bundle["contradiction_examples"],
        "location_spans": detail_bundle["location_spans"],
        "new_location_spans": detail_bundle["new_location_spans"],
        "novel_detail_tokens": detail_bundle["novel_detail_tokens"],
        "confident_tone_risk": features["confident_tone_risk"],
        "claim_density_risk": features["claim_density_risk"],
    }


def _sep_sample_bundle(
    *,
    question: str,
    answer: str,
    sample_metrics: dict[str, Any] | None,
    main_features: dict[str, Any],
) -> dict[str, Any]:
    sample_answers = sample_metrics["sample_answers"] if sample_metrics is not None else [safe_text(answer)]
    sample_profiles = [
        _sep_text_profile(_text_features(question, sample_answer), sample_answer, question=question)
        for sample_answer in sample_answers
    ]
    main_profile = _sep_text_profile(main_features, answer, question=question)
    simple_fact_plurality = _fact_vote_bundle(question, sample_answers) if sample_metrics is not None else None

    content_sets = [_sep_tokens(sample_answer) for sample_answer in sample_answers]
    paraphrase_stability = _pairwise_set_overlap(content_sets, empty_score=1.0)
    slot_bundle = _sep_slot_bundle(question, sample_answers)
    entity_score = slot_bundle["entity_consistency_score"]
    time_score = slot_bundle["temporal_consistency_score"]
    num_score = slot_bundle["numeric_consistency_score"]
    polarity_score = slot_bundle["polarity_consistency_score"]
    core_score = slot_bundle["core_claim_consistency_score"]

    question_anchors = _sep_anchor_tokens(question)
    if question_anchors:
        anchor_coverages = [len(question_anchors & content_set) / len(question_anchors) for content_set in content_sets]
        target_score = float(np.mean(anchor_coverages)) if anchor_coverages else 0.0
    else:
        target_score = 1.0

    if sample_metrics is None:
        semantic_score = slot_bundle["semantic_consensus_score"]
        instability_penalty = 0.0
    else:
        semantic_score = _clip_probability(
            (0.65 * slot_bundle["semantic_consensus_score"])
            + (0.20 * paraphrase_stability)
            + (0.15 * (1.0 - float(sample_metrics["mean_disagreement"])))
        )
        instability_penalty = _clip_probability(
            (0.52 * slot_bundle["explicit_contradiction_score"])
            + (0.30 * slot_bundle["factual_slot_inconsistency_score"])
            + (0.18 * (1.0 - semantic_score))
        )

    sample_score = _clip_probability(
        (0.36 * (1.0 - slot_bundle["explicit_contradiction_score"]))
        + (0.28 * (1.0 - slot_bundle["factual_slot_inconsistency_score"]))
        + (0.20 * semantic_score)
        + (0.08 * paraphrase_stability)
        + (0.08 * target_score)
    )
    sample_detail_mean = float(np.mean([profile["suspicious_specificity_score"] for profile in sample_profiles]))
    detail_amp_score = _clip_probability(
        (0.55 * sample_detail_mean)
        + (0.25 * main_profile["suspicious_specificity_score"])
        + (0.20 * main_profile["weak_support_risk"])
    )
    suspect_flag = bool(
        sample_metrics is not None
        and max(main_profile["suspicious_specificity_score"], main_profile["weak_support_risk"]) >= 0.46
        and sample_score >= 0.48
        and slot_bundle["factual_slot_consistency_score"] >= 0.72
        and slot_bundle["semantic_consensus_score"] >= 0.45
    )
    suspect_consensus = _clip_probability(sample_score * detail_amp_score)
    suspect_floor = 0.0
    if suspect_flag:
        suspect_floor = _clip_probability(
            (0.60 * main_profile["suspicious_specificity_score"])
            + (0.20 * main_profile["weak_support_risk"])
            + (0.20 * sample_score)
        )

    vote_disagree_score = 0.0 if simple_fact_plurality is None else float(simple_fact_plurality["disagreement_score"])
    main_loses_vote = bool(
        simple_fact_plurality is not None and simple_fact_plurality["majority_disagrees_with_main"]
    )

    fact_boost = 0.0
    if sample_metrics is not None and not main_loses_vote:
        fact_boost = _clip_probability(
            sample_score
            * (1.0 - max(main_profile["suspicious_specificity_score"], main_profile["weak_support_risk"]))
            * main_profile["simplicity_signal"]
        )
        if slot_bundle["explicit_contradiction_score"] > 0.0:
            fact_boost *= 0.25

    return {
        "answer_target_consistency_score": target_score,
        "core_claim_consistency_score": core_score,
        "dominant_sample_risk_source": slot_bundle["dominant_sample_risk_source"],
        "entity_consistency_score": entity_score,
        "entity_inconsistency_score": slot_bundle["entity_inconsistency_score"],
        "explicit_contradiction_score": slot_bundle["explicit_contradiction_score"],
        "factual_slot_consistency_score": slot_bundle["factual_slot_consistency_score"],
        "factual_slot_inconsistency_score": slot_bundle["factual_slot_inconsistency_score"],
        "instability_penalty": instability_penalty,
        "main_profile": main_profile,
        "numeric_consistency_score": num_score,
        "numeric_inconsistency_score": slot_bundle["numeric_inconsistency_score"],
        "pairwise_slot_comparisons": slot_bundle["comparisons"],
        "polarity_consistency_score": polarity_score,
        "sample_answers": sample_answers,
        "sample_consistency_score": sample_score,
        "sample_count": len(sample_answers),
        "sample_specificity_mean": sample_detail_mean,
        "semantic_agreement_score": semantic_score,
        "semantic_consensus_score": slot_bundle["semantic_consensus_score"],
        "simple_fact_plurality": simple_fact_plurality,
        "simple_fact_reassurance": fact_boost,
        "specificity_amplification_score": detail_amp_score,
        "structural_paraphrase_stability": paraphrase_stability,
        "suspicious_consensus_flag": suspect_flag,
        "suspicious_consensus_floor": suspect_floor,
        "suspicious_consensus_score": suspect_consensus,
        "suspicious_specificity_score": main_profile["suspicious_specificity_score"],
        "temporal_consistency_score": time_score,
        "temporal_inconsistency_score": slot_bundle["temporal_inconsistency_score"],
        "unsupported_detail_cluster": main_profile["unsupported_detail_cluster"],
        "weak_support_risk": main_profile["weak_support_risk"],
        "uncertainty_language_risk": main_profile["uncertainty_language_risk"],
        "contradiction_risk": main_profile["contradiction_risk"],
        "plurality_disagreement_score": vote_disagree_score,
        "main_answer_conflicts_with_sample_plurality": main_loses_vote,
    }


def _text_proxy_sep_score(
    features: dict[str, Any],
    *,
    question: str = "",
    answer_text: str = "",
) -> tuple[float, list[dict[str, Any]]]:
    profile = _sep_text_profile(features, answer_text, question=question)
    sep_sub_signals = [
        {
            "signal": "Suspicious specificity in main answer",
            "value": round(profile["suspicious_specificity_score"], 4),
            "risk": round(profile["suspicious_specificity_score"] * 100.0, 1),
            "explanation": "Highly specific claims about obscure entities, dates, counts, or institutional outcomes keep SEP risk elevated.",
        },
        {
            "signal": "Internal contradiction pressure",
            "value": round(profile["contradiction_risk"], 4),
            "risk": round(profile["contradiction_risk"] * 100.0, 1),
            "explanation": "Conflicting claim details inside one answer raise SEP risk even before any external checking.",
        },
        {
            "signal": "Unsupported detail cluster",
            "value": round(profile["unsupported_detail_cluster"], 4),
            "risk": round(profile["unsupported_detail_cluster"] * 100.0, 1),
            "explanation": "Clusters of new entities, dates, locations, and numbers raise risk when the question itself did not anchor them.",
        },
        {
            "signal": "Overconfident specifics with weak support",
            "value": round(profile["weak_support_risk"], 4),
            "risk": round(profile["weak_support_risk"] * 100.0, 1),
            "explanation": "Specific factual tone with little attribution or support language is riskier than a short direct answer.",
        },
        {
            "signal": "Hedging / uncertainty language",
            "value": round(profile["uncertainty_language_risk"], 4),
            "risk": round(profile["uncertainty_language_risk"] * 100.0, 1),
            "explanation": "Noticeable hedging can act as a local uncertainty cue when the answer still carries factual detail.",
        },
        {
            "signal": "Simple-fact simplicity signal",
            "value": round(profile["simplicity_signal"], 4),
            "risk": round((1.0 - profile["simplicity_signal"]) * 100.0, 1),
            "explanation": "Short, direct answers to common factual questions lower SEP risk when suspicious detail load is minimal.",
        },
    ]
    return profile["text_score"], sep_sub_signals


def _sep_score_bundle(
    *,
    question: str,
    answer: str,
    text_features: dict[str, Any],
    sample_metrics: dict[str, Any] | None,
    simple_fact_sanity: dict[str, Any] | None,
) -> dict[str, Any]:
    text_score, text_sub_signals = _text_proxy_sep_score(text_features, question=question, answer_text=answer)
    sep_bundle = _sep_sample_bundle(
        question=question,
        answer=answer,
        sample_metrics=sample_metrics,
        main_features=text_features,
    )
    risk_floor = 0.0
    risk_cap: float | None = None
    if sample_metrics is None:
        proxy_score = text_score
        if simple_fact_sanity is not None and simple_fact_sanity["verdict"] == "incorrect":
            risk_floor = 0.72
            proxy_score = max(proxy_score, risk_floor)
    else:
        plurality_floor = 0.0
        if sep_bundle["main_answer_conflicts_with_sample_plurality"] and text_features["question_is_simple_factoid"]:
            plurality_floor = 0.62 if (sep_bundle["simple_fact_plurality"] or {}).get("plurality_matches_canonical") else 0.48
        if simple_fact_sanity is not None and simple_fact_sanity["verdict"] == "incorrect":
            plurality_floor = max(plurality_floor, 0.72)

        contra_pressure = max(sep_bundle["explicit_contradiction_score"], sep_bundle["plurality_disagreement_score"])
        temporal_drift_floor = 0.67 if sep_bundle["temporal_inconsistency_score"] >= 0.58 and sep_bundle["core_claim_consistency_score"] >= 0.35 else 0.0
        contradiction_floor = 0.72 if contra_pressure >= 0.48 else 0.0
        risk_floor = max(plurality_floor, temporal_drift_floor, contradiction_floor, sep_bundle["suspicious_consensus_floor"])
        proxy_score = _clip_probability(
            max(text_score, risk_floor)
            + (0.12 * contra_pressure)
            + (0.10 * sep_bundle["factual_slot_inconsistency_score"])
            + (0.06 * (1.0 - sep_bundle["semantic_agreement_score"]))
            + (0.06 * sep_bundle["unsupported_detail_cluster"])
            + (0.04 * sep_bundle["weak_support_risk"])
            - (0.24 * sep_bundle["simple_fact_reassurance"])
        )
        proxy_score = max(proxy_score, risk_floor)

        if (
            contra_pressure <= 0.05
            and sep_bundle["temporal_consistency_score"] >= 0.88
            and sep_bundle["numeric_consistency_score"] >= 0.82
            and sep_bundle["entity_consistency_score"] >= 0.82
            and sep_bundle["polarity_consistency_score"] >= 0.82
            and sep_bundle["core_claim_consistency_score"] >= 0.82
            and sep_bundle["semantic_agreement_score"] >= 0.55
            and sep_bundle["semantic_consensus_score"] >= 0.68
            and sep_bundle["suspicious_specificity_score"] <= 0.38
            and sep_bundle["weak_support_risk"] <= 0.28
            and not sep_bundle["suspicious_consensus_flag"]
        ):
            risk_cap = 0.32
            proxy_score = min(proxy_score, risk_cap)

    sep_bundle["paraphrase_safety_cap"] = risk_cap
    sep_bundle["paraphrase_safety_cap_applied"] = risk_cap is not None
    return {
        "score": proxy_score,
        "text_score": text_score,
        "text_sub_signals": text_sub_signals,
        "sep_bundle": sep_bundle,
        "risk_floor": risk_floor,
        "risk_cap": risk_cap,
    }


def _join_reasons(fragments: list[str]) -> str:
    cleaned = [fragment for fragment in fragments if fragment]
    if not cleaned:
        return "the internal instability signals stayed limited"
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + ", and " + cleaned[-1]


def _sep_backend_reasons(features: dict[str, Any]) -> list[str]:
    fragments: list[str] = []
    hidden_drift_risk = _clip_probability(features["hidden_drift_mean"] / 0.35)
    layer_dispersion_risk = _clip_probability(features["layer_centroid_dispersion"] / 0.35)
    token_risk = _clip_probability(
        (0.45 * _clip_probability(features["mean_negative_log_prob"] / 5.5))
        + (0.30 * _clip_probability(1.0 - features["mean_token_probability"]))
        + (0.25 * _clip_probability(1.0 - min(features["top2_margin_mean"] / 0.45, 1.0)))
    )
    if hidden_drift_risk >= 0.45:
        fragments.append("hidden-state drift across the answer tokens was elevated")
    if layer_dispersion_risk >= 0.45:
        fragments.append("selected hidden layers disagreed on the answer representation")
    if token_risk >= 0.52:
        fragments.append("token-level uncertainty stayed elevated")
    return fragments[:2]


def _sep_proxy_reasons(
    *,
    profile: dict[str, Any],
    sep_bundle: dict[str, Any] | None,
    simple_fact_sanity: dict[str, Any] | None,
) -> list[str]:
    fragments: list[str] = []
    if simple_fact_sanity is not None and simple_fact_sanity["verdict"] == "incorrect":
        fragments.append("a small explicit local fact bank disagreed with the answer")
    elif simple_fact_sanity is not None and simple_fact_sanity["verdict"] == "correct" and profile["simplicity_signal"] >= 0.55:
        fragments.append("a small explicit local fact bank agreed with the answer")

    if sep_bundle is not None and sep_bundle["main_answer_conflicts_with_sample_plurality"]:
        fragments.append("sampled alternatives converged on a different answer than the main answer")
    elif sep_bundle is not None and sep_bundle["dominant_sample_risk_source"] == "temporal inconsistency" and sep_bundle["temporal_inconsistency_score"] >= 0.40:
        fragments.append("sampled alternatives drifted on dates or years")
    elif sep_bundle is not None and sep_bundle["dominant_sample_risk_source"] == "numeric inconsistency" and sep_bundle["numeric_inconsistency_score"] >= 0.40:
        fragments.append("sampled alternatives drifted on numeric details")
    elif sep_bundle is not None and sep_bundle["dominant_sample_risk_source"] == "entity inconsistency" and sep_bundle["entity_inconsistency_score"] >= 0.40:
        fragments.append("sampled alternatives drifted on named entities")
    elif sep_bundle is not None and sep_bundle["dominant_sample_risk_source"] == "low semantic consensus" and sep_bundle["semantic_consensus_score"] <= 0.45:
        fragments.append("sampled alternatives showed low semantic consensus")
    elif sep_bundle is not None and sep_bundle["explicit_contradiction_score"] >= 0.25:
        fragments.append("sampled alternatives contained explicit contradictions")
    elif sep_bundle is not None and sep_bundle["suspicious_consensus_flag"]:
        fragments.append("sampled answers consistently reinforced the same highly specific claim")
    elif sep_bundle is not None and sep_bundle["semantic_consensus_score"] <= 0.45:
        fragments.append("sampled alternatives showed low semantic consensus")
    elif sep_bundle is not None and sep_bundle["paraphrase_safety_cap_applied"]:
        fragments.append("sampled alternatives stayed semantically close and agreed on the same facts")
    elif sep_bundle is not None and sep_bundle["instability_penalty"] >= 0.40:
        fragments.append("sampled answers disagreed on core entities, numbers, dates, or outcomes")

    if profile["contradiction_risk"] >= 0.25:
        fragments.append("the answer contains internally conflicting claim details")
    if profile["weak_support_risk"] >= 0.32 or profile["unsupported_detail_cluster"] >= 0.40:
        fragments.append("it states specific entities, dates, locations, or numbers with weak support cues")
    elif profile["uncertainty_language_risk"] >= 0.25:
        fragments.append("it uses notable hedging or uncertainty language")
    elif profile["confident_tone_risk"] >= 0.45 and profile["weak_support_risk"] >= 0.18:
        fragments.append("it uses an overconfident factual tone without support cues")

    if not fragments and sep_bundle is not None and sep_bundle["paraphrase_safety_cap_applied"]:
        fragments.append("sampled alternatives were stable paraphrases with aligned facts")
    elif not fragments and sep_bundle is not None and sep_bundle["simple_fact_reassurance"] >= 0.45 and profile["simplicity_signal"] >= 0.55:
        fragments.append("the answer stays short and low-detail, and sampled alternatives are stable paraphrases")
    elif not fragments and profile["simplicity_signal"] >= 0.55 and profile["suspicious_specificity_score"] <= 0.25:
        fragments.append("the answer stays short, direct, and low on unsupported detail")
    return fragments[:3]


def _sep_explanation_text(
    *,
    score: float,
    fallback: bool,
    profile: dict[str, Any],
    sep_bundle: dict[str, Any] | None,
    simple_fact_sanity: dict[str, Any] | None,
    backend_features: dict[str, Any] | None = None,
    probe_loaded: bool = False,
) -> str:
    if fallback:
        prefix = "The local Hugging Face backend was unavailable, so SEP-inspired mode used a deterministic local approximation."
    else:
        prefix = "SEP-inspired mode used hidden-state statistics plus token uncertainty"
        if probe_loaded:
            prefix += " and a lightweight logistic probe"
        prefix += "."

    reason_fragments = _sep_proxy_reasons(
        profile=profile,
        sep_bundle=sep_bundle,
        simple_fact_sanity=simple_fact_sanity,
    )
    if backend_features is not None:
        reason_fragments = _sep_backend_reasons(backend_features) + reason_fragments
    level = "high" if score >= 0.67 else "medium" if score >= 0.34 else "low"
    explanation = f"{prefix} The final SEP-style risk is {level} because {_join_reasons(reason_fragments)}."
    if fallback:
        explanation += " This fallback still estimates internal suspiciousness rather than proving factual correctness."
    return explanation


def _text_claims(
    *,
    question: str,
    claims: list[str],
    mode: str,
    backend_error: Exception,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for claim in claims[:8]:
        claim_features = _text_features(question, claim)
        if mode == "uncertainty_baseline":
            claim_score, _ = _text_base_score(claim_features)
        else:
            claim_score, _ = _text_proxy_sep_score(claim_features, answer_text=claim)
        claim_score = max(claim_score, _fallback_floor(text_features=claim_features, sample_metrics=None, mode=mode))
        if claim_score >= 0.67:
            status = "high_risk"
        elif claim_score >= 0.34:
            status = "mixed"
        else:
            status = "low_risk"
        findings.append(
            {
                "claim": claim,
                "status": status,
                "score": round(claim_score * 100.0, 1),
                "best_match": "Text-only fallback proxy; no external evidence is used.",
                "reason": (
                    "Deterministic text-only fallback estimated "
                    f"{claim_score:.2f} risk from specificity, named-entity load, and unsupported confident tone "
                    f"because the local HF backend was unavailable: {backend_error}"
                ),
            }
        )
    return findings


def _text_fallback(
    *,
    question: str,
    answer: str,
    sampled_answers_text: str,
    method_name: str,
    mode: str,
    family: str,
    config: SignalConfig,
    backend_error: Exception,
    start_time: float,
    token_logprobs: list[list[float]] | None = None,
) -> dict[str, Any]:
    text_features = _text_features(question, answer)
    sample_metrics = _sample_metrics(question, answer, sampled_answers_text)
    simple_fact_sanity = text_features.get("simple_fact_sanity")
    logprob_input = _logprob_bundle(token_logprobs)

    if mode == "uncertainty_baseline":
        text_score, sub_signals, baseline_diagnostics = _text_base_score(text_features, return_diagnostics=True)
        final_score = text_score
        if sample_metrics is not None:
            sample_score, sample_signals = _sample_proxy_score(sample_metrics, mode=mode)
            final_score = _clip_probability((0.70 * text_score) + (0.30 * sample_score))
            sub_signals = sub_signals + sample_signals
        if logprob_input is not None:
            final_score = _clip_probability((0.76 * final_score) + (0.24 * logprob_input["uncertainty_score"]))
            sub_signals = sub_signals + [
                {
                    "signal": "External token logprob uncertainty",
                    "value": round(logprob_input["uncertainty_score"], 4),
                    "risk": round(logprob_input["uncertainty_score"] * 100.0, 1),
                    "explanation": "Externally supplied token logprob uncertainty was blended into fallback risk calibration.",
                }
            ]
        if simple_fact_sanity is not None and simple_fact_sanity["verdict"] == "incorrect":
            final_score = max(final_score, 0.72)
        # Nonzero floor.
        uncertainty_floor = _fallback_floor(text_features=text_features, sample_metrics=sample_metrics, mode=mode)
        floor_applied = uncertainty_floor > 0.0 and final_score < uncertainty_floor
        if floor_applied:
            final_score = uncertainty_floor
            sub_signals = sub_signals + [
                {
                    "signal": "Fallback uncertainty floor",
                    "value": round(uncertainty_floor, 4),
                    "risk": round(uncertainty_floor * 100.0, 1),
                    "explanation": "Because this deterministic no-evidence fallback cannot verify factual correctness from wording alone, it retains a small non-zero uncertainty floor instead of showing zero risk.",
                }
            ]
        if simple_fact_sanity is not None and simple_fact_sanity["verdict"] == "incorrect":
            explanation = (
                "The local Hugging Face backend was unavailable, so this run used a deterministic local approximation. "
                "A small explicit local fact bank matched this short factoid question and the provided answer disagreed with the curated answer, so the fallback treated it as genuinely risky rather than automatically safe."
            )
        elif simple_fact_sanity is not None and simple_fact_sanity["verdict"] == "correct":
            explanation = (
                "The local Hugging Face backend was unavailable, so this run used a deterministic local approximation. "
                "A small explicit local fact bank matched this short factoid question and the provided answer aligned with that curated answer, which kept the fallback risk low without pretending to verify arbitrary world knowledge."
            )
        else:
            explanation = (
                "The local Hugging Face backend was unavailable, so this run used a deterministic local approximation based on "
                "internal contradiction cues, temporal alignment checks, numeric/date/entity inconsistency checks, and calibrated support-vs-reassurance signals. "
                "It does not know factual correctness from wording alone; it estimates how internally risky the answer appears in a no-evidence setting."
            )
        if floor_applied:
            explanation += " A small non-zero uncertainty floor was retained because deterministic fallback cannot verify factual correctness from wording alone."
        explanation += f" Strongest internal signals in this run were {_top_signal_line(sub_signals)}."
        confidence_base = _clip_probability(
            0.28
            + min(text_features["word_count"], 60) / 420.0
            + (0.10 if sample_metrics is not None else 0.0)
        )
        confidence_cap = 0.62 if sample_metrics is not None else 0.52
        confidence = min(confidence_base, confidence_cap)
        fallback_type = "text_proxy_with_sample_consistency" if sample_metrics is not None else "text_only_proxy"
        backend_error_text = _offline_error(backend_error)
        sampled_answers = [] if sample_metrics is None else sample_metrics["sample_answers"]
        if simple_fact_sanity is not None and simple_fact_sanity["verdict"] == "incorrect":
            summary = (
                f"{method_name} used a deterministic fallback approximation because {config.model_name} could not be loaded; "
                f"a curated simple-fact sanity check disagreed with the answer and pushed the risk to {final_score:.2f}."
            )
        else:
            summary = (
                f"{method_name} used a deterministic fallback approximation because {config.model_name} could not be loaded; "
                f"the suspiciousness estimate was {final_score:.2f}."
            )
        if floor_applied:
            summary += " A small non-zero uncertainty floor was retained because this fallback cannot verify correctness from wording alone."
        limitations = (
            "This fallback does not use token probabilities or hidden states. It is a deterministic text-only approximation based on "
            "internal contradiction cues, numeric/date/entity inconsistency checks, temporal alignment checks, an explicit small simple-fact bank for a few canonical checks, "
            "and optional sampled-answer disagreement, so it should be read as suspiciousness estimation rather than general factual verification."
        )
        intermediate_steps = [
            {
                "stage": "fallback_activation",
                "output": {
                    "mode": mode,
                    "fallback_type": fallback_type,
            "used_external_token_logprobs": logprob_input is not None,
            "external_token_logprob_score": None if logprob_input is None else round(logprob_input["uncertainty_score"], 6),
            "external_token_logprob_token_count": 0 if logprob_input is None else int(logprob_input["token_count"]),
                    "backend_available": False,
                    "backend_error": backend_error_text,
                    "model_name": config.model_name,
                },
            },
            {
                "stage": "text_proxy_features",
                "output": {
                    "claim_count": text_features["claim_count"],
                    "word_count": text_features["word_count"],
                    "detail_count": text_features["detail_count"],
                    "new_entity_spans": text_features["new_entity_spans"],
                    "certainty_count": text_features["certainty_count"],
                    "hedge_count": text_features["hedge_count"],
                    "attribution_count": text_features["attribution_count"],
                    "simple_fact_sanity": simple_fact_sanity,
                    "contradiction_risk": round(baseline_diagnostics["profile"]["contradiction_risk"], 4),
                    "date_inconsistency_risk": round(baseline_diagnostics["profile"]["date_inconsistency_risk"], 4),
                    "numeric_inconsistency_risk": round(baseline_diagnostics["profile"]["numeric_inconsistency_risk"], 4),
                    "entity_inconsistency_risk": round(baseline_diagnostics["profile"]["entity_inconsistency_risk"], 4),
                    "temporal_mismatch_risk": round(baseline_diagnostics["temporal_bundle"]["temporal_mismatch_risk"], 4),
                    "direct_answer_reassurance": round(baseline_diagnostics["direct_answer_reassurance"], 4),
                },
            },
        ]
        if sample_metrics is not None:
            intermediate_steps.append(
                {
                    "stage": "sample_consistency_proxy",
                    "output": {
                        "primary_claim": sample_metrics["primary_claim"],
                        "comparisons": sample_metrics["comparisons"],
                        "support_ratio": round(float(sample_metrics["support_ratio"]), 4),
                        "contradiction_ratio": round(float(sample_metrics["contradiction_ratio"]), 4),
                        "pairwise_instability": round(float(sample_metrics["pairwise_instability"]), 4),
                        "unique_claim_ratio": round(float(sample_metrics["unique_claim_ratio"]), 4),
                    },
                }
            )
        metadata = {
            "mode": mode,
            "hf_model_name": config.model_name,
            "backend_model_name": config.model_name,
            "backend_available": False,
            "backend_status": "unavailable",
            "backend_status_label": "HF backend unavailable",
            "backend_error": backend_error_text,
            "fallback_mode": True,
            "fallback_type": fallback_type,
            "used_external_token_logprobs": logprob_input is not None,
            "external_token_logprob_score": None if logprob_input is None else round(logprob_input["uncertainty_score"], 6),
            "external_token_logprob_token_count": 0 if logprob_input is None else int(logprob_input["token_count"]),
            "result_origin": "deterministic_fallback_approximation",
            "result_origin_label": "Deterministic fallback approximation",
            "num_samples": len(sampled_answers) if sampled_answers else 1,
            "sample_count": len(sampled_answers) if sampled_answers else 1,
            "confidence_cap": confidence_cap,
            "confidence_capped": sample_metrics is None,
            "uncertainty_floor": round(uncertainty_floor, 6),
            "uncertainty_floor_applied": floor_applied,
            "simple_fact_sanity": simple_fact_sanity,
            "baseline_diagnostics": {
                "structural_inconsistency": round(baseline_diagnostics["structural_inconsistency"], 6),
                "weak_support_pressure": round(baseline_diagnostics["weak_support_pressure"], 6),
                "contextual_uncertainty_risk": round(baseline_diagnostics["contextual_uncertainty_risk"], 6),
                "direct_answer_reassurance": round(baseline_diagnostics["direct_answer_reassurance"], 6),
                "temporal_reassurance": round(baseline_diagnostics["temporal_reassurance"], 6),
                "compound_inconsistency_pressure": round(baseline_diagnostics["compound_inconsistency_pressure"], 6),
                "temporal_mismatch_risk": round(baseline_diagnostics["temporal_bundle"]["temporal_mismatch_risk"], 6),
                "bare_temporal_answer": baseline_diagnostics["temporal_bundle"]["bare_temporal_answer"],
                "date_only_uncertainty_risk": round(baseline_diagnostics["temporal_bundle"]["date_only_uncertainty_risk"], 6),
                "internal_claim_count": baseline_diagnostics["profile"]["internal_claim_count"],
                "fragment_split_used": baseline_diagnostics["profile"]["fragment_split_used"],
            },
            "text_feature_summary": {
                "claim_count": text_features["claim_count"],
                "word_count": text_features["word_count"],
                "detail_count": text_features["detail_count"],
                "new_entity_count": text_features["new_entity_count"],
                "certainty_count": text_features["certainty_count"],
                "attribution_count": text_features["attribution_count"],
                "hedge_count": text_features["hedge_count"],
                "question_is_simple_factoid": text_features["question_is_simple_factoid"],
                "simple_fact_reassurance": round(text_features["simple_fact_reassurance"], 6),
            },
        }
    else:
        sep_proxy_bundle = _sep_score_bundle(
            question=question,
            answer=answer,
            text_features=text_features,
            sample_metrics=sample_metrics,
            simple_fact_sanity=simple_fact_sanity,
        )
        final_score = sep_proxy_bundle["score"]
        text_sub_signals = sep_proxy_bundle["text_sub_signals"]
        sep_bundle = sep_proxy_bundle["sep_bundle"]
        explanation = _sep_explanation_text(
            score=final_score,
            fallback=True,
            profile=sep_bundle["main_profile"],
            sep_bundle=sep_bundle if sample_metrics is not None else None,
            simple_fact_sanity=simple_fact_sanity,
        )
        if logprob_input is not None:
            final_score = _clip_probability((0.72 * final_score) + (0.28 * logprob_input["uncertainty_score"]))
            explanation += " External token logprob statistics from the backend were also available, so SEP fallback blended those model-side uncertainty signals into the final score."

        # Nonzero floor.
        uncertainty_floor = _fallback_floor(text_features=text_features, sample_metrics=sample_metrics, mode=mode)
        floor_applied = uncertainty_floor > 0.0 and final_score < uncertainty_floor
        if floor_applied:
            final_score = uncertainty_floor
            explanation += " A small non-zero uncertainty floor was retained because deterministic fallback cannot verify factual correctness from wording alone."
        if sep_proxy_bundle["risk_floor"] > 0.0:
            final_score = max(final_score, sep_proxy_bundle["risk_floor"])
        if sep_proxy_bundle["risk_cap"] is not None:
            final_score = min(final_score, float(sep_proxy_bundle["risk_cap"]))
        confidence_base = _clip_probability(
            0.30
            + min(text_features["word_count"], 60) / 430.0
            + (0.18 if sample_metrics is not None else 0.0)
        )
        confidence_cap = 0.68 if sample_metrics is not None else 0.54
        confidence = min(confidence_base, confidence_cap)
        fallback_type = "text_proxy_with_sample_consistency" if sample_metrics is not None else "text_only_proxy"
        sampled_answers = [] if sample_metrics is None else sep_bundle["sample_answers"]
        backend_error_text = _offline_error(backend_error)
        if sample_metrics is None:
            summary = (
                f"{method_name} used a deterministic fallback approximation because {config.model_name} could not be loaded; "
                f"the suspiciousness estimate was {final_score:.2f}."
            )
        elif sep_bundle["main_answer_conflicts_with_sample_plurality"]:
            summary = (
                f"{method_name} detected that the main answer conflicted with the sample plurality on a short factoid and scored it at {final_score:.2f}."
            )
        elif sep_bundle["suspicious_consensus_flag"]:
            summary = (
                f"{method_name} detected a stable but suspicious sample consensus and scored the answer at {final_score:.2f} hallucination risk."
            )
        elif sep_bundle["simple_fact_reassurance"] >= 0.45 and sep_bundle["suspicious_specificity_score"] <= 0.25:
            summary = (
                f"{method_name} found strong low-risk sample stability around a simple factual answer and scored it at {final_score:.2f}."
            )
        else:
            summary = (
                f"{method_name} combined text suspiciousness with sampled-answer consistency signals and scored the answer at {final_score:.2f}."
            )
        if floor_applied:
            summary += " A small non-zero uncertainty floor was retained because this fallback cannot verify correctness from wording alone."
        limitations = (
            "This fallback does not use hidden states. It is a deterministic SEP-inspired approximation based on explicit contradiction checks, "
            "temporal, numeric, entity, polarity, and core-claim consistency across samples, a small explicit simple-fact bank for a few canonical checks, "
            "and lower-priority semantic drift signals, so it should be read as suspiciousness estimation rather than general factual verification."
        )
        sub_signals = text_sub_signals + [
            {
                "signal": "Explicit contradiction across samples",
                "value": round(sep_bundle["explicit_contradiction_score"], 4),
                "risk": round(sep_bundle["explicit_contradiction_score"] * 100.0, 1),
                "explanation": "Explicit contradictions across sampled alternatives are weighted more heavily than generic wording drift.",
            },
            {
                "signal": "Temporal consistency across samples",
                "value": round(sep_bundle["temporal_consistency_score"], 4),
                "risk": round(sep_bundle["temporal_inconsistency_score"] * 100.0, 1),
                "explanation": "Year and month-year agreement is treated as mostly consistent, while year drift sharply raises SEP risk.",
            },
            {
                "signal": "Numeric consistency across samples",
                "value": round(sep_bundle["numeric_consistency_score"], 4),
                "risk": round(sep_bundle["numeric_inconsistency_score"] * 100.0, 1),
                "explanation": "Conflicting non-temporal numbers raise SEP risk more than harmless paraphrase variation.",
            },
            {
                "signal": "Entity consistency across samples",
                "value": round(sep_bundle["entity_consistency_score"], 4),
                "risk": round(sep_bundle["entity_inconsistency_score"] * 100.0, 1),
                "explanation": "Central entity drift across samples raises risk, while alias-level agreement stays reassuring.",
            },
            {
                "signal": "Negation / polarity consistency",
                "value": round(sep_bundle["polarity_consistency_score"], 4),
                "risk": round((1.0 - sep_bundle["polarity_consistency_score"]) * 100.0, 1),
                "explanation": "Negation mismatches only raise risk strongly when the samples are otherwise answering the same claim.",
            },
            {
                "signal": "Core claim consistency",
                "value": round(sep_bundle["core_claim_consistency_score"], 4),
                "risk": round((1.0 - sep_bundle["core_claim_consistency_score"]) * 100.0, 1),
                "explanation": "Stable short factual targets and aligned core claims reduce SEP risk for paraphrases.",
            },
            {
                "signal": "Semantic consensus across samples",
                "value": round(sep_bundle["semantic_consensus_score"], 4),
                "risk": round((1.0 - sep_bundle["semantic_consensus_score"]) * 100.0, 1),
                "explanation": "Low semantic consensus matters, but less than explicit contradiction or factual drift.",
            },
            {
                "signal": "Structural paraphrase stability",
                "value": round(sep_bundle["structural_paraphrase_stability"], 4),
                "risk": round((1.0 - sep_bundle["structural_paraphrase_stability"]) * 100.0, 1),
                "explanation": "Harmless paraphrase variation is lower-priority once factual slots agree.",
            },
            {
                "signal": "Answer target consistency",
                "value": round(sep_bundle["answer_target_consistency_score"], 4),
                "risk": round((1.0 - sep_bundle["answer_target_consistency_score"]) * 100.0, 1),
                "explanation": "This checks whether the sampled answers keep addressing the same target implied by the question.",
            },
            {
                "signal": "Specificity amplification across samples",
                "value": round(sep_bundle["specificity_amplification_score"], 4),
                "risk": round(sep_bundle["specificity_amplification_score"] * 100.0, 1),
                "explanation": "When multiple samples repeat specific institutions, locations, or exact outcomes, SEP fallback keeps the risk elevated.",
            },
            {
                "signal": "Sample instability penalty",
                "value": round(sep_bundle["instability_penalty"], 4),
                "risk": round(sep_bundle["instability_penalty"] * 100.0, 1),
                "explanation": "SEP aggregates contradiction first, then factual-slot inconsistency, then generic semantic drift.",
            },
            {
                "signal": "Sample plurality disagreement",
                "value": round(sep_bundle["plurality_disagreement_score"], 4),
                "risk": round(sep_bundle["plurality_disagreement_score"] * 100.0, 1),
                "explanation": "When the main answer disagrees with the plurality of sampled short factoid answers, SEP fallback treats that as a strong instability cue.",
            },
            {
                "signal": "Suspicious consensus across samples",
                "value": round(sep_bundle["suspicious_consensus_score"], 4),
                "risk": round(sep_bundle["suspicious_consensus_score"] * 100.0, 1),
                "explanation": "If the sample set consistently reinforces the same obscure specific claim, SEP fallback treats that as stable but suspicious rather than automatically safe.",
            },
        ]
        if floor_applied:
            sub_signals = sub_signals + [
                {
                    "signal": "Fallback uncertainty floor",
                    "value": round(uncertainty_floor, 4),
                    "risk": round(uncertainty_floor * 100.0, 1),
                    "explanation": "Because this deterministic no-evidence fallback cannot verify factual correctness from wording alone, it retains a small non-zero uncertainty floor instead of showing zero risk.",
                }
            ]
        if simple_fact_sanity is not None:
            sub_signals.append(
                {
                    "signal": "Curated simple-fact sanity",
                    "value": simple_fact_sanity["candidate"],
                    "risk": 92.0 if simple_fact_sanity["verdict"] == "incorrect" else 8.0,
                    "explanation": f"A small explicit local fact bank matched this question and expected '{simple_fact_sanity['canonical_answer']}'.",
                }
            )
        intermediate_steps = [
            {
                "stage": "fallback_activation",
                "output": {
                    "mode": mode,
                    "fallback_type": fallback_type,
            "used_external_token_logprobs": logprob_input is not None,
            "external_token_logprob_score": None if logprob_input is None else round(logprob_input["uncertainty_score"], 6),
            "external_token_logprob_token_count": 0 if logprob_input is None else int(logprob_input["token_count"]),
                    "backend_available": False,
                    "backend_error": backend_error_text,
                    "model_name": config.model_name,
                },
            },
            {
                "stage": "sep_text_proxy_features",
                "output": {
                    "claim_count": text_features["claim_count"],
                    "word_count": text_features["word_count"],
                    "detail_count": text_features["detail_count"],
                    "new_entity_spans": text_features["new_entity_spans"],
                    "suspicious_specificity_score": round(sep_bundle["suspicious_specificity_score"], 4),
                    "obscure_event_risk": round(sep_bundle["main_profile"]["obscure_event_risk"], 4),
                    "simple_fact_sanity": simple_fact_sanity,
                },
            },
        ]
        if sample_metrics is not None:
            intermediate_steps.append(
                {
                    "stage": "sep_sample_consistency_proxy",
                    "output": {
                        "sample_count": sep_bundle["sample_count"],
                        "sample_consistency_score": round(sep_bundle["sample_consistency_score"], 4),
                        "semantic_agreement_score": round(sep_bundle["semantic_agreement_score"], 4),
                        "semantic_consensus_score": round(sep_bundle["semantic_consensus_score"], 4),
                        "explicit_contradiction_score": round(sep_bundle["explicit_contradiction_score"], 4),
                        "temporal_consistency_score": round(sep_bundle["temporal_consistency_score"], 4),
                        "numeric_consistency_score": round(sep_bundle["numeric_consistency_score"], 4),
                        "entity_consistency_score": round(sep_bundle["entity_consistency_score"], 4),
                        "polarity_consistency_score": round(sep_bundle["polarity_consistency_score"], 4),
                        "core_claim_consistency_score": round(sep_bundle["core_claim_consistency_score"], 4),
                        "factual_slot_inconsistency_score": round(sep_bundle["factual_slot_inconsistency_score"], 4),
                        "structural_paraphrase_stability": round(sep_bundle["structural_paraphrase_stability"], 4),
                        "answer_target_consistency_score": round(sep_bundle["answer_target_consistency_score"], 4),
                        "specificity_amplification_score": round(sep_bundle["specificity_amplification_score"], 4),
                        "suspicious_consensus_flag": sep_bundle["suspicious_consensus_flag"],
                        "paraphrase_safety_cap_applied": sep_bundle["paraphrase_safety_cap_applied"],
                        "dominant_sample_risk_source": sep_bundle["dominant_sample_risk_source"],
                        "instability_penalty": round(sep_bundle["instability_penalty"], 4),
                        "simple_fact_plurality": sep_bundle["simple_fact_plurality"],
                        "pairwise_slot_comparisons": sep_bundle["pairwise_slot_comparisons"],
                        "plurality_disagreement_score": round(sep_bundle["plurality_disagreement_score"], 4),
                    },
                }
            )
        metadata = {
            "mode": mode,
            "hf_model_name": config.model_name,
            "backend_model_name": config.model_name,
            "backend_available": False,
            "backend_status": "unavailable",
            "backend_status_label": "HF backend unavailable",
            "backend_error": backend_error_text,
            "fallback_mode": True,
            "fallback_type": fallback_type,
            "used_external_token_logprobs": logprob_input is not None,
            "external_token_logprob_score": None if logprob_input is None else round(logprob_input["uncertainty_score"], 6),
            "external_token_logprob_token_count": 0 if logprob_input is None else int(logprob_input["token_count"]),
            "result_origin": "deterministic_fallback_approximation",
            "result_origin_label": "Deterministic fallback approximation",
            "sample_count": sep_bundle["sample_count"],
            "num_samples": sep_bundle["sample_count"],
            "confidence_cap": confidence_cap,
            "confidence_capped": sample_metrics is None,
            "uncertainty_floor": round(uncertainty_floor, 6),
            "uncertainty_floor_applied": floor_applied,
            "sample_consistency_score": round(sep_bundle["sample_consistency_score"], 6),
            "explicit_contradiction_score": round(sep_bundle["explicit_contradiction_score"], 6),
            "temporal_consistency_score": round(sep_bundle["temporal_consistency_score"], 6),
            "temporal_inconsistency_score": round(sep_bundle["temporal_inconsistency_score"], 6),
            "entity_consistency_score": round(sep_bundle["entity_consistency_score"], 6),
            "entity_inconsistency_score": round(sep_bundle["entity_inconsistency_score"], 6),
            "numeric_consistency_score": round(sep_bundle["numeric_consistency_score"], 6),
            "numeric_inconsistency_score": round(sep_bundle["numeric_inconsistency_score"], 6),
            "polarity_consistency_score": round(sep_bundle["polarity_consistency_score"], 6),
            "core_claim_consistency_score": round(sep_bundle["core_claim_consistency_score"], 6),
            "factual_slot_inconsistency_score": round(sep_bundle["factual_slot_inconsistency_score"], 6),
            "structural_paraphrase_stability": round(sep_bundle["structural_paraphrase_stability"], 6),
            "semantic_agreement_score": round(sep_bundle["semantic_agreement_score"], 6),
            "semantic_consensus_score": round(sep_bundle["semantic_consensus_score"], 6),
            "answer_target_consistency_score": round(sep_bundle["answer_target_consistency_score"], 6),
            "suspicious_specificity_score": round(sep_bundle["suspicious_specificity_score"], 6),
            "specificity_amplification_score": round(sep_bundle["specificity_amplification_score"], 6),
            "suspicious_consensus_flag": sep_bundle["suspicious_consensus_flag"],
            "paraphrase_safety_cap_applied": sep_bundle["paraphrase_safety_cap_applied"],
            "dominant_sample_risk_source": sep_bundle["dominant_sample_risk_source"],
            "sample_instability_penalty": round(sep_bundle["instability_penalty"], 6),
            "plurality_disagreement_score": round(sep_bundle["plurality_disagreement_score"], 6),
            "main_answer_conflicts_with_sample_plurality": sep_bundle["main_answer_conflicts_with_sample_plurality"],
            "simple_fact_plurality": sep_bundle["simple_fact_plurality"],
            "simple_fact_sanity": simple_fact_sanity,
            "text_feature_summary": {
                "claim_count": text_features["claim_count"],
                "word_count": text_features["word_count"],
                "detail_count": text_features["detail_count"],
                "new_entity_count": text_features["new_entity_count"],
                "certainty_count": text_features["certainty_count"],
                "attribution_count": text_features["attribution_count"],
                "hedge_count": text_features["hedge_count"],
                "question_is_simple_factoid": text_features["question_is_simple_factoid"],
                "simple_fact_reassurance": round(text_features["simple_fact_reassurance"], 6),
            },
        }
    claim_findings = _text_claims(
        question=question,
        claims=extract_claims(answer),
        mode=mode,
        backend_error=backend_error,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return make_result(
        method_name=method_name,
        family=family,
        score=final_score,
        confidence=confidence,
        summary=summary,
        explanation=explanation,
        evidence_used="Text-only answer features and optional sampled-answer consistency; no external evidence retrieval.",
        evidence=[],
        citations=[],
        intermediate_steps=intermediate_steps,
        claim_findings=claim_findings,
        limitations=limitations,
        impl_status="approximate",
        latency_ms=latency_ms,
        metadata=metadata,
        mode_used=mode,
        sub_signals=sub_signals if mode != "uncertainty_baseline" else sub_signals,
        sampled_answers=sampled_answers,
    )


def _sample_fallback(
    *,
    question: str,
    answer: str,
    sampled_answers_text: str,
    method_name: str,
    mode: str,
    family: str,
    config: SignalConfig,
    backend_error: Exception,
    start_time: float,
    token_logprobs: list[list[float]] | None = None,
) -> dict[str, Any] | None:
    if not safe_text(answer):
        return None
    return _text_fallback(
        question=question,
        answer=answer,
        sampled_answers_text=sampled_answers_text,
        method_name=method_name,
        mode=mode,
        family=family,
        config=config,
        backend_error=backend_error,
        start_time=start_time,
        token_logprobs=token_logprobs,
    )

def _layer_indices(layer_spec: str, hidden_count: int) -> list[int]:
    max_index = hidden_count - 1
    resolved: list[int] = []
    for raw_part in layer_spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        index = int(part)
        if index < 0:
            index = hidden_count + index
        index = max(1, min(max_index, index))
        if index not in resolved:
            resolved.append(index)
    return resolved or [max_index]


def _cosine_drift(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    left = vectors[:-1]
    right = vectors[1:]
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    normalized_left = np.divide(left, np.clip(left_norm, 1e-12, None))
    normalized_right = np.divide(right, np.clip(right_norm, 1e-12, None))
    cosine = np.sum(normalized_left * normalized_right, axis=1)
    return float(np.mean(1.0 - cosine))


def _centroid_spread(centroids: list[np.ndarray]) -> float:
    if len(centroids) < 2:
        return 0.0
    distances: list[float] = []
    for left_index in range(len(centroids)):
        for right_index in range(left_index + 1, len(centroids)):
            left = centroids[left_index]
            right = centroids[right_index]
            left = left / max(np.linalg.norm(left), 1e-12)
            right = right / max(np.linalg.norm(right), 1e-12)
            distances.append(float(1.0 - np.dot(left, right)))
    return float(np.mean(distances)) if distances else 0.0


def _hidden_features(hidden_states: tuple[Any, ...], answer_start: int, layer_spec: str) -> dict[str, float | list[int]]:
    selected_layers = _layer_indices(layer_spec, len(hidden_states))
    mean_norms: list[float] = []
    variances: list[float] = []
    drifts: list[float] = []
    centroids: list[np.ndarray] = []

    for layer_index in selected_layers:
        layer_tensor = hidden_states[layer_index][0, answer_start:, :].detach().cpu().numpy()
        if layer_tensor.size == 0:
            continue
        token_norms = np.linalg.norm(layer_tensor, axis=1)
        mean_norms.append(float(np.mean(token_norms)))
        variances.append(float(np.var(token_norms)))
        drifts.append(_cosine_drift(layer_tensor))
        centroids.append(np.mean(layer_tensor, axis=0))

    return {
        "selected_layers": selected_layers,
        "hidden_norm_mean": float(np.mean(mean_norms)) if mean_norms else 0.0,
        "hidden_norm_var": float(np.mean(variances)) if variances else 0.0,
        "hidden_drift_mean": float(np.mean(drifts)) if drifts else 0.0,
        "layer_centroid_dispersion": _centroid_spread(centroids),
    }


def _prepare_model_inputs(tokenizer: Any, question: str, answer: str, config: SignalConfig) -> tuple[list[int], list[int]]:
    prompt_text = f"Question: {safe_text(question)}\nAnswer:" if safe_text(question) else "Answer:"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + safe_text(answer), add_special_tokens=False)
    if not answer_ids:
        raise ValueError("The answer could not be tokenized into any model tokens.")

    answer_ids = answer_ids[: config.max_answer_tokens]
    prompt_budget = max(0, config.max_total_tokens - len(answer_ids))
    prompt_ids = prompt_ids[-prompt_budget:]
    return prompt_ids, answer_ids


def _extract_features(
    *,
    question: str,
    answer: str,
    config: SignalConfig,
) -> dict[str, Any]:
    import torch

    backend = _init_hf(config.model_name, config.local_files_only)
    if not backend.backend_available or backend.tokenizer is None or backend.model is None:
        raise RuntimeError(backend.backend_error or f"HF backend is unavailable for {config.model_name}.")
    tokenizer = backend.tokenizer
    model = backend.model
    prompt_ids, answer_ids = _prepare_model_inputs(tokenizer, question, answer, config)
    input_ids = prompt_ids + answer_ids
    answer_start = len(prompt_ids)

    tensor = torch.tensor([input_ids], dtype=torch.long, device=backend.device)
    with torch.no_grad():
        outputs = model(input_ids=tensor, output_hidden_states=True)

    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = tensor[:, 1:]
    answer_slice_start = max(answer_start - 1, 0)
    answer_slice_end = answer_slice_start + len(answer_ids)
    answer_logits = shift_logits[:, answer_slice_start:answer_slice_end, :]
    answer_labels = shift_labels[:, answer_slice_start:answer_slice_end]

    log_probs = torch.log_softmax(answer_logits, dim=-1)
    probabilities = torch.exp(log_probs)
    token_log_probs = log_probs.gather(-1, answer_labels.unsqueeze(-1)).squeeze(-1)
    token_probabilities = torch.exp(token_log_probs)
    entropy = -(probabilities * log_probs).sum(dim=-1)
    top_two = torch.topk(probabilities, k=2, dim=-1).values
    top2_margin = top_two[..., 0] - top_two[..., 1]

    hidden_features = _hidden_features(outputs.hidden_states, answer_start, config.layer_spec)
    token_strings = tokenizer.convert_ids_to_tokens(answer_ids)

    features = {
        "mean_negative_log_prob": float((-token_log_probs).mean().item()),
        "token_log_prob_std": float(token_log_probs.std(unbiased=False).item()),
        "entropy_mean": float(entropy.mean().item()),
        "entropy_std": float(entropy.std(unbiased=False).item()),
        "mean_token_probability": float(token_probabilities.mean().item()),
        "top2_margin_mean": float(top2_margin.mean().item()),
        "num_answer_tokens": float(len(answer_ids)),
        **hidden_features,
    }
    return {
        "features": features,
        "tokens": token_strings,
        "answer_token_count": len(answer_ids),
    }


def _baseline_score(features: dict[str, float]) -> tuple[float, list[dict[str, Any]]]:
    mean_nll = _clip_probability(features["mean_negative_log_prob"] / 5.5)
    entropy_mean = _clip_probability(features["entropy_mean"] / 8.0)
    entropy_std = _clip_probability(features["entropy_std"] / 4.0)
    low_token_probability = _clip_probability(1.0 - features["mean_token_probability"])
    weak_margin = _clip_probability(1.0 - min(features["top2_margin_mean"] / 0.45, 1.0))

    score = (
        (0.34 * mean_nll)
        + (0.24 * entropy_mean)
        + (0.12 * entropy_std)
        + (0.18 * low_token_probability)
        + (0.12 * weak_margin)
    )
    sub_signals = [
        {
            "signal": "Mean negative log probability",
            "value": round(features["mean_negative_log_prob"], 4),
            "risk": round(mean_nll * 100.0, 1),
            "explanation": "Higher token surprisal indicates lower model confidence in the provided answer tokens.",
        },
        {
            "signal": "Token entropy mean",
            "value": round(features["entropy_mean"], 4),
            "risk": round(entropy_mean * 100.0, 1),
            "explanation": "Higher predictive entropy means the model spread probability mass across many alternatives.",
        },
        {
            "signal": "Token entropy variability",
            "value": round(features["entropy_std"], 4),
            "risk": round(entropy_std * 100.0, 1),
            "explanation": "Large swings in token entropy can indicate unstable confidence within the answer.",
        },
        {
            "signal": "Mean token probability",
            "value": round(features["mean_token_probability"], 4),
            "risk": round(low_token_probability * 100.0, 1),
            "explanation": "Lower token probability raises hallucination risk under the baseline uncertainty view.",
        },
        {
            "signal": "Top-2 margin mean",
            "value": round(features["top2_margin_mean"], 4),
            "risk": round(weak_margin * 100.0, 1),
            "explanation": "Small gaps between the top two next-token probabilities indicate local uncertainty.",
        },
    ]
    return _clip_probability(score), sub_signals


def _probe_vector(features: dict[str, float], num_samples: int, feature_var: float) -> np.ndarray:
    values = {
        **features,
        "feature_sample_variance": feature_var,
        "num_samples": float(num_samples),
    }
    return np.asarray([float(values.get(name, 0.0)) for name in PROBE_FEATURE_ORDER], dtype=float)


def _sample_variance(feature_dicts: list[dict[str, float]]) -> float:
    if len(feature_dicts) < 2:
        return 0.0
    vectors = np.asarray([[float(item.get(name, 0.0)) for name in PROBE_FEATURE_ORDER[:-2]] for item in feature_dicts], dtype=float)
    return float(np.mean(np.var(vectors, axis=0)))


def _load_probe_bundle(path: str) -> dict[str, Any] | None:
    if not safe_text(path):
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    with open(resolved, "rb") as handle:
        return pickle.load(handle)


def _sep_score(
    *,
    feature_dicts: list[dict[str, float]],
    probe_bundle: dict[str, Any] | None,
) -> tuple[float, list[dict[str, Any]], bool]:
    base_score, base_sub_signals = _baseline_score(feature_dicts[0])
    feature_var = _sample_variance(feature_dicts)
    hidden_norm_var = _clip_probability(feature_dicts[0]["hidden_norm_var"] / 40.0)
    hidden_drift = _clip_probability(feature_dicts[0]["hidden_drift_mean"] / 0.35)
    layer_dispersion = _clip_probability(feature_dicts[0]["layer_centroid_dispersion"] / 0.35)
    sample_variance = _clip_probability(feature_var / 4.0)

    hidden_score = (0.4 * hidden_norm_var) + (0.35 * hidden_drift) + (0.25 * layer_dispersion)
    heuristic_score = _clip_probability((0.55 * base_score) + (0.35 * hidden_score) + (0.10 * sample_variance))

    probe_loaded = False
    final_score = heuristic_score
    if probe_bundle is not None:
        vector = _probe_vector(feature_dicts[0], len(feature_dicts), feature_var)
        model = probe_bundle["model"]
        final_score = float(model.predict_proba([vector])[0][1])
        probe_loaded = True

    sep_sub_signals = base_sub_signals + [
        {
            "signal": "Hidden-state norm variance",
            "value": round(feature_dicts[0]["hidden_norm_var"], 4),
            "risk": round(hidden_norm_var * 100.0, 1),
            "explanation": "Higher norm variance across answer tokens suggests less stable internal representations.",
        },
        {
            "signal": "Hidden-state token drift",
            "value": round(feature_dicts[0]["hidden_drift_mean"], 4),
            "risk": round(hidden_drift * 100.0, 1),
            "explanation": "Larger adjacent-token hidden-state drift suggests more unstable token-to-token trajectories.",
        },
        {
            "signal": "Cross-layer centroid dispersion",
            "value": round(feature_dicts[0]["layer_centroid_dispersion"], 4),
            "risk": round(layer_dispersion * 100.0, 1),
            "explanation": "Greater disagreement across selected hidden layers increases SEP-inspired risk.",
        },
        {
            "signal": "Cross-sample feature variance",
            "value": round(feature_var, 4),
            "risk": round(sample_variance * 100.0, 1),
            "explanation": "When multiple sampled answers are provided, feature variance across them acts as an instability cue.",
        },
    ]
    return _clip_probability(final_score), sep_sub_signals, probe_loaded


def append_probe_example(
    *,
    output_path: str,
    question: str,
    answer: str,
    label: int,
    sampled_answers_text: str = "",
    config: SignalConfig | None = None,
    token_logprobs: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Extract SEP-lite features and append them to a JSONL training file."""
    resolved_config = config or SignalConfig()
    examples = [safe_text(answer)] + _split_sample_blocks(sampled_answers_text)
    feature_records = [
        _extract_features(question=question, answer=item, config=resolved_config)
        for item in examples
        if safe_text(item)
    ]
    if not feature_records:
        raise ValueError("At least one non-empty answer is required to extract probe features.")

    feature_dicts = [record["features"] for record in feature_records]
    sample_variance = _sample_variance(feature_dicts)
    vector = _probe_vector(feature_dicts[0], len(feature_dicts), sample_variance)
    row = {
        "question": question,
        "answer": answer,
        "label": int(label),
        "feature_names": PROBE_FEATURE_ORDER,
        "features": vector.tolist(),
    }
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return row


def train_probe_jsonl(*, feature_path: str, output_path: str) -> dict[str, Any]:
    """Train and save a simple logistic probe for SEP-lite inference."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    with open(feature_path, "r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = safe_text(line)
            if not cleaned:
                continue
            rows.append(json.loads(cleaned))
    if len(rows) < 2:
        raise ValueError("At least two labeled rows are required to train the probe.")

    feature_names = rows[0].get("feature_names") or PROBE_FEATURE_ORDER
    features = np.asarray([row["features"] for row in rows], dtype=float)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(features, labels)
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "trained_rows": len(rows),
    }
    with open(output_path, "wb") as handle:
        pickle.dump(bundle, handle)
    return {"output_path": output_path, "trained_rows": len(rows), "feature_names": feature_names}


def _claim_rows(
    *,
    question: str,
    claims: list[str],
    config: SignalConfig,
    mode: str,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for claim in claims[:8]:
        features = _extract_features(question=question, answer=claim, config=config)["features"]
        if mode == "uncertainty_baseline":
            backend_claim_score, _ = _baseline_score(features)
            claim_text_features = _text_features(question, claim)
            text_claim_score, _, claim_diagnostics = _text_base_score(claim_text_features, return_diagnostics=True)
            claim_score = _clip_probability((0.80 * backend_claim_score) + (0.20 * text_claim_score))
            if claim_diagnostics["profile"]["contradiction_risk"] >= 0.40:
                claim_score = max(claim_score, 0.70)
        else:
            claim_score, _, _ = _sep_score(feature_dicts=[features], probe_bundle=None)
        if claim_score >= 0.67:
            status = "high_risk"
        elif claim_score >= 0.34:
            status = "mixed"
        else:
            status = "low_risk"
        findings.append(
            {
                "claim": claim,
                "status": status,
                "score": round(claim_score * 100.0, 1),
                "best_match": "Model-internal analysis only; no external evidence is used.",
                "reason": (
                    f"Teacher-forced {mode} score from the local Hugging Face model was {claim_score:.2f}."
                ),
            }
        )
    return findings


def run_signal_detector(
    *,
    question: str,
    answer: str,
    method_name: str,
    mode: str,
    family: str = "internal-signal",
    sampled_answers_text: str = "",
    config: SignalConfig | None = None,
    token_logprobs: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Run the local Hugging Face internal-signal detector."""
    start_time = time.perf_counter()
    resolved_config = config or SignalConfig()
    backend_status = get_signal_status(resolved_config)
    cleaned_answer = safe_text(answer)
    if not cleaned_answer:
        return make_unavailable(
            method_name=method_name,
            family=family,
            summary="The answer is empty, so no model-internal uncertainty features could be extracted.",
            explanation="The answer is empty, so no model-internal uncertainty features could be extracted.",
            evidence_used="No answer text was provided.",
            limitations="Internal-signal methods need answer text so a local language model can score it.",
            impl_status="unavailable",
            metadata={"mode": mode, "model_name": resolved_config.model_name},
        )

    # Text fallback.
    if not backend_status["backend_available"]:
        backend_error = RuntimeError(
            _offline_error(
                backend_status.get("backend_error") or f"HF backend is unavailable for {resolved_config.model_name}."
            )
        )
        fallback_result = _sample_fallback(
            question=question,
            answer=cleaned_answer,
            sampled_answers_text=sampled_answers_text,
            method_name=method_name,
            mode=mode,
            family=family,
            config=resolved_config,
            backend_error=backend_error,
            start_time=start_time,
            token_logprobs=token_logprobs,
        )
        if fallback_result is not None:
            return fallback_result
        return make_unavailable(
            method_name=method_name,
            family=family,
            summary="The local Hugging Face backend could not be loaded for internal-signal scoring.",
            explanation=str(backend_error),
            evidence_used="No external evidence is used by this family.",
            limitations=(
                "This detector needs a locally available causal language model from Hugging Face. "
                "Run the app with the same Python interpreter that has torch, transformers, and the model cache available."
            ),
            impl_status="unavailable",
            metadata={
                "mode": mode,
                "model_name": resolved_config.model_name,
                "local_files_only": resolved_config.local_files_only,
                **backend_status,
                "backend_error": _offline_error(backend_status.get("backend_error")),
            },
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )

    try:
        primary_record = _extract_features(
            question=question,
            answer=cleaned_answer,
            config=resolved_config,
        )
        sample_answers = [cleaned_answer] + _split_sample_blocks(sampled_answers_text)
        sample_records = [primary_record]
        for sample in sample_answers[1:4]:
            sample_records.append(
                _extract_features(
                    question=question,
                    answer=sample,
                    config=resolved_config,
                )
            )
    except Exception as exc:
        fallback_result = _sample_fallback(
            question=question,
            answer=cleaned_answer,
            sampled_answers_text=sampled_answers_text,
            method_name=method_name,
            mode=mode,
            family=family,
            config=resolved_config,
            backend_error=exc,
            start_time=start_time,
            token_logprobs=token_logprobs,
        )
        if fallback_result is not None:
            return fallback_result
        return make_unavailable(
            method_name=method_name,
            family=family,
            summary="The local Hugging Face backend could not be loaded for internal-signal scoring.",
            explanation=f"Model loading or feature extraction failed: {exc}",
            evidence_used="No external evidence is used by this family.",
            limitations=(
                "This detector needs a locally available causal language model from Hugging Face. "
                "By default it looks for cached weights so the prototype still works offline."
            ),
            impl_status="unavailable",
            metadata={
                "mode": mode,
                "model_name": resolved_config.model_name,
                "local_files_only": resolved_config.local_files_only,
                **backend_status,
            },
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )

    primary_features = primary_record["features"]
    feature_dicts = [record["features"] for record in sample_records]

    if mode == "uncertainty_baseline":
        backend_score, backend_sub_signals = _baseline_score(primary_features)
        text_features = _text_features(question, cleaned_answer)
        baseline_text_score, text_signals, text_diags = _text_base_score(
            text_features,
            return_diagnostics=True,
        )
        # Model dominates.
        backend_weight = 0.78
        score = _clip_probability((backend_weight * backend_score) + ((1.0 - backend_weight) * baseline_text_score))
        if text_diags["profile"]["contradiction_risk"] >= 0.40 and text_features["claim_count"] >= 2:
            score = max(score, 0.72)
        if text_diags["temporal_bundle"]["explicit_year_mismatch"] or text_diags["profile"]["date_inconsistency_risk"] >= 0.44:
            score = max(score, 0.68)
        simple_fact_sanity = text_features.get("simple_fact_sanity")
        if simple_fact_sanity is not None:
            if simple_fact_sanity["verdict"] == "incorrect":
                score = max(score, 0.82)
            else:
                score = min(score, 0.18)
        sub_signals = backend_sub_signals + [
            {
                "signal": "Text-structure baseline cross-check",
                "value": round(baseline_text_score, 4),
                "risk": round(baseline_text_score * 100.0, 1),
                "explanation": "Teacher-forced uncertainty was calibrated against contradiction, numeric/date/entity inconsistency, and temporal-alignment text signals.",
            }
        ] + text_signals
        impl_status = "implemented"
        probe_loaded = False
        result_origin = "full_backend_scoring"
        result_origin_label = "Full backend scoring"
        explanation = (
            "Implemented uncertainty baseline using teacher-forced token probabilities from a local Hugging Face causal language model, "
            "then calibrated that uncertainty with deterministic text-structure checks for contradiction, numeric/date/entity inconsistency, and temporal alignment."
        )
    else:
        probe_bundle = _load_probe_bundle(resolved_config.probe_path)
        score, sub_signals, probe_loaded = _sep_score(
            feature_dicts=feature_dicts,
            probe_bundle=probe_bundle,
        )
        sep_text_features = _text_features(question, cleaned_answer)
        sep_sample_metrics = _sample_metrics(question, cleaned_answer, sampled_answers_text)
        sep_proxy_bundle = _sep_score_bundle(
            question=question,
            answer=cleaned_answer,
            text_features=sep_text_features,
            sample_metrics=sep_sample_metrics,
            simple_fact_sanity=sep_text_features.get("simple_fact_sanity"),
        )
        # Backend dominates.
        backend_weight = 0.88 if probe_loaded else 0.82
        score = _clip_probability((backend_weight * score) + ((1.0 - backend_weight) * sep_proxy_bundle["score"]))
        if sep_proxy_bundle["risk_floor"] > 0.0:
            score = max(score, sep_proxy_bundle["risk_floor"])
        if sep_proxy_bundle["risk_cap"] is not None:
            score = min(score, float(sep_proxy_bundle["risk_cap"]))
        sub_signals = sub_signals + sep_proxy_bundle["text_sub_signals"]
        if sep_sample_metrics is not None:
            sub_signals = sub_signals + [
                {
                    "signal": "Explicit contradiction across samples",
                    "value": round(sep_proxy_bundle["sep_bundle"]["explicit_contradiction_score"], 4),
                    "risk": round(sep_proxy_bundle["sep_bundle"]["explicit_contradiction_score"] * 100.0, 1),
                    "explanation": "Explicit contradictions across sampled alternatives are weighted more heavily than generic wording drift.",
                },
                {
                    "signal": "Temporal consistency across samples",
                    "value": round(sep_proxy_bundle["sep_bundle"]["temporal_consistency_score"], 4),
                    "risk": round(sep_proxy_bundle["sep_bundle"]["temporal_inconsistency_score"] * 100.0, 1),
                    "explanation": "Year and month-year agreement is treated as mostly consistent, while year drift sharply raises SEP risk.",
                },
                {
                    "signal": "Numeric consistency across samples",
                    "value": round(sep_proxy_bundle["sep_bundle"]["numeric_consistency_score"], 4),
                    "risk": round(sep_proxy_bundle["sep_bundle"]["numeric_inconsistency_score"] * 100.0, 1),
                    "explanation": "Conflicting non-temporal numbers raise SEP risk more than harmless paraphrase variation.",
                },
                {
                    "signal": "Entity consistency across samples",
                    "value": round(sep_proxy_bundle["sep_bundle"]["entity_consistency_score"], 4),
                    "risk": round(sep_proxy_bundle["sep_bundle"]["entity_inconsistency_score"] * 100.0, 1),
                    "explanation": "Central entity drift across samples raises risk, while alias-level agreement stays reassuring.",
                },
                {
                    "signal": "Core claim consistency",
                    "value": round(sep_proxy_bundle["sep_bundle"]["core_claim_consistency_score"], 4),
                    "risk": round((1.0 - sep_proxy_bundle["sep_bundle"]["core_claim_consistency_score"]) * 100.0, 1),
                    "explanation": "Stable short factual targets and aligned core claims reduce SEP risk for paraphrases.",
                },
                {
                    "signal": "Semantic consensus across samples",
                    "value": round(sep_proxy_bundle["sep_bundle"]["semantic_consensus_score"], 4),
                    "risk": round((1.0 - sep_proxy_bundle["sep_bundle"]["semantic_consensus_score"]) * 100.0, 1),
                    "explanation": "Low semantic consensus matters, but less than explicit contradiction or factual drift.",
                },
                {
                    "signal": "Sample instability penalty",
                    "value": round(sep_proxy_bundle["sep_bundle"]["instability_penalty"], 4),
                    "risk": round(sep_proxy_bundle["sep_bundle"]["instability_penalty"] * 100.0, 1),
                    "explanation": "SEP aggregates contradiction first, then factual-slot inconsistency, then generic semantic drift.",
                },
                {
                    "signal": "Sample plurality disagreement",
                    "value": round(sep_proxy_bundle["sep_bundle"]["plurality_disagreement_score"], 4),
                    "risk": round(sep_proxy_bundle["sep_bundle"]["plurality_disagreement_score"] * 100.0, 1),
                    "explanation": "When the main answer conflicts with the plurality of short sampled answers, SEP risk stays elevated.",
                },
                {
                    "signal": "Suspicious consensus across samples",
                    "value": round(sep_proxy_bundle["sep_bundle"]["suspicious_consensus_score"], 4),
                    "risk": round(sep_proxy_bundle["sep_bundle"]["suspicious_consensus_score"] * 100.0, 1),
                    "explanation": "Stable agreement on the same unsupported specific claim should not be treated as automatically safe.",
                },
            ]
        impl_status = "approximate"
        result_origin = "sep_lite_probe_path" if probe_loaded else "full_backend_scoring"
        result_origin_label = "SEP-lite probe path" if probe_loaded else "Full backend scoring"
        explanation = _sep_explanation_text(
            score=score,
            fallback=False,
            profile=sep_proxy_bundle["sep_bundle"]["main_profile"],
            sep_bundle=sep_proxy_bundle["sep_bundle"] if sep_sample_metrics is not None else None,
            simple_fact_sanity=sep_text_features.get("simple_fact_sanity"),
            backend_features=primary_features,
            probe_loaded=probe_loaded,
        )

    confidence = _clip_probability(0.45 + min(primary_features["num_answer_tokens"], 80.0) / 200.0)
    claims = extract_claims(cleaned_answer)
    claim_findings = _claim_rows(
        question=question,
        claims=claims,
        config=resolved_config,
        mode=mode,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    if result_origin == "sep_lite_probe_path":
        summary = (
            f"{method_name} used the SEP-lite probe path with {resolved_config.model_name} and scored the answer at {score:.2f} "
            f"hallucination risk over {int(primary_features['num_answer_tokens'])} answer tokens."
        )
    else:
        summary = (
            f"{method_name} used full backend scoring with {resolved_config.model_name} and estimated {score:.2f} hallucination risk "
            f"over {int(primary_features['num_answer_tokens'])} answer tokens."
        )
    limitations = (
        "The baseline uses teacher-forced uncertainty from one local causal LM and the SEP-inspired path uses hidden-state heuristics plus an optional simple probe. "
        "Neither mode is a full reproduction of large-scale research setups, and SEP-inspired mode is explicitly SEP-lite."
    )
    intermediate_steps = [
        {
            "stage": "feature_extraction",
            "mode": mode,
            "model_name": resolved_config.model_name,
            "selected_layers": primary_features["selected_layers"],
            "num_tokens": int(primary_features["num_answer_tokens"]),
            "features": {key: round(float(value), 6) if isinstance(value, (float, int)) else value for key, value in primary_features.items()},
            "sample_count": len(sample_records),
        },
        {
            "stage": "sub_signal_scoring",
            "sub_signals": sub_signals,
            "probe_loaded": probe_loaded,
        },
    ]

    return make_result(
        method_name=method_name,
        family=family,
        score=score,
        confidence=confidence,
        summary=summary,
        explanation=explanation,
        evidence_used="Teacher-forced local model uncertainty and hidden states; no external evidence retrieval.",
        evidence=[],
        citations=[],
        intermediate_steps=intermediate_steps,
        claim_findings=claim_findings,
        limitations=limitations,
        impl_status=impl_status,
        latency_ms=latency_ms,
        metadata={
            "mode": mode,
            "hf_model_name": resolved_config.model_name,
            "backend_model_name": resolved_config.model_name,
            "backend_available": backend_status["backend_available"],
            "backend_status": backend_status["backend_status"],
            "backend_status_label": backend_status["backend_status_label"],
            "python_executable": backend_status["python_executable"],
            "torch_version": backend_status["torch_version"],
            "transformers_version": backend_status["transformers_version"],
            "device": backend_status["device"],
            "local_files_only": backend_status["local_files_only"],
            "result_origin": result_origin,
            "result_origin_label": result_origin_label,
            "selected_layers": primary_features["selected_layers"],
            "probe_loaded": probe_loaded,
            "num_samples": len(sample_records),
            "feature_export_supported": True,
            "probe_training_supported": True,
            "baseline_backend_score": round(backend_score, 6) if mode == "uncertainty_baseline" else None,
            "baseline_text_proxy_score": round(baseline_text_score, 6) if mode == "uncertainty_baseline" else None,
            "baseline_backend_weight": backend_weight if mode == "uncertainty_baseline" else None,
        },
        mode_used=mode,
        sub_signals=sub_signals,
        sampled_answers=sample_answers[: len(sample_records)],
    )


