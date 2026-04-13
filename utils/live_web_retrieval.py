"""Live Wikipedia retrieval helpers for web-backed evidence gathering."""

from functools import lru_cache
import html
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from utils.text_utils import chunk_text, safe_text


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "QualitativeHallucinationPrototype/1.0 (educational prototype)"
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
PRON_PAREN_RE = re.compile(
    r"\s*\([^)]*(?:/|\[|French:|pronounced|listen)[^)]*\)\s*",
    flags=re.IGNORECASE,
)


class LiveRetrievalError(RuntimeError):
    """Raised when live web retrieval fails before any evidence can be returned."""


def _clean_wikipedia_text(text: str | None) -> str:
    """Normalize Wikipedia snippets and extracts into cleaner matching text."""
    cleaned = html.unescape(safe_text(text)).replace("\u00a0", " ")
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = PRON_PAREN_RE.sub(" ", cleaned, count=1)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return WHITESPACE_PATTERN.sub(" ", cleaned).strip()


@lru_cache(maxsize=128)
def _request_json(url: str) -> dict:
    """Fetch and decode one JSON response from the Wikipedia API."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=8) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        raise LiveRetrievalError(f"Wikipedia API returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise LiveRetrievalError("Wikipedia API could not be reached from this runtime.") from exc
    except Exception as exc:
        raise LiveRetrievalError("Live web retrieval failed before any evidence could be returned.") from exc

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LiveRetrievalError("Wikipedia API returned unreadable JSON.") from exc


@lru_cache(maxsize=128)
def _search_wikipedia(query: str, limit: int) -> list[dict]:
    """Run a live Wikipedia search for one query string."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max(1, min(limit, 5)),
        "srnamespace": 0,
        "format": "json",
        "utf8": 1,
    }
    url = f"{WIKIPEDIA_API_URL}?{urlencode(params)}"
    data = _request_json(url)
    return list(data.get("query", {}).get("search", []))


@lru_cache(maxsize=128)
def _fetch_page_extracts(titles: tuple[str, ...], extract_chars: int) -> dict[str, dict]:
    """Fetch plain-text extracts for a set of Wikipedia titles."""
    if not titles:
        return {}

    params = {
        "action": "query",
        "prop": "extracts",
        "titles": "|".join(titles),
        "redirects": 1,
        "explaintext": 1,
        "exchars": max(240, min(extract_chars, 1200)),
        "format": "json",
        "utf8": 1,
    }
    url = f"{WIKIPEDIA_API_URL}?{urlencode(params)}"
    data = _request_json(url)
    pages = data.get("query", {}).get("pages", {})
    return {
        page.get("title", ""): page
        for page in pages.values()
        if safe_text(page.get("title"))
    }


def _title_to_url(title: str) -> str:
    """Construct a stable Wikipedia page URL from a page title."""
    normalized = title.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{quote(normalized)}"


def fetch_wiki_evidence(
    query: str,
    chunk_prefix: str = "W",
    max_pages: int = 3,
    extract_chars: int = 700,
) -> dict:
    """Fetch live Wikipedia extracts and return chunked evidence records."""
    cleaned_query = safe_text(query)
    if not cleaned_query:
        return {
            "status": "no_results",
            "backend": "wikipedia-live",
            "query": cleaned_query,
            "message": "No live web query text was available.",
            "chunks": [],
            "sources": [],
        }

    try:
        search_results = _search_wikipedia(cleaned_query, max_pages)
    except LiveRetrievalError as exc:
        return {
            "status": "error",
            "backend": "wikipedia-live",
            "query": cleaned_query,
            "message": str(exc),
            "chunks": [],
            "sources": [],
        }

    if not search_results:
        return {
            "status": "no_results",
            "backend": "wikipedia-live",
            "query": cleaned_query,
            "message": "No Wikipedia pages matched this live retrieval query.",
            "chunks": [],
            "sources": [],
        }

    titles = tuple(item.get("title", "") for item in search_results if safe_text(item.get("title")))
    try:
        page_map = _fetch_page_extracts(titles, extract_chars)
    except LiveRetrievalError as exc:
        return {
            "status": "error",
            "backend": "wikipedia-live",
            "query": cleaned_query,
            "message": str(exc),
            "chunks": [],
            "sources": [],
        }

    chunks = []
    sources = []
    chunk_index = 1
    seen_urls = set()

    for item in search_results:
        title = safe_text(item.get("title"))
        if not title:
            continue

        page = page_map.get(title, {})
        extract_text = _clean_wikipedia_text(page.get("extract")) or _clean_wikipedia_text(item.get("snippet"))
        if not extract_text:
            continue

        source_url = _title_to_url(title)
        if source_url not in seen_urls:
            seen_urls.add(source_url)
            sources.append(
                {
                    "page_title": title,
                    "source_url": source_url,
                    "search_query": cleaned_query,
                }
            )

        text_chunks = chunk_text(extract_text, max_sentences=2, max_chars=340, overlap=1)[:2] or [extract_text]
        for text_chunk in text_chunks:
            chunks.append(
                {
                    "chunk_id": f"{chunk_prefix}{chunk_index}",
                    "source_label": "web",
                    "text": text_chunk,
                    "page_title": title,
                    "source_url": source_url,
                    "search_query": cleaned_query,
                }
            )
            chunk_index += 1

    if not chunks:
        return {
            "status": "no_results",
            "backend": "wikipedia-live",
            "query": cleaned_query,
            "message": "Wikipedia returned search hits, but no usable extracts were available for them.",
            "chunks": [],
            "sources": sources,
        }

    return {
        "status": "ok",
        "backend": "wikipedia-live",
        "query": cleaned_query,
        "message": f"Retrieved {len(chunks)} live Wikipedia chunk(s) from {len(sources)} page(s).",
        "chunks": chunks,
        "sources": sources,
    }