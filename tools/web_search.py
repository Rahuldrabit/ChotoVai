"""
MCP tool: web_search — simple public web search (read-only).

Implementation: lightweight HTML scraping against DuckDuckGo's HTML endpoint.
This is intentionally minimal; swap the backend later without changing the schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import httpx


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


_RE_RESULT = re.compile(
    r'(?is)<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
)


def _strip_tags(s: str) -> str:
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def web_search(
    query: str,
    max_results: int = 5,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """
    Return top search results for a query.
    Output is intentionally small and stable.
    """
    q = (query or "").strip()
    if not q:
        return {"results": []}

    max_results = max(1, min(int(max_results), 10))

    url = "https://duckduckgo.com/html/?q=" + quote_plus(q)
    headers = {"User-Agent": "slm-agent/0.1 (read-only)"}
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers) as c:
            r = c.get(url)
    except Exception as e:
        return {"results": [], "error": f"search failed: {type(e).__name__}: {e}"}

    if r.status_code >= 400:
        return {"results": [], "error": f"search http {r.status_code}"}

    html = r.text[:400_000]
    results: list[SearchResult] = []
    for m in _RE_RESULT.finditer(html):
        results.append(
            SearchResult(
                title=_strip_tags(m.group("title"))[:200],
                url=m.group("url")[:500],
                snippet=_strip_tags(m.group("snippet"))[:300],
            )
        )
        if len(results) >= max_results:
            break

    return {"results": [r.__dict__ for r in results]}


TOOL_DESCRIPTOR = {
    "name": "web_search",
    "description": "Search the public web (read-only) and return top results.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
            "timeout_s": {"type": "number", "default": 10.0},
        },
        "required": ["query"],
    },
}

