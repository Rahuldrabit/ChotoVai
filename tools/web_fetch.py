"""
MCP tool: web_fetch — fetch a public web page (read-only) with SSRF guards.
"""

from __future__ import annotations

import re
from typing import Literal

import httpx

from tools._net_safety import validate_public_http_url


def _strip_html_to_text(html: str) -> str:
    # Very small, dependency-free readability pass.
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p\s*>", "\n\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = re.sub(r"[ \t\r\f\v]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def web_fetch(
    url: str,
    max_bytes: int = 500_000,
    timeout_s: float = 15.0,
    user_agent: str = "slm-agent/0.1 (read-only)",
    mode: Literal["text", "raw"] = "text",
) -> dict[str, str | int]:
    """
    Fetch a URL and return cleaned text.

    Safety:
    - only http(s)
    - blocks localhost/private IPs (SSRF)
    - caps bytes returned
    """
    safety = validate_public_http_url(url)
    if not safety.ok:
        return {"status_code": 0, "content_type": "", "text": f"ERROR: blocked URL: {safety.reason}"}

    headers = {"User-Agent": user_agent, "Accept": "text/html,application/json;q=0.9,*/*;q=0.1"}
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout_s,
            headers=headers,
        ) as client:
            r = client.get(url)
    except Exception as e:
        return {"status_code": 0, "content_type": "", "text": f"ERROR: fetch failed: {type(e).__name__}: {e}"}

    content_type = r.headers.get("content-type", "")
    raw = r.text if isinstance(r.text, str) else (r.content.decode("utf-8", errors="replace"))
    raw = raw[: max_bytes]

    text = raw if mode == "raw" else _strip_html_to_text(raw)
    text = text[:8000]  # keep tool outputs bounded for small models

    return {"status_code": int(r.status_code), "content_type": content_type, "text": text}


TOOL_DESCRIPTOR = {
    "name": "web_fetch",
    "description": "Fetch a public URL (read-only), returning cleaned text. SSRF-protected.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Public http(s) URL"},
            "max_bytes": {"type": "integer", "default": 500000},
            "timeout_s": {"type": "number", "default": 15.0},
            "mode": {"type": "string", "enum": ["text", "raw"], "default": "text"},
        },
        "required": ["url"],
    },
}

