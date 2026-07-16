"""Public page fetch helpers.

Best-effort stack:
1. ``httpx`` — always available path for static / server-rendered HTML
2. ``playwright`` (optional) — JS-rendered public pages

Authenticated sites (including grok.com private chats) are **not** supported
here on purpose: session automation is fragile and not the primary bridge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import List, Optional
from urllib.parse import urlparse


@dataclass
class FetchResult:
    url: str
    ok: bool
    backend: str
    status_code: Optional[int] = None
    title: str = ""
    text: str = ""
    error: str = ""
    notes: List[str] = field(default_factory=list)

    def preview(self, n: int = 500) -> str:
        body = self.text.strip()
        if len(body) > n:
            return body[:n] + "…"
        return body


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []
        self._skip = False
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip = True
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data
        text = data.strip()
        if text:
            self._chunks.append(text + " ")

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        # Extremely broken HTML — crude strip
        stripped = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        stripped = re.sub(r"(?is)<style.*?>.*?</style>", " ", stripped)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        return "", re.sub(r"\s+", " ", stripped).strip()
    return parser.title.strip(), parser.text()


def _looks_auth_walled(url: str, text: str, title: str) -> Optional[str]:
    host = urlparse(url).netloc.lower()
    blob = f"{title}\n{text}".lower()
    if "grok.com" in host or "x.ai" in host:
        if len(text) < 400 or "sign in" in blob or "log in" in blob:
            return (
                "This looks like an authenticated Grok/xAI surface. "
                "Use the file bridge (`grmc bridge receive`) instead of fetch."
            )
    return None


def fetch_httpx(url: str, timeout: float = 20.0) -> FetchResult:
    try:
        import httpx
    except ImportError:
        return FetchResult(
            url=url,
            ok=False,
            backend="httpx",
            error="httpx is not installed. pip install httpx",
        )

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "GRMC-Bridge/0.2 (+local research agent)"},
        ) as client:
            resp = client.get(url)
        title, text = _html_to_text(resp.text)
        result = FetchResult(
            url=str(resp.url),
            ok=resp.is_success,
            backend="httpx",
            status_code=resp.status_code,
            title=title,
            text=text,
        )
        if not resp.is_success:
            result.error = f"HTTP {resp.status_code}"
        note = _looks_auth_walled(url, text, title)
        if note:
            result.notes.append(note)
            result.ok = False
            result.error = result.error or "auth-walled-or-empty"
        return result
    except Exception as exc:
        return FetchResult(url=url, ok=False, backend="httpx", error=str(exc))


def fetch_playwright(url: str, timeout_ms: int = 20000) -> FetchResult:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return FetchResult(
            url=url,
            ok=False,
            backend="playwright",
            error=(
                "playwright not installed. "
                "pip install playwright && playwright install chromium"
            ),
        )

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            html = page.content()
            title = page.title()
            browser.close()
        _, text = _html_to_text(html)
        result = FetchResult(
            url=url,
            ok=True,
            backend="playwright",
            status_code=200,
            title=title,
            text=text,
        )
        note = _looks_auth_walled(url, text, title)
        if note:
            result.notes.append(note)
            result.ok = False
            result.error = "auth-walled-or-empty"
        return result
    except Exception as exc:
        return FetchResult(url=url, ok=False, backend="playwright", error=str(exc))


def fetch_url(url: str, backend: str = "auto") -> FetchResult:
    """Fetch a *public* URL.

    backend: auto | httpx | playwright
    auto tries httpx first, then playwright if installed and content looks empty.
    """
    backend = (backend or "auto").lower()
    if backend == "httpx":
        return fetch_httpx(url)
    if backend == "playwright":
        return fetch_playwright(url)

    # auto
    primary = fetch_httpx(url)
    if primary.ok and len(primary.text.strip()) >= 80:
        return primary
    # Only attempt playwright when httpx failed or returned a near-empty body
    secondary = fetch_playwright(url)
    if secondary.ok and len(secondary.text.strip()) > len(primary.text.strip()):
        secondary.notes.append(
            "Fell back to playwright after httpx was thin/failed."
        )
        return secondary
    if primary.ok or primary.text or primary.status_code:
        if not primary.ok and secondary.error:
            primary.notes.append(f"playwright also failed: {secondary.error}")
        return primary
    return secondary
