"""HTML -> plain text with block structure kept as newlines. Stdlib only."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_BLOCK = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "table", "section", "article", "blockquote", "pre", "hr",
}  # fmt: skip


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n- " if tag == "li" else "\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str | None) -> str | None:
    if not html or not html.strip():
        return None
    p = _TextExtractor()
    p.feed(html)
    p.close()
    lines = (re.sub(r"[ \t ]+", " ", ln).strip() for ln in "".join(p.parts).splitlines())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text or None


def clean_inline(s: str | None) -> str | None:
    """Collapse whitespace in a short field (title, location). Empty -> None."""
    if s is None:
        return None
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s or None
