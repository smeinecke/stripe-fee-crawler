"""Rich-text rendering and link extraction."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

from .models import Link


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace and normalize non-breaking spaces."""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text(element: Any) -> str:
    """Extract normalized text from an lxml element."""
    if element is None:
        return ""
    text = " ".join(element.itertext())
    return _normalize_whitespace(text)


def extract_links(element: Any, base_url: str) -> list[Link]:
    """Extract hyperlinks from an element tree."""
    links: list[Link] = []
    if element is None:
        return links
    for anchor in element.iter("a"):
        href = anchor.get("href")
        if not href:
            continue
        text = _normalize_whitespace("".join(anchor.itertext()))
        uri = urljoin(base_url, href)
        links.append(Link(text=text or None, uri=uri))
    return links


def clean_fee_text(text: str) -> str:
    """Clean a fee phrase for downstream parsing."""
    text = _normalize_whitespace(text)
    # Remove common UI noise that is not part of the fee statement.
    noise = [
        "Show additional fees",
        "Hide additional fees",
        "Learn more",
        "Start now",
        "Contact sales",
        "Request early access",
    ]
    for phrase in noise:
        text = text.replace(phrase, "")
    return _normalize_whitespace(text)
