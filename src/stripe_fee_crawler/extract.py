"""High-level page extraction and metadata extraction for Stripe pages."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from lxml import html

from .components import extract_sections, split_section_body_into_entries
from .models import PricingEntry, Section, Source
from .pricing_tokens import parse_fee_value
from .rich_text import clean_fee_text, extract_text

logger = logging.getLogger(__name__)


def _extract_page_title(tree: Any) -> str | None:
    title_node = tree.find(".//title")
    if title_node is not None and title_node.text:
        return title_node.text.strip()
    html_tag = tree.xpath("//html")
    if html_tag:
        return html_tag[0].get("data-page-title")
    return None


def _extract_page_id(tree: Any) -> str | None:
    html_tag = tree.xpath("//html")
    if html_tag:
        return html_tag[0].get("data-page-id")
    return None


def _extract_page_locale(tree: Any) -> str | None:
    html_tag = tree.xpath("//html")
    if html_tag:
        lang = html_tag[0].get("lang")
        if lang:
            return lang.lower().replace("_", "-")
    return None


def _extract_canonical_url(tree: Any, base_url: str) -> str | None:
    link = tree.xpath("//link[@rel='canonical']/@href")
    if link:
        return link[0]
    return base_url


def _extract_update_time(tree: Any) -> str | None:
    # Stripe does not currently expose a reliable update date in the markup.
    # Preserve the attribute if it appears in the future.
    html_tag = tree.xpath("//html")
    if html_tag:
        return html_tag[0].get("data-page-updated")
    return None


def _entry_id_for(section: Section, phrase: str, index: int) -> str:
    normalized = f"{section.section_id}:{phrase}:{index}"
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _fee_phrase_in_section(phrase: str) -> bool:
    """Return True if a phrase contains a fee-like pattern."""
    if not re.search(r"[0-9]", phrase):
        return False
    return bool(
        re.search(r"[0-9]\s*%", phrase)
        or re.search(r"[€£$¥A-Z]{1,3}\s*[0-9]", phrase)
        or re.search(r"included|free", phrase, re.IGNORECASE)
    )


def extract_page_source(response: Any) -> Source:
    """Build a Source record from an HTTP response and parsed HTML."""
    tree = html.fromstring(response.text)
    page_title = _extract_page_title(tree)
    page_id = _extract_page_id(tree)
    canonical_url = _extract_canonical_url(tree, response.url)
    return Source(
        requested_url=response.url,
        canonical_url=canonical_url,
        page_id=page_id,
        page_title=page_title,
        source_updated_at=_extract_update_time(tree),
        etag=response.headers.get("etag"),
        last_modified=response.headers.get("last-modified"),
        content_sha256=response.content_sha256,
        evidence_text=_extract_evidence_text(tree),
    )


def _extract_evidence_text(tree: Any) -> str:
    """Extract a compact evidence snippet for the whole page."""
    # Remove scripts and styles to keep evidence clean.
    for bad in tree.iter("script", "style", "nav"):
        bad.clear(keep_tail=True)
    text = clean_fee_text(extract_text(tree))
    return text[:2000]


def extract_pricing_entries(
    html_text: str,
    source_url: str,
    page_kind: str = "pricing",
) -> tuple[list[PricingEntry], list[Section]]:
    """Extract pricing entries and sections from a Stripe page.

    Returns a tuple of (entries, sections). Each entry is a candidate pricing
    fact that can later be classified into a fee rule.
    """
    sections = extract_sections(html_text, source_url, page_kind=page_kind)
    entries: list[PricingEntry] = []

    for section in sections:
        phrases = split_section_body_into_entries(section)
        if not phrases and section.heading and _fee_phrase_in_section(section.heading):
            phrases = [(section.heading, parse_fee_value(section.heading)["tokens"])]

        for index, (phrase, tokens) in enumerate(phrases):
            product = section.section_path[0] if section.section_path else None
            fee_category = section.section_path[-1] if section.section_path else None
            payment_method = _infer_payment_method(section, phrase)
            entry_id = _entry_id_for(section, phrase, index)
            entries.append(
                PricingEntry(
                    entry_id=entry_id,
                    product=product,
                    product_category=page_kind,
                    section_path=section.section_path,
                    fee_category=fee_category,
                    payment_method=payment_method,
                    channel=_infer_channel(product, fee_category),
                    source_text=phrase,
                    source_url=source_url,
                    source_evidence=section.body,
                    tokens=tokens,
                    links=section.links,
                    source_order=section.source_order + index,
                )
            )
    return entries, sections


def _infer_payment_method(section: Section, phrase: str) -> str | None:
    """Infer a payment method identifier from the section heading or phrase."""
    candidates = section.section_path + [phrase]
    for text in candidates:
        text = text.lower()
        for method in [
            "sepa direct debit",
            "sepa bank transfer",
            "ach direct debit",
            "bacs direct debit",
            "bancontact",
            "bizum",
            "blik",
            "eps",
            "ideal",
            "wero",
            "przelewy24",
            "swish",
            "twint",
            "pay by bank",
            "mb way",
            "pix",
            "upi",
            "klarna",
            "billie",
            "scalapay",
            "multibanco",
            "alipay",
            "mobilepay",
            "paypal",
            "revolut pay",
            "wechat pay",
            "amazon pay",
            "satispay",
            "konbini",
            "link",
            "card",
            "terminal",
        ]:
            if method in text:
                return method.replace(" ", "_")
    return None


def _infer_channel(product: str | None, fee_category: str | None) -> str | None:
    text = " ".join(filter(None, [product, fee_category])).lower()
    if "terminal" in text or "in-person" in text or "tap to pay" in text:
        return "in_person"
    if "online" in text or "checkout" in text or "payment link" in text:
        return "online"
    return None
