"""HTML section extraction and component parsing."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from lxml import html

from .models import FeeToken, Section
from .pricing_tokens import tokenize_fee_text
from .rich_text import clean_fee_text, extract_links, extract_text

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _section_id_from_path(path: list[str]) -> str:
    normalized = "|".join(p.lower().replace(" ", "_") for p in path if p)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _is_heading(element: Any) -> bool:
    return element.tag in HEADING_TAGS


def _heading_level(element: Any) -> int:
    if element.tag in HEADING_TAGS:
        return int(element.tag[1])
    return 0


def _build_section_tree(
    tree: Any,
    base_url: str,
    page_title: str | None = None,
) -> list[Section]:
    """Build a deterministic section tree from heading hierarchy.

    This collects all headings in source order and groups subsequent content
    under the nearest preceding heading of higher or equal level.
    """
    sections: list[Section] = []
    stack: list[tuple[int, Section]] = []
    order = 0
    for element in tree.iter():
        if element.tag in {"script", "style", "nav", "footer", "header"}:
            continue
        if _is_heading(element):
            level = _heading_level(element)
            text = clean_fee_text(extract_text(element))
            if not text:
                continue

            # Pop stack until we find a parent with a lower level.
            while stack and stack[-1][0] >= level:
                stack.pop()

            parent_path = [s.heading for _, s in stack if s.heading]
            section_path = parent_path + [text]
            section_id = _section_id_from_path(section_path)
            links = extract_links(element, base_url)
            section = Section(
                section_id=section_id,
                heading=text,
                level=level,
                section_path=section_path,
                source_order=order,
                links=links,
            )
            order += 1
            sections.append(section)
            stack.append((level, section))
            continue

        # Collect body text for all sections currently on the stack.
        body_text = clean_fee_text(extract_text(element))
        if not body_text or not stack:
            continue
        for _level, section in stack:
            if section.heading and body_text == section.heading:
                continue
        # Update all stacked sections with the new body text.
        new_stack: list[tuple[int, Section]] = []
        for level, section in stack:
            updated = section.model_copy(update={"body": f"{section.body}\n{body_text}" if section.body else body_text})
            for i, s in enumerate(sections):
                if s.section_id == updated.section_id:
                    sections[i] = updated
                    break
            new_stack.append((level, updated))
        stack = new_stack

    return sections


def _extract_local_payment_method_cards(tree: Any, base_url: str) -> list[Section]:
    """Extract method cards from the local payment methods page.

    Each card has a data-js-controller="LocalPaymentMethodsPricingCard" and
    contains a heading with the method name and fee text. The cards are grouped
    by preceding section headings (Cards, Wallets, etc.).
    """
    sections: list[Section] = []
    cards = tree.xpath("//*[@data-js-controller='LocalPaymentMethodsPricingCard']")

    current_family = "Payment methods"
    for order, card in enumerate(cards):
        heading_button = card.xpath(".//h3//button | .//h3")
        if not heading_button:
            continue
        heading_text = clean_fee_text(extract_text(heading_button[0]))
        if not heading_text:
            continue

        # Determine family from the nearest preceding h2 or from the heading text.
        family = current_family
        for prev in card.itersiblings(preceding=True):
            if prev.tag == "h2" or prev.tag == "h3":
                prev_text = clean_fee_text(extract_text(prev))
                if prev_text and not re.search(r"[0-9]", prev_text):
                    family = prev_text
                    current_family = family
                    break

        section_path = [family, heading_text]
        section_id = _section_id_from_path(section_path)
        body = clean_fee_text(extract_text(card))
        links = extract_links(card, base_url)
        sections.append(
            Section(
                section_id=section_id,
                heading=heading_text,
                level=3,
                body=body,
                section_path=section_path,
                source_order=order,
                links=links,
            )
        )
    return sections


def _extract_product_sections(tree: Any, base_url: str) -> list[Section]:
    """Extract product sections from the main pricing page.

    Uses data-js-controller="PricingProductFeatureGroup" as a product boundary
    hint, but falls back to heading hierarchy if those markers are absent.
    """
    sections: list[Section] = []
    feature_groups = tree.xpath("//*[@data-js-controller='PricingProductFeatureGroup']")
    if not feature_groups:
        return _build_section_tree(tree, base_url)

    for order, group in enumerate(feature_groups):
        # Find the nearest preceding h2 as the product name.
        product = "Standard pricing"
        for prev in group.itersiblings(preceding=True):
            if prev.tag in {"h2", "h3"}:
                product = clean_fee_text(extract_text(prev))
                break
        body = clean_fee_text(extract_text(group))
        if not body:
            continue
        section_path = [product]
        section_id = _section_id_from_path(section_path)
        links = extract_links(group, base_url)
        sections.append(
            Section(
                section_id=section_id,
                heading=product,
                level=2,
                body=body,
                section_path=section_path,
                source_order=order,
                links=links,
            )
        )
    return sections


def extract_sections(html_text: str, base_url: str, page_kind: str = "pricing") -> list[Section]:
    """Extract sections from a Stripe pricing HTML page.

    ``page_kind`` is either ``pricing`` or ``local-payment-methods``.
    """
    try:
        tree = html.fromstring(html_text)
    except Exception as exc:
        raise ValueError(f"Invalid HTML: {exc}") from exc

    if page_kind == "local-payment-methods":
        return _extract_local_payment_method_cards(tree, base_url)
    return _build_section_tree(tree, base_url)


def split_section_body_into_entries(section: Section) -> list[tuple[str, list[FeeToken]]]:
    """Split a section body into candidate fee phrases.

    Returns a list of (phrase, tokens) pairs. Consecutive lines are merged when
    a fee line is immediately followed by a qualifier line (e.g. starting
    with "for", "if", "per", "starting at", "up to"). This preserves the
    context needed for classification.
    """
    if not section.body:
        return []
    raw_lines = [clean_fee_text(line) for line in section.body.splitlines()]
    lines = [line for line in raw_lines if line]

    merged: list[str] = []
    qualifier_pattern = re.compile(
        r"^(for|if|per|starting at|up to|minimum|maximum|cap|capped|custom|contact sales)", re.IGNORECASE
    )
    fee_pattern = re.compile(r"[0-9]\s*%|[0-9]\s*[€£$¥A-Z]|included|free")

    for line in lines:
        if not merged:
            merged.append(line)
            continue
        previous = merged[-1]
        if qualifier_pattern.match(line) and fee_pattern.search(previous):
            merged[-1] = f"{previous} {line}"
        elif fee_pattern.search(line):
            merged.append(line)
        else:
            merged.append(line)

    results: list[tuple[str, list[FeeToken]]] = []
    for phrase in merged:
        if not re.search(r"[0-9]", phrase) and not re.search(r"included|free", phrase, re.IGNORECASE):
            continue
        tokens = tokenize_fee_text(phrase)
        if tokens:
            results.append((phrase, tokens))
    return results
