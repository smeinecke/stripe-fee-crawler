"""HTML section extraction and component parsing."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from lxml import html

from .models import FeeToken, Section
from .pricing_tokens import parse_fee_value, tokenize_fee_text
from .rich_text import clean_fee_text, extract_links, extract_text

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# HTML classes that identify product-group headings on Stripe pricing pages.
_FEATURE_GROUP_HEADING_CLASSES = {
    "PricingProductFeatureGroup__headingButton",
}


def _element_classes(element: Any) -> set[str]:
    """Return the CSS classes on an element as a set."""
    cls = element.get("class") or ""
    return set(cls.split())


def _is_feature_group_heading(element: Any) -> bool:
    """Return True for product-group heading buttons that act as section boundaries."""
    return bool(_FEATURE_GROUP_HEADING_CLASSES & _element_classes(element))


def _section_id_from_path(path: list[str]) -> str:
    normalized = "|".join(p.lower().replace(" ", "_") for p in path if p)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _is_heading(element: Any) -> bool:
    return element.tag in HEADING_TAGS or _is_feature_group_heading(element)


def _heading_level(element: Any) -> int:
    if element.tag in HEADING_TAGS:
        return int(element.tag[1])
    if _is_feature_group_heading(element):
        # Treat collapsible product-group headings as h1-level boundaries so
        # they sit alongside the feature headings they contain.
        return 1
    return 0


def _build_section_tree(
    tree: Any,
    base_url: str,
    page_title: str | None = None,
) -> list[Section]:
    """Build a deterministic section tree from heading hierarchy.

    This collects all headings in source order and groups subsequent content
    under the nearest preceding heading of higher or equal level. Body text is
    collected from each element's own ``text`` and ``tail`` only, not from
    descendants, so parent containers and child elements never contribute the
    same content twice.
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

        # Collect only this element's direct text and tail. Descendant text will
        # be handled when those descendant elements are visited, so each visible
        # text node has exactly one canonical source section.
        text_parts: list[str] = []
        if element.text:
            text_parts.append(element.text)
        if element.tail:
            text_parts.append(element.tail)
        body_text = clean_fee_text("".join(text_parts))
        if not body_text or not stack:
            continue
        if any(section.heading and body_text == section.heading for _, section in stack):
            continue

        # Assign the fragment to the deepest (most specific) open section.
        deepest_level, deepest_section = stack[-1]
        updated_body = f"{deepest_section.body}\n{body_text}" if deepest_section.body else body_text
        updated = deepest_section.model_copy(update={"body": updated_body})
        for i, s in enumerate(sections):
            if s.section_id == updated.section_id:
                sections[i] = updated
                break
        stack[-1] = (deepest_level, updated)

    return sections


def _is_pricing_grid_price(element: Any) -> bool:
    """Return True for the pricing-grid price containers used on LPM pages."""
    return element.tag == "section" and "PricingGridPrice" in (element.get("class") or "")


def _price_row_text(element: Any) -> str:
    """Extract caption, amount and label text from a PricingGridPrice section."""
    parts: list[str] = []
    for cls, tag in (
        ("PricingGridPrice__caption", None),
        ("PricingGridPrice__amount", "strong"),
        ("PricingGridPrice__label", "span"),
    ):
        if tag:
            nodes = element.xpath(f".//{tag}[contains(@class,'{cls}')]")
        else:
            nodes = element.xpath(f".//*[contains(@class,'{cls}')]")
        if nodes:
            text = clean_fee_text(extract_text(nodes[0]))
            if text:
                parts.append(text)
    return " ".join(parts)


def _method_title_from_card(card: Any) -> str | None:
    """Return the method title from an LPM card heading."""
    for xpath in (
        ".//h4[contains(@class,'LocalPaymentMethodsPricingCardHeader__title')]/text()",
        ".//h4",
        ".//h3//button",
        ".//h3",
    ):
        try:
            nodes = card.xpath(xpath)
        except Exception:  # nosec B112 - malformed xpaths are intentionally skipped
            continue
        if nodes:
            text = clean_fee_text(extract_text(nodes[0]) if not isinstance(nodes[0], str) else nodes[0])
            if text:
                return text
    return None


def _extract_local_payment_method_cards(tree: Any, base_url: str) -> list[Section]:
    """Extract method cards from the local payment methods page.

    Each card has a data-js-controller="LocalPaymentMethodsPricingCard" and
    contains a heading with the method name and fee text. The cards are grouped
    by preceding section headings (Cards, Wallets, etc.).

    Cards built with the newer ``PricingGridPrice`` layout emit one section per
    price row so that multiple variants (e.g. Link domestic, Instant Bank
    Payments, Klarna) are not squashed into a single concatenated phrase.
    Subprices inside a ``PricingGridPrice`` (e.g. ``+ 1.5% for international
    transactions``) are extracted as separate rows with the same caption context.
    """
    sections: list[Section] = []
    cards = tree.xpath("//*[@data-js-controller='LocalPaymentMethodsPricingCard']")

    current_family = "Payment methods"
    order = 0
    for card in cards:
        # Determine family from the nearest preceding h2 or from the heading text.
        family = current_family
        for prev in card.itersiblings(preceding=True):
            if prev.tag == "h2" or prev.tag == "h3":
                prev_text = clean_fee_text(extract_text(prev))
                if prev_text and not re.search(r"[0-9]", prev_text):
                    family = prev_text
                    current_family = family
                    break

        method_title = _method_title_from_card(card)
        price_sections = [el for el in card.iter() if _is_pricing_grid_price(el)]

        if not price_sections:
            # Legacy/simple fixture layout: fee text lives in <p>/<li> children.
            heading_button = card.xpath(".//h3//button | .//h3")
            if not heading_button:
                continue
            heading_text = clean_fee_text(extract_text(heading_button[0]))
            if not heading_text:
                continue
            section_path = [family, heading_text]
            paragraphs: list[str] = []
            if heading_text and re.search(r"[0-9]\s*%|[0-9]\s*[€£$¥A-Z]|included|free", heading_text):
                paragraphs.append(heading_text)
            for para in card.iter("p", "li"):
                text = clean_fee_text(extract_text(para))
                if text:
                    paragraphs.append(text)
            body = "\n".join(paragraphs) if paragraphs else clean_fee_text(extract_text(card))
            links = extract_links(card, base_url)
            sections.append(
                Section(
                    section_id=_section_id_from_path(section_path),
                    heading=heading_text,
                    level=3,
                    body=body,
                    section_path=section_path,
                    source_order=order,
                    links=links,
                )
            )
            order += 1
            continue

        # Newer grid layout: one section per PricingGridPrice/base row and one per
        # PricingGridSubprice surcharge row.
        current_price: Any | None = None
        current_body_parts: list[str] = []

        for element in card.iter():
            if _is_pricing_grid_price(element):
                # Finalise the body for the previous price row.
                if current_price is not None and current_body_parts:
                    sections[-1] = sections[-1].model_copy(update={"body": "\n".join(current_body_parts)})
                    current_body_parts = []

                base_text = _price_row_text(element)
                if base_text:
                    heading = base_text
                    if method_title and not heading.lower().startswith(method_title.lower()):
                        heading = f"{method_title} {base_text}"
                    section_path = [family, heading]
                    sections.append(
                        Section(
                            section_id=_section_id_from_path(section_path),
                            heading=heading,
                            level=3,
                            body="",
                            section_path=section_path,
                            source_order=order,
                            links=extract_links(card, base_url),
                        )
                    )
                    order += 1
                current_price = element

                # Subprices belong to the same pricing row but are separate fees.
                caption = clean_fee_text(
                    extract_text(element.xpath(".//*[contains(@class,'PricingGridPrice__caption')]")[0])
                    if element.xpath(".//*[contains(@class,'PricingGridPrice__caption')]")
                    else ""
                )
                for sub in element.xpath(".//div[contains(@class,'PricingGridSubprice')]"):
                    sub_text = clean_fee_text(extract_text(sub))
                    if sub_text:
                        sub_heading = f"{caption} {sub_text}" if caption else sub_text
                        heading = sub_heading
                        if method_title and not heading.lower().startswith(method_title.lower()):
                            heading = f"{method_title} {sub_heading}"
                        section_path = [family, heading]
                        sections.append(
                            Section(
                                section_id=_section_id_from_path(section_path),
                                heading=heading,
                                level=3,
                                body="",
                                section_path=section_path,
                                source_order=order,
                                links=extract_links(card, base_url),
                            )
                        )
                        order += 1
                continue

            if current_price is None:
                continue
            # Skip nested elements of the current price section; their text has
            # already been captured by the base/subprice sections.
            if element in current_price.iterdescendants():
                continue
            if element.tag in {"p", "li"}:
                text = clean_fee_text(extract_text(element))
                if text and text not in current_body_parts:
                    current_body_parts.append(text)

        if current_price is not None and current_body_parts:
            sections[-1] = sections[-1].model_copy(update={"body": "\n".join(current_body_parts)})

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


def _is_feeish_line(line: str) -> bool:
    """Return True when a line contains a parseable fee or included/free statement."""
    parsed = parse_fee_value(line)
    return bool(parsed["percentage"] or parsed["fixed_amount"] or parsed["exactness"] in {"included", "free"})


_MARKET_SHARE_PHRASES = (
    "share",
    "market share",
    "most popular payment method",
    "used in more than",
    "used by over",
    "adoption",
    "customers use",
    "active monthly users",
    "active global customers",
)


def _is_market_share_statistic(line: str) -> bool:
    """Return True for lines where a percentage describes market share, not a fee."""
    parsed = parse_fee_value(line)
    if not parsed["percentage"]:
        return False
    lower = line.lower()
    for phrase in _MARKET_SHARE_PHRASES:
        if phrase in lower:
            return True
    # Broad pattern: "more than X% share / used in ..."
    return bool(re.search(r"\b(more than|over)\s+[0-9]+%?\s*(share|of)\b", lower))


def _is_trusted_feeish_line(line: str) -> bool:
    """Return True when a line is a genuine fee value, not a market-share statistic."""
    return _is_feeish_line(line) and not _is_market_share_statistic(line)


def split_section_body_into_entries(section: Section) -> list[tuple[str, list[FeeToken]]]:
    """Split a section body into candidate fee phrases.

    Returns a list of (phrase, tokens) pairs. Consecutive lines are merged when
    a fee line is immediately followed by a qualifier line (e.g. starting
    with "for", "if", "per", "starting at", "up to", "of"). Short label lines
    ending in "fee" or "price" (such as "Smart Disputes fee") are also merged
    with the following fee amount so the resulting phrase keeps its context.

    Qualifiers are anchored to the most recent trusted fee-value line in the
    same pricing container; they do not attach to marketing prose such as
    "... X% share of online payments".
    """
    if not section.body:
        return []
    raw_lines = [clean_fee_text(line) for line in section.body.splitlines()]
    # If the section heading itself is a fee phrase, make it the first line so
    # that trailing qualifiers in the body (e.g. "per successful transaction for
    # domestic cards") attach to the heading fee.  Avoid duplicating the heading
    # when LPM cards have already inlined it in the body.
    if section.heading and _is_feeish_line(section.heading):
        heading = clean_fee_text(section.heading)
        if raw_lines[0:1] != [heading]:
            raw_lines.insert(0, heading)
    lines = [line for line in raw_lines if line]

    qualifier_pattern = re.compile(
        r"^(of|for|if|per|starting at|up to|minimum|maximum|cap|capped|custom|contact sales)\b",
        re.IGNORECASE,
    )
    # Lines that end with a range/start qualifier introduce the next fee value.
    prefix_qualifier_ends = (
        "starting at",
        "starting from",
        "up to",
        "minimum",
        "maximum",
        "from",
    )
    label_pattern = re.compile(r"^[^\d]{1,40}\b(?:fee|price)\b$", re.IGNORECASE)

    merged: list[str] = []
    last_trusted_idx: int | None = None
    pending_label: str | None = None
    pending_prefix: str | None = None
    pending_suffix: str | None = None

    def _flush_suffix() -> None:
        nonlocal pending_suffix
        if pending_suffix and last_trusted_idx is not None:
            merged[last_trusted_idx] = f"{merged[last_trusted_idx]} {pending_suffix}"
        pending_suffix = None

    for line in lines:
        is_feeish = _is_feeish_line(line)
        is_trusted_feeish = is_feeish and not _is_market_share_statistic(line)
        is_label = bool(label_pattern.match(line)) and not is_feeish
        is_qualifier = bool(qualifier_pattern.match(line))
        parsed = parse_fee_value(line)
        lower = line.lower()
        is_prefix_qualifier = (
            not is_feeish
            and not is_label
            and not is_qualifier
            and (parsed["exactness"] in {"from", "range"} or lower.endswith(prefix_qualifier_ends))
        )

        if is_label:
            _flush_suffix()
            if pending_label:
                merged.append(pending_label)
            pending_label = line
            continue

        if is_trusted_feeish:
            _flush_suffix()
            phrase = line
            if pending_prefix:
                phrase = f"{pending_prefix} {phrase}"
                pending_prefix = None
            if pending_label:
                phrase = f"{pending_label} {phrase}"
                pending_label = None
            merged.append(phrase)
            last_trusted_idx = len(merged) - 1
            continue

        if is_feeish and not is_trusted_feeish:
            # A market-share/marketing line that contains a percentage. Keep it
            # as a separate entry so downstream classification can reject it,
            # but do not let trailing qualifiers attach to it.
            _flush_suffix()
            if pending_label:
                merged.append(pending_label)
                pending_label = None
            merged.append(line)
            continue

        if is_qualifier:
            pending_suffix = f"{pending_suffix} {line}" if pending_suffix else line
            continue

        if is_prefix_qualifier:
            _flush_suffix()
            if pending_label:
                merged.append(pending_label)
                pending_label = None
            pending_prefix = f"{pending_prefix} {line}" if pending_prefix else line
            continue

        # Any other prose line.
        _flush_suffix()
        if pending_label:
            merged.append(pending_label)
            pending_label = None
        merged.append(line)

    _flush_suffix()
    if pending_label:
        merged.append(pending_label)
    if pending_prefix:
        merged.append(pending_prefix)

    results: list[tuple[str, list[FeeToken]]] = []
    for phrase in merged:
        if not re.search(r"[0-9]", phrase) and not re.search(r"included|free", phrase, re.IGNORECASE):
            continue
        tokens = tokenize_fee_text(phrase)
        if tokens:
            results.append((phrase, tokens))
    return results
