"""Span resolver: from overlapping candidates to one clean, deterministic set of spans (03 §2).

Recognizers report a union of candidates, overlaps included (the RIB inside a Tunisian IBAN, a phone and a
CIN on the same 8 digits). The resolver:

1. drops spans below `min_score`, and spans whose text is on the allowlist (FR-5);
2. adds every denylist term as a `CUSTOM` span (FR-5), and every known surface form from earlier messages of
   the conversation (forward propagation, ADR-0004);
3. keeps non-overlapping spans greedily: longest first, then type priority, then score, then `Span.sort_key`.

The result never depends on the order recognizers ran in (ADR-0002: placeholders are numbered by first
appearance, so any instability would renumber them).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from nona.core.normalize import NormalizedDoc, normalize
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span

# Higher wins when two spans have the same length. Explicit and checksum-validated types first, then
# context-gated ones, then the rest. A CIN beats a phone on the same 8 digits because its context word
# says what it is.
PRIORITY: dict[EntityType, int] = {
    EntityType.CUSTOM: 100,
    EntityType.IBAN: 90,
    EntityType.RIB: 85,
    EntityType.CARD: 80,
    EntityType.NATIONAL_ID: 70,
    EntityType.EMAIL: 60,
    EntityType.IP_ADDRESS: 55,
    EntityType.PHONE: 50,
    EntityType.PERSON: 40,
    EntityType.ORGANIZATION: 30,
    EntityType.LOCATION: 20,
}


def _fold(term: str) -> str:
    return " ".join(term.casefold().split())


def denylist_spans(doc: NormalizedDoc, terms: Iterable[str]) -> list[Span]:
    """Every occurrence of a denylist term, matched on the normalized view at word boundaries."""
    spans: list[Span] = []
    for term in terms:
        needle = _fold(term)
        if not needle:
            continue
        pattern = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, needle.split())) + r"(?!\w)")
        for m in pattern.finditer(doc.text):
            spans.append(to_span(doc, m.start(), m.end(), EntityType.CUSTOM, 1.0, "denylist"))
    return spans


def known_spans(doc: NormalizedDoc, known: Iterable[tuple[EntityType, str]]) -> list[Span]:
    """Forward propagation (ADR-0004, step 2): each occurrence of a surface form linked in an earlier message.

    Multi-word forms match on the normalized view (case, spacing and digits don't matter). Single-word forms
    match the original text exactly, at word boundaries, so a known `Amal` doesn't mask *amal* ("hope").
    """
    spans: list[Span] = []
    for etype, surface in dict.fromkeys(known):
        words = normalize(surface).text.split()
        if len(words) > 1:
            pattern = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)")
            spans.extend(
                to_span(doc, m.start(), m.end(), etype, 1.0, "propagation")
                for m in pattern.finditer(doc.text)
            )
        elif words:
            pattern = re.compile(r"(?<!\w)" + re.escape(surface.strip()) + r"(?!\w)")
            spans.extend(
                Span(m.start(), m.end(), etype, 1.0, "propagation") for m in pattern.finditer(doc.original)
            )
    return spans


def _rank(span: Span) -> tuple[int, int, float, tuple[int, int, str, float, str]]:
    return (-len(span), -PRIORITY[span.type], -span.score, span.sort_key())


def resolve(
    spans: Iterable[Span],
    original: str,
    *,
    min_score: float = 0.5,
    allowlist: Sequence[str] = (),
) -> list[Span]:
    """Non-overlapping spans, sorted by position. `original` is the text the spans index into."""
    allowed = {_fold(term) for term in allowlist}
    candidates = [s for s in spans if s.score >= min_score and _fold(s.text_in(original)) not in allowed]
    kept: list[Span] = []
    for span in sorted(candidates, key=_rank):
        if not any(span.overlaps(k) for k in kept):
            kept.append(span)
    return sorted(kept, key=Span.sort_key)
