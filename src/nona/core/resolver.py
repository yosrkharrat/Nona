"""Span resolver: from overlapping candidates to one clean, deterministic set of spans (03 §2).

Recognizers report a union of candidates, overlaps included (the RIB inside a Tunisian IBAN, a phone and a
CIN on the same 8 digits). The resolver:

1. drops spans below `min_score`, and spans whose text is on the allowlist (FR-5);
2. adds every denylist term as a `CUSTOM` span (FR-5), and every known surface form from earlier messages of
   the conversation (forward propagation, ADR-0004);
3. cuts every span at a placeholder the user typed (`Sami Trabelsi PERSON_1` → `Sami Trabelsi`): a detection
   that swallowed one would replace it, and its text would end up in the mapping;
4. keeps non-overlapping spans greedily: longest first, then type priority, then score, then `Span.sort_key`.

The result never depends on the order recognizers ran in (ADR-0002: placeholders are numbered by first
appearance, so any instability would renumber them).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Sequence

from nona.core.boundaries import LEFT, LOOSE_LEFT, LOOSE_RIGHT, NAME_TYPES, RIGHT, TYPED_PLACEHOLDER
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


def known_spans(
    doc: NormalizedDoc, known: Iterable[tuple[EntityType, str]], detector: str = "propagation"
) -> list[Span]:
    """Forward propagation (ADR-0004, step 2): each occurrence of a surface form linked in an earlier message.

    Multi-word forms match on the normalized view (case, spacing and digits don't matter). Single-word forms
    match the original text exactly, at word boundaries, so a known `Amal` doesn't mask *amal* ("hope").
    Boundaries are the ones restoring uses (`core/boundaries.py`): `_` separates, and Arabic proclitics may
    be attached (`لكنزة`), so a restored value is always caught again.
    """
    spans: list[Span] = []
    for etype, surface in dict.fromkeys(known):
        pattern, on_normalized = _known_pattern(surface, etype in NAME_TYPES)
        if on_normalized:
            spans.extend(
                to_span(doc, m.start(), m.end(), etype, 1.0, detector) for m in pattern.finditer(doc.text)
            )
        else:
            spans.extend(
                Span(m.start(), m.end(), etype, 1.0, detector) for m in pattern.finditer(doc.original)
            )
    return spans


@functools.lru_cache(maxsize=4096)
def _known_pattern(surface: str, name: bool) -> tuple[re.Pattern[str], bool]:
    """The pattern for one known form, and whether it runs on the normalized view. Names use the strict
    boundaries, structured values the loose ones (`core/boundaries.py`). Cached, because the proxy re-runs the
    whole conversation on every request."""
    left, right = (LEFT, RIGHT) if name else (LOOSE_LEFT, LOOSE_RIGHT)
    words = normalize(surface).text.split()
    if len(words) > 1:
        return re.compile(left + r"\s+".join(map(re.escape, words)) + right), True
    if not words:
        return re.compile(r"(?!)"), False  # matches nothing
    return re.compile(left + re.escape(surface.strip()) + right), False


def _rank(span: Span) -> tuple[int, int, float, tuple[int, int, str, float, str]]:
    return (-len(span), -PRIORITY[span.type], -span.score, span.sort_key())


def _outside(span: Span, original: str, holes: Sequence[tuple[int, int]]) -> Span | None:
    """The span cut back to the text before the first placeholder it overlaps (or after one at its start),
    spaces trimmed. `None` if nothing is left."""
    start, end = span.start, span.end
    for h_start, h_end in holes:
        if h_end <= start or end <= h_start:
            continue
        if start < h_start:
            end = h_start
            break
        start = h_end
    while start < end and original[start].isspace():
        start += 1
    while end > start and original[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    if (start, end) == (span.start, span.end):
        return span
    return Span(start, end, span.type, span.score, span.detector)


def resolve(
    spans: Iterable[Span],
    original: str,
    *,
    min_score: float = 0.5,
    allowlist: Sequence[str] = (),
) -> list[Span]:
    """Non-overlapping spans, sorted by position. `original` is the text the spans index into."""
    allowed = {_fold(term) for term in allowlist}
    holes = [m.span() for m in TYPED_PLACEHOLDER.finditer(original)]
    cut = [_outside(s, original, holes) for s in spans] if holes else list(spans)
    candidates = [
        s for s in cut if s is not None and s.score >= min_score and _fold(s.text_in(original)) not in allowed
    ]
    kept: list[Span] = []
    for span in sorted(candidates, key=_rank):
        if not any(span.overlaps(k) for k in kept):
            kept.append(span)
    return sorted(kept, key=Span.sort_key)
