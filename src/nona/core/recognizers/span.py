"""Turn a match in the normalized view into a `Span` over the original text."""

from __future__ import annotations

from nona.core.normalize import NormalizedDoc
from nona.core.types import EntityType, Span


def to_span(doc: NormalizedDoc, start: int, end: int, etype: EntityType, score: float, detector: str) -> Span:
    o_start, o_end = doc.to_original(start, end)
    return Span(o_start, o_end, etype, score, detector)
