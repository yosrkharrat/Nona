"""Recognizers: each one finds one family of personal data (02 §3, deep dive A).

A recognizer reads the normalized view (`NormalizedDoc.text`) and returns `Span`s in **original** offsets,
using `to_span`. Recognizers don't resolve overlaps or apply thresholds; the span resolver does that, so
recognizers can be combined freely (union first, recall first: NFR-5).
"""

from __future__ import annotations

from typing import Protocol

from nona.core.normalize import NormalizedDoc
from nona.core.recognizers.email import EmailRecognizer
from nona.core.recognizers.ip import IpRecognizer
from nona.core.recognizers.phone import PhoneRecognizer
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span


class Recognizer(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def supported_types(self) -> frozenset[EntityType]: ...

    def detect(self, doc: NormalizedDoc) -> list[Span]: ...


def default_recognizers() -> list[Recognizer]:
    """The structured recognizers of FR-1, in a fixed order (determinism, ADR-0002)."""
    return [EmailRecognizer(), PhoneRecognizer(), IpRecognizer()]


__all__ = [
    "EmailRecognizer",
    "IpRecognizer",
    "PhoneRecognizer",
    "Recognizer",
    "default_recognizers",
    "to_span",
]
