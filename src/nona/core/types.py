"""Core value types shared by every stage of the pipeline (docs/design/02-api.md, section 3).

All types are immutable. Offsets are Python string indices into the *original* text, half-open: [start, end).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EntityType(StrEnum):
    """Kinds of personal data Nona detects. The value is the placeholder prefix (`PERSON` → `PERSON_1`)."""

    PERSON = "PERSON"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    IBAN = "IBAN"
    RIB = "RIB"  # Tunisian bank account number
    CARD = "CARD"  # payment card, Luhn-validated
    IP_ADDRESS = "IP_ADDRESS"
    NATIONAL_ID = "NATIONAL_ID"  # Tunisian CIN, only with a context word (deep dive A)
    CUSTOM = "CUSTOM"  # denylist terms, and later custom recognizers from configuration (FR-4, FR-5)


@dataclass(frozen=True, slots=True)
class Span:
    """One detection: a typed range of the original text, as reported by a single recognizer."""

    start: int
    end: int
    type: EntityType
    score: float
    detector: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span offsets [{self.start}, {self.end})")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        if not self.detector:
            raise ValueError("detector name must not be empty")

    def __len__(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def sort_key(self) -> tuple[int, int, str, float, str]:
        """Total, deterministic order: by position, longest first, then type, score, detector.

        Determinism is a requirement, not a nicety: placeholders are numbered by order of first
        appearance (ADR-0002, ADR-0004), so any unstable ordering would renumber them.
        """
        return (self.start, -self.end, self.type.value, -self.score, self.detector)

    def text_in(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass(frozen=True, slots=True)
class Entity:
    """A real-world thing (one person, one phone number) and every mention of it in a conversation."""

    id: int
    type: EntityType
    mentions: tuple[Span, ...]
    canonical: str  # longest surface form seen so far; what a placeholder is restored to (ADR-0004)

    def __post_init__(self) -> None:
        if self.id < 1:
            raise ValueError("entity ids start at 1, matching placeholder numbers")
        if not self.mentions:
            raise ValueError("an entity needs at least one mention")
        if any(m.type is not self.type for m in self.mentions):
            raise ValueError("all mentions must have the entity's type")

    @property
    def placeholder(self) -> str:
        return f"{self.type.value}_{self.id}"
