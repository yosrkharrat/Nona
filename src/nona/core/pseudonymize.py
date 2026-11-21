"""Placeholders in, real values out: the pseudonymizer and the non-streaming restorer (ADR-0001, FR-6, FR-8).

Placeholders are `TYPE_n`, numbered per type by order of first appearance. In S2 one entity is one exact
surface string; linking variants (`Ahmed` ≈ `Ahmed Ben Salem`) comes with ADR-0004 in S3.

Restoration is tolerant, as models mangle placeholders (`person 1`, `PERSON 1`), but it only touches numbers
that were actually issued. Numbers that already appear in the input (a user who literally wrote `PERSON_1`)
are never issued, so restoring can't rewrite what the user typed.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from enum import StrEnum

from nona.core.types import Entity, EntityType, Span


class Policy(StrEnum):
    PSEUDONYMIZE = "pseudonymize"  # reversible: TYPE_n
    REDACT = "redact"  # irreversible: [TYPE]
    KEEP = "keep"  # left as is


# 02 §5. Types not listed here are pseudonymized.
DEFAULT_POLICIES: AbcMapping[EntityType, Policy] = {
    EntityType.CARD: Policy.REDACT,
    EntityType.ORGANIZATION: Policy.KEEP,
}


def _type_pattern(etype: EntityType) -> str:
    return etype.value.replace("_", "[ _]")


def placeholder_regex(types: Iterable[EntityType]) -> re.Pattern[str]:
    """`TYPE_n`, case-insensitive, also `TYPE n` and `TYPEn`. Group 1 is the type, group 2 the number."""
    names = sorted({_type_pattern(t) for t in types}, key=len, reverse=True)
    if not names:
        return re.compile(r"(?!)")  # matches nothing
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(names) + r")[ _]?(\d+)(?!\d)", re.IGNORECASE)


_ANY_PLACEHOLDER = placeholder_regex(EntityType)


def _canonical_type(raw: str) -> EntityType:
    return EntityType(re.sub(r"[ _]", "_", raw.upper()))


class Mapping:
    """Placeholder → original value. Opaque on purpose: it never prints its values (deep dive G)."""

    __slots__ = ("_values",)

    def __init__(self, values: AbcMapping[str, str]) -> None:
        self._values = dict(values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, placeholder: object) -> bool:
        return placeholder in self._values

    def __repr__(self) -> str:
        return f"<Mapping with {len(self._values)} placeholder(s)>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and self._values == other._values

    __hash__ = None  # type: ignore[assignment]  # mutable-looking value semantics, not hashable

    def value(self, placeholder: str) -> str:
        return self._values[placeholder]

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple(self._values)


@dataclass(frozen=True, slots=True)
class Anonymized:
    text: str
    entities: tuple[Entity, ...]  # pseudonymized entities, in placeholder order
    redacted: tuple[Span, ...]  # irreversibly removed spans
    mapping: Mapping


def pseudonymize(
    original: str,
    spans: Iterable[Span],
    policies: AbcMapping[EntityType, Policy] = DEFAULT_POLICIES,
) -> Anonymized:
    """Replace resolved (non-overlapping) spans according to their type's policy."""
    ordered = sorted(spans, key=Span.sort_key)
    for a, b in itertools.pairwise(ordered):
        if a.overlaps(b):
            raise ValueError("spans must not overlap; run the resolver first")

    reserved: dict[EntityType, set[int]] = {}
    for m in _ANY_PLACEHOLDER.finditer(original):
        reserved.setdefault(_canonical_type(m.group(1)), set()).add(int(m.group(2)))

    next_number: dict[EntityType, int] = {}
    numbers: dict[tuple[EntityType, str], int] = {}  # (type, surface) → number
    mentions: dict[tuple[EntityType, str], list[Span]] = {}
    redacted: list[Span] = []
    parts: list[str] = []
    pos = 0
    for span in ordered:
        policy = policies.get(span.type, Policy.PSEUDONYMIZE)
        if policy is Policy.KEEP:
            continue
        parts.append(original[pos : span.start])
        surface = span.text_in(original)
        if policy is Policy.REDACT:
            parts.append(f"[{span.type.value}]")
            redacted.append(span)
        else:
            key = (span.type, surface)
            if key not in numbers:
                n = next_number.get(span.type, 1)
                while n in reserved.get(span.type, set()):
                    n += 1
                numbers[key] = n
                next_number[span.type] = n + 1
            mentions.setdefault(key, []).append(span)
            parts.append(f"{span.type.value}_{numbers[key]}")
        pos = span.end
    parts.append(original[pos:])

    entities = tuple(
        Entity(id=numbers[key], type=key[0], mentions=tuple(mentions[key]), canonical=key[1])
        for key in sorted(numbers, key=lambda k: (mentions[k][0].start, k[0].value))
    )
    mapping = Mapping({e.placeholder: e.canonical for e in entities})
    return Anonymized("".join(parts), entities, tuple(redacted), mapping)


def restore(text: str, mapping: Mapping) -> str:
    """Put the original values back for every issued placeholder; leave anything else untouched."""
    if not len(mapping):
        return text
    types = {_canonical_type(p.rsplit("_", 1)[0]) for p in mapping.placeholders}

    def replace(m: re.Match[str]) -> str:
        placeholder = f"{_canonical_type(m.group(1)).value}_{int(m.group(2))}"
        return mapping.value(placeholder) if placeholder in mapping else m.group(0)

    return placeholder_regex(types).sub(replace, text)
