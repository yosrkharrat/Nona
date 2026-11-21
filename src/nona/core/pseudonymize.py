"""Placeholders in, real values out: the pseudonymizer and the non-streaming restorer (ADR-0001, FR-6, FR-8).

Placeholders are `TYPE_n`, numbered per type by order of first appearance. Which mentions share a placeholder
is decided by the causal linker (`core/linking.py`, ADR-0004): `Ahmed` then `Ahmed Ben Salem` are one person,
and a placeholder is restored to the entity's longest surface form. A `Pseudonymizer` keeps that state for one
conversation; `pseudonymize()` is the one-message case.

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

from nona.core.linking import LITERAL_DETECTOR, LinkedEntity, Linker
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


@dataclass(frozen=True, slots=True)
class AnonymizedConversation:
    """Every message of a conversation, anonymized with one shared mapping (ADR-0004)."""

    messages: tuple[Anonymized, ...]
    mapping: Mapping

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(m.text for m in self.messages)


def reserved_numbers(text: str) -> list[tuple[EntityType, int]]:
    """Placeholders the user literally wrote (`PERSON_1`, `person 2`): their numbers are never issued."""
    return [(_canonical_type(m.group(1)), int(m.group(2))) for m in _ANY_PLACEHOLDER.finditer(text)]


@dataclass(frozen=True, slots=True)
class _Message:
    original: str
    text: str
    linked: tuple[tuple[Span, LinkedEntity], ...]
    redacted: tuple[Span, ...]


class Pseudonymizer:
    """Pseudonymizes the messages of one conversation, in order, with one mapping (ADR-0004)."""

    def __init__(self, policies: AbcMapping[EntityType, Policy] = DEFAULT_POLICIES) -> None:
        self._policies = dict(policies)
        self._linker = Linker()
        self._messages: list[_Message] = []

    def policy(self, etype: EntityType) -> Policy:
        return self._policies.get(etype, Policy.PSEUDONYMIZE)

    def known_surfaces(self) -> list[tuple[EntityType, str]]:
        """Every surface form of every pseudonymized entity so far, for forward propagation."""
        return [(e.type, s) for e in self._linker.entities if not e.literal for s in e.surfaces]

    def add(self, original: str, spans: Iterable[Span]) -> str:
        """Pseudonymize the next message from its resolved (non-overlapping) spans; return its text."""
        ordered = sorted(spans, key=Span.sort_key)
        for a, b in itertools.pairwise(ordered):
            if a.overlaps(b):
                raise ValueError("spans must not overlap; run the resolver first")
        kept = [s for s in ordered if self.policy(s.type) is not Policy.KEEP]
        kept = sorted([*kept, *self._colliding_literals(original, kept)], key=Span.sort_key)
        reversible = [s for s in kept if self.policy(s.type) is Policy.PSEUDONYMIZE]
        entities = dict(
            zip(reversible, self._linker.link(original, reversible, reserved_numbers(original)), strict=True)
        )

        parts: list[str] = []
        pos = 0
        for span in kept:
            parts.append(original[pos : span.start])
            entity = entities.get(span)
            parts.append(entity.placeholder if entity else f"[{span.type.value}]")
            pos = span.end
        parts.append(original[pos:])
        text = "".join(parts)
        redacted = tuple(s for s in kept if s not in entities)
        self._messages.append(_Message(original, text, tuple(entities.items()), redacted))
        return text

    def _colliding_literals(self, original: str, kept: list[Span]) -> list[Span]:
        """Placeholders the user wrote whose number was already issued (see `core/linking.py`)."""
        out: list[Span] = []
        for m in _ANY_PLACEHOLDER.finditer(original):
            span = Span(m.start(), m.end(), _canonical_type(m.group(1)), 1.0, LITERAL_DETECTOR)
            if self._linker.issued(span.type, int(m.group(2))) and not any(span.overlaps(k) for k in kept):
                out.append(span)
        return out

    def mapping(self) -> Mapping:
        return Mapping({e.placeholder: e.canonical for e in self._linker.entities})

    def result(self) -> AnonymizedConversation:
        mapping = self.mapping()
        return AnonymizedConversation(tuple(self._result(m, mapping) for m in self._messages), mapping)

    @staticmethod
    def _result(message: _Message, mapping: Mapping) -> Anonymized:
        mentions: dict[LinkedEntity, list[Span]] = {}
        for span, entity in message.linked:
            mentions.setdefault(entity, []).append(span)
        entities = tuple(
            Entity(id=e.number, type=e.type, mentions=tuple(spans), canonical=e.canonical)
            for e, spans in sorted(mentions.items(), key=lambda item: (item[0].type.value, item[0].number))
        )
        return Anonymized(message.text, entities, message.redacted, mapping)


def pseudonymize(
    original: str,
    spans: Iterable[Span],
    policies: AbcMapping[EntityType, Policy] = DEFAULT_POLICIES,
) -> Anonymized:
    """Replace resolved (non-overlapping) spans by their type's policy: a one-message conversation."""
    pseudonymizer = Pseudonymizer(policies)
    pseudonymizer.add(original, spans)
    return pseudonymizer.result().messages[0]


def restore(text: str, mapping: Mapping) -> str:
    """Put the original values back for every issued placeholder; leave anything else untouched."""
    if not len(mapping):
        return text
    types = {_canonical_type(p.rsplit("_", 1)[0]) for p in mapping.placeholders}

    def replace(m: re.Match[str]) -> str:
        placeholder = f"{_canonical_type(m.group(1)).value}_{int(m.group(2))}"
        return mapping.value(placeholder) if placeholder in mapping else m.group(0)

    return placeholder_regex(types).sub(replace, text)
