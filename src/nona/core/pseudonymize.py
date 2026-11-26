"""Placeholders in, real values out: the pseudonymizer and the non-streaming restorer (ADR-0001, FR-6, FR-8).

Placeholders are `TYPE_n`, numbered per type by order of first appearance. Which mentions share a placeholder
is decided by the causal linker (`core/linking.py`, ADR-0004): `Ahmed` then `Ahmed Ben Salem` are one person,
and a placeholder is restored to the entity's longest surface form. A `Pseudonymizer` keeps that state for one
conversation; `pseudonymize()` is the one-message case.

Restoration is tolerant, as models mangle placeholders (`person 1`, `PERSON 1`), but it only touches numbers
that were actually issued. Numbers that already appear in the input (a user who literally wrote `PERSON_1`)
are never issued, so restoring can't rewrite what the user typed. A placeholder typed *after* its number was
issued (`PHONE_1` in message 3) can't be reserved any more: it becomes an entity whose value is the literal
text, goes upstream as a fresh placeholder (`PHONE_2`) and is restored to exactly what the user wrote.
"""

from __future__ import annotations

import bisect
import itertools
import re
from collections.abc import Iterable
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from enum import StrEnum

from nona.core.boundaries import (
    ANY_PLACEHOLDER,
    NAME_TYPES,
    composes_after,
    loose_at,
    placeholder_regex,
    strict_at,
)
from nona.core.linking import LITERAL_DETECTOR, LinkedEntity, Linker
from nona.core.normalize import NormalizedDoc, normalize
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


# Letters the case-insensitive placeholder regex takes for an ASCII one, which `upper()` doesn't map to it.
_ASCII_CASE = str.maketrans({"\u0130": "I", "\u0131": "I", "\u017f": "S", "\u212a": "K"})


def _canonical_type(raw: str) -> EntityType:
    return EntityType(re.sub(r"[ _]", "_", raw.translate(_ASCII_CASE).upper()))


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


def fold_surface(surface: str) -> str:
    """How `Pseudonymizer.seeds` keys a surface form: normalized, spacing collapsed."""
    return " ".join(normalize(surface).text.split())


def reserved_numbers(text: str) -> list[tuple[EntityType, int]]:
    """Placeholders the user literally wrote (`PERSON_1`, `person 2`): their numbers are never issued."""
    return [(_canonical_type(m.group(1)), int(m.group(2))) for m in ANY_PLACEHOLDER.finditer(text)]


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
        self._redacted: dict[tuple[EntityType, str], None] = {}  # redacted surfaces, for propagation

    def policy(self, etype: EntityType) -> Policy:
        return self._policies.get(etype, Policy.PSEUDONYMIZE)

    def known_surfaces(self) -> list[tuple[EntityType, str]]:
        """Every surface form of every pseudonymized entity so far, and every redacted value, for forward
        propagation: a redacted card must stay redacted when it comes back."""
        linked = [(e.type, s) for e in self._linker.entities if e.propagates for s in e.surfaces]
        return [*linked, *self._redacted]

    def seeds(self) -> dict[str, frozenset[str]]:
        """Folded surface form → the tokens look-ahead may spread when that form is propagated (the seeds of
        every person with that form, `core/linking.py`)."""
        out: dict[str, frozenset[str]] = {}
        for e in self._linker.entities:
            if e.seeds and e.propagates:
                for surface in e.surfaces:
                    key = fold_surface(surface)
                    out[key] = out.get(key, frozenset()) | e.seeds
        return out

    def add(
        self, original: str, spans: Iterable[Span], seeds: AbcMapping[Span, frozenset[str]] | None = None
    ) -> str:
        """Pseudonymize the next message from its resolved (non-overlapping) spans; return its text.

        `seeds` gives, for PERSON spans, the tokens look-ahead may spread for their entity (`seeds()`).
        """
        ordered = sorted(spans, key=Span.sort_key)
        for a, b in itertools.pairwise(ordered):
            if a.overlaps(b):
                raise ValueError("spans must not overlap; run the resolver first")
        kept = [s for s in ordered if self.policy(s.type) is not Policy.KEEP]
        kept = sorted([*kept, *self._colliding_literals(original, kept)], key=Span.sort_key)
        reversible = [s for s in kept if self.policy(s.type) is Policy.PSEUDONYMIZE]
        seeds = seeds or {}
        linked = self._linker.link(
            original,
            reversible,
            reserved_numbers(original),
            [seeds.get(s, frozenset()) for s in reversible],
        )
        entities = dict(zip(reversible, linked, strict=True))

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
        for span in redacted:
            if span.detector != LITERAL_DETECTOR:
                self._redacted[(span.type, span.text_in(original))] = None
        self._messages.append(_Message(original, text, tuple(entities.items()), redacted))
        return text

    def _colliding_literals(self, original: str, kept: list[Span]) -> list[Span]:
        """Placeholders the user wrote whose number was already issued (see `core/linking.py`)."""
        out: list[Span] = []
        for m in ANY_PLACEHOLDER.finditer(original):
            span = Span(m.start(), m.end(), _canonical_type(m.group(1)), 1.0, LITERAL_DETECTOR)
            if self._linker.issued(span.type, int(m.group(2))) and not any(span.overlaps(k) for k in kept):
                out.append(span)
        return out

    def mapping(self) -> Mapping:
        return Mapping({e.placeholder: e.canonical for e in self._linker.entities})

    def result(self) -> AnonymizedConversation:
        mapping = self.mapping()
        owners = {e: o.placeholder for e in self._linker.entities if (o := self._linker.owner(e))}
        return AnonymizedConversation(
            tuple(self._result(m, mapping, owners) for m in self._messages), mapping
        )

    @staticmethod
    def _result(message: _Message, mapping: Mapping, owners: dict[LinkedEntity, str]) -> Anonymized:
        mentions: dict[LinkedEntity, list[Span]] = {}
        for span, entity in message.linked:
            if (
                not entity.literal
            ):  # a placeholder the user wrote is in the mapping, but it isn't personal data
                mentions.setdefault(entity, []).append(span)
        entities = tuple(
            Entity(e.number, e.type, tuple(spans), e.canonical, owners.get(e))
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
    """Put the original values back for every issued placeholder; leave anything else untouched.

    A value is written only where forward propagation would find it again (`core/boundaries.py`): a name at
    strict boundaries (`PERSON_1abc` stays, propagation couldn't catch `Samiabc`), and a multi-word value,
    which propagation looks for on the normalized view, at its boundaries there too (`سَـلPERSON_1` stays:
    normalized, `سل` glues to the name).
    """
    if not len(mapping):
        return text
    types = {_canonical_type(p.rsplit("_", 1)[0]) for p in mapping.placeholders}
    view = _NormalizedView(text)
    multi_word: dict[str, bool] = {}  # per call: values are personal data, never kept across calls

    def replace(m: re.Match[str]) -> str:
        etype = _canonical_type(m.group(1))
        placeholder = f"{etype.value}_{int(m.group(2))}"
        if placeholder not in mapping:
            return m.group(0)
        name = etype in NAME_TYPES
        if name and not strict_at(m.string, m.start(), m.end()):
            return m.group(0)
        value = mapping.value(placeholder)
        if value not in multi_word:
            multi_word[value] = len(normalize(value).text.split()) > 1  # found on the normalized view
        if multi_word[value] and not view.at(m.start(), m.end(), name):
            return m.group(0)
        return value

    return placeholder_regex(types).sub(replace, text)


class _NormalizedView:
    """Boundaries of a placeholder on the normalized view of the text, normalized once, when first needed."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._doc: NormalizedDoc | None = None
        self._starts: list[int] = []

    def at(self, start: int, end: int, name: bool) -> bool:
        if name and composes_after(self._text, end):
            return False
        if self._doc is None:
            self._doc = normalize(self._text)
            self._starts = [o_start for o_start, _ in self._doc.origin]
        # A placeholder is ASCII, one normalized character per character, even if a mark follows it.
        n_start = bisect.bisect_left(self._starts, start)
        return (strict_at if name else loose_at)(self._doc.text, n_start, n_start + end - start)
