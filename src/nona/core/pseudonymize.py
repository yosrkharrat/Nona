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

import itertools
import re
from collections.abc import Iterable
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from enum import StrEnum

from nona.core.boundaries import (
    ANY_PLACEHOLDER,
    RESERVED_PLACEHOLDER,
    placeholder_parts,
    placeholder_regex,
)
from nona.core.linking import LITERAL_DETECTOR, LinkedEntity, Linker
from nona.core.normalize import normalize
from nona.core.resolver import known_spans
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


def fold_surface(surface: str) -> str:
    """How `Pseudonymizer.seeds` keys a surface form: normalized, spacing collapsed."""
    return " ".join(normalize(surface).text.split())


def reserved_numbers(text: str) -> list[tuple[EntityType, int]]:
    """Placeholders the user literally wrote (`PERSON_1`, `person 2`, glued `SamiPERSON_1`): their numbers are
    never issued."""
    return [(_canonical_type(m.group(1)), int(m.group(2))) for m in RESERVED_PLACEHOLDER.finditer(text)]


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
            type_name, number = placeholder_parts(m)
            span = Span(m.start(), m.end(), _canonical_type(type_name), 1.0, LITERAL_DETECTOR)
            if self._linker.issued(span.type, number) and not any(span.overlaps(k) for k in kept):
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

    A value is written only where forward propagation finds it again in the restored text, so the client
    can't re-send it in clear (`core/boundaries.py`): `PERSON_1abc` stays, and so does `PERSON_1PERSON_2`,
    which would read `SamiAmal`. A placeholder the user wrote (a literal) is written only where it is found
    again as one. Checked on the output, again after each round of placeholders put back. A round puts back
    every value that fails, so a value that failed only because of a glued neighbour put back with it stays
    a placeholder too (`PERSON_2PERSON_1abc`): never a leak, and one round per value would be quadratic.

    Glued to a letter, a placeholder counts in its exact issued shape, so `IPHONE_1` from the model is read
    as `PHONE_1`: the user's own value in the user's own answer, a corruption but no leak. Telling it from an
    issued `PHONE_1` glued to a capital (`S+216…`) would need the anonymized text, which restore doesn't have.
    """
    if not len(mapping):
        return text
    types = {_canonical_type(p.rsplit("_", 1)[0]) for p in mapping.placeholders}
    found: list[_Found] = []
    for m in placeholder_regex(types).finditer(text):
        type_name, number = placeholder_parts(m)
        etype = _canonical_type(type_name)
        placeholder = f"{etype.value}_{number}"
        if placeholder in mapping:
            found.append((m.start(), m.end(), etype, mapping.value(placeholder)))
    while True:
        out, placed = _write(text, found)
        kept = _found_again(out, found, placed)
        if len(kept) == len(found):
            return out
        found = kept


_Found = tuple[int, int, EntityType, str]  # placeholder start and end, type, value


def _found_again(out: str, found: list[_Found], placed: list[tuple[int, int]]) -> list[_Found]:
    """The values that forward propagation finds again where restore wrote them in `out` (`placed`), and the
    literals found again as placeholders. One propagation pass per distinct value."""
    doc = normalize(out)
    literals = {m.span() for m in ANY_PLACEHOLDER.finditer(out)}
    spans = {
        (etype, value): {(s.start, s.end) for s in known_spans(doc, [(etype, value)])}
        for etype, value in dict.fromkeys((f[2], f[3]) for f in found)
        if not RESERVED_PLACEHOLDER.fullmatch(value)
    }
    kept: list[_Found] = []
    for f, (start, end) in zip(found, placed, strict=True):
        etype, value = f[2], f[3]
        if RESERVED_PLACEHOLDER.fullmatch(value):
            ok = (start, end) in literals
        else:
            at = spans[etype, value]
            ok = (start, end) in at or any(s <= start and end <= e for s, e in at)
        if ok:
            kept.append(f)
    return kept


def _write(text: str, found: list[_Found]) -> tuple[str, list[tuple[int, int]]]:
    """`text` with each placeholder replaced by its value, and where each value went, without spaces."""
    parts: list[str] = []
    placed: list[tuple[int, int]] = []
    pos = size = 0
    for start, end, _, value in found:
        parts.append(text[pos:start])
        size += start - pos
        lead, trail = len(value) - len(value.lstrip()), len(value) - len(value.rstrip())
        placed.append((size + lead, size + len(value) - trail))
        parts.append(value)
        size += len(value)
        pos = end
    parts.append(text[pos:])
    return "".join(parts), placed
