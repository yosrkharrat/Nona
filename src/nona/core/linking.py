"""Causal entity linking: which entity each mention belongs to, and its placeholder number (ADR-0004).

A `Linker` lives for one conversation and sees its messages in order. For each message it decides, for every
resolved span, which entity the span is a mention of:

- **PERSON** mentions are compared by canonical token set (`core/canonical.py`). A mention is *compatible*
  with an entity when one token set contains the other.
  - Exactly one compatible entity: the mention joins it, and the entity **absorbs** any new tokens
    (`{ahmed}` → `{ahmed, bensalem}`). Its number doesn't change.
  - None: a new entity.
  - Two or more: the mention is **ambiguous**. It gets its own entity, and every later mention with the same
    token set reuses it. Ambiguous entities are matched only by an identical token set, never by containment,
    so they never make a later full name ambiguous.
- **Other types** join the entity with the same value: compared on letters and digits for numbers
  (`22 123 456` = `22.123.456`), and on folded text for the rest.

Inside one message the linker may look ahead, as a message never changes once sent: PERSON mentions are linked
longest first, so `Ahmed … Ahmed Karoui … Ahmed Ben Salem` makes the bare `Ahmed` ambiguous instead of
letting it absorb whichever full name came next.

**Freeze.** Entities are never merged, split or renumbered, so placeholders issued for a message stay the same
whatever comes after it: `link(msgs[:k])` is a prefix of `link(msgs)`. New entities are numbered per type by
first appearance, skipping numbers reserved because the user literally wrote that placeholder.

**Literal placeholders.** When the user writes `PHONE_1` after `PHONE_1` was issued, the number can't be
renumbered, so the literal text becomes an entity of its own (spans from `LITERAL_DETECTOR`): it goes upstream
as a fresh placeholder and is restored to exactly what the user wrote. Such entities are never propagated.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from nona.core.canonical import canonical_tokens, compatible
from nona.core.types import EntityType, Span

LITERAL_DETECTOR = "literal-placeholder"  # a placeholder the user wrote, colliding with an issued one

# Values compared on letters and digits only, so formatting doesn't split one number into two entities.
COMPACT_TYPES = frozenset(
    {EntityType.PHONE, EntityType.IBAN, EntityType.RIB, EntityType.CARD, EntityType.NATIONAL_ID}
)


def value_key(etype: EntityType, surface: str) -> str:
    """What makes two non-PERSON mentions the same entity."""
    folded = " ".join(surface.casefold().split())
    if etype in COMPACT_TYPES:
        return "".join(c for c in folded if c.isalnum())
    return folded


@dataclass(slots=True, eq=False)
class LinkedEntity:
    """One entity of the conversation. Mutable only by absorbing tokens and surface forms."""

    number: int
    type: EntityType
    tokens: frozenset[str] = frozenset()  # PERSON only
    key: str = ""  # everything else, and PERSON mentions with no tokens (an honorific alone)
    ambiguous: bool = False
    surfaces: list[str] = field(default_factory=list[str])

    @property
    def placeholder(self) -> str:
        return f"{self.type.value}_{self.number}"

    @property
    def canonical(self) -> str:
        """The longest surface form seen so far (the first one among equals): what restoring writes back."""
        return max(self.surfaces, key=len)

    @property
    def literal(self) -> bool:
        """A placeholder the user wrote, not personal data: restored as is, never propagated."""
        return self.key.startswith("\0")

    def add_surface(self, surface: str) -> None:
        if surface not in self.surfaces:
            self.surfaces.append(surface)


class Linker:
    def __init__(self) -> None:
        self._entities: list[LinkedEntity] = []
        self._by_key: dict[tuple[EntityType, str], LinkedEntity] = {}
        self._next: dict[EntityType, int] = {}
        self._reserved: dict[EntityType, set[int]] = {}

    @property
    def entities(self) -> tuple[LinkedEntity, ...]:
        """Every entity so far, in the order it was created."""
        return tuple(self._entities)

    def link(
        self,
        original: str,
        spans: Sequence[Span],
        reserved: Iterable[tuple[EntityType, int]] = (),
    ) -> list[LinkedEntity]:
        """The entity of each span of one message, in the order of `spans`. Call once per message, in order.

        `reserved` lists placeholder numbers already written in this message; they are never issued.
        """
        for etype, number in reserved:
            self._reserved.setdefault(etype, set()).add(number)

        surfaces = [s.text_in(original) for s in spans]
        found: list[LinkedEntity | None] = [None] * len(spans)
        fresh: list[LinkedEntity] = []  # created in this message, numbered once the message is linked

        tokens = {
            i: canonical_tokens(surfaces[i])
            for i, s in enumerate(spans)
            if s.type is EntityType.PERSON and s.detector != LITERAL_DETECTOR
        }
        person_order = sorted(tokens, key=lambda i: (-len(tokens[i]), spans[i].start))
        other = [i for i in range(len(spans)) if i not in tokens]
        for i in [*person_order, *other]:
            span = spans[i]
            if span.detector == LITERAL_DETECTOR:
                entity = self._link_value(span.type, "\0" + surfaces[i], fresh)
            elif span.type is EntityType.PERSON and tokens[i]:
                entity = self._link_person(tokens[i], fresh)
            else:
                entity = self._link_value(span.type, value_key(span.type, surfaces[i]), fresh)
            entity.add_surface(surfaces[i])
            found[i] = entity

        by_position = sorted(range(len(spans)), key=lambda i: spans[i].sort_key())
        for i in by_position:
            entity = found[i]
            if entity is not None and entity in fresh and entity.number == 0:
                entity.number = self._issue(entity.type)
                self._entities.append(entity)
        return [e for e in found if e is not None]

    def issued(self, etype: EntityType, number: int) -> bool:
        return any(e.type is etype and e.number == number for e in self._entities)

    def _link_person(self, tokens: frozenset[str], fresh: list[LinkedEntity]) -> LinkedEntity:
        people = [e for e in [*self._entities, *fresh] if e.type is EntityType.PERSON and e.tokens]
        for entity in people:
            if entity.ambiguous and entity.tokens == tokens:
                return entity
        candidates = [e for e in people if not e.ambiguous and compatible(e.tokens, tokens)]
        if len(candidates) == 1:
            entity = candidates[0]
            entity.tokens = entity.tokens | tokens
            return entity
        entity = LinkedEntity(0, EntityType.PERSON, tokens=tokens, ambiguous=len(candidates) > 1)
        fresh.append(entity)
        return entity

    def _link_value(self, etype: EntityType, key: str, fresh: list[LinkedEntity]) -> LinkedEntity:
        entity = self._by_key.get((etype, key))
        if entity is None:
            entity = LinkedEntity(0, etype, key=key)
            self._by_key[(etype, key)] = entity
            fresh.append(entity)
        return entity

    def _issue(self, etype: EntityType) -> int:
        n = self._next.get(etype, 1)
        while n in self._reserved.get(etype, set()):
            n += 1
        self._next[etype] = n + 1
        return n
