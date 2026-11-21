"""Conversation-level anonymization: causal linking and forward propagation (ADR-0004)."""

from __future__ import annotations

import re
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nona.core import DetectorError, EntityType, Nona, NormalizedDoc, Policy, Span
from nona.core.linking import value_key
from nona.core.pseudonymize import Pseudonymizer
from nona.core.recognizers import default_recognizers

ALL_REVERSIBLE = {t: Policy.PSEUDONYMIZE for t in EntityType}


class Names:
    """A stand-in for the name recognizers of #37/S4: exact names from a list, longest first.

    Text containing one of `blind_to` is not looked at, like a model that misses names in an assistant turn.
    """

    name = "test:names"
    supported_types = frozenset({EntityType.PERSON})

    def __init__(self, names: Sequence[str], blind_to: Sequence[str] = ()) -> None:
        self._names = sorted(names, key=len, reverse=True)
        self._blind_to = tuple(blind_to)

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        if any(marker in doc.original for marker in self._blind_to):
            return []
        out: list[Span] = []
        for name in self._names:
            for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", doc.original):
                if not any(m.start() < s.end and s.start < m.end() for s in out):
                    out.append(Span(m.start(), m.end(), EntityType.PERSON, 0.9, self.name))
        return out


NAMES = ["Ahmed", "Ahmed Ben Salem", "Ahmed Karoui", "M. Ben Salem", "Sami", "Sami Trabelsi", "Amal", "Mme"]


def nona(*, blind_to: Sequence[str] = (), names: Sequence[str] = NAMES, **kwargs: object) -> Nona:
    return Nona([*default_recognizers(), Names(names, blind_to)], **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- the ADR-0004 worked example

ADR_EXAMPLE = [
    "Rappelle Ahmed demain.",
    "Ahmed Ben Salem a envoyé le devis, son mail ahmed.bs@gmail.com",
    "Ahmed Karoui veut aussi un rdv.",
    "J'ai relancé Ahmed Ben Salem et Ahmed Karoui.",
    "Ahmed a confirmé ?",
]


def test_adr_0004_worked_example() -> None:
    conv = nona(blind_to=["relancé"]).anonymize_messages(ADR_EXAMPLE)
    assert conv.texts == (
        "Rappelle PERSON_1 demain.",  # one Ahmed so far
        "PERSON_1 a envoyé le devis, son mail EMAIL_1",  # ⊇ {ahmed}: joins PERSON_1 and absorbs its tokens
        "PERSON_2 veut aussi un rdv.",  # not compatible with {ahmed, bensalem}
        "J'ai relancé PERSON_1 et PERSON_2.",  # the detector is blind here: forward propagation catches both
        "PERSON_3 a confirmé ?",  # compatible with both: ambiguous, its own placeholder
    )
    assert {p: conv.mapping.value(p) for p in conv.mapping.placeholders} == {
        "PERSON_1": "Ahmed Ben Salem",  # restored to the longest form
        "EMAIL_1": "ahmed.bs@gmail.com",
        "PERSON_2": "Ahmed Karoui",
        "PERSON_3": "Ahmed",
    }


def test_restored_answer_cannot_leak_on_the_next_turn() -> None:
    n = nona(blind_to=["relancé"])
    first = n.anonymize_messages(ADR_EXAMPLE[:3])
    answer = n.restore("J'ai relancé PERSON_1 et PERSON_2.", first.mapping)
    assert answer == ADR_EXAMPLE[3]
    again = n.anonymize_messages([*ADR_EXAMPLE[:3], answer])
    assert "Ahmed" not in again.texts[3]


# --------------------------------------------------------------------------- linking rules


def test_ambiguous_form_is_reused_and_never_blocks_a_full_name() -> None:
    conv = nona().anonymize_messages(
        ["Ahmed Ben Salem et Ahmed Karoui", "Ahmed ?", "Oui, Ahmed.", "Non, Ahmed Ben Salem."]
    )
    assert conv.texts == ("PERSON_1 et PERSON_2", "PERSON_3 ?", "Oui, PERSON_3.", "Non, PERSON_1.")


def test_inside_a_message_longer_names_are_linked_first() -> None:
    conv = nona().anonymize_messages(["Ahmed, Ahmed Karoui et Ahmed Ben Salem"])
    # Position order would let the bare `Ahmed` absorb `Karoui`; looking ahead shows it's ambiguous.
    assert conv.texts == ("PERSON_1, PERSON_2 et PERSON_3",)
    assert conv.mapping.value("PERSON_1") == "Ahmed"


def test_honorific_and_particle_forms_link() -> None:
    conv = nona().anonymize_messages(["Ahmed Ben Salem a appelé.", "Rappelle M. Ben Salem."])
    assert conv.texts == ("PERSON_1 a appelé.", "Rappelle PERSON_1.")


def test_honorific_alone_is_its_own_entity() -> None:
    conv = nona().anonymize_messages(["Mme a appelé, Sami aussi.", "Mme encore."])
    assert conv.texts == ("PERSON_1 a appelé, PERSON_2 aussi.", "PERSON_1 encore.")


def test_values_are_linked_whatever_their_formatting() -> None:
    conv = nona().anonymize_messages(["Tél 22 123 456", "ou 22.123.456 ?", "ou plutôt 98 765 432"])
    assert conv.texts == ("Tél PHONE_1", "ou PHONE_1 ?", "ou plutôt PHONE_2")
    assert value_key(EntityType.PHONE, "22 123 456") == value_key(EntityType.PHONE, "22.123.456")


def test_numbers_written_by_the_user_are_never_issued() -> None:
    conv = nona(policies=ALL_REVERSIBLE).anonymize_messages(["Sami", "Ahmed et PERSON_2", "Amal"])
    assert conv.texts == ("PERSON_1", "PERSON_3 et PERSON_2", "PERSON_4")


def test_a_literal_placeholder_after_its_number_was_issued_round_trips() -> None:
    n = nona(policies=ALL_REVERSIBLE)
    conv = n.anonymize_messages(["Tél 22 123 456", "j'ai écrit PHONE_1 et phone 1", "PHONE_1 encore"])
    assert conv.texts == ("Tél PHONE_1", "j'ai écrit PHONE_2 et PHONE_3", "PHONE_2 encore")
    restored = [n.restore(t, conv.mapping) for t in conv.texts[1:]]
    assert restored == ["j'ai écrit PHONE_1 et phone 1", "PHONE_1 encore"]


# --------------------------------------------------------------------------- forward propagation


def test_single_word_forms_propagate_with_the_same_capitalization_only() -> None:
    conv = nona(names=["Amal"], blind_to=["plein"]).anonymize_messages(
        ["Amal arrive.", "Amal est pleine d'amal"]
    )
    assert conv.texts[1] == "PERSON_1 est pleine d'amal"


def test_multi_word_forms_propagate_on_the_normalized_text() -> None:
    conv = nona(blind_to=["relance"]).anonymize_messages(["Sami Trabelsi", "relance SAMI  TRABELSI"])
    assert conv.texts[1] == "relance PERSON_1"


def test_structured_values_propagate_too() -> None:
    class Nothing:
        name = "nothing"
        supported_types = frozenset({EntityType.PHONE})

        def detect(self, doc: NormalizedDoc) -> list[Span]:
            return []

    conv = Nona([Nothing()]).anonymize_messages(["a"])
    assert conv.texts == ("a",)  # nothing known, nothing propagated

    n = nona(blind_to=["encore"])
    assert n.anonymize_messages(["22 123 456", "encore 22 123 456"]).texts == ("PHONE_1", "encore PHONE_1")


def test_kept_and_redacted_values_are_not_propagated() -> None:
    pseudonymizer = Pseudonymizer()  # defaults: CARD redacted, ORGANIZATION kept
    text = "carte 4242 4242 4242 4242 chez Acme"
    spans = [Span(6, 25, EntityType.CARD, 1.0, "t"), Span(31, 35, EntityType.ORGANIZATION, 1.0, "t")]
    assert pseudonymizer.add(text, spans) == "carte [CARD] chez Acme"
    assert pseudonymizer.known_surfaces() == []


def test_allowlist_beats_propagation() -> None:
    conv = nona(allowlist=["Sami"]).anonymize_messages(["Sami Trabelsi", "Sami"])
    assert conv.texts == ("PERSON_1", "Sami")


def test_fails_closed_on_any_message() -> None:
    class Flaky:
        name = "flaky"
        supported_types = frozenset({EntityType.PERSON})

        def detect(self, doc: NormalizedDoc) -> list[Span]:
            if "boom" in doc.original:
                raise RuntimeError("boom")
            return []

    with pytest.raises(DetectorError):
        Nona([Flaky()]).anonymize_messages(["ok", "boom"])


def test_message_entities_list_only_their_own_mentions() -> None:
    conv = nona().anonymize_messages(["Ahmed", "Ahmed Ben Salem et Sami"])
    first, second = conv.messages
    assert [(e.placeholder, e.canonical, len(e.mentions)) for e in first.entities] == [
        ("PERSON_1", "Ahmed Ben Salem", 1)
    ]
    assert [e.placeholder for e in second.entities] == ["PERSON_1", "PERSON_2"]
    assert first.mapping is conv.mapping


# --------------------------------------------------------------------------- properties

PIECES = [
    *" ,.?\n",
    "Ahmed", "Ahmed Ben Salem", "Ahmed Karoui", "M. Ben Salem", "Sami", "Sami Trabelsi", "Amal", "amal",
    "Mme",
    "22 123 456", "22.123.456", "98 765 432", "a@example.com", "PERSON_1", "person 2", "PHONE_1", "relancé",
    "4242 4242 4242 4242",
]  # fmt: skip
message = st.lists(st.sampled_from(PIECES), max_size=8).map(" ".join)
conversation = st.lists(message, min_size=1, max_size=6)


@settings(max_examples=200, deadline=None)
@given(conversation, st.integers(min_value=0, max_value=6))
def test_prefix_stability(msgs: list[str], k: int) -> None:
    """ADR-0004: anonymize(msgs[:k]) == anonymize(msgs)[:k]. Earlier placeholders never change."""
    n = nona(blind_to=["relancé"])
    assert n.anonymize_messages(msgs[:k]).texts == n.anonymize_messages(msgs).texts[:k]


@settings(max_examples=200, deadline=None)
@given(conversation)
def test_conversation_round_trip(msgs: list[str]) -> None:
    """FR-8 at conversation level: restoring a message gives it back, with every mention written as its
    entity's canonical (longest) form, and nothing else changed."""
    n = nona(blind_to=["relancé"], policies=ALL_REVERSIBLE)
    conv = n.anonymize_messages(msgs)
    for original, result in zip(msgs, conv.messages, strict=True):
        expected, pos = [], 0
        mentions = [(m, e) for e in result.entities for m in e.mentions]
        for span, entity in sorted(mentions, key=lambda me: me[0].sort_key()):
            expected += [original[pos : span.start], entity.canonical]
            pos = span.end
        assert n.restore(result.text, conv.mapping) == "".join([*expected, original[pos:]])


@settings(max_examples=100, deadline=None)
@given(conversation)
def test_single_message_anonymize_is_the_one_message_conversation(msgs: list[str]) -> None:
    n = nona()
    for text in msgs:
        assert n.anonymize(text) == n.anonymize_messages([text]).messages[0]
