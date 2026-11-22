"""Conversation-level anonymization: causal linking and forward propagation (ADR-0004)."""

from __future__ import annotations

import itertools
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


# The structured recognizers, without the gazetteer: these tests are about linking, and the `Names` stub
# decides exactly which names are detected.
STRUCTURED = [r for r in default_recognizers() if r.name != "gazetteer"]


def nona(*, blind_to: Sequence[str] = (), names: Sequence[str] = NAMES, **kwargs: object) -> Nona:
    return Nona([*STRUCTURED, Names(names, blind_to)], **kwargs)  # type: ignore[arg-type]


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


def test_honorific_alone_is_its_own_entity_and_is_not_propagated() -> None:
    conv = nona().anonymize_messages(["Mme a appelé, Sami aussi.", "Mme encore."])
    assert conv.texts == ("PERSON_1 a appelé, PERSON_2 aussi.", "PERSON_1 encore.")
    # Not propagated: `Mme` in front of two different surnames must not make them one person.
    later = nona(names=["Mme"], blind_to=["Trabelsi"]).anonymize_messages(
        ["Mme a appelé.", "Mme Trabelsi et Mme X"]
    )
    assert later.texts[1] == "Mme Trabelsi et Mme X"


@pytest.mark.parametrize("second", ["Ahmed Karoui et Ahmed Ben Salem", "Ahmed Ben Salem et Ahmed Karoui"])
def test_an_earlier_name_two_mentions_fit_is_absorbed_by_neither(second: str) -> None:
    conv = nona().anonymize_messages(["Rappelle Ahmed.", second, "Ahmed ?"])
    assert conv.texts[1] == "PERSON_2 et PERSON_3"  # whatever the word order
    assert conv.mapping.value("PERSON_1") == "Ahmed"
    assert conv.texts[2] == "PERSON_4 ?"  # three Ahmeds now: ambiguous


def test_canonical_is_the_first_longest_form_with_spacing_collapsed() -> None:
    conv = nona(blind_to=["relance"]).anonymize_messages(["Sami Trabelsi", "relance SAMI  TRABELSI"])
    assert conv.mapping.value("PERSON_1") == "Sami Trabelsi"


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
    assert conv.messages[1].entities == ()  # in the mapping, but not personal data


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


def test_redacted_values_are_propagated_and_kept_ones_are_not() -> None:
    pseudonymizer = Pseudonymizer()  # defaults: CARD redacted, ORGANIZATION kept
    text = "carte 4242 4242 4242 4242 chez Acme"
    spans = [Span(6, 25, EntityType.CARD, 1.0, "t"), Span(31, 35, EntityType.ORGANIZATION, 1.0, "t")]
    assert pseudonymizer.add(text, spans) == "carte [CARD] chez Acme"
    assert pseudonymizer.known_surfaces() == [(EntityType.CARD, "4242 4242 4242 4242")]


def test_a_redacted_name_stays_redacted_when_the_detector_misses_it_later() -> None:
    n = nona(names=["Kenza"], blind_to=["rappelle"], policies={EntityType.PERSON: Policy.REDACT})
    assert n.anonymize_messages(["Kenza a appelé.", "Kenza rappelle demain."]).texts == (
        "[PERSON] a appelé.",
        "[PERSON] rappelle demain.",
    )


def test_a_kept_organization_does_not_hide_a_known_name() -> None:
    class Clinic:
        name = "clinic"
        supported_types = frozenset({EntityType.ORGANIZATION})

        def detect(self, doc: NormalizedDoc) -> list[Span]:
            i = doc.original.find("Clinique")
            return [Span(i, len(doc.original) - 1, EntityType.ORGANIZATION, 0.9, self.name)] if i >= 0 else []

    n = Nona([*STRUCTURED, Names(["Ahmed Ben Salem"], blind_to=["Clinique"]), Clinic()])
    conv = n.anonymize_messages(["Ahmed Ben Salem a appelé.", "Rendez-vous à la Clinique Ahmed Ben Salem."])
    assert conv.texts[1] == "Rendez-vous à la Clinique PERSON_1."
    # the same in one message: the name found there wins over the organization
    seeing = Nona([*STRUCTURED, Names(["Ahmed Ben Salem"]), Clinic()])
    one = seeing.anonymize_messages(["Ahmed Ben Salem a appelé de la Clinique Ahmed Ben Salem."])
    assert one.texts == ("PERSON_1 a appelé de la Clinique PERSON_1.",)


def test_a_typed_placeholder_glued_to_a_word_still_reserves_its_number() -> None:
    conv = nona(names=["Sami"]).anonymize_messages(["عندPERSON_1 خبر", "Sami a appelé"])
    assert conv.texts == ("عندPERSON_1 خبر", "PERSON_2 a appelé")


AFFIXES = ["", "ل", "و", "وب", "_", "'", "-", "(", "«"]
SUFFIXES = ["", "_2024", "'s", ".", ")", "»", "-"]


@pytest.mark.parametrize(
    ("value", "prefix", "suffix"),
    list(
        itertools.product(
            ["Kenza Boukadida", "كنزة", "Sami", "22 123 456", "a@example.com"], AFFIXES, SUFFIXES
        )
    ),
)
def test_a_restored_value_never_leaks_back(value: str, prefix: str, suffix: str) -> None:
    """ADR-0004: whatever the model glues to a placeholder, the restored text is masked again when re-sent."""
    n = nona(names=["Kenza Boukadida", "كنزة", "Sami"], blind_to=["merci", "شكرا"])
    first = n.anonymize_messages([f"contact : {value} ."])
    (placeholder,) = first.mapping.placeholders
    reply = n.restore(f"merci {prefix}{placeholder}{suffix} شكرا", first.mapping)
    resent = n.anonymize_messages([f"contact : {value} .", reply]).texts[1]
    assert value not in resent, (reply, resent)


# --------------------------------------------------------------------------- look-ahead inside a message


def test_a_bare_first_name_before_the_full_name_is_masked() -> None:
    conv = nona(names=["Chirine Boukadida"]).anonymize_messages(["Chirine. Chirine Boukadida travaille ici."])
    assert conv.texts == ("PERSON_1. PERSON_1 travaille ici.",)
    assert conv.mapping.value("PERSON_1") == "Chirine Boukadida"


def test_a_bare_surname_after_the_full_name_is_masked() -> None:
    conv = nona(names=["Sami Trabelsi"], blind_to=["merci"]).anonymize_messages(
        ["Sami Trabelsi a appelé, Trabelsi rappellera."]
    )
    assert conv.texts == ("PERSON_1 a appelé, PERSON_1 rappellera.",)


def test_look_ahead_skips_particle_halves_lowercase_and_short_words() -> None:
    text = "Salem et amal: Ahmed Ben Salem, et Al. amal Trabelsi."
    conv = nona(names=["Ahmed Ben Salem", "amal Trabelsi"]).anonymize_messages([text])
    assert conv.texts == ("Salem et amal: PERSON_1, et Al. PERSON_2.",)


def test_look_ahead_words_become_known_forms_for_later_messages() -> None:
    conv = nona(names=["Chirine Boukadida"], blind_to=["rappelle"]).anonymize_messages(
        ["Chirine, Chirine Boukadida.", "rappelle Chirine"]
    )
    assert conv.texts == ("PERSON_1, PERSON_1.", "rappelle PERSON_1")


def test_a_shared_first_name_found_by_look_ahead_is_ambiguous() -> None:
    conv = nona(names=["Ahmed Karoui", "Ahmed Jlassi"]).anonymize_messages(
        ["Ahmed Karoui et Ahmed Jlassi, puis Ahmed."]
    )
    assert conv.texts == ("PERSON_1 et PERSON_2, puis PERSON_3.",)


def test_look_ahead_leaves_names_that_are_ordinary_words_to_context() -> None:
    # `Mabrouk` is "congratulations" in Darija: only the gazetteer's context rules may decide on it alone.
    conv = nona(names=["Mabrouk Jlassi"]).anonymize_messages(["Mabrouk ! Mabrouk Jlassi a eu son bac."])
    assert conv.texts == ("Mabrouk ! PERSON_1 a eu son bac.",)


def test_look_ahead_ignores_weak_person_spans() -> None:
    class Weak:
        name = "weak"
        supported_types = frozenset({EntityType.PERSON})

        def detect(self, doc: NormalizedDoc) -> list[Span]:
            i = doc.original.find("Bonne Route")
            return [Span(i, i + 11, EntityType.PERSON, 0.55, self.name)] if i >= 0 else []

    conv = Nona([Weak()]).anonymize_messages(["Bonne Route ! Route barrée.", "Route libre."])
    assert conv.texts == ("PERSON_1 ! Route barrée.", "Route libre.")


def test_look_ahead_does_not_retype_a_detected_span() -> None:
    class Place:
        name = "place"
        supported_types = frozenset({EntityType.LOCATION})

        def detect(self, doc: NormalizedDoc) -> list[Span]:
            return [
                Span(m.start(), m.end(), EntityType.LOCATION, 0.8, self.name)
                for m in re.finditer("Gabes", doc.original)
            ]

    n = Nona([Place(), Names(["Sami Gabes"])], policies=ALL_REVERSIBLE)
    assert n.anonymize("Sami Gabes habite Gabes.").text == "PERSON_1 habite LOCATION_1."


def test_allowlist_beats_look_ahead() -> None:
    conv = nona(names=["Chirine Boukadida"], allowlist=["Chirine"]).anonymize_messages(
        ["Chirine. Chirine Boukadida."]
    )
    assert conv.texts == ("Chirine. PERSON_1.",)


# --------------------------------------------------------------------------- owners (deep dive B, step 6)


def _owners(conv: object) -> dict[str, str | None]:
    return {e.placeholder: e.owner for m in conv.messages for e in m.entities if e.type is EntityType.EMAIL}  # type: ignore[attr-defined]


def test_an_email_belongs_to_the_person_it_spells() -> None:
    conv = nona().anonymize_messages(["Ahmed Ben Salem a envoyé le devis, son mail ahmed.bs@gmail.com"])
    assert conv.texts == ("PERSON_1 a envoyé le devis, son mail EMAIL_1",)  # metadata only
    assert _owners(conv) == {"EMAIL_1": "PERSON_1"}


def test_an_email_given_before_the_name_still_links() -> None:
    conv = nona().anonymize_messages(["écris à s.trabelsi@example.com", "c'est Sami Trabelsi"])
    assert _owners(conv) == {"EMAIL_1": "PERSON_1"}


def test_two_candidate_people_mean_no_owner() -> None:
    conv = nona().anonymize_messages(["Ahmed Ben Salem et Ahmed Karoui : ahmed@example.com"])
    assert _owners(conv) == {"EMAIL_1": None}


def test_an_ambiguous_person_owns_nothing() -> None:
    conv = nona(names=["Ahmed Karoui", "Ahmed Jlassi", "Ahmed"]).anonymize_messages(
        ["Ahmed Karoui, Ahmed Jlassi.", "Ahmed : ahmed@example.com"]
    )
    assert conv.texts[1] == "PERSON_3 : EMAIL_1"
    assert _owners(conv) == {"EMAIL_1": None}


def test_an_owner_never_changes_once_given() -> None:
    msgs = ["Ahmed Ben Salem : ahmed@example.com", "et Ahmed Karoui"]
    n = nona()
    assert _owners(n.anonymize_messages(msgs[:1])) == {"EMAIL_1": "PERSON_1"}
    assert _owners(n.anonymize_messages(msgs)) == {"EMAIL_1": "PERSON_1"}  # a second Ahmed came too late


def test_a_superset_naming_someone_else_has_no_owner() -> None:
    conv = nona(names=["Sami", "Ahmed Karoui"]).anonymize_messages(
        ["Sami et Ahmed Karoui : sami_ahmed@example.com"]
    )
    assert _owners(conv) == {"EMAIL_1": None}
    conv = nona(names=["Wassim"]).anonymize_messages(["Wassim : wassim.ferchichi@example.com"])
    assert _owners(conv) == {"EMAIL_1": "PERSON_1"}


def test_a_surname_written_without_its_particle_links() -> None:
    conv = nona().anonymize_messages(["Ahmed Ben Salem : ahmed.salem@example.com"])
    assert _owners(conv) == {"EMAIL_1": "PERSON_1"}


def test_an_unrelated_email_has_no_owner_and_other_types_never_do() -> None:
    conv = nona().anonymize_messages(["Sami Trabelsi : contact@example.com, 22 123 456"])
    assert _owners(conv) == {"EMAIL_1": None}
    assert all(e.owner is None for e in conv.messages[0].entities)


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
    "Mme", "Karoui", "Trabelsi", "Salem",
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


# --------------------------------------------------------------------------- S3 review, linking bugs (#55)


def _mapping(conv: object) -> dict[str, str]:
    m = conv.mapping  # type: ignore[attr-defined]
    return {p: m.value(p) for p in m.placeholders}


def test_naming_a_person_again_in_pieces_does_not_split_them() -> None:
    # `Sami` and `M. Trabelsi` add nothing to {sami, trabelsi}: they don't contest it.
    conv = Nona().anonymize_messages(
        ["Sami Trabelsi a appelé.", "Sami Trabelsi ? Oui, Sami. M. Trabelsi rappellera demain."]
    )
    assert conv.texts == ("PERSON_1 a appelé.", "PERSON_1 ? Oui, PERSON_1. M. PERSON_1 rappellera demain.")
    assert _mapping(conv) == {"PERSON_1": "Sami Trabelsi"}


def test_an_unlisted_word_of_a_gazetteer_run_does_not_spread() -> None:
    conv = Nona().anonymize_messages(
        ["Merci Sami Bonne journée, Bonne chance !", "Bonne nouvelle : le dossier est prêt."]
    )
    assert conv.texts == ("Merci PERSON_1 journée, Bonne chance !", "Bonne nouvelle : le dossier est prêt.")
    assert Nona.restore(conv.texts[1], conv.mapping) == "Bonne nouvelle : le dossier est prêt."


def test_a_run_after_a_strong_cue_spreads_its_unlisted_words() -> None:
    conv = Nona().anonymize_messages(["je m'appelle Kenza Haddad. Haddad, c'est mon nom.", "Haddad Kenza"])
    assert conv.texts[0] == "je m'appelle PERSON_1. PERSON_1, c'est mon nom."


class _Fixed:
    """Detects one name at a fixed score."""

    supported_types = frozenset({EntityType.PERSON})

    def __init__(self, text: str, score: float) -> None:
        self.name = f"fixed-{score}"
        self._text, self._score = text, score

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        return [
            Span(m.start(), m.end(), EntityType.PERSON, self._score, self.name)
            for m in re.finditer(re.escape(self._text), doc.original)
        ]


def test_a_weak_span_does_not_spread_once_it_is_propagated() -> None:
    msgs = ["Kenza Haddad a écrit.", "Kenza Haddad a rappelé. Haddad attend."]
    weak = Nona([_Fixed("Kenza Haddad", 0.55)]).anonymize_messages(msgs)  # type: ignore[list-item]
    assert weak.texts == ("PERSON_1 a écrit.", "PERSON_1 a rappelé. Haddad attend.")
    # A confident detection's words do spread when its form comes back
    strong = Nona([_Fixed("Kenza Haddad", 0.9)]).anonymize_messages(msgs[:1])  # type: ignore[list-item]
    blind = Nona([_Fixed("Kenza Haddad", 0.9)]).anonymize_messages(msgs)  # type: ignore[list-item]
    assert strong.texts == ("PERSON_1 a écrit.",)
    assert blind.texts == ("PERSON_1 a écrit.", "PERSON_1 a rappelé. PERSON_1 attend.")


def test_a_propagated_form_spreads_only_its_seeds() -> None:
    # `Sami Bonne` comes back in message 2 as a known form; `Bonne` was never vouched for.
    conv = Nona().anonymize_messages(["Merci Sami Bonne journée.", "Sami Bonne, Bonne chance."])
    assert conv.texts == ("Merci PERSON_1 journée.", "PERSON_1, Bonne chance.")


def test_a_typed_placeholder_is_never_swallowed_into_a_name() -> None:
    conv = Nona().anonymize_messages(["Sami a appelé.", "Écris à Sami Trabelsi PERSON_1 stp."])
    assert conv.texts == ("PERSON_1 a appelé.", "Écris à PERSON_1 PERSON_2 stp.")
    assert _mapping(conv) == {"PERSON_1": "Sami Trabelsi", "PERSON_2": "PERSON_1"}
    assert Nona.restore(conv.texts[0], conv.mapping) == "Sami Trabelsi a appelé."
    assert Nona.restore(conv.texts[1], conv.mapping) == "Écris à Sami Trabelsi PERSON_1 stp."


def test_a_superset_owner_is_dropped_when_the_person_turns_out_to_be_someone_else() -> None:
    msgs = ["Ahmed : ahmed.bensalem@gmail.com", "Ahmed Karoui a signé."]
    n = nona(names=["Ahmed", "Ahmed Karoui"])
    assert _owners(n.anonymize_messages(msgs[:1])) == {"EMAIL_1": "PERSON_1"}  # provisional
    assert _owners(n.anonymize_messages(msgs)) == {"EMAIL_1": None}
    assert _owners(n.anonymize_messages(msgs[::-1])) == {"EMAIL_1": None}  # the order doesn't matter


def test_a_superset_owner_that_still_fits_stays() -> None:
    conv = nona(names=["Wassim"]).anonymize_messages(["Wassim : wassim.ferchichi@example.com", "Wassim ok"])
    assert _owners(conv) == {"EMAIL_1": "PERSON_1"}


def test_one_value_in_two_digit_scripts_is_one_entity() -> None:
    assert value_key(EntityType.PHONE, "22123456") == value_key(EntityType.PHONE, "٢٢١٢٣٤٥٦")
    conv = Nona().anonymize_messages(["Appelle 22123456.", "Ou ٢٢١٢٣٤٥٦."])
    assert conv.texts == ("Appelle PHONE_1.", "Ou PHONE_1.")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Yassine Trabelsi", "Yacine Trabelsi"),
        ("Cyrine Gharbi", "Syrine Gharbi"),
        ("Karim Jlassi", "Karim Jelassi"),
    ],
)
def test_spelling_variants_link(first: str, second: str) -> None:
    conv = Nona().anonymize_messages([f"{first} a appelé.", f"{second} rappellera."])
    assert conv.texts == ("PERSON_1 a appelé.", "PERSON_1 rappellera.")


GAZETTEER_PIECES = [
    *" ,.?\n",
    "Merci", "Sami", "Sami Trabelsi", "Trabelsi", "M.", "Bonne", "journée", "Yacine", "Yassine", "Ahmed",
    "Karoui", "je m'appelle", "Kenza", "Haddad", "22123456", "٢٢١٢٣٤٥٦", "PERSON_1", "PERSON_2", "كريم",
    "ahmed.bensalem@gmail.com",
]  # fmt: skip
gazetteer_conversation = st.lists(
    st.lists(st.sampled_from(GAZETTEER_PIECES), max_size=8).map(" ".join), min_size=1, max_size=5
)


@settings(max_examples=150, deadline=None)
@given(gazetteer_conversation, st.integers(min_value=0, max_value=5))
def test_prefix_stability_with_the_default_recognizers(msgs: list[str], k: int) -> None:
    """Seeds carry state across messages too: earlier placeholders still never change."""
    n = Nona()
    assert n.anonymize_messages(msgs[:k]).texts == n.anonymize_messages(msgs).texts[:k]


@settings(max_examples=150, deadline=None)
@given(gazetteer_conversation)
def test_round_trip_with_the_default_recognizers(msgs: list[str]) -> None:
    n = Nona(policies=ALL_REVERSIBLE)
    conv = n.anonymize_messages(msgs)
    for original, result in zip(msgs, conv.messages, strict=True):
        expected, pos = [], 0
        mentions = [(m, e) for e in result.entities for m in e.mentions]
        for span, entity in sorted(mentions, key=lambda me: me[0].sort_key()):
            expected += [original[pos : span.start], entity.canonical]
            pos = span.end
        assert n.restore(result.text, conv.mapping) == "".join([*expected, original[pos:]])


@pytest.mark.parametrize("text", ["كريم وصل", "خالد وصل", "شفت ريم اليوم"])
def test_an_arabic_name_keeps_its_own_common_word_status(text: str) -> None:
    # `كريم` now links to `Karim`, which is on the common-word list for its Darija meaning in Latin letters.
    # That must not change how the Arabic word is detected: the list decides for Arabic forms separately.
    assert Nona().anonymize(text).text.startswith(("PERSON_1", "شفت PERSON_1"))


def test_an_arabic_name_links_to_its_latin_form() -> None:
    conv = Nona().anonymize_messages(["كريم الطرابلسي وصل", "Karim Trabelsi a confirmé."])
    assert conv.texts == ("PERSON_1 وصل", "PERSON_1 a confirmé.")
