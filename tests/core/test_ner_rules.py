"""NER quality rules (#72 part 2, ADR-0008 conditions 1 and 2): the de-Arabizi'd view, span cleaning, types,
cue-backed low scores, and the conversation-level corrections in the pipeline. With a fake model."""

from __future__ import annotations

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nona.core import EntityType, Nona, normalize
from nona.core.canonical import ABD_PREFIXES, PARTICLES, canonical_tokens
from nona.core.recognizers import NerPrediction, NerRecognizer, default_recognizers
from nona.core.recognizers.gazetteer import NAMES
from nona.core.recognizers.gazetteer_data import NEVER_NAMES, SPEECH_VERBS
from nona.core.recognizers.ner import arabizi, arabizi_cuts, clean, model_view, never_name
from nona.core.resolver import resolve
from nona.core.types import Span
from tests.core.test_ner import FakeModel


def fake(names: dict[str, tuple[str, float]]) -> FakeModel:
    """A fake model keyed by surfaces as written: it receives the de-Arabizi'd view, so keys are converted."""
    return FakeModel({model_view(k): v for k, v in names.items()})


def detect(names: dict[str, tuple[str, float]], text: str) -> list[tuple[str, EntityType, float]]:
    spans = NerRecognizer(fake(names)).detect(normalize(text))
    return [(text[s.start : s.end], s.type, s.score) for s in spans]


def pipeline(names: dict[str, tuple[str, float]]) -> Nona:
    return Nona([*default_recognizers(), NerRecognizer(fake(names))])


def per_message(found: dict[str, list[tuple[str, str, float]]]) -> Nona:
    """A pipeline whose fake model gives, for each message, the listed (surface, label, score) predictions."""

    def predict(view: str) -> list[NerPrediction]:
        text = next(t for t in found if model_view(t) == view)
        return [NerPrediction(text.index(w), text.index(w) + len(w), lab, sc) for w, lab, sc in found[text]]

    return Nona([*default_recognizers(), NerRecognizer(FakeModel(predict=predict))])


# --------------------------------------------------------------------------- the de-Arabizi'd view


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("3andi", True), ("m3a", True), ("9alli", True), ("y5allik", True), ("sa7a", True), ("5ti", True),
        ("15h", False), ("3h", False), ("hamza23", False), ("4G", False), ("22", False), ("Sami", False),
        ("b2", False), ("3", False), ("سامي", False), ("ab3athli", True), ("B2B", False),
    ],
)  # fmt: skip
def test_arabizi_words(word: str, expected: bool) -> None:
    assert arabizi(word) is expected


def test_the_model_reads_the_de_arabizi_view() -> None:
    model = FakeModel()
    NerRecognizer(model).detect(normalize("3andi rdv m3a Sami à 15h, nomrou 22 123 456"))
    assert model.calls == ["aandi rdv maa Sami à 15h, nomrou 22 123 456"]


@given(st.text(alphabet="ab 37925h1-.'سا", max_size=40))
def test_the_view_keeps_every_offset(text: str) -> None:
    view = model_view(text)
    assert len(view) == len(text)
    assert all(v == t or t in "23579" for v, t in zip(view, text, strict=True))


# --------------------------------------------------------------------------- cleaning


@pytest.mark.parametrize(
    "word",
    ["I", "Je", "Il", "nomrou", "Behi", "باهي", "نومرو", "9alli", "y3aychek", "chnowa", "بالسيدة", "Mme"],
)
def test_words_that_are_never_names(word: str) -> None:
    assert never_name(word)


@pytest.mark.parametrize(
    "word",
    ["Sami", "Amal", "7amza", "mo7amed", "سامي", "Montaser", "Ben", "زكرياء", "Wala", "Bahi", "يسر", "Rani"],
)
def test_names_are_not_never_names(word: str) -> None:
    assert not never_name(word)


def test_a_place_keeps_its_honorific() -> None:
    assert never_name("Dr")
    assert not never_name("Dr", honorific=False)


def test_no_never_name_word_is_a_listed_name() -> None:
    listed = [w for w in NEVER_NAMES if canonical_tokens(w) and canonical_tokens(w) <= NAMES]
    assert listed == []


@pytest.mark.parametrize(
    ("span", "pieces"),
    [
        ("Chnowa el mail mta3 Hamza", ["Hamza"]),
        ("Aymen 9alli", ["Aymen"]),
        ("rabbi y5allik", []),
        ("I", []),
        ("Hamza el", ["Hamza"]),
        ("Ahmed Ben Salem", ["Ahmed Ben Salem"]),
        ("montaser rajel behi", ["montaser"]),
        ("نومرو زكرياء", ["زكرياء"]),
        ("موعد سامي", ["سامي"]),
        ("بالسيدة الشتيوي", ["الشتيوي"]),
        ("وسامي", ["سامي"]),  # a listed name behind a clitic
        ("وَسامي", ["سامي"]),  # the clitic carries a mark
        ("وليد", ["وليد"]),  # a listed name that starts like a clitic
        ("لزكرياء", ["لزكرياء"]),  # unlisted: the pipeline aligns it with a known form
        ("Mme Islem", ["Islem"]),
        ("Sami et Ines", ["Sami", "Ines"]),
    ],
)
def test_clean(span: str, pieces: list[str]) -> None:
    assert [span[a:b] for a, b in clean(span, 0, len(span))] == pieces


# --------------------------------------------------------------------------- types


@pytest.mark.parametrize(
    ("text", "predicted", "label", "found", "etype"),
    [
        ("3andi rdv m3a Bilel Gharbi ghodwa", "Bilel Gharbi ghodwa", "location", "Bilel Gharbi", "PERSON"),
        ("Booked Bilel Sassi on Monday", "Bilel Sassi", "organization", "Bilel Sassi", "PERSON"),
        ("rendez-vous m3a Hazem Drira lundi", "Hazem Drira", "location", "Hazem Drira", "PERSON"),
        ("a meeting with Hazem Drira", "Hazem Drira", "location", "Hazem Drira", "PERSON"),
        ("ena fi Sfax", "Sfax", "location", "Sfax", "LOCATION"),
        ("nemchi l sousse ghodwa", "sousse", "location", "sousse", "LOCATION"),  # a listed city, lowercase
        ("rendez-vous m3a Ooredoo", "Ooredoo", "organization", "Ooredoo", "ORGANIZATION"),
        ("rue Mohamed Ali", "Mohamed Ali", "location", "Mohamed Ali", "LOCATION"),
        ("habite à Sbeitla", "Sbeitla", "location", "Sbeitla", "LOCATION"),
        ("3andi Amal Bank", "Amal Bank", "organization", "Amal Bank", "ORGANIZATION"),  # a common word
    ],
)
def test_types(text: str, predicted: str, label: str, found: str, etype: str) -> None:
    assert detect({predicted: (label, 0.8)}, text) == [(found, EntityType(etype), 0.8)]


@pytest.mark.parametrize("text", ["sbe7 el jardin", "nomrou fel commune", "ykhdmou 24/24 fel bureau"])
def test_a_lowercase_location_is_dropped_unless_listed(text: str) -> None:
    words = text.split()
    assert detect({w: ("location", 0.8) for w in words if len(w) > 3}, text) == []


# --------------------------------------------------------------------------- cue-backed low scores


@pytest.mark.parametrize(
    ("text", "predicted", "found"),
    [
        ("Hazem 9alli bech yjib el contrat.", "Hazem", "Hazem"),  # a speech verb after
        ("Si Aymen kallamni el bera7.", "Aymen", "Aymen"),
        ("سأتصل بالسيدة الغرياني غدا", "الغرياني", "الغرياني"),  # an honorific behind a clitic
        ("Bonjour Chaima, merci", "Chaima", "Chaima"),
        ("3andi rdv m3a Mootez", "Mootez", "Mootez"),
    ],
)
def test_a_low_score_needs_a_cue(text: str, predicted: str, found: str) -> None:
    assert detect({predicted: ("location", 0.3)}, text) == [(found, EntityType.PERSON, 0.5)]


@pytest.mark.parametrize(
    ("text", "predicted"),
    [
        ("Hazem bech yjib el contrat.", "Hazem"),  # no cue
        ("Kallem el service client", "el service client"),  # lowercase: not name-shaped
        ("m3a montaser ghodwa", "montaser"),
        ("Fhemtek. Nchoufou", "Fhemtek"),
    ],
)
def test_a_low_score_without_cue_or_capital_is_dropped(text: str, predicted: str) -> None:
    assert detect({predicted: ("person", 0.3)}, text) == []


def test_a_cue_joins_the_pieces_of_one_name() -> None:
    text = "3andi rendez-vous m3a Mootez Jaouadi ghodwa"
    found = detect({"Mootez": ("location", 0.25), "Jaouadi": ("location", 0.42)}, text)
    assert found == [("Mootez Jaouadi", EntityType.PERSON, 0.5)]
    assert detect({"Jaouadi": ("location", 0.42)}, text) == []  # an uncued piece alone


def test_below_the_cued_threshold_nothing_is_kept() -> None:
    assert detect({"Hazem": ("person", 0.1)}, "Hazem 9alli") == []
    with pytest.raises(ValueError, match="cued_threshold"):
        NerRecognizer(FakeModel(), threshold=0.5, cued_threshold=0.6)


def test_the_fingerprint_covers_the_rules() -> None:
    a = NerRecognizer(FakeModel()).fingerprint
    assert "rules=" in a
    assert a != NerRecognizer(FakeModel(), cued_threshold=0.3).fingerprint


# --------------------------------------------------------------------------- conversation (pipeline)


def test_a_place_that_fits_a_known_person_is_that_person() -> None:
    nona = pipeline({"Bacem": ("person", 0.9), "Bacem Hachicha": ("location", 0.7)})
    conv = nona.anonymize_messages(["Rappelle Bacem demain.", "Oui, c'est Bacem Hachicha, au bureau."])
    assert conv.texts == ("Rappelle PERSON_1 demain.", "Oui, c'est PERSON_1, au bureau.")


def test_a_place_listed_with_a_person_is_a_person() -> None:
    nona = pipeline({"Hazem Boukadida": ("person", 0.9), "Hazem Drira": ("location", 0.7)})
    conv = nona.anonymize_messages(["3andi rdv m3a Hazem Boukadida w Hazem Drira lundi."])
    assert conv.texts == ("3andi rdv m3a PERSON_1 w PERSON_2 lundi.",)


def test_an_organization_becomes_a_person_only_when_every_token_is_known() -> None:
    nona = per_message(
        {
            "Appelle Oussema Sellami.": [("Oussema Sellami", "person", 0.9)],
            "Booked Oussema Sellami.": [("Oussema Sellami", "organization", 0.7)],
            "à la Clinique Oussema Sellami": [("Clinique Oussema Sellami", "organization", 0.9)],
        }
    )
    texts = nona.anonymize_messages(
        ["Appelle Oussema Sellami.", "Booked Oussema Sellami.", "à la Clinique Oussema Sellami"]
    ).texts
    assert texts[1] == "Booked PERSON_1."
    assert texts[2] == "à la Clinique PERSON_1"  # still an organization; the known person in it is masked


def test_a_listed_city_never_becomes_a_person() -> None:
    nona = pipeline({"Sami Gabes": ("person", 0.9), "Gabès": ("location", 0.9)})
    conv = nona.anonymize_messages(["Sami Gabes habite à Gabès."])
    assert conv.texts == ("PERSON_1 habite à LOCATION_1.",)


def test_a_known_name_behind_a_clitic_links_to_it() -> None:
    nona = pipeline({"زكرياء": ("person", 0.9), "لزكرياء": ("person", 0.9)})
    conv = nona.anonymize_messages(["كلّم زكرياء غدوة", "وقلّو لزكرياء الاجتماع تأخّر"])
    assert conv.texts == ("كلّم PERSON_1 غدوة", "وقلّو لPERSON_1 الاجتماع تأخّر")
    assert conv.mapping.value("PERSON_1") == "زكرياء"


def test_a_name_keeps_its_first_placeholder_whatever_the_model_says_later() -> None:
    nona = per_message(
        {
            "Chiheb Zghal w Chiheb Haddar": [
                ("Chiheb Zghal", "location", 0.8), ("Chiheb Haddar", "location", 0.8)
            ],
            "Lequel, Chiheb Zghal ?": [("Chiheb Zghal", "person", 0.9)],
        }
    )  # fmt: skip
    conv = nona.anonymize_messages(["Chiheb Zghal w Chiheb Haddar", "Lequel, Chiheb Zghal ?"])
    assert conv.texts == ("LOCATION_1 w LOCATION_2", "Lequel, LOCATION_1 ?")


def test_the_resolver_keeps_a_propagated_name_over_another_type_at_the_same_place() -> None:
    known = Span(0, 12, EntityType.LOCATION, 1.0, "propagation")
    model = Span(0, 12, EntityType.PERSON, 0.9, "ner")
    assert resolve([model, known], "Chiheb Zghal") == [known]
    phone = Span(0, 8, EntityType.PHONE, 1.0, "propagation")  # structured values keep the priority rule
    cin = Span(0, 8, EntityType.NATIONAL_ID, 0.9, "cin")
    assert resolve([phone, cin], "12345678") == [cin]


# --------------------------------------------------------------------------- review findings (PR #79)


def test_a_name_scored_low_in_one_window_and_high_in_another_is_kept() -> None:
    filler = " ".join(f"mot{i}" for i in range(20))
    text = f"{filler} Karim Jlassi {filler}"
    scores = iter([0.4, 0.9, 0.9, 0.9])

    def predict(view: str) -> list[NerPrediction]:
        i = view.find("Karim Jlassi")
        return [] if i < 0 else [NerPrediction(i, i + 12, "person", next(scores))]

    spans = NerRecognizer(FakeModel(predict=predict, max_words=30), overlap_words=10).detect(normalize(text))
    assert [(text[s.start : s.end], s.score) for s in spans] == [("Karim Jlassi", 0.9)]


@pytest.mark.parametrize(
    ("text", "predicted", "label"),
    [
        ("J'habite rue Mohamed Ali.", "rue Mohamed Ali", "location"),
        ("rdv à la Clinique Ahmed Ben Salem", "Clinique Ahmed Ben Salem", "organization"),
        ("rdv à la Clinique Ahmed Ben Salem", "Ahmed Ben Salem", "organization"),
        ("نسكن في شارع محمد علي", "شارع محمد علي", "location"),
    ],
)
def test_a_place_or_organization_named_after_someone_keeps_its_type(
    text: str, predicted: str, label: str
) -> None:
    assert detect({predicted: (label, 0.9)}, text) == [(predicted, EntityType(label.upper()), 0.9)]


def test_a_street_is_not_merged_with_a_person_of_the_same_name() -> None:
    nona = per_message(
        {
            "Mohamed passe demain.": [("Mohamed", "person", 0.9)],
            "J'habite rue Mohamed Ali.": [("rue Mohamed Ali", "location", 0.9)],
        }
    )
    texts = nona.anonymize_messages(["Mohamed passe demain.", "J'habite rue Mohamed Ali."]).texts
    assert texts[1] == "J'habite LOCATION_1."


@pytest.mark.parametrize(
    ("text", "predicted", "found"),
    [
        ("3andi rdv m3a Mootez Jaouadi", "m3a Mootez Jaouadi", "Mootez Jaouadi"),  # the cue inside the span
        ("demain Mme Islem passe", "Mme Islem", "Islem"),  # a leading honorific
    ],
)
def test_a_cue_inside_a_place_span_makes_it_a_person(text: str, predicted: str, found: str) -> None:
    assert detect({predicted: ("location", 0.9)}, text) == [(found, EntityType.PERSON, 0.9)]


@pytest.mark.parametrize(
    ("text", "predicted"),
    [
        ("rendez-vous avec Orange demain", "Orange"),
        ("موعد مع الشركة", "الشركة"),
        ("avec la Clinique Sami", "Sami"),
    ],
)
def test_a_cue_does_not_make_a_low_organization_a_person(text: str, predicted: str) -> None:
    assert detect({predicted: ("organization", 0.3)}, text) == []


def test_a_cue_does_not_cross_a_line() -> None:
    assert detect({"Yassine": ("person", 0.3)}, "Merci Dr\nYassine passe demain") == []
    assert detect({"Yassine": ("person", 0.3)}, "Merci Dr Yassine") == [("Yassine", EntityType.PERSON, 0.5)]


def test_a_denylist_term_keeps_its_policy_over_a_propagated_name() -> None:
    from nona.core.pseudonymize import Policy

    known = Span(0, 4, EntityType.PERSON, 1.0, "propagation")
    denied = Span(0, 4, EntityType.CUSTOM, 1.0, "denylist")
    assert resolve([known, denied], "Sami") == [denied]
    nona = Nona(
        denylist=["Sami"], policies={EntityType.CUSTOM: Policy.REDACT, EntityType.PERSON: Policy.PSEUDONYMIZE}
    )
    texts = nona.anonymize_messages(["Bonjour Sami Trabelsi", "Sami a appelé"]).texts
    assert "Sami" not in texts[1]
    assert "PERSON" not in texts[1]


# --------------------------------------------------------------------------- security review (PR #79)
# Cleaning cuts a span at Arabizi words for its rules, but an Arabizi word may be a name: what is cut out is
# masked again, each word on its own. Only the closed lists ever unmask a word.


@pytest.mark.parametrize(
    ("text", "predicted", "found"),
    [
        ("Mo3ez passe demain", "Mo3ez", ["Mo3ez"]),
        ("Mo7sen kallamni el bera7", "Mo7sen kallamni", ["Mo7sen"]),
        ("3li bech yji", "3li", ["3li"]),
        ("Na3ima Ka3bi", "Na3ima Ka3bi", ["Na3ima", "Ka3bi"]),
        ("Sa3ida Bel7aj fil bureau", "Sa3ida Bel7aj", ["Sa3ida", "Bel7aj"]),
        ("Rdv m3a na3ima ka3bi Ben Salah", "na3ima ka3bi Ben Salah", ["na3ima", "ka3bi", "Ben Salah"]),
        ("Sami y7eb Ali", "Sami y7eb Ali", ["Sami", "Ali"]),  # two people kept apart; `y7eb` is listed
        ("Sami y7ebha Ali", "Sami y7ebha Ali", ["Sami", "y7ebha", "Ali"]),  # unlisted: masked on its own
    ],
)
def test_a_confident_word_with_digits_is_never_unmasked(text: str, predicted: str, found: list[str]) -> None:
    assert [f for f, _, _ in detect({predicted: ("person", 0.95)}, text)] == found


def test_a_digit_name_is_masked_through_the_pipeline() -> None:
    nona = pipeline({"Na3ima Ka3bi": ("person", 0.95)})
    # One span since #50 lists `Naima`: the gazetteer takes `Na3ima Ka3bi` as one name after `m3a`
    assert nona.anonymize_messages(["Rdv m3a Na3ima Ka3bi ghodwa"]).texts == ("Rdv m3a PERSON_1 ghodwa",)


def test_a_known_name_written_later_with_a_digit_is_not_sent_in_clear() -> None:
    nona = per_message(
        {
            "Mohsen Trabelsi called": [("Mohsen Trabelsi", "person", 0.95)],
            "Mo7sen 9alli barcha": [("Mo7sen", "person", 0.95)],
        }
    )
    assert "Mo7sen" not in nona.anonymize_messages(["Mohsen Trabelsi called", "Mo7sen 9alli barcha"]).texts[1]


@pytest.mark.parametrize(
    ("text", "predicted", "found"),
    [
        # What the real model gives: a place at 0.29 and 0.31, backed by the speech verb and the cue.
        ("Mo7sen 9alli barcha", "Mo7sen 9alli barcha", ["Mo7sen"]),
        ("Rdv m3a Na3ima Ka3bi ghodwa", "Na3ima Ka3bi ghodwa", ["Na3ima", "Ka3bi"]),  # one word at a time
        ("Rdv m3a 3li ghodwa", "3li ghodwa", ["3li"]),
    ],
)
def test_a_cued_low_name_with_digits_is_masked(text: str, predicted: str, found: list[str]) -> None:
    assert [f for f, _, _ in detect({predicted: ("location", 0.3)}, text)] == found


@pytest.mark.parametrize(
    ("text", "predicted"),
    [
        ("ok mo3ez", "mo3ez"),  # lowercase and no cue: not name-shaped
        ("Mo3ez passe", "Mo3ez"),  # no cue
        ("m3a Sfax", "Sfax"),
        ("m3a Pharmacie Mo3ez", "Pharmacie Mo3ez"),  # named after a head word: a place
    ],
)
def test_a_low_word_with_digits_needs_a_cue_and_a_name_shape(text: str, predicted: str) -> None:
    assert detect({predicted: ("person", 0.3)}, text) == []


def test_an_ordinal_is_cut_but_not_masked_again() -> None:
    assert arabizi_cuts("Sami 3eme 2nd Mo3ez", 0, 19) == [(14, 19)]
    assert [f for f, _, _ in detect({"Sami 3eme": ("person", 0.9)}, "Sami 3eme etage")] == ["Sami"]


_LIST_WORDS = ["Mme", "Dr", "بالسيدة", "I", "Je", "el", "nomrou", "Behi", "9alli", "mta3", "y3aychek", "باهي"]
_OPEN_WORDS = ["Mo3ez", "Mo7sen", "Na3ima", "Ka3bi", "Bel7aj", "3li", "yfar7ek", "7attli", "Zorgati", "mo3ez"]
_PARTICLES = PARTICLES | ABD_PREFIXES


_WORDS = [*_LIST_WORDS, *_OPEN_WORDS, "Sami", "Hamza", "سامي", "Ben", "Sfax", "Clinique", "m3a", "kallemni"]


@given(st.lists(st.sampled_from(_WORDS), min_size=1, max_size=6), st.sampled_from([0.95, 0.6]))
def test_only_closed_list_words_of_a_confident_person_are_unmasked(words: list[str], score: float) -> None:
    """The security invariant, through the whole pipeline: every word of a confident PERSON span stays
    masked, unless a closed list says it is never a name or it is a trailing particle. The gazetteer and the
    other recognizers run too, so the resolver's choices are covered."""
    text = " ".join(words)
    covered = {i for s in pipeline({text: ("person", score)}).detect(text) for i in range(s.start, s.end)}
    for m in re.finditer(r"\S+", text):
        if not any(i in covered for i in range(m.start(), m.end())):
            word = m.group()
            assert never_name(word) or normalize(word).text in _PARTICLES, (text, word)


@pytest.mark.parametrize("first", ["Zorgati", "Clinique"])
def test_a_confident_name_partly_inside_a_longer_gazetteer_name_stays_masked(first: str) -> None:
    """#89, found by the property above: the gazetteer's `Sami Mo3ez Sami` is longer than the model's
    `Zorgati Sami`, and the resolver dropped the model's span whole."""
    text = f"{first} Sami Mo3ez Sami"
    out = pipeline({text: ("person", 0.95)}).anonymize_messages([text]).texts[0]
    assert first not in out
    assert "Sami" not in out


def test_a_losing_organization_never_hides_a_person_it_holds() -> None:
    """Final review of #91: an ORGANIZATION piece (kept as is) swallowed the person `Ahmed`."""
    text = "Pharmacie Ahmed, Sami Ben Ali Trabelsi"
    out = per_message({text: [("Pharmacie Ahmed, Sami", "organization", 0.95)]}).anonymize_messages([text])
    assert "Ahmed" not in out.texts[0]


@pytest.mark.parametrize("symbol", ["№", "™"])
def test_a_piece_next_to_a_symbol_that_normalizes_to_letters_restores(symbol: str) -> None:
    """Final review of #91: `№` is `No` on the normalized view, so a piece glued to it never restored."""
    text = f"voir {symbol}Kousksou Farfoura Hlaiem Zgolli Bouslimi Ghariani"
    found = {
        text: [
            ("Kousksou Farfoura Hlaiem", "person", 0.95),
            ("Hlaiem Zgolli Bouslimi Ghariani", "person", 0.95),
        ]
    }
    conv = per_message(found).anonymize_messages([text])
    assert Nona.restore(conv.texts[0], conv.mapping) == text
    assert "Hlaiem" not in conv.texts[0]


@pytest.mark.parametrize(
    "found",
    [
        # A piece of a particle (`de`) spread to every later `de` and linked into `Ines de Montfort`.
        {
            "Ines de Montfort": [("Ines de Montfort", "person", 0.95)],
            "merci de Sami Ben Ali Trabelsi": [("de Sami", "person", 0.9)],
            "merci de rien": [],
        },
        # A piece kept the full name's entity: once `Zorgati Sami` was written, `Sami` came back twice.
        {
            "Zorgati Sami Mo3ez Sami": [("Zorgati Sami", "person", 0.95)],
            "Zorgati Sami est venu": [("Zorgati Sami", "person", 0.95)],
        },
        # A place typed PERSON from `Sami` left a PERSON piece `Menzah`, which retyped the next LOCATION.
        {
            "Menzah Sami Ben Ali Trabelsi": [("Menzah Sami", "location", 0.9)],
            "Je vis au Menzah": [("Menzah", "location", 0.9)],
        },
    ],
)
def test_a_piece_never_shapes_the_conversation(found: dict[str, list[tuple[str, str, float]]]) -> None:
    """Final review of #91: a piece is masked where it is, and is no entity later messages link to."""
    messages = list(found)
    conv = per_message(found).anonymize_messages(messages)
    assert [Nona.restore(text, conv.mapping) for text in conv.texts] == messages
    if "Je vis au Menzah" in found:
        assert conv.texts[1] == "Je vis au LOCATION_1"
    if "merci de rien" in found:
        assert conv.texts[2] == "merci de rien"


def test_a_re_masked_word_is_never_joined_into_a_longer_span() -> None:
    """A re-masked word joined to a longer span could partly overlap the gazetteer's name, and the resolver
    keeps only the longer span (`Sami` leaked)."""
    text = "ena w Sami Ka3bi Bel7aj kallamni"
    out = pipeline({"Sami Ka3bi Bel7aj": ("person", 0.3)}).anonymize_messages([text]).texts[0]
    assert "Sami" not in out
    assert "Ka3bi" not in out


def test_an_arabizi_word_from_a_span_holding_a_person_is_a_person() -> None:
    nona = pipeline({"Ka3bi Sami": ("organization", 0.9)})
    assert "Ka3bi" not in nona.anonymize_messages(["el contrat ta3 Ka3bi Sami jeh"]).texts[0]


@pytest.mark.parametrize("verb", [v for v in SPEECH_VERBS if any(c.isdigit() for c in v)])
def test_a_speech_verb_with_digits_is_cut_and_not_masked_again(verb: str) -> None:
    assert never_name(verb)
    assert arabizi_cuts(verb, 0, len(verb)) == []


@pytest.mark.parametrize(("text", "view"), [("3ers m3a Sami", "aers maa Sami"), ("5er", "ker")])
def test_the_view_de_arabizes_darija_words(text: str, view: str) -> None:
    assert model_view(text) == view


@pytest.mark.parametrize("text", ["خالد وصل", "كريم وصل", "نبيل هنا"])
def test_a_model_span_on_an_arabic_common_word_name_is_still_retyped(text: str) -> None:
    """Review of #97: the #50 rule (a bare `كريم` needs a cue) is for finding a word without a model. A span
    the model flagged is still retyped to PERSON when it is a listed first name in Arabic script."""
    word = text.split()[0]
    assert detect({word: ("location", 0.6)}, text) == [(word, EntityType.PERSON, 0.6)]


@pytest.mark.parametrize("text", ["b3atht l Canada", "chouft Montreal lbare7", "kallamt Tunisie Telecom"])
def test_a_place_after_a_speech_verb_stays_a_place(text: str) -> None:
    """Re-review of #97: speech verbs aren't person cues for the model; a place often follows them."""
    place = text.split(" ", 2)[-1] if text.startswith("b3atht") else text.split()[1]
    if text.startswith("kallamt"):
        place = "Tunisie Telecom"
    assert all(t is not EntityType.PERSON for _, t, _ in detect({place: ("location", 0.9)}, text))
