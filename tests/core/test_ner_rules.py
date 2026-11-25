"""NER quality rules (#72 part 2, ADR-0008 conditions 1 and 2): the de-Arabizi'd view, span cleaning, types,
cue-backed low scores, and the conversation-level corrections in the pipeline. With a fake model."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nona.core import EntityType, Nona, normalize
from nona.core.canonical import canonical_tokens
from nona.core.recognizers import NerPrediction, NerRecognizer, default_recognizers
from nona.core.recognizers.gazetteer import NAMES
from nona.core.recognizers.gazetteer_data import NEVER_NAMES
from nona.core.recognizers.ner import arabizi, clean, model_view, never_name
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
