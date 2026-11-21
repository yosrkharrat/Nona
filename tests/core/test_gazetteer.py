from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nona.core import EntityType, Nona, normalize
from nona.core.recognizers import GazetteerRecognizer

P, L = EntityType.PERSON, EntityType.LOCATION


def kept(text: str) -> list[tuple[str, EntityType]]:
    """What the default pipeline masks from this recognizer: spans at or above the default threshold."""
    return [(s.text_in(text), s.type) for s in Nona([GazetteerRecognizer()]).detect(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rappelle Ahmed demain.", [("Ahmed", P)]),
        ("Ahmed Ben Salem a envoyé le devis", [("Ahmed Ben Salem", P)]),
        ("Ahmed Karoui veut un rdv", [("Ahmed Karoui", P)]),  # a capitalized word after a listed name
        ("Mohammed et Med sont là", [("Mohammed", P), ("Med", P)]),  # spelling variants
        ("Kallem mo7amed 3al telifoun", [("mo7amed", P)]),  # Arabizi, lowercase
        ("ena w skander mchina", [("skander", P)]),
        ("ena w 3ali mchina", []),  # `3ali` is also "high": a lowercase common word needs a cue
        ("Ben-Salem a appelé", [("Ben-Salem", P)]),
        ("Cordialement,\nMariem Ouertani\nTél", [("Mariem Ouertani", P)]),  # a newline ends the run
        ("Je pense à Amal", [("Amal", P)]),  # common-word name, capitalized mid-sentence
        ("Amal arrive demain", [("Amal", P)]),  # at the start of a sentence: masked, a leak costs more
        ("أرسل الملف إلى سامي", [("سامي", P)]),
        ("بعثلي لسامي", [("سامي", P)]),  # clitic ل left out of the span
        (
            "Ahmed, Ahmed Karoui et Ahmed Ben Salem",
            [("Ahmed", P), ("Ahmed Karoui", P), ("Ahmed Ben Salem", P)],
        ),
    ],
)
def test_gazetteer_names(text: str, expected: list[tuple[str, EntityType]]) -> None:
    assert kept(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dr Khemiri a confirmé", [("Khemiri", P)]),  # the honorific isn't part of the span
        ("Mme Lefort et M. Ben Salem", [("Lefort", P), ("Ben Salem", P)]),  # unlisted name after a cue
        ("Si Hamza 9alli bech yfout", [("Hamza", P)]),
        ("khalti Zina jet", [("Zina", P)]),
        ("My name is Emily Stone", [("Emily Stone", P)]),
        ("je m'appelle Kenza Haddad", [("Kenza Haddad", P)]),
        ("3andi rdv m3a Ghofrane", [("Ghofrane", P)]),
        ("Merci Gaspard", [("Gaspard", P)]),  # weak cue, capitalized
    ],
)
def test_context_cues(text: str, expected: list[tuple[str, EntityType]]) -> None:
    assert kept(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "J'ai l'amal que ça marche",  # amal = hope, lowercase
        "nour el 9amra",  # light, lowercase
        "du thé à l'anis",
        "Bonjour Madame, merci beaucoup",  # an honorific is not a name
        "si tu veux on y va",  # `si` = if
        "La réunion de Lundi est annulée",
        "Envoie le PDF et le RIB",
        "M Bricolage est fermé",  # `M` without its dot is not a cue
        "le vent gharbi souffle",  # gharbi = western
        "ahmed.bs@example.com",  # resolved as an email by the pipeline, not here
    ],
)
def test_not_names(text: str) -> None:
    assert [t for t, ty in kept(text) if ty is P and "@" not in text] == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Il habite à Sfax depuis 2019", [("Sfax", L)]),
        ("Il habite à Sidi Bouzid", [("Sidi Bouzid", L)]),  # not `sidi` + a name
        ("rendez-vous à La Marsa", [("La Marsa", L)]),
        ("Nice weather in Nice today", [("Nice", L)]),  # a common-word city isn't matched at a sentence start
        ("I live in sfax", []),  # Latin script needs its capital
        ("ساكن في صفاقس", [("صفاقس", L)]),
    ],
)
def test_cities(text: str, expected: list[tuple[str, EntityType]]) -> None:
    assert kept(text) == expected


def test_email_beats_the_name_inside_it() -> None:
    assert Nona().anonymize("écris à ahmed.bs@example.com").text == "écris à EMAIL_1"


@given(st.text(max_size=80))
def test_spans_are_valid_on_any_text(text: str) -> None:
    for span in GazetteerRecognizer().detect(normalize(text)):
        assert 0 <= span.start < span.end <= len(text)
        assert span.text_in(text).strip() == span.text_in(text)
