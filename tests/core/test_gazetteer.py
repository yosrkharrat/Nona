from __future__ import annotations

import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nona.core import EntityType, Nona, normalize
from nona.core.recognizers import GazetteerRecognizer, gazetteer_data

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
        ("3andi rdv m3a Sami", [("Sami", P)]),  # `m3a` is a weak cue (#56): listed names only
        ("3andi rdv m3a Ghofrane", []),  # an unlisted name after it is left to S4's model
        ("Merci Skander", [("Skander", P)]),  # weak cue, listed name
        ("السيدة سعاد وصلت", [("سعاد", P)]),  # Arabic honorific (teh marbuta), unlisted name
        ("M. ben salem a appelé", [("ben salem", P)]),  # `M.` is strong once its dot is there
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
        "Merci Beaucoup pour tout",  # a weak cue needs a listed name
        "Hello World",
        "Hi Team, see you",
        "chnowa 3amel lyoum?",  # 3amel = doing: a name only after the Arabizi step
        "sana jdida sa3ida",  # sana = year
        "sadiki el aziz",  # el + a common word
        "Bonjour\nGaspard",  # a cue doesn't cross a line break
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ahmed - Sami ont appelé", [("Ahmed", P), ("Sami", P)]),  # a spaced dash separates people
        ("Ahmed Karoui - Directeur Commercial", [("Ahmed Karoui", P)]),
        ("Meeting With Ahmed Tomorrow Morning", [("Ahmed", P)]),
        ("Ahmed Lefort Dupont Martin", [("Ahmed Lefort", P)]),  # one unlisted word per run
        ("سلم على أحمد", [("أحمد", P)]),  # على (on) is not Ali
        ("سلم على سامي", [("سامي", P)]),
    ],
)
def test_run_boundaries(text: str, expected: list[tuple[str, EntityType]]) -> None:
    assert kept(text) == expected


def test_seed_lists_have_no_duplicates() -> None:
    for name in ("FIRST_NAMES_TN", "FIRST_NAMES_FR_EN", "SURNAMES_TN", "NAMES_AR", "COMMON_WORDS", "CITIES"):
        values: tuple[str, ...] = getattr(gazetteer_data, name)
        assert len(values) == len(set(values)), name


def test_detection_is_linear_in_text_length() -> None:
    small, large = "Bonjour Ahmed, voici le rapport. " * 200, "Bonjour Ahmed, voici le rapport. " * 2000
    recognizer = GazetteerRecognizer()
    timings = []
    for text in (small, large):
        start = time.perf_counter()
        recognizer.detect(normalize(text))
        timings.append(time.perf_counter() - start)
    assert timings[1] < timings[0] * 30  # 10x the text; quadratic would be ~100x


def test_email_beats_the_name_inside_it() -> None:
    assert Nona().anonymize("écris à ahmed.bs@example.com").text == "écris à EMAIL_1"


@given(st.text(max_size=80))
def test_spans_are_valid_on_any_text(text: str) -> None:
    for span in GazetteerRecognizer().detect(normalize(text)):
        assert 0 <= span.start < span.end <= len(text)
        assert span.text_in(text).strip() == span.text_in(text)


# --------------------------------------------------------------------------- S3 review leaks (#54)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("السيد كمال الزغلامي وصل", [("كمال الزغلامي", P)]),  # an Arabic family name of the الـ…ي kind
        ("اتصل بيا محمد الطرابلسي اليوم", [("محمد الطرابلسي", P)]),
        ("محمد اليوم مريض", [("محمد", P)]),  # `اليوم` (today) is not a family name
        ("رقم محمد الثاني", [("محمد", P)]),  # an adjective shaped like one (`الثاني`, the second)
        ("السيدة سعاد وصلت", [("سعاد", P)]),  # no capitals in Arabic: a cue doesn't take the verb
        ("J'ai vu M'barek hier", [("M'barek", P)]),
        ("J'ai vu Abdel Kader Trabelsi", [("Abdel Kader Trabelsi", P)]),
        ("Abdel Aziz est arrivé", [("Abdel Aziz", P)]),
        ("c'est Abd El Aziz qui vient", [("Abd El Aziz", P)]),
        ("ولسامي", [("سامي", P)]),  # two clitics
        ("للطرابلسي", [("لطرابلسي", P)]),  # ل + ال: the first ل is left out
        ("مشيت لصفاقس", [("صفاقس", L)]),
        ("مشيت للمنستير", [("لمنستير", L)]),
        ("Je suis à Gabes demain", [("Gabes", L)]),  # the list has `Gabès`
        ("le stade de Radès", [("Radès", L)]),  # the list has `Rades`
        ("El Kef c'est loin", [("El Kef", L)]),
    ],
)
def test_review_leaks(text: str, expected: list[tuple[str, EntityType]]) -> None:
    assert kept(text) == expected


# --------------------------------------------------------------------------- over-masking (S3 review, #56)


@pytest.mark.parametrize(
    "text",
    [
        "je suis marié",  # folds to `marie`, but an accented word isn't that name
        "une eau claire et une pierre",  # lowercase French first names are words
        "une explication claire",
        "pierre claire",
        "ya sahbi",  # Darija: my friend
        "lazem saber",  # patient
        "fi seif",  # summer
        "el anouar",  # the lights
        "m3a Ooredoo",  # `m3a` (with) is a weak cue: no unlisted word after it
        "Ena Tounsi",  # `ena` (I)
        "2 M. Dinars",  # `M.` after a number is a unit
        "je m'appelle\nKenza Haddad",  # cues don't cross a line break
        "Grâce à vous",
        "Mabrouk !",
        "Mabrouk 3lik",
        "mabrouk 3likom",
    ],
)
def test_ordinary_words_are_not_names(text: str) -> None:
    assert kept(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("j'ai vu sami", [("sami", P)]),  # Arabizi writes Tunisian names lowercase
        ("m3a mo7amed", [("mo7amed", P)]),  # Arabizi digits are not accents
        ("مرحبا نور", [("نور", P)]),  # a greeting lifts an Arabic common-word name, like `Merci Amal`
        ("شكرا أمال", [("أمال", P)]),
        ("Mabrouk Jlassi a eu son bac", [("Mabrouk Jlassi", P)]),
        ("Saber Trabelsi a appelé", [("Saber Trabelsi", P)]),  # a common word in a full name stays a name
    ],
)
def test_names_still_found(text: str, expected: list[tuple[str, EntityType]]) -> None:
    assert kept(text) == expected


def test_a_cue_before_a_line_break_does_not_lift_the_next_line() -> None:
    spans = [s for s in GazetteerRecognizer().detect(normalize("Bonjour\nKarim est là")) if s.score >= 0.5]
    assert [s.score for s in spans] == [0.5]  # a common-word name starting a line, no cue
