from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nona.core.canonical import HONORIFICS, canonical_tokens, compatible, name_words


def toks(*words: str) -> frozenset[str]:
    return frozenset(words)


@pytest.mark.parametrize(
    "surface",
    ["M. Ben Salem", "ben salem", "BEN SALEM", "Ben-Salem", "Bensalem", "Mme Ben Salem", "Dr. ben  salem"],
)
def test_surname_forms_are_one_token(surface: str) -> None:
    assert canonical_tokens(surface) == toks("bensalem")


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("Ahmed Ben Salem", toks("ahmed", "bensalem")),
        ("Ahmed", toks("ahmed")),
        ("Ahmad", toks("ahmed")),
        ("Mohammed Trabelsi", toks("mohamed", "trabelsi")),
        ("Med Trabelsi", toks("mohamed", "trabelsi")),
        ("Muhammad", toks("mohamed")),
        ("Meriem", toks("mariem")),
        ("Hélène Dupont", toks("helene", "dupont")),
        ("Mohamed El Abed", toks("mohamed", "elabed")),
        ("si Moncef", toks("monsef")),
        ("khalti Fatma", toks("fatma")),
        ("3amm Hédi", toks("hedi")),
        ("3ali", toks("ali")),
        ("m7ammed", toks("mhammed")),
        ("Sami O'Neil", toks("sami", "o", "neil")),
        ("سامي", toks("sami")),
        ("السيد أحمد بن سالم", toks("ahmed", "bensalem")),
        ("فاطمة", toks("fatma")),
        ("Ahmed Ben", toks("ahmed", "ben")),  # a trailing particle stays a token
        ("Ben", toks("ben")),
    ],
)
def test_canonical_tokens(surface: str, expected: frozenset[str]) -> None:
    assert canonical_tokens(surface) == expected


@pytest.mark.parametrize("surface", ["M.", "Mme", "Dr", "si", "السيد", "", "  "])
def test_honorific_alone_has_no_tokens(surface: str) -> None:
    assert canonical_tokens(surface) == frozenset()


def test_cross_script_and_spelling_variants_are_compatible() -> None:
    assert compatible(canonical_tokens("سامي"), canonical_tokens("Sami Trabelsi"))
    assert compatible(canonical_tokens("Mohammed"), canonical_tokens("Med Ben Salem"))


def test_compatible_is_subset_or_superset() -> None:
    a, ab, ac = toks("ahmed"), toks("ahmed", "bensalem"), toks("ahmed", "karoui")
    assert compatible(a, ab)
    assert compatible(ab, a)
    assert compatible(ab, ab)
    assert not compatible(ab, ac)
    assert not compatible(frozenset(), a)  # nothing to link on


@given(st.text(max_size=40))
def test_canonical_tokens_is_total_and_honorific_free(text: str) -> None:
    tokens = canonical_tokens(text)
    assert not tokens & HONORIFICS
    assert all(t and not t.isdigit() for t in tokens)


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz -.", max_size=30))
def test_case_and_honorific_prefix_dont_change_tokens(text: str) -> None:
    assert canonical_tokens(text.upper()) == canonical_tokens(text)
    assert canonical_tokens("M. " + text) == canonical_tokens(text)


# --------------------------------------------------------------------------- name words (look-ahead)


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("Chirine Boukadida", ["Chirine", "Boukadida"]),
        ("Mme Chirine Boukadida", ["Chirine", "Boukadida"]),  # the honorific names no one
        ("Ahmed Ben Salem", ["Ahmed"]),  # `Salem` alone is not `bensalem`, and `Ben` is a particle
        ("Ahmed Ben-Salem", ["Ahmed"]),
        ("SAMI TRABELSI", ["SAMI", "TRABELSI"]),
        ("amal trabelsi", []),  # lowercase Latin words are as likely ordinary words
        ("Sami", []),  # the whole mention is already a span
        ("Al Jaziri", []),  # `al` is a particle: the surname is `aljaziri`
        ("Mohamed Ali", ["Mohamed", "Ali"]),
        ("Ali Ali", ["Ali"]),
        ("سامي الطرابلسي", ["سامي", "الطرابلسي"]),  # no case in Arabic script
    ],
)
def test_name_words(surface: str, expected: list[str]) -> None:
    assert name_words(surface) == expected


@given(st.text(max_size=40))
def test_name_words_are_words_of_the_mention_inside_its_tokens(text: str) -> None:
    whole = canonical_tokens(text)
    for word in name_words(text):
        assert word in text
        assert canonical_tokens(word) <= whole
        assert len(word) >= 3
