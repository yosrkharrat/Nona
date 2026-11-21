from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nona.core.canonical import HONORIFICS, canonical_tokens, compatible


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
