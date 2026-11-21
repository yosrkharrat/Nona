from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nona.core.normalize import NormalizedDoc, normalize, normalize_cluster


@pytest.mark.parametrize(
    ("original", "normalized"),
    [
        ("appelle sami au 22 123 456", "appelle sami au 22 123 456"),
        ("Appelle SAMI", "appelle sami"),
        ("٩٨ ٧٦٥ ٤٣٢", "98 765 432"),
        ("۰۹۱۲", "0912"),  # Persian digits
        ("ＡＢＣ１２", "abc12"),  # noqa: RUF001 - full-width forms on purpose
        ("ﬁchier", "fichier"),
        ("Straße", "strasse"),
        ("été", "été"),  # decomposed accents are composed
        ("ســـامي", "سامي"),  # tatweel
        ("سَامِي", "سامي"),  # short-vowel diacritics
        ("", ""),
    ],
)
def test_normalize(original: str, normalized: str) -> None:
    assert normalize(original).text == normalized


def test_plain_ascii_maps_one_to_one() -> None:
    doc = normalize("sami 22 123 456")
    assert doc.origin == tuple((i, i + 1) for i in range(len(doc.text)))


def test_spans_map_back_to_the_original_text() -> None:
    doc = normalize("Appelle ســامي au ٢٢ ١٢٣ ٤٥٦")
    name = doc.text.index("سامي")
    assert doc.original_text(name, name + 4) == "ســامي"
    phone = doc.text.index("22 123 456")
    assert doc.original_text(phone, phone + 10) == "٢٢ ١٢٣ ٤٥٦"


def test_an_expansion_maps_back_to_the_whole_source_character() -> None:
    doc = normalize("ﬁn")
    assert doc.text == "fin"
    assert doc.to_original(0, 1) == (0, 1)  # `f` alone still points at the whole ligature
    assert doc.to_original(1, 3) == (0, 2)


def test_combining_marks_stay_with_their_base() -> None:
    doc = normalize("Léa")
    assert doc.text == "léa"
    assert doc.original_text(1, 2) == "é"


@pytest.mark.parametrize(("start", "end"), [(0, 0), (-1, 2), (2, 1), (0, 99)])
def test_to_original_rejects_invalid_ranges(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="invalid normalized range"):
        normalize("sami").to_original(start, end)


def test_origin_must_match_text_length() -> None:
    with pytest.raises(ValueError, match="one entry per normalized character"):
        NormalizedDoc("ab", "ab", ((0, 1),))


# --------------------------------------------------------------------------- properties

texts = st.text(st.characters(codec="utf-8", exclude_categories=("Cs",)), max_size=60)


@given(texts)
def test_origin_is_valid_and_monotonic(text: str) -> None:
    doc = normalize(text)
    previous_start = 0
    for start, end in doc.origin:
        assert 0 <= start < end <= len(text)
        assert start >= previous_start
        previous_start = start


@given(texts, st.data())
def test_any_normalized_range_maps_to_an_original_range_with_the_same_normal_form(
    text: str, data: st.DataObject
) -> None:
    """The core guarantee: what a recognizer finds is backed by original text that normalizes to it."""
    doc = normalize(text)
    if not doc.text:
        return
    start = data.draw(st.integers(0, len(doc.text) - 1))
    end = data.draw(st.integers(start + 1, len(doc.text)))
    o_start, o_end = doc.to_original(start, end)
    assert doc.text[start:end] in normalize(text[o_start:o_end]).text


@given(texts)
def test_normalization_is_built_cluster_by_cluster(text: str) -> None:
    doc = normalize(text)
    seen: list[tuple[int, int]] = list(dict.fromkeys(doc.origin))
    assert "".join(normalize_cluster(text[s:e]) for s, e in seen) == doc.text


@given(st.text(st.characters(min_codepoint=0x20, max_codepoint=0x7E), max_size=60))
def test_lowercase_printable_ascii_is_unchanged(text: str) -> None:
    assert normalize(text.lower()).text == text.lower()
