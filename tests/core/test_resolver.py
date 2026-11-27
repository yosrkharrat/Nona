from __future__ import annotations

import random

from hypothesis import given
from hypothesis import strategies as st

from nona.core import EntityType, Span, normalize
from nona.core.boundaries import NAME_TYPES
from nona.core.resolver import denylist_spans, resolve

P, PH, CIN, IBAN, RIB = (
    EntityType.PERSON,
    EntityType.PHONE,
    EntityType.NATIONAL_ID,
    EntityType.IBAN,
    EntityType.RIB,
)
TEXT = "x" * 100


def s(start: int, end: int, etype: EntityType = P, score: float = 1.0, detector: str = "d") -> Span:
    return Span(start, end, etype, score, detector)


def test_longest_span_wins() -> None:
    assert resolve([s(5, 29, RIB), s(0, 29, IBAN)], TEXT) == [s(0, 29, IBAN)]


def test_priority_breaks_ties_on_equal_length() -> None:
    assert resolve([s(4, 12, PH), s(4, 12, CIN)], TEXT) == [s(4, 12, CIN)]


def test_score_breaks_ties_on_equal_type_and_length() -> None:
    assert resolve([s(0, 4, score=0.6, detector="a"), s(0, 4, score=0.9, detector="b")], TEXT) == [
        s(0, 4, score=0.9, detector="b")
    ]


def test_non_overlapping_spans_are_all_kept_in_order() -> None:
    assert resolve([s(10, 12), s(0, 4), s(5, 8)], TEXT) == [s(0, 4), s(5, 8), s(10, 12)]


def test_partial_overlap_keeps_the_longer() -> None:
    assert resolve([s(0, 6, PH), s(4, 12, PH)], TEXT) == [s(4, 12, PH)]


def test_a_name_keeps_its_words_outside_a_longer_partial_overlap() -> None:
    # #89: the gazetteer's longer span won, and the model's confident `Zorgati` was sent in clear.
    text = "Zorgati Sami Mo3ez Sami"
    model, gazetteer = s(0, 12, score=0.95, detector="ner"), s(8, 23, score=0.9, detector="gazetteer")
    result = resolve([model, gazetteer], text)
    assert [r.text_in(text) for r in result] == ["Zorgati", "Sami Mo3ez Sami"]
    assert result[0] == s(0, 7, score=0.95, detector="ner")


def test_a_name_inside_a_longer_span_is_dropped_whole() -> None:
    text = "Sami Ben Ali"
    assert resolve([s(0, 4), s(0, 12, EntityType.LOCATION)], text) == [s(0, 12, EntityType.LOCATION)]


def test_a_clitic_left_in_front_of_a_name_is_not_masked_alone() -> None:
    text = "لسامي الجلاصي"
    full = s(1, len(text))
    assert resolve([s(0, 5), full], text) == [full]


def test_a_piece_on_the_allowlist_stays_in_clear() -> None:
    text = "Clinique Sami Mo3ez Sami"
    gazetteer = s(9, 24)
    assert resolve([s(0, 13), gazetteer], text, allowlist=["clinique"]) == [gazetteer]


def test_low_scores_are_dropped() -> None:
    assert resolve([s(0, 4, score=0.3)], TEXT) == []
    assert resolve([s(0, 4, score=0.3)], TEXT, min_score=0.2) == [s(0, 4, score=0.3)]


def test_allowlist_drops_matching_spans_case_insensitively() -> None:
    text = "Acme Corp et Sami"
    spans = [s(0, 9, EntityType.ORGANIZATION), s(13, 17)]
    assert resolve(spans, text, allowlist=["acme  CORP"]) == [s(13, 17)]


def test_denylist_finds_every_occurrence_at_word_boundaries() -> None:
    text = "Le Projet Jasmin avance; projet  jasmin bis, mais pas ProjetJasmin."
    found = [sp.text_in(text) for sp in denylist_spans(normalize(text), ["Projet Jasmin"])]
    assert found == ["Projet Jasmin", "projet  jasmin"]
    assert {sp.type for sp in denylist_spans(normalize(text), ["Projet Jasmin"])} == {EntityType.CUSTOM}


def test_denylist_ignores_empty_terms() -> None:
    assert denylist_spans(normalize("abc"), ["", "  "]) == []


spans_strategy = st.lists(
    st.builds(
        lambda a, n, t, sc: s(a, a + n, t, sc),
        st.integers(0, 80),
        st.integers(1, 15),
        st.sampled_from(list(EntityType)),
        st.floats(0.0, 1.0),
    ),
    max_size=25,
)


@given(spans_strategy, st.randoms(use_true_random=False))
def test_result_is_non_overlapping_and_independent_of_input_order(
    spans: list[Span], rng: random.Random
) -> None:
    result = resolve(spans, TEXT)
    assert all(not a.overlaps(b) for i, a in enumerate(result) for b in result[i + 1 :])
    # Each span kept is a candidate, or a piece of a name candidate it lost a partial overlap with (#89).
    assert all(
        r in spans
        or any(
            c.type in NAME_TYPES
            and (c.type, c.score, c.detector) == (r.type, r.score, r.detector)
            and c.start <= r.start
            and r.end <= c.end
            for c in spans
        )
        for r in result
    )
    shuffled = spans[:]
    rng.shuffle(shuffled)
    assert resolve(shuffled, TEXT) == result


@given(spans_strategy)
def test_no_name_candidate_loses_two_characters_in_a_row(spans: list[Span]) -> None:
    """#89: what a name span loses to a partial overlap is masked as its own pieces, down to one character
    between two other spans."""
    covered = {i for r in resolve(spans, TEXT) for i in range(r.start, r.end)}
    for c in spans:
        if c.type in NAME_TYPES and c.score >= 0.5:
            gap = 0
            for i in range(c.start, c.end):
                gap = 0 if i in covered else gap + 1
                assert gap < 2, (c, sorted(covered))


def test_a_span_is_cut_at_a_typed_placeholder() -> None:
    text = "Écris à Sami Trabelsi PERSON_1 stp."
    start = text.index("Sami")
    swallowing = Span(start, text.index(" stp"), EntityType.PERSON, 0.9, "g")
    assert [s.text_in(text) for s in resolve([swallowing], text)] == ["Sami Trabelsi"]
    inside = Span(text.index("PERSON"), text.index("_1"), EntityType.PERSON, 0.95, "g")
    assert resolve([inside], text) == []
    after = "PERSON_1 Trabelsi"
    assert [s.text_in(after) for s in resolve([Span(0, len(after), EntityType.PERSON, 0.9, "g")], after)] == [
        "Trabelsi"
    ]


def test_loose_placeholder_lookalikes_are_not_cut() -> None:
    # `card 4242…` and `RIB 0812…` read as `CARD_n` and `RIB_n` with the loose pattern: they are real values.
    for text in ["card 4242424242424242", "RIB 08123456789012345678"]:
        span = Span(text.index(" ") + 1, len(text), EntityType.CARD, 1.0, "c")
        assert resolve([span], text) == [span]


def test_a_placeholder_inside_a_span_leaves_both_sides() -> None:
    text = "Sami PERSON_1 Trabelsi"
    span = Span(0, len(text), EntityType.PERSON, 0.9, "ner")
    assert [s.text_in(text) for s in resolve([span], text)] == ["Sami", "Trabelsi"]


def test_a_structured_value_holding_a_placeholder_shape_is_not_cut() -> None:
    text = "PERSON_1@example.com"
    email = Span(0, len(text), EntityType.EMAIL, 1.0, "email")
    assert resolve([email], text) == [email]
