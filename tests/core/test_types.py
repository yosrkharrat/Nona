from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nona.core import Entity, EntityType, Span

TEXT = "3andi rendez-vous m3a Sami Trabelsi ghodwa, a3tih 22 123 456"


def span(start: int, end: int, type_: EntityType = EntityType.PERSON, score: float = 0.9) -> Span:
    return Span(start=start, end=end, type=type_, score=score, detector="test")


@st.composite
def spans(draw: st.DrawFn) -> Span:
    start = draw(st.integers(min_value=0, max_value=500))
    end = draw(st.integers(min_value=start + 1, max_value=start + 50))
    return Span(
        start=start,
        end=end,
        type=draw(st.sampled_from(EntityType)),
        score=draw(st.floats(min_value=0.0, max_value=1.0)),
        detector=draw(st.sampled_from(["regex:phone_tn", "gliner", "gazetteer"])),
    )


class TestSpan:
    def test_text_in_returns_the_original_substring(self) -> None:
        s = span(22, 35)
        assert s.text_in(TEXT) == "Sami Trabelsi"
        assert len(s) == 13

    @pytest.mark.parametrize(("start", "end"), [(-1, 3), (5, 5), (7, 3)])
    def test_rejects_invalid_offsets(self, start: int, end: int) -> None:
        with pytest.raises(ValueError, match="offsets"):
            span(start, end)

    @pytest.mark.parametrize("score", [-0.01, 1.01])
    def test_rejects_scores_outside_unit_interval(self, score: float) -> None:
        with pytest.raises(ValueError, match="score"):
            span(0, 1, score=score)

    def test_rejects_empty_detector(self) -> None:
        with pytest.raises(ValueError, match="detector"):
            Span(start=0, end=1, type=EntityType.EMAIL, score=1.0, detector="")

    def test_is_immutable(self) -> None:
        s = span(0, 4)
        with pytest.raises(AttributeError):
            s.start = 1  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ((0, 5), (5, 9), False),  # touching, half-open: no overlap
            ((0, 5), (4, 9), True),
            ((2, 3), (0, 9), True),  # contained
            ((0, 2), (3, 9), False),
        ],
    )
    def test_overlaps(self, a: tuple[int, int], b: tuple[int, int], expected: bool) -> None:
        assert span(*a).overlaps(span(*b)) is expected
        assert span(*b).overlaps(span(*a)) is expected

    @given(st.lists(spans(), max_size=30), st.randoms())
    def test_sort_order_is_independent_of_input_order(self, items: list[Span], rnd: object) -> None:
        # Placeholder numbering depends on this order (ADR-0002, ADR-0004): it must be total and stable.
        shuffled = list(items)
        rnd.shuffle(shuffled)  # type: ignore[attr-defined]
        assert sorted(items, key=Span.sort_key) == sorted(shuffled, key=Span.sort_key)

    def test_longest_span_sorts_first_at_same_start(self) -> None:
        short, long = span(22, 26), span(22, 35)
        assert sorted([short, long], key=Span.sort_key) == [long, short]


class TestEntity:
    def test_placeholder_uses_type_and_id(self) -> None:
        e = Entity(id=1, type=EntityType.PERSON, mentions=(span(22, 35),), canonical="Sami Trabelsi")
        assert e.placeholder == "PERSON_1"

    def test_ids_start_at_one(self) -> None:
        with pytest.raises(ValueError, match="start at 1"):
            Entity(id=0, type=EntityType.PERSON, mentions=(span(0, 4),), canonical="Sami")

    def test_needs_a_mention(self) -> None:
        with pytest.raises(ValueError, match="at least one mention"):
            Entity(id=1, type=EntityType.PERSON, mentions=(), canonical="Sami")

    def test_mentions_must_share_the_entity_type(self) -> None:
        with pytest.raises(ValueError, match="type"):
            Entity(
                id=1,
                type=EntityType.PERSON,
                mentions=(span(0, 4), span(10, 20, EntityType.PHONE)),
                canonical="Sami",
            )
