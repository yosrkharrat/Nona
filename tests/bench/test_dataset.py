from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nona.bench.dataset import (
    Conversation,
    DatasetError,
    DatasetWarning,
    Stats,
    dataset_hash,
    dumps_jsonl,
    load_dataset,
    load_jsonl,
    main,
    parse_conversation,
)
from nona.core import EntityType

FIXTURE = Path(__file__).parent / "fixtures" / "authoring.jsonl"


@pytest.fixture
def convs() -> dict[str, Conversation]:
    return {c.id: c for c in load_jsonl(FIXTURE)}


@pytest.fixture
def raw() -> dict[str, Any]:
    """A valid conversation in authoring form, to mutate in error tests."""
    return copy.deepcopy(json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0]))


# --------------------------------------------------------------------------- happy path


def test_fixture_loads(convs: dict[str, Conversation]) -> None:
    assert set(convs) == {"ex-arabizi", "ex-two-ahmeds", "ex-arabic-clitic"}


def test_offsets_are_filled_from_text(convs: dict[str, Conversation]) -> None:
    turn = convs["ex-arabizi"].turns[0]
    sami, phone = turn.mentions
    assert (sami.start, sami.end) == (22, 35)
    assert turn.text[phone.start : phone.end] == "22 123 456"


def test_arabic_clitic_is_outside_the_span(convs: dict[str, Conversation]) -> None:
    turn = convs["ex-arabic-clitic"].turns[0]
    sami = turn.mentions[0]
    assert turn.text[sami.start : sami.end] == "سامي"
    assert turn.text[sami.start - 1] == "ل"  # the clitic stays in the text, not in the label


def test_distractors_get_offsets(convs: dict[str, Conversation]) -> None:
    (amal,) = convs["ex-arabic-clitic"].turns[0].distractors
    assert amal.text == "أمل"
    assert "hope" in amal.note


def test_conversation_level_entities_and_flags(convs: dict[str, Conversation]) -> None:
    conv = convs["ex-two-ahmeds"]
    assert conv.entity("e2").type is EntityType.EMAIL
    assert conv.entity("e2").owner == "e1"
    assert [t.role for t in conv.turns] == ["user", "user", "user", "assistant", "user"]
    assert conv.turns[-1].mentions[0].ambiguous is True
    assert not conv.code_switched
    assert convs["ex-arabizi"].code_switched


def test_canonical_round_trip_is_identical(convs: dict[str, Conversation]) -> None:
    original = list(convs.values())
    text = dumps_jsonl(original)
    reparsed = [parse_conversation(json.loads(line)) for line in text.splitlines()]
    assert reparsed == original
    assert dumps_jsonl(reparsed) == text


def test_hash_ignores_formatting_but_not_content(convs: dict[str, Conversation], tmp_path: Path) -> None:
    pretty = tmp_path / "pretty.jsonl"
    pretty.write_text(FIXTURE.read_text(encoding="utf-8").replace("\n", "\n\n"), encoding="utf-8")
    assert dataset_hash(load_jsonl(pretty)) == dataset_hash(convs.values())

    changed = FIXTURE.read_text(encoding="utf-8").replace("ghodwa", "lyoum")
    edited = tmp_path / "edited.jsonl"
    edited.write_text(changed, encoding="utf-8")
    assert dataset_hash(load_jsonl(edited)) != dataset_hash(convs.values())


# --------------------------------------------------------------------------- validation errors

Mutation = Callable[[dict[str, Any]], None]


def _set(path: list[Any], value: Any) -> Mutation:
    def mutate(raw: dict[str, Any]) -> None:
        target: Any = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _mention(**fields: Any) -> Mutation:
    return _set(["turns", 0, "mentions", 0], {"text": "Sami Trabelsi", "entity": "e1", **fields})


CASES: list[tuple[str, Mutation, str]] = [
    ("typo in a field name", _set(["turnz"], []), "unknown field"),
    ("missing turns", lambda r: r.pop("turns"), "missing field"),
    ("unknown language", _set(["languages"], ["tn"]), "languages"),
    ("duplicate language", _set(["languages"], ["fr", "fr"]), "languages"),
    ("unknown source", _set(["source"], ["scraped"]), "source"),
    ("unknown entity type", _set(["entities", 0, "type"], "NAME"), "unknown type"),
    ("mention text not in turn", _mention(text="Samir Trabelsi"), "not found"),
    ("offsets disagree with text", _mention(start=0, end=13), "expected"),
    ("offsets out of range", _mention(start=50, end=500), "outside"),
    ("only one offset", _mention(start=22), "integers"),
    ("undeclared entity", _mention(entity="e9"), "not declared"),
    ("leading whitespace", _mention(text=" Sami Trabelsi"), "whitespace"),
    (
        "unused entity",
        lambda r: r["entities"].append({"id": "e3", "type": "EMAIL", "value": "x@example.com"}),
        "never",
    ),
    ("duplicate entity id", _set(["entities", 1, "id"], "e1"), "duplicate entity"),
    ("owner is not a person", _set(["entities", 0, "owner"], "e2"), "owner"),
    ("public non-person", _set(["entities", 1, "public"], True), "public"),
    ("bad role", _set(["turns", 0, "role"], "bot"), "role"),
    ("no-pii with entities", _set(["tags"], ["no-pii"]), "no-pii"),
    ("empty turns", _set(["turns"], []), "at least one turn"),
    (
        "overlapping mentions",
        _set(["turns", 0, "mentions", 1], {"text": "Sami", "entity": "e1"}),
        "overlaps",
    ),
    (
        "distractor overlaps a mention",
        _set(["turns", 0, "distractors"], [{"text": "Trabelsi"}]),
        "overlaps",
    ),
]


@pytest.mark.parametrize(("mutate", "message"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_rejects_invalid_conversations(raw: dict[str, Any], mutate: Mutation, message: str) -> None:
    mutate(raw)
    with pytest.raises(DatasetError, match=message):
        parse_conversation(raw)


def test_repeated_text_needs_an_occurrence() -> None:
    raw: dict[str, Any] = {
        "id": "rep",
        "languages": ["fr"],
        "source": "handwritten",
        "entities": [{"id": "e1", "type": "PERSON", "value": "Nour"}],
        "turns": [
            {
                "role": "user",
                "text": "Nour a dit que Nour viendra.",
                "mentions": [{"text": "Nour", "entity": "e1"}],
            }
        ],
    }
    with pytest.raises(DatasetError, match="appears 2 times"):
        parse_conversation(raw)
    raw["turns"][0]["mentions"] = [
        {"text": "Nour", "entity": "e1", "occurrence": 0},
        {"text": "Nour", "entity": "e1", "occurrence": 1},
    ]
    starts = [m.start for m in parse_conversation(raw).turns[0].mentions]
    assert starts == [0, 15]


def test_unknown_tag_warns_but_loads(raw: dict[str, Any]) -> None:
    raw["tags"] = ["arabizi", "sarcasm"]
    with pytest.warns(DatasetWarning, match="sarcasm"):
        parse_conversation(raw)


def test_invalid_json_reports_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(FIXTURE.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"bad\.jsonl:4"):
        load_jsonl(path)


def test_ids_must_be_unique_across_files(tmp_path: Path) -> None:
    copy_path = tmp_path / "copy.jsonl"
    copy_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(DatasetError, match="already used"):
        load_dataset([FIXTURE, copy_path])


# --------------------------------------------------------------------------- CLI


def test_cli_fmt_writes_canonical_form_with_offsets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["fmt", str(path)]) == 0
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["turns"][0]["mentions"][0] == {
        "text": "Sami Trabelsi",
        "entity": "e1",
        "start": 22,
        "end": 35,
    }

    assert main(["check", str(path)]) == 0
    assert "3 conversations, 7 turns, 11 mentions" in capsys.readouterr().out


def test_cli_reports_errors_without_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    assert main(["check", str(path)]) == 1
    assert "missing field" in capsys.readouterr().err


def test_stats_count_the_05_section_7_quantities(convs: dict[str, Conversation]) -> None:
    stats = Stats.of(convs.values())
    assert (stats.conversations, stats.turns, stats.long) == (3, 7, 1)
    assert stats.code_switched == 1
    assert stats.languages["fr"] == 2
    assert stats.languages["en"] == 0
    assert stats.sources["handwritten"] == 3
    assert stats.share(stats.languages["fr"]) == pytest.approx(2 / 3)
    assert "3 conversations, 7 turns, 1 of 4+ turns" in stats.format()


def test_cli_stats(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stats", str(FIXTURE)]) == 0
    assert "code-switched" in capsys.readouterr().out
