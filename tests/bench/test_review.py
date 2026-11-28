from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from nona.bench.dataset import Conversation, load_jsonl
from nona.bench.review import main, mark, render, review, select, summary

FIXTURE = Path(__file__).parent / "fixtures" / "authoring.jsonl"


@pytest.fixture
def convs() -> list[Conversation]:
    return load_jsonl(FIXTURE)


def _by_id(convs: list[Conversation], conv_id: str) -> Conversation:
    return next(c for c in convs if c.id == conv_id)


def test_mentions_and_distractors_are_marked_inline(convs: list[Conversation]) -> None:
    conv = _by_id(convs, "ex-arabizi")
    assert mark(conv.turns[0], conv) == (
        "3andi rendez-vous m3a [Sami Trabelsi|e1 PERSON] ghodwa, a3tih [22 123 456|e2 PHONE]"
    )
    clitic = _by_id(convs, "ex-arabic-clitic")
    marked = mark(clitic.turns[0], clitic)
    assert "ل[سامي|e1 PERSON]" in marked  # the clitic stays outside the label
    assert "{أمل|~ " in marked


def test_ambiguous_mentions_and_owners_are_shown(convs: list[Conversation]) -> None:
    text = render(_by_id(convs, "ex-two-ahmeds"))
    assert "PERSON?]" in text
    assert "owner=e1" in text
    assert text.startswith("== ex-two-ahmeds  [fr]  source=handwritten")


def test_select_filters_by_source_language_and_id(convs: list[Conversation]) -> None:
    assert [c.id for c in select(convs, languages=["aeb-Arab"])] == ["ex-arabic-clitic"]
    assert len(select(convs, sources=["handwritten"])) == 3
    assert select(convs, sources=["llm"]) == []
    assert [c.id for c in select(convs, ids=["ex-arabizi"], languages=["fr"])] == ["ex-arabizi"]


def _answers(*values: str) -> Iterator[str]:
    yield from values


def test_review_records_decisions_and_resumes(convs: list[Conversation], tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    answers = _answers("x", "a", "f", "Ahmed is a nickname here", "q")
    shown: list[str] = []
    decisions = review(convs, path, ask=lambda _: next(answers), out=shown.append)
    assert decisions == {
        "ex-arabizi": {"decision": "accept"},
        "ex-two-ahmeds": {"decision": "fix", "note": "Ahmed is a nickname here"},
    }
    assert json.loads(path.read_text(encoding="utf-8")) == decisions  # saved after each answer
    assert any("[Sami Trabelsi|e1 PERSON]" in s for s in shown)

    answers = _answers("d", "")
    decisions = review(convs, path, ask=lambda _: next(answers), out=lambda _: None)
    assert decisions["ex-arabic-clitic"] == {"decision": "drop"}
    assert "drop: 1  (ex-arabic-clitic)" in summary(decisions)
    assert "fix: 1  (ex-two-ahmeds)" in summary(decisions)


def test_skip_leaves_no_decision(convs: list[Conversation], tmp_path: Path) -> None:
    answers = _answers("s", "s", "s")
    assert review(convs, tmp_path / "d.json", ask=lambda _: next(answers), out=lambda _: None) == {}


def test_cli_prints_every_selected_conversation(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(FIXTURE), "--lang", "aeb-Latn"]) == 0
    out = capsys.readouterr().out
    assert "== ex-arabizi" in out
    assert "ex-two-ahmeds" not in out
    assert out.rstrip().endswith("1 conversations")


def test_cli_reports_dataset_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\n', encoding="utf-8")
    assert main([str(bad)]) == 1
    assert "missing field" in capsys.readouterr().err
