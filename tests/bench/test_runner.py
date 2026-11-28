from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from nona.bench.dataset import Conversation, dumps_jsonl, load_jsonl, parse_conversation
from nona.bench.metrics import Rate
from nona.bench.runner import causal_payloads, run, summary
from nona.bench.systems import IdentitySystem, NonaSystem
from nona.cli import is_frozen, main
from nona.core import Entity, EntityType, Nona

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "bench" / "data" / "dev.jsonl"
FIXTURE = Path(__file__).parent / "fixtures" / "authoring.jsonl"


# --------------------------------------------------------------------------- the committed dev split


def test_dev_split_is_valid_and_canonical() -> None:
    assert dumps_jsonl(load_jsonl(DEV)) == DEV.read_text(encoding="utf-8")


def test_dev_split_meets_the_v0_shape() -> None:
    """Build plan S1: ~80 conversations in FR, EN and Arabizi, with no-PII controls."""
    convs = load_jsonl(DEV)
    assert len(convs) >= 80
    languages = {lang for c in convs for lang in c.languages}
    assert {"fr", "en", "aeb-Latn"} <= languages
    assert sum("no-pii" in c.tags for c in convs) >= 0.10 * len(convs)
    assert all(c.id.startswith("dev-") for c in convs)


def test_identity_leaks_100_percent_on_dev() -> None:
    """The S1 exit criterion: the control system leaks everything, so the metrics can see a leak."""
    report = run(IdentitySystem(), "dev", DEV).report
    assert report.headline == Rate(report.headline.total, report.headline.total)
    assert report.headline.total > 0
    assert report.masked_tokens.hits == 0


# --------------------------------------------------------------------------- summary and CLI


def test_summary_is_deterministic() -> None:
    assert summary(run(IdentitySystem(), "x", FIXTURE)) == summary(run(IdentitySystem(), "x", FIXTURE))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    shutil.copy(FIXTURE, tmp_path / "mini.jsonl")
    return tmp_path


def test_bench_prints_a_table_and_writes_results(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "results"
    assert (
        main(
            ["bench", "--systems", "identity", "--split", "mini", "--data", str(data_dir), "--out", str(out)]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "Leak rate, direct identifiers   100.0% (9/9)" in printed
    assert "aeb-Arab" in printed

    result = json.loads((out / "mini-identity.json").read_text(encoding="utf-8"))
    assert result["system"] == "identity"
    assert result["leak_rate"] == {"rate": 1.0, "hits": 9, "total": 9}
    assert result["leak_rate_by_language_and_type"]["fr"]["PERSON"]["total"] == 4
    raw = (out / "raw" / "mini-identity.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw) == result["dataset"]["mentions"]
    assert {json.loads(line)["outcome"] for line in raw} == {"leaked"}


def test_bench_rejects_unknown_systems(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["bench", "--systems", "identity,presidio"])
    assert exit_info.value.code == 2
    assert "unknown system(s) ['presidio']" in capsys.readouterr().err


def test_bench_reports_a_missing_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "--split", "nope", "--data", str(tmp_path)]) == 1
    assert "no dataset file" in capsys.readouterr().err


@pytest.mark.parametrize("split", ["test", "./test", "../data/test"])
def test_bench_refuses_the_frozen_split_without_published(
    split: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#75: `test` is never used for tuning, so a run on it has to say it is the published one. Recognized
    by its bytes, however the path is written."""
    with pytest.raises(SystemExit) as exit_info:
        main(["bench", "--split", split, "--data", str(ROOT / "bench" / "data"), "--out", str(tmp_path)])
    assert exit_info.value.code == 2
    assert "the test split is frozen" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_the_committed_test_split_is_the_frozen_one() -> None:
    assert is_frozen(ROOT / "bench" / "data" / "test.jsonl")
    assert not is_frozen(DEV)


@pytest.fixture
def frozen_mini(data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The `mini` fixture stands for the frozen split, so the flag is tested without scoring the real one."""
    digest = hashlib.sha256((data_dir / "mini.jsonl").read_bytes()).hexdigest()
    (tmp_path / "frozen.sha256").write_text(f"{digest}  mini.jsonl\n", encoding="utf-8")
    monkeypatch.setattr("nona.cli.FROZEN_HASH", tmp_path / "frozen.sha256")
    return data_dir


def test_bench_runs_the_frozen_split_only_when_published(frozen_mini: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    args = ["bench", "--split", "mini", "--data", str(frozen_mini), "--out", str(out)]
    with pytest.raises(SystemExit):
        main(args)
    assert main([*args, "--published"]) == 0
    assert (out / "mini-identity.json").is_file()


def test_another_test_file_is_not_frozen(tmp_path: Path) -> None:
    """Only the committed bytes are frozen: a `test.jsonl` of one's own runs without the flag."""
    shutil.copy(FIXTURE, tmp_path / "test.jsonl")
    out = tmp_path / "results"
    assert main(["bench", "--split", "test", "--data", str(tmp_path), "--out", str(out)]) == 0


def test_bench_reports_an_invalid_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "bad.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    assert main(["bench", "--split", "bad", "--data", str(tmp_path), "--out", str(tmp_path / "r")]) == 1
    assert "missing field" in capsys.readouterr().err


# --------------------------------------------------------------------------- the nona system (S2)

STRUCTURED = {"EMAIL", "PHONE", "IBAN", "RIB", "CARD", "IP_ADDRESS", "NATIONAL_ID"}


def test_nona_leaks_no_structured_value_on_dev() -> None:
    """S2 exit criterion: the structured leak rate on dev drops from 100% to 0%."""
    report = run(NonaSystem(), "dev", DEV).report
    structured = [m for m in report.mentions if m.type.value in STRUCTURED]
    assert structured
    # Partial leaks remain possible: `bilel.me@…` is masked, but the unmasked name `Bilel` matches it.
    assert [m.text for m in structured if m.outcome.value == "leaked"] == []


def test_nona_over_masks_little_on_dev() -> None:
    report = run(NonaSystem(), "dev", DEV).report
    tokens = report.masked_tokens.value
    assert tokens is not None
    assert tokens < 0.01


def test_bench_runs_several_systems(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "results"
    args = [
        "bench",
        "--systems",
        "identity,nona",
        "--split",
        "mini",
        "--data",
        str(data_dir),
        "--out",
        str(out),
    ]
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "system=identity" in printed
    assert "system=nona" in printed
    assert (out / "mini-nona.json").is_file()


def test_nona_links_people_consistently_on_dev() -> None:
    """ADR-0004: on the mentions Nona masks, no two people share a placeholder and no one gets two."""
    r = run(NonaSystem(), "dev", DEV)
    people = r.report.linking((EntityType.PERSON,))
    assert people.placed.hits > 0
    assert people.wrong_merge.hits == 0
    assert people.wrong_split.hits == 0
    linking = summary(r)["linking"]
    assert set(linking) == {"all", "PERSON"}
    assert set(linking["PERSON"]) == {
        "mentions_with_placeholder", "partly_leaking", "wrong_merge", "wrong_split", "ambiguity",
    }  # fmt: skip


def _touching(
    entities: tuple[Entity, ...], start: int, end: int, etype: EntityType
) -> list[tuple[str, str | None]]:
    return [
        (e.placeholder, e.owner)
        for e in entities
        if e.type is etype and any(s.start < end and start < s.end for s in e.mentions)
    ]


def test_nona_never_gives_an_email_the_wrong_owner_on_dev() -> None:
    """Deep dive B, step 6: owners are conservative. Checked against the gold `owner` of each email."""
    correct = 0
    nona = Nona()  # the owner is metadata, not in the payload the bench sees, so this calls Nona directly
    for conv in load_jsonl(DEV):
        result = nona.anonymize_messages([t.text for t in conv.turns])
        people: dict[str, set[str]] = {}
        for i, turn in enumerate(conv.turns):
            for m in turn.mentions:
                for placeholder, _ in _touching(
                    result.messages[i].entities, m.start, m.end, EntityType.PERSON
                ):
                    people.setdefault(m.entity, set()).add(placeholder)
        for i, turn in enumerate(conv.turns):
            for m in turn.mentions:
                gold = conv.entity(m.entity)
                for _, owner in _touching(result.messages[i].entities, m.start, m.end, EntityType.EMAIL):
                    if owner is None:
                        continue
                    assert gold.owner is not None, (conv.id, m.text)
                    assert owner in people.get(gold.owner, set()), (conv.id, m.text, owner)
                    correct += 1
    assert correct >= 8


# --------------------------------------------------------------------------- harness validity (#58)


def test_a_system_that_looks_ahead_gains_nothing() -> None:
    """Turn `i` is computed from turns `0..i` only (ADR-0004): a system that masks `Sami` because a *later*
    turn says it's a name can't use that for the earlier turn."""

    class LooksAhead:
        name = "looks-ahead"

        def payloads(self, conversation: Conversation) -> list[str]:
            told = any("NAME:" in t.text for t in conversation.turns)
            return [t.text.replace("Sami", "PERSON_1") if told else t.text for t in conversation.turns]

    conv = parse_conversation(
        {
            "id": "c", "languages": ["fr"], "source": "handwritten", "tags": [],
            "entities": [{"id": "e1", "type": "PERSON", "value": "Sami"}],
            "turns": [
                {"role": "user", "text": "Sami arrive.",
                 "mentions": [{"text": "Sami", "entity": "e1", "start": 0, "end": 4}]},
                {"role": "user", "text": "NAME: Sami",
                 "mentions": [{"text": "Sami", "entity": "e1", "start": 6, "end": 10}]},
            ],
        }
    )  # fmt: skip
    assert LooksAhead().payloads(conv)[0] == "PERSON_1 arrive."  # what it would send with hindsight
    assert causal_payloads(LooksAhead(), conv) == ["Sami arrive.", "NAME: PERSON_1"]


def test_the_summary_does_not_depend_on_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    r = run(IdentitySystem(), "dev", DEV)
    s = summary(r)
    assert s["dataset"]["path"] == "bench/data/dev.jsonl"
    assert s["python"].count(".") == 1  # major.minor only
