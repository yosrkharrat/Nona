from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from nona.bench.dataset import dumps_jsonl, load_jsonl
from nona.bench.metrics import Rate
from nona.bench.runner import run, summary
from nona.bench.systems import IdentitySystem, NonaSystem
from nona.cli import main

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
