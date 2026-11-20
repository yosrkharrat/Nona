from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from nona.bench.dataset import dumps_jsonl, load_jsonl
from nona.bench.metrics import Rate
from nona.bench.runner import run, summary
from nona.bench.systems import IdentitySystem
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
