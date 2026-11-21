"""`nona bench`: run systems over a split, print a language-by-type table, write JSON (04 §F, NFR-9, NFR-11).

Outputs, per system:
- `results/<split>-<system>.json`: the summary. Deterministic (no timestamps), so two runs on the same code
  and data give the same file, and a diff shows exactly what a change moved.
- `results/raw/<split>-<system>.jsonl`: one line per gold mention with its outcome, to list failures and
  coincidental matches (05 §6). Ignored by git.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nona import __version__
from nona.bench.dataset import Conversation, dataset_hash, load_jsonl
from nona.bench.metrics import DIRECT_IDENTIFIERS, Outcome, Rate, Report, evaluate
from nona.bench.systems import System

# Short column headers, so the table fits in a terminal.
_ABBREVIATIONS = {
    "PERSON": "PER", "LOCATION": "LOC", "ORGANIZATION": "ORG", "EMAIL": "EMAIL", "PHONE": "PHONE",
    "IBAN": "IBAN", "RIB": "RIB", "CARD": "CARD", "IP_ADDRESS": "IP", "NATIONAL_ID": "CIN",
    "CUSTOM": "CUSTOM",
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class Run:
    system: str
    split: str
    dataset: Path
    conversations: Sequence[Conversation]
    report: Report

    @property
    def dataset_sha256(self) -> str:
        return dataset_hash(self.conversations)


def run(system: System, split: str, dataset: Path) -> Run:
    conversations = load_jsonl(dataset)
    results = [evaluate(c, system.payloads(c)) for c in conversations]
    return Run(system.name, split, dataset, conversations, Report.from_results(results))


def _rate_json(rate: Rate) -> dict[str, Any]:
    value = rate.value
    return {"rate": None if value is None else round(value, 6), "hits": rate.hits, "total": rate.total}


def summary(r: Run) -> dict[str, Any]:
    report = r.report
    by_language: dict[str, dict[str, Any]] = {}
    for language in report.languages:
        row: dict[str, Any] = {}
        for etype in report.types:
            mentions = report.select(types=(etype,), language=language)
            if mentions:
                row[etype.value] = _rate_json(report.rate(mentions))
        by_language[language] = row
    return {
        "system": r.system,
        "split": r.split,
        "dataset": {
            "path": r.dataset.as_posix(),
            "sha256": r.dataset_sha256,
            "conversations": len(r.conversations),
            "turns": sum(len(c.turns) for c in r.conversations),
            "mentions": len(report.mentions),
        },
        "nona_version": __version__,
        "python": platform.python_version(),
        "leak_rate": _rate_json(report.headline),
        "partial_leak_rate": _rate_json(report.partial),
        "assistant_turn_leak_rate": _rate_json(report.assistant),
        "location_organization_leak_rate": _rate_json(report.location_organization),
        "over_masking": {
            "tokens": _rate_json(report.masked_tokens),
            "distractors": _rate_json(report.masked_distractors),
        },
        "leak_rate_by_language_and_type": by_language,
    }


def _cell(rate: Rate) -> str:
    value = rate.value
    return "" if value is None else f"{100 * value:.0f}% /{rate.total}"


def format_table(r: Run) -> str:
    """Leak rate per language and type, then the summary lines. Cells read `leaked% /mentions`."""
    report = r.report
    types = report.types
    header = ["language", *(_ABBREVIATIONS[t.value] for t in types), "direct"]
    rows: list[list[str]] = []
    for language in [*report.languages, None]:
        row = ["all" if language is None else language]
        for etype in types:
            row.append(_cell(report.rate(report.select(types=(etype,), language=language))))
        row.append(_cell(report.rate(report.select(language=language))))
        rows.append(row)
    widths = [max(len(line[i]) for line in [header, *rows]) for i in range(len(header))]

    def fmt(line: list[str]) -> str:
        first = line[0].ljust(widths[0])
        rest = (cell.rjust(w) for cell, w in zip(line[1:], widths[1:], strict=True))
        return "  ".join([first, *rest]).rstrip()

    rule = "-" * len(fmt(header))
    lines = [
        f"system={r.system}  split={r.split}  conversations={len(r.conversations)}  "
        f"dataset sha256={r.dataset_sha256[:12]}",
        "",
        "Leak rate by language and type (user/system/tool turns, public figures excluded)",
        fmt(header),
        rule,
        *(fmt(row) for row in rows[:-1]),
        rule,
        fmt(rows[-1]),
        "",
        f"Leak rate, direct identifiers   {report.headline}",
        f"Partial leak rate               {report.partial}",
        f"Leak rate, assistant turns      {report.assistant}",
        f"Leak rate, LOCATION and ORG     {report.location_organization}",
        f"Over-masking, non-PII tokens    {report.masked_tokens}",
        f"Over-masking, distractors       {report.masked_distractors}",
    ]
    direct = ", ".join(sorted(_ABBREVIATIONS[t.value] for t in DIRECT_IDENTIFIERS))
    lines.append(f"(direct = {direct})")
    return "\n".join(lines)


def write_outputs(r: Run, out_dir: Path) -> Path:
    """Write the summary and the per-mention file. Returns the summary path."""
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{r.split}-{r.system}"
    summary_path = out_dir / f"{stem}.json"
    summary_path.write_text(json.dumps(summary(r), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raw_lines = [
        json.dumps(
            {
                "conversation": m.conversation,
                "turn": m.turn,
                "role": m.role,
                "language": m.language,
                "type": m.type.value,
                "public": m.public,
                "text": m.text,
                "outcome": m.outcome.value,
            },
            ensure_ascii=False,
        )
        for m in r.report.mentions
    ]
    (raw_dir / f"{stem}.jsonl").write_text("".join(f"{line}\n" for line in raw_lines), encoding="utf-8")
    return summary_path


__all__ = ["Outcome", "Run", "format_table", "run", "summary", "write_outputs"]
