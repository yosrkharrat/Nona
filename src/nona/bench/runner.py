"""`nona bench`: run systems over a split, print a language-by-type table, write JSON (04 §F, NFR-9, NFR-11).

Outputs, per system:
- `results/<split>-<system>.json`: the summary. Deterministic (no timestamps), so two runs on the same code
  and data give the same file, and a diff shows exactly what a change moved.
- `results/raw/<split>-<system>.jsonl`: one line per gold mention with its outcome, to list failures and
  coincidental matches (05 §6). Ignored by git.
"""

from __future__ import annotations

import dataclasses
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nona import __version__
from nona.bench.dataset import Conversation, dataset_hash, load_jsonl
from nona.bench.metrics import DIRECT_IDENTIFIERS, Linking, Outcome, Rate, Report, evaluate
from nona.bench.systems import System
from nona.core import EntityType

# The repository root, when running from a checkout: dataset paths in the summary are written relative to it,
# so the summary doesn't depend on the directory `nona bench` was run from.
REPO_ROOT = Path(__file__).resolve().parents[3]

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


def requests(system: System, conversation: Conversation) -> list[list[str]]:
    """What the upstream receives in each request of the stateless flow (ADR-0002): request `k` re-sends
    turns `0..k`, computed from them only. So every copy of a turn is causal, and every copy is scored."""
    return [
        list(system.payloads(dataclasses.replace(conversation, turns=conversation.turns[: k + 1])))
        for k in range(len(conversation.turns))
    ]


def causal_payloads(system: System, conversation: Conversation) -> list[str]:
    """Turn `i` as first sent, from turns `0..i` only (ADR-0004): a system that looks ahead gains nothing."""
    return [request[k] for k, request in enumerate(requests(system, conversation))]


def run(system: System, split: str, dataset: Path) -> Run:
    conversations = load_jsonl(dataset)
    results = []
    for c in conversations:
        sent = requests(system, c)
        last = sent[-1] if sent else []
        copies = [[request[i] for request in sent[i:]] for i in range(len(c.turns))]
        results.append(evaluate(c, last, copies))
    return Run(system.name, split, dataset, conversations, Report.from_results(results))


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _rate_json(rate: Rate) -> dict[str, Any]:
    value = rate.value
    return {"rate": None if value is None else round(value, 6), "hits": rate.hits, "total": rate.total}


def _linking_json(linking: Linking) -> dict[str, Any]:
    return {
        "mentions_with_placeholder": _rate_json(linking.placed),
        "partly_leaking": _rate_json(linking.partly_leaking),
        "wrong_merge": _rate_json(linking.wrong_merge),
        "wrong_split": _rate_json(linking.wrong_split),
        "ambiguity": _rate_json(linking.ambiguity),
    }


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
            "path": _display_path(r.dataset),
            "sha256": r.dataset_sha256,
            "conversations": len(r.conversations),
            "turns": sum(len(c.turns) for c in r.conversations),
            "mentions": len(report.mentions),
        },
        "nona_version": __version__,
        "python": ".".join(platform.python_version_tuple()[:2]),
        "leak_rate": _rate_json(report.headline),
        "obfuscated_rate": _rate_json(report.obfuscated),
        "partial_leak_rate": _rate_json(report.partial),
        "deleted_rate": _rate_json(report.deleted),
        "held_out_person_leak_rate": _rate_json(report.held_out),
        "assistant_turn_leak_rate": _rate_json(report.assistant),
        "location_organization_leak_rate": _rate_json(report.location_organization),
        "over_masking": {
            "tokens": _rate_json(report.masked_tokens),
            "distractors": _rate_json(report.masked_distractors),
        },
        "linking": {
            "all": _linking_json(report.linking()),
            "PERSON": _linking_json(report.linking((EntityType.PERSON,))),
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
        f"  of which obfuscated           {report.obfuscated}",
        f"Partial leak rate               {report.partial}",
        f"Deleted (no placeholder)        {report.deleted}",
        f"Leak rate, PERSON held out      {report.held_out}",
        f"Leak rate, assistant turns      {report.assistant}",
        f"Leak rate, LOCATION and ORG     {report.location_organization}",
        f"Over-masking, non-PII tokens    {report.masked_tokens}",
        f"Over-masking, distractors       {report.masked_distractors}",
        "",
        "Linking (all roles, ambiguous and partly leaking mentions skipped)   all types / PERSON",
    ]
    linked, people = report.linking(), report.linking((EntityType.PERSON,))
    for label, field in (
        ("Mentions with a placeholder", "placed"),
        ("  of which partly leaking", "partly_leaking"),
        ("Wrong merge, entities", "wrong_merge"),
        ("Wrong split, entities", "wrong_split"),
        ("Ambiguity, later mentions", "ambiguity"),
    ):
        lines.append(f"{label:<32}{getattr(linked, field)} / {getattr(people, field)}")
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
                "entity": m.entity,
                "ambiguous": m.ambiguous,
                "placeholder": m.placeholder,
            },
            ensure_ascii=False,
        )
        for m in r.report.mentions
    ]
    (raw_dir / f"{stem}.jsonl").write_text("".join(f"{line}\n" for line in raw_lines), encoding="utf-8")
    return summary_path


__all__ = ["Outcome", "Run", "format_table", "run", "summary", "write_outputs"]
