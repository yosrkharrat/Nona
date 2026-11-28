"""Review benchmark conversations with their gold labels (docs/design/05-dataset.md, D3).

LLM-written messages are accepted only after a human has read them with their labels (05 §1). This script
prints each conversation with its mentions and distractors marked inline, and can record a decision for each:
accept, fix (edit the JSONL line by hand, then review it again) or drop.

Usage:
    python -m nona.bench.review bench/data/dev.jsonl --source llm             # print, no questions
    python -m nona.bench.review bench/data/*.jsonl --lang aeb-Arab --source handwritten
    python -m nona.bench.review bench/data/test.jsonl --source llm --decisions review.json   # ask for each

Marks: `[Sami|e1 PERSON]` is a gold mention (`?` after the type when it is ambiguous), `{amal|~ hope}` is a
distractor, with its note. Decisions are saved after every answer as JSON (id → decision and note), so a
review can stop and resume; conversations that already have a decision are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from nona.bench.dataset import Conversation, DatasetError, Turn, load_dataset

DECISIONS = {"a": "accept", "f": "fix", "d": "drop"}


def mark(turn: Turn, conv: Conversation) -> str:
    """The turn text with its gold mentions and distractors marked inline."""
    spans: list[tuple[int, int, str]] = []
    for m in turn.mentions:
        flag = "?" if m.ambiguous else ""
        spans.append((m.start, m.end, f"[{m.text}|{m.entity} {conv.entity(m.entity).type.value}{flag}]"))
    for d in turn.distractors:
        spans.append((d.start, d.end, f"{{{d.text}|~ {d.note}}}" if d.note else f"{{{d.text}|~}}"))
    out: list[str] = []
    pos = 0
    for start, end, label in sorted(spans):
        out.append(turn.text[pos:start])
        out.append(label)
        pos = end
    out.append(turn.text[pos:])
    return "".join(out)


def render(conv: Conversation) -> str:
    """One conversation as a block of text: header, entities, then each turn with its labels."""
    tags = ", ".join(conv.tags) or "-"
    lines = [f"== {conv.id}  [{' + '.join(conv.languages)}]  source={conv.source}  tags={tags}"]
    for e in conv.entities:
        extra = (f" owner={e.owner}" if e.owner else "") + (" public" if e.public else "")
        lines.append(f"   {e.id} {e.type.value}: {e.value}{extra}")
    if not conv.entities:
        lines.append("   (no entities)")
    for turn in conv.turns:
        body = mark(turn, conv).replace("\n", "\n" + " " * 13)
        lines.append(f"   {turn.role:<9} | {body}")
    return "\n".join(lines)


def select(
    convs: Sequence[Conversation],
    sources: Sequence[str] = (),
    languages: Sequence[str] = (),
    ids: Sequence[str] = (),
) -> list[Conversation]:
    """Conversations matching every filter given: one of the sources, one of the languages, one of the ids."""
    return [
        c
        for c in convs
        if (not sources or c.source in sources)
        and (not languages or set(languages) & set(c.languages))
        and (not ids or c.id in ids)
    ]


def load_decisions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object of id → decision")
    return raw  # pyright: ignore[reportUnknownVariableType]


def save_decisions(path: Path, decisions: dict[str, dict[str, str]]) -> None:
    text = json.dumps(dict(sorted(decisions.items())), indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def review(
    convs: Sequence[Conversation],
    decisions_path: Path,
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> dict[str, dict[str, str]]:
    """Ask for a decision on each conversation that has none yet. `q` stops; `s` skips one."""
    decisions = load_decisions(decisions_path)
    pending = [c for c in convs if c.id not in decisions]
    for i, conv in enumerate(pending, start=1):
        out(f"\n({i}/{len(pending)})")
        out(render(conv))
        while True:
            answer = ask("accept / fix / drop / skip / quit [a/f/d/s/q]: ").strip().lower()
            if answer in {"a", "f", "d", "s", "q"}:
                break
        if answer == "q":
            break
        if answer == "s":
            continue
        decision = {"decision": DECISIONS[answer]}
        if answer in {"f", "d"}:
            note = ask("note (what is wrong): ").strip()
            if note:
                decision["note"] = note
        decisions[conv.id] = decision
        save_decisions(decisions_path, decisions)
    return decisions


def summary(decisions: dict[str, dict[str, str]]) -> str:
    lines: list[str] = []
    for kind in ("accept", "fix", "drop"):
        ids = sorted(i for i, d in decisions.items() if d.get("decision") == kind)
        lines.append(f"{kind}: {len(ids)}" + (f"  ({', '.join(ids)})" if ids and kind != "accept" else ""))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nona.bench.review", description="Review conversations with their gold labels (05 D3)."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--source", action="append", default=[], help="template, llm or handwritten")
    parser.add_argument("--lang", action="append", default=[], help="keep conversations in this language")
    parser.add_argument("--id", action="append", default=[], help="keep this conversation id")
    parser.add_argument("--decisions", type=Path, help="ask for a decision on each, saved to this JSON")
    args = parser.parse_args(argv)
    try:
        convs = select(load_dataset(args.files), args.source, args.lang, args.id)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.decisions is None:
        for conv in convs:
            print(render(conv), end="\n\n")
        print(f"{len(convs)} conversations")
        return 0
    print(summary(review(convs, args.decisions)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
