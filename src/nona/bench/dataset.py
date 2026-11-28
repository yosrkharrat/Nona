"""Benchmark dataset: strict loading, validation and canonical writing (docs/design/05-dataset.md).

Usage:
    python -m nona.bench.dataset check bench/data/dev.jsonl bench/data/test.jsonl
    python -m nona.bench.dataset fmt bench/data/dev.jsonl      # fill offsets, rewrite in canonical form
    python -m nona.bench.dataset hash bench/data/test.jsonl
    python -m nona.bench.dataset stats bench/data/dev.jsonl    # the 05 §7 size and balance counts
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nona.core import EntityType

LANGUAGES = frozenset({"fr", "en", "ar", "aeb-Arab", "aeb-Latn"})
ROLES = frozenset({"system", "user", "assistant", "tool"})
SOURCES = frozenset({"template", "llm", "handwritten"})
KNOWN_TAGS = frozenset(
    {
        "arabizi", "code-switching", "multi-turn", "common-word-name", "clitic", "eastern-digits",
        "spelling-variant", "honorific", "ambiguous-reference", "context-cue", "no-pii",
    }
)  # fmt: skip


class DatasetError(ValueError):
    """A dataset file breaks the spec. The message says where (file:line, turn, mention)."""


class DatasetWarning(UserWarning):
    """Allowed, but probably a mistake (for example an unknown difficulty tag)."""


# --------------------------------------------------------------------------- model


@dataclass(frozen=True, slots=True)
class GoldEntity:
    id: str
    type: EntityType
    value: str
    owner: str | None = None
    public: bool = False


@dataclass(frozen=True, slots=True)
class Mention:
    start: int
    end: int
    text: str
    entity: str
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class Distractor:
    start: int
    end: int
    text: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    text: str
    mentions: tuple[Mention, ...] = ()
    distractors: tuple[Distractor, ...] = ()


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    languages: tuple[str, ...]
    source: str
    tags: tuple[str, ...]
    entities: tuple[GoldEntity, ...]
    turns: tuple[Turn, ...]

    def entity(self, entity_id: str) -> GoldEntity:
        for e in self.entities:
            if e.id == entity_id:
                return e
        raise KeyError(entity_id)

    @property
    def code_switched(self) -> bool:
        return len(self.languages) > 1

    def to_json(self) -> dict[str, Any]:
        """Canonical form: fixed key order, offsets always present, defaults omitted."""

        def entity(e: GoldEntity) -> dict[str, Any]:
            out: dict[str, Any] = {"id": e.id, "type": e.type.value, "value": e.value}
            if e.owner is not None:
                out["owner"] = e.owner
            if e.public:
                out["public"] = True
            return out

        def mention(m: Mention) -> dict[str, Any]:
            out: dict[str, Any] = {"text": m.text, "entity": m.entity, "start": m.start, "end": m.end}
            if m.ambiguous:
                out["ambiguous"] = True
            return out

        def distractor(d: Distractor) -> dict[str, Any]:
            out: dict[str, Any] = {"text": d.text, "start": d.start, "end": d.end}
            if d.note:
                out["note"] = d.note
            return out

        def turn(t: Turn) -> dict[str, Any]:
            out: dict[str, Any] = {
                "role": t.role,
                "text": t.text,
                "mentions": [mention(m) for m in t.mentions],
            }
            if t.distractors:
                out["distractors"] = [distractor(d) for d in t.distractors]
            return out

        return {
            "id": self.id,
            "languages": list(self.languages),
            "source": self.source,
            "tags": list(self.tags),
            "entities": [entity(e) for e in self.entities],
            "turns": [turn(t) for t in self.turns],
        }


# --------------------------------------------------------------------------- parsing helpers


def _keys(raw: Mapping[str, Any], where: str, required: set[str], optional: set[str]) -> None:
    missing = required - raw.keys()
    if missing:
        raise DatasetError(f"{where}: missing field(s) {sorted(missing)}")
    unknown = raw.keys() - required - optional
    if unknown:
        raise DatasetError(f"{where}: unknown field(s) {sorted(unknown)}")


def _str(raw: Mapping[str, Any], key: str, where: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{where}: '{key}' must be a non-empty string")
    return value


def _bool(raw: Mapping[str, Any], key: str, where: str) -> bool:
    value = raw.get(key, False)
    if not isinstance(value, bool):
        raise DatasetError(f"{where}: '{key}' must be true or false")
    return value


def _list(raw: Mapping[str, Any], key: str, where: str) -> list[Any]:
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise DatasetError(f"{where}: '{key}' must be a list")
    return value


def _locate(turn_text: str, raw: Mapping[str, Any], where: str) -> tuple[int, int, str]:
    """Return (start, end, text): checks given offsets, or finds them from `text` (+ `occurrence`)."""
    text = _str(raw, "text", where)
    if "start" in raw or "end" in raw:
        start, end = raw.get("start"), raw.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or isinstance(start, bool):
            raise DatasetError(f"{where}: 'start' and 'end' must both be integers")
        if not 0 <= start < end <= len(turn_text):
            raise DatasetError(f"{where}: offsets [{start}, {end}) outside the turn text")
        if turn_text[start:end] != text:
            raise DatasetError(f"{where}: text[{start}:{end}] is {turn_text[start:end]!r}, expected {text!r}")
        return start, end, text

    hits: list[int] = []
    i = turn_text.find(text)
    while i != -1:
        hits.append(i)
        i = turn_text.find(text, i + 1)
    if not hits:
        raise DatasetError(f"{where}: {text!r} not found in the turn text")
    occurrence = raw.get("occurrence")
    if occurrence is None:
        if len(hits) > 1:
            raise DatasetError(f"{where}: {text!r} appears {len(hits)} times; set 'occurrence' (0-based)")
        occurrence = 0
    if not isinstance(occurrence, int) or isinstance(occurrence, bool) or not 0 <= occurrence < len(hits):
        raise DatasetError(f"{where}: 'occurrence' must be an integer in [0, {len(hits) - 1}]")
    return hits[occurrence], hits[occurrence] + len(text), text


# --------------------------------------------------------------------------- parsing


def _parse_entity(raw: Any, where: str) -> GoldEntity:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: must be an object")
    _keys(raw, where, {"id", "type", "value"}, {"owner", "public"})
    type_name = _str(raw, "type", where)
    try:
        etype = EntityType(type_name)
    except ValueError:
        raise DatasetError(f"{where}: unknown type {type_name!r}") from None
    owner = raw.get("owner")
    if owner is not None and (not isinstance(owner, str) or not owner):
        raise DatasetError(f"{where}: 'owner' must be an entity id")
    return GoldEntity(
        id=_str(raw, "id", where),
        type=etype,
        value=_str(raw, "value", where),
        owner=owner,
        public=_bool(raw, "public", where),
    )


def _parse_turn(raw: Any, where: str, entity_ids: set[str]) -> Turn:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: must be an object")
    _keys(raw, where, {"role", "text"}, {"mentions", "distractors"})
    role = _str(raw, "role", where)
    if role not in ROLES:
        raise DatasetError(f"{where}: role {role!r} not in {sorted(ROLES)}")
    text = _str(raw, "text", where)

    mentions: list[Mention] = []
    for j, m in enumerate(_list(raw, "mentions", where)):
        mw = f"{where} mention {j}"
        if not isinstance(m, dict):
            raise DatasetError(f"{mw}: must be an object")
        _keys(m, mw, {"text", "entity"}, {"start", "end", "occurrence", "ambiguous"})
        start, end, mtext = _locate(text, m, mw)
        if mtext != mtext.strip():
            raise DatasetError(f"{mw}: {mtext!r} has leading or trailing whitespace")
        ref = _str(m, "entity", mw)
        if ref not in entity_ids:
            raise DatasetError(f"{mw}: entity {ref!r} is not declared in 'entities'")
        mentions.append(Mention(start, end, mtext, ref, _bool(m, "ambiguous", mw)))

    distractors: list[Distractor] = []
    for j, d in enumerate(_list(raw, "distractors", where)):
        dw = f"{where} distractor {j}"
        if not isinstance(d, dict):
            raise DatasetError(f"{dw}: must be an object")
        _keys(d, dw, {"text"}, {"start", "end", "occurrence", "note"})
        start, end, dtext = _locate(text, d, dw)
        note = d.get("note", "")
        if not isinstance(note, str):
            raise DatasetError(f"{dw}: 'note' must be a string")
        distractors.append(Distractor(start, end, dtext, note))

    mentions.sort(key=lambda m: (m.start, m.end))
    ranges = sorted(
        [(m.start, m.end, "mention") for m in mentions]
        + [(d.start, d.end, "distractor") for d in distractors]
    )
    for (s1, e1, k1), (s2, e2, k2) in itertools.pairwise(ranges):
        if s2 < e1:
            raise DatasetError(f"{where}: {k1} [{s1}, {e1}) overlaps {k2} [{s2}, {e2})")
    return Turn(role, text, tuple(mentions), tuple(distractors))


def parse_conversation(raw: Any, where: str = "<input>") -> Conversation:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: a conversation must be a JSON object")
    _keys(raw, where, {"id", "languages", "source", "turns"}, {"tags", "entities"})
    conv_id = _str(raw, "id", where)
    where = f"{where} ({conv_id})"

    languages = _list(raw, "languages", where)
    if (
        not languages
        or any(lang not in LANGUAGES for lang in languages)
        or len(set(languages)) != len(languages)
    ):
        raise DatasetError(f"{where}: 'languages' must be distinct values from {sorted(LANGUAGES)}")
    source = _str(raw, "source", where)
    if source not in SOURCES:
        raise DatasetError(f"{where}: source {source!r} not in {sorted(SOURCES)}")
    tags = _list(raw, "tags", where)
    if not all(isinstance(t, str) and t for t in tags):
        raise DatasetError(f"{where}: 'tags' must be non-empty strings")
    for tag in tags:
        if tag not in KNOWN_TAGS:
            warnings.warn(f"{where}: unknown tag {tag!r}", DatasetWarning, stacklevel=2)

    entities = [_parse_entity(e, f"{where} entity {i}") for i, e in enumerate(_list(raw, "entities", where))]
    by_id = {e.id: e for e in entities}
    if len(by_id) != len(entities):
        raise DatasetError(f"{where}: duplicate entity ids")
    for e in entities:
        if e.owner is not None:
            owner = by_id.get(e.owner)
            if owner is None or owner.type is not EntityType.PERSON or owner.id == e.id:
                raise DatasetError(f"{where}: entity {e.id} 'owner' must name another PERSON entity")
        if e.public and e.type is not EntityType.PERSON:
            raise DatasetError(f"{where}: entity {e.id}: only PERSON entities can be 'public'")

    raw_turns = _list(raw, "turns", where)
    if not raw_turns:
        raise DatasetError(f"{where}: needs at least one turn")
    turns = [_parse_turn(t, f"{where} turn {i}", set(by_id)) for i, t in enumerate(raw_turns)]

    mentioned = {m.entity for t in turns for m in t.mentions}
    unused = sorted(set(by_id) - mentioned)
    if unused:
        raise DatasetError(f"{where}: entities {unused} are never mentioned")
    if "no-pii" in tags and entities:
        raise DatasetError(f"{where}: tagged 'no-pii' but declares entities")

    return Conversation(conv_id, tuple(languages), source, tuple(tags), tuple(entities), tuple(turns))


# --------------------------------------------------------------------------- files


def load_jsonl(path: Path) -> list[Conversation]:
    conversations: list[Conversation] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{lineno}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{where}: invalid JSON ({exc.msg})") from None
        conv = parse_conversation(raw, where)
        if conv.id in seen:
            raise DatasetError(f"{where}: duplicate conversation id {conv.id!r}")
        seen.add(conv.id)
        conversations.append(conv)
    return conversations


def load_dataset(paths: Iterable[Path]) -> list[Conversation]:
    """Load several files (e.g. dev + test) and check that ids are unique across all of them."""
    out: list[Conversation] = []
    owner: dict[str, Path] = {}
    for path in paths:
        for conv in load_jsonl(path):
            if conv.id in owner:
                raise DatasetError(f"{path}: conversation id {conv.id!r} already used in {owner[conv.id]}")
            owner[conv.id] = path
            out.append(conv)
    return out


def dumps_jsonl(conversations: Iterable[Conversation]) -> str:
    lines = [json.dumps(c.to_json(), ensure_ascii=False, separators=(",", ":")) for c in conversations]
    return "\n".join(lines) + "\n" if lines else ""


@dataclass(frozen=True, slots=True)
class Stats:
    """The 05 §7 size and balance counts of a set of conversations."""

    conversations: int
    turns: int
    long: int  # conversations of 4+ turns
    code_switched: int
    no_pii: int
    languages: Mapping[str, int]  # conversations in which each language appears
    sources: Mapping[str, int]

    @classmethod
    def of(cls, conversations: Iterable[Conversation]) -> Stats:
        convs = list(conversations)
        return cls(
            conversations=len(convs),
            turns=sum(len(c.turns) for c in convs),
            long=sum(len(c.turns) >= 4 for c in convs),
            code_switched=sum(c.code_switched for c in convs),
            no_pii=sum("no-pii" in c.tags for c in convs),
            languages={lang: sum(lang in c.languages for c in convs) for lang in sorted(LANGUAGES)},
            sources={src: sum(c.source == src for c in convs) for src in sorted(SOURCES)},
        )

    def share(self, count: int) -> float:
        return count / self.conversations if self.conversations else 0.0

    def format(self) -> str:
        def row(label: str, count: int) -> str:
            return f"  {label:<16}{count:>5}  {100 * self.share(count):5.1f}%"

        return "\n".join(
            [
                f"{self.conversations} conversations, {self.turns} turns, {self.long} of 4+ turns",
                *(row(lang, n) for lang, n in self.languages.items()),
                row("code-switched", self.code_switched),
                row("no-pii", self.no_pii),
                *(row(src, n) for src, n in self.sources.items()),
            ]
        )


def dataset_hash(conversations: Iterable[Conversation]) -> str:
    """sha256 of the canonical form: formatting changes don't change it, content changes do."""
    return hashlib.sha256(dumps_jsonl(conversations).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nona.bench.dataset")
    parser.add_argument("command", choices=["check", "fmt", "hash", "stats"])
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    files: list[Path] = args.files
    try:
        if args.command == "check":
            convs = load_dataset(files)
            turns = sum(len(c.turns) for c in convs)
            mentions = sum(len(t.mentions) for c in convs for t in c.turns)
            print(f"ok: {len(convs)} conversations, {turns} turns, {mentions} mentions")
        elif args.command == "fmt":
            for path in files:
                path.write_text(dumps_jsonl(load_jsonl(path)), encoding="utf-8")
                print(f"formatted {path}")
        elif args.command == "hash":
            for path in files:
                print(f"{dataset_hash(load_jsonl(path))}  {path}")
        else:
            for path in files:
                print(f"{path}: {Stats.of(load_jsonl(path)).format()}")
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
