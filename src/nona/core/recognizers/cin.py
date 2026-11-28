"""Tunisian national ID (CIN): 8 digits, reported **only** next to a context word (deep dive A, layer 1).

Eight digits alone collide with order numbers, prices and dates, so a number counts as a CIN only when a
context word (`cin`, `carte d'identité`, `بطاقة`…) sits within 5 word tokens before or after it.

A cue belongs to **one** number (#100): the nearest 8-digit number on either side, never one reached through
another. A cue between two numbers goes to the side without a clause break (`,` `،` `;`), then to the nearer
one, then to the number after it (a cue usually comes first). A number joined to a CIN by a list joiner only
(`CIN 08123456 et 01234567`, `/`, `,`, `و`) is a CIN too. Deep dive A
allows a low-score span without context; we report nothing instead, because the resolver's threshold would
drop such a span anyway, and it keeps the dataset's order numbers out of the results.
"""

from __future__ import annotations

import re

from nona.core.normalize import NormalizedDoc
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span

WINDOW = 5  # word tokens on each side

_NUMBER = re.compile(r"(?<![\w.,/-])\d{8}(?![\w]|[.,/-]\d)")
_WORD = re.compile(r"\w+")
# Matched on the normalized (lower-case) view. `carte` alone is in the list on purpose (deep dive A): a bank
# card number is 13-19 digits, so it can't be mistaken for 8.
CONTEXT = re.compile(
    r"(?<!\w)(?:"
    r"cin|c\.i\.n\.?|carte|identité|identite|national id|identity card|id card"
    r"|بطاقة|ب\.ت\.و|التعريف"
    r")(?!\w)"
)

_CLAUSE_BREAK = re.compile(r"[,،;؛]")
# Only separators and a coordinating word between two numbers: one list, so they share a cue.
_LIST_JOINER = re.compile(r"[\s,،/&-]*(?:(?:et|ou|and|or|w|و)[\s,،]*)?")


def _owner(text: str, cue: re.Match[str], numbers: list[re.Match[str]]) -> int | None:
    """The index of the number this cue belongs to, or None when no number is within the window."""
    before = [i for i, n in enumerate(numbers) if n.end() <= cue.start()]
    after = [i for i, n in enumerate(numbers) if n.start() >= cue.end()]
    # (clause break between, words between, cue after the number, index): min() gives the owner.
    sides: list[tuple[bool, int, bool, int]] = []
    if before:
        sides.append((*_gap(text[numbers[before[-1]].end() : cue.start()]), True, before[-1]))
    if after:
        sides.append((*_gap(text[cue.end() : numbers[after[0]].start()]), False, after[0]))
    near = [side for side in sides if side[1] < WINDOW]
    return min(near)[3] if near else None


def _gap(text: str) -> tuple[bool, int]:
    return bool(_CLAUSE_BREAK.search(text)), len(_WORD.findall(text))


class CinRecognizer:
    @property
    def fingerprint(self) -> str:
        """No configuration: every instance detects the same (detection cache, `core/cache.py`)."""
        return "default"

    @property
    def name(self) -> str:
        return "context:cin"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset({EntityType.NATIONAL_ID})

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        text = doc.text
        numbers = list(_NUMBER.finditer(text))
        if not numbers:
            return []
        cued = {_owner(text, cue, numbers) for cue in CONTEXT.finditer(text)} - {None}
        if not cued:
            return []
        # Spread along lists: a run of numbers joined only by list joiners is cued as a whole.
        runs: list[list[int]] = [[0]]
        for i in range(1, len(numbers)):
            if _LIST_JOINER.fullmatch(text[numbers[i - 1].end() : numbers[i].start()]):
                runs[-1].append(i)
            else:
                runs.append([i])
        return [
            to_span(doc, numbers[i].start(), numbers[i].end(), EntityType.NATIONAL_ID, 0.95, self.name)
            for run in runs
            if cued.intersection(run)
            for i in run
        ]
