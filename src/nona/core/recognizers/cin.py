"""Tunisian national ID (CIN): 8 digits, reported **only** next to a context word (deep dive A, layer 1).

Eight digits alone collide with order numbers, prices and dates, so a number counts as a CIN only when a
context word (`cin`, `carte d'identité`, `بطاقة`…) sits within 5 word tokens before or after it. Deep dive A
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


def _window(text: str, start: int, end: int, tokens: list[tuple[int, int]]) -> str:
    """The text from 5 tokens before the number to 5 tokens after it."""
    before = [t for t in tokens if t[1] <= start][-WINDOW:]
    after = [t for t in tokens if t[0] >= end][:WINDOW]
    lo = before[0][0] if before else start
    hi = after[-1][1] if after else end
    return text[lo:start] + " " + text[end:hi]


class CinRecognizer:
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
        tokens = [(m.start(), m.end()) for m in _WORD.finditer(text)]
        return [
            to_span(doc, m.start(), m.end(), EntityType.NATIONAL_ID, 0.95, self.name)
            for m in numbers
            if CONTEXT.search(_window(text, m.start(), m.end(), tokens))
        ]
