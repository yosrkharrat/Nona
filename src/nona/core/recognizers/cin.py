"""Tunisian national ID (CIN): 8 digits, reported **only** next to a context word (deep dive A, layer 1).

Eight digits alone collide with order numbers, prices and dates, so a number counts as a CIN only when a
context word (`cin`, `carte d'identité`, `بطاقة`…) sits within 5 word tokens before or after it.

A cue followed by two numbers serves the first one only when the second has its own label right before it:
`CIN 08123456, commande 45678912`, `رقم بطاقة التعريف …، ورقم الطلبية …` (#100). Nothing else is dropped: a
missed CIN leaks, while an extra one only over-masks.

Deep dive A allows a low-score span without context; we report nothing instead, because the resolver's
threshold would drop such a span anyway, and it keeps the dataset's order numbers out of the results.
"""

from __future__ import annotations

import re
from bisect import bisect_left

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

# Labels of other 8-digit numbers: order, invoice, parcel, tracking, receipt, reference. Only nouns that
# can't be read another way, since a label that drops a real CIN leaks it: `de suivi`, not `suivi`;
# `الوصل` (receipt), not `وصل` (also "arrived"); `رقم الطلب`, not `الطلب` (also "the request").
# Arabic labels take a و/ف conjunction, then ب/ل, the article, or both (`بال`, `لل`); `رقم`/`عدد` takes the
# conjunction and ب/ل but no article, since it heads a construct phrase (`رقم الطلب`). The words of a two-word
# label are separated by spaces or tabs only: a label wrapped onto the next line is missed, which over-masks.
_SP = r"[ \t\u00a0]+"
_CONJ = r"[وف]?"
_ART = r"(?:[بل]?ال|لل)"
OTHER_LABEL = re.compile(
    r"(?<!\w)(?:"
    rf"commande|facture|colis|de{_SP}suivi|référence|reference|réf|ref|ticket|order|invoice|tracking"
    rf"|{_CONJ}{_ART}?(?:طلبية|فاتورة|طرد|مرجع)|{_CONJ}{_ART}وصل|{_CONJ}[بل]?(?:رقم|عدد){_SP}(?:ال)?طلب"
    r")(?!\w)"
)
LABEL_GAP = 2  # a label at most this many words before a number is its own


def _own_label(gap: str) -> bool:
    """`gap` (the text between two numbers) ends with a label for the second one, with no cue after it."""
    for label in OTHER_LABEL.finditer(gap):
        tail = gap[label.end() :]
        if len(_WORD.findall(tail)) <= LABEL_GAP and not CONTEXT.search(tail):
            return True
    return False


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
        starts = [m.start() for m in _WORD.finditer(text)]

        def words(lo: int, hi: int) -> int:  # word tokens starting in [lo, hi)
            return bisect_left(starts, hi) - bisect_left(starts, lo)

        cued: set[int] = set()
        first_after = [n.start() for n in numbers]
        for cue in CONTEXT.finditer(text):
            # The whole cue must fit in the 5 tokens next to the number.
            j = bisect_left(first_after, cue.end())
            k = j
            while k < len(numbers) and words(cue.start(), numbers[k].start()) <= WINDOW:
                if k == j or not _own_label(text[numbers[k - 1].end() : numbers[k].start()]):
                    cued.add(k)
                k += 1
            k = j - 1
            while k >= 0 and words(numbers[k].end(), cue.end()) <= WINDOW:
                cued.add(k)
                k -= 1
        return [
            to_span(doc, n.start(), n.end(), EntityType.NATIONAL_ID, 0.95, self.name)
            for i, n in enumerate(numbers)
            if i in cued
        ]
