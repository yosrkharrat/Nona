"""Tunisian national ID (CIN): 8 digits, reported **only** next to a context word (deep dive A, layer 1).

Eight digits alone collide with order numbers, prices and dates, so a number counts as a CIN only when a
context word (`cin`, `carte d'identité`, `بطاقة`…) sits within 5 word tokens before or after it.

One cue can sit near two numbers: `CIN 08123456, commande 45678912` (#100). A number is left out when another
label claims it (`commande`, `facture`, `الطلبية`… between the cue and it, or just before it) **and** the same
cue already has a number that no label claims. Without such a number the cue still counts, so
`ma CIN pour la commande : 08123456` stays masked: a missed CIN leaks, an extra one only over-masks.

Deep dive A allows a low-score span without context; we report nothing instead, because the resolver's
threshold would drop such a span anyway, and it keeps the dataset's order numbers out of the results.
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

# Labels of other 8-digit numbers (order, invoice, parcel, reference). A leading و/ب/ل/ف clitic is allowed.
OTHER_LABEL = re.compile(
    r"(?<!\w)(?:"
    r"commande|facture|colis|suivi|référence|reference|réf|ref|ticket|order|invoice|tracking"
    r"|[وبلف]?(?:الطلبية|طلبية|الطلب|الفاتورة|فاتورة|الطرد|المرجع|الوصل)"
    r")(?!\w)"
)
CLAIM = 2  # a label at most this many words before a number claims it


def _words(text: str) -> int:
    return len(_WORD.findall(text))


def _between(a: re.Match[str], b: re.Match[str]) -> tuple[int, int]:
    return (a.end(), b.start()) if a.end() <= b.start() else (b.end(), a.start())


def _claimed(text: str, cue: re.Match[str], number: re.Match[str], labels: list[re.Match[str]]) -> bool:
    """Another label owns this number: it sits between the cue and the number, or right before the number
    with no number and no cue in between (`colis : CIN 08123456` is still the CIN's)."""
    lo, hi = _between(cue, number)
    for label in labels:
        if lo <= label.start() and label.end() <= hi:
            return True
        gap = text[label.end() : number.start()]
        if (
            label.end() <= number.start()
            and _words(gap) <= CLAIM
            and not (_NUMBER.search(gap) or CONTEXT.search(gap))
        ):
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
        labels = list(OTHER_LABEL.finditer(text))
        cued: set[int] = set()
        for cue in CONTEXT.finditer(text):
            near = [i for i, n in enumerate(numbers) if _words(text[slice(*_between(cue, n))]) < WINDOW]
            claimed = {i for i in near if _claimed(text, cue, numbers[i], labels)}
            # Drop the claimed numbers only when the cue has another one to describe.
            cued.update(set(near) - claimed if set(near) - claimed else near)
        return [
            to_span(doc, n.start(), n.end(), EntityType.NATIONAL_ID, 0.95, self.name)
            for i, n in enumerate(numbers)
            if i in cued
        ]
