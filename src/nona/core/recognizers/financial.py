"""IBAN, Tunisian RIB and payment cards: found only when their checksum is valid (FR-1, deep dive A).

The numbers are often written in groups (`TN59 1000 6035…`, `4111-1111-…`), and the text right after them
can look like one more group (`… 8831 merci`, `… 1111 12/27`). So from each token start, a recognizer joins
the following tokens one separator at a time and keeps the **longest** run whose checksum is valid. Runs are
capped in length, so the cost stays linear in the text.

Overlaps between these recognizers (the RIB inside a Tunisian IBAN, a Luhn-valid prefix of an IBAN) are
expected. The span resolver keeps the longest span.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

from nona.core.checksums import compact, iban_valid, luhn_valid, tn_rib_valid
from nona.core.normalize import NormalizedDoc
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span

_ALNUM = re.compile(r"[a-z0-9]+")
_DIGITS = re.compile(r"[0-9]+")
_SEPARATORS = frozenset(" -")
_MAX_RUN = 42  # longest IBAN (34) with a separator every 4 characters


def _token_starts(text: str, token: re.Pattern[str]) -> Iterator[int]:
    """Start of every token that isn't glued to a preceding letter or digit. `_` separates words, as in
    `core/boundaries.py`: `card_4242…` and `4242…_` are numbers (#93), though after `x_` a short token
    stands alone (`_scan`)."""
    for m in token.finditer(text):
        if m.start() == 0 or not text[m.start() - 1].isalnum():
            yield m.start()


def _run_ends(text: str, start: int, token: re.Pattern[str]) -> list[int]:
    """End offsets of the growing runs `t1`, `t1 t2`, `t1 t2 t3`… joined by single spaces or dashes."""
    ends: list[int] = []
    pos = start
    while (m := token.match(text, pos)) is not None and m.end() - start <= _MAX_RUN:
        end = m.end()
        if end < len(text) and text[end].isalnum():
            break  # the token continues with other characters: `1234abc` is not a number
        ends.append(end)
        if end + 1 < len(text) and text[end] in _SEPARATORS and token.match(text, end + 1):
            pos = end + 1
        else:
            break
    return ends


def _scan(
    doc: NormalizedDoc,
    token: re.Pattern[str],
    accept: Callable[[str], bool],
    etype: EntityType,
    detector: str,
) -> list[Span]:
    spans: list[Span] = []
    covered = 0
    text = doc.text
    for start in _token_starts(text, token):
        if start < covered:
            continue
        ends = _run_ends(text, start, token)
        if (
            start >= 2
            and text[start - 1] == "_"
            and text[start - 2].isalnum()
            and ends
            and ends[0] - start < 4
        ):
            # A short first token after `x_` stands alone: in `PERSON_2 4139 4411 7715 1620` the `2` is the
            # placeholder's. A full group starts a run (`card_4242 4242 4242 4242`).
            ends = ends[:1]
        for end in reversed(ends):
            if accept(text[start:end]):
                spans.append(to_span(doc, start, end, etype, 1.0, detector))
                covered = end
                break
    return spans


def _iban(candidate: str) -> bool:
    return re.match(r"[a-z]{2}\d{2}", candidate) is not None and iban_valid(candidate)


def card_number(candidate: str) -> bool:
    """13 to 19 digits, maybe grouped, from a card network, Luhn-valid."""
    return _card(candidate)


def _card(candidate: str) -> bool:
    digits = compact(candidate)
    # Card networks start with 2-6 (Mastercard 2/5, Amex and Diners 3, Visa 4, Discover 6).
    return 13 <= len(digits) <= 19 and digits[0] in "23456" and luhn_valid(digits)


class IbanRecognizer:
    @property
    def fingerprint(self) -> str:
        """No configuration: every instance detects the same (detection cache, `core/cache.py`)."""
        return "default"

    @property
    def name(self) -> str:
        return "checksum:iban"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset({EntityType.IBAN})

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        return _scan(doc, _ALNUM, _iban, EntityType.IBAN, self.name)


class RibRecognizer:
    """Tunisian RIB: 20 digits (bank 2, branch 3, account 13, key 2), divisible by 97."""

    @property
    def fingerprint(self) -> str:
        """No configuration: every instance detects the same (detection cache, `core/cache.py`)."""
        return "default"

    @property
    def name(self) -> str:
        return "checksum:rib"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset({EntityType.RIB})

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        return _scan(doc, _DIGITS, tn_rib_valid, EntityType.RIB, self.name)


class CardRecognizer:
    @property
    def fingerprint(self) -> str:
        """No configuration: every instance detects the same (detection cache, `core/cache.py`)."""
        return "default"

    @property
    def name(self) -> str:
        return "checksum:card"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset({EntityType.CARD})

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        return _scan(doc, _DIGITS, _card, EntityType.CARD, self.name)
