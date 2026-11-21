"""Phone numbers, validated by libphonenumber (`phonenumbers`), so `22 123 456` and `+33 6 12 34 56 78`
are found but dates, prices and card numbers are not (deep dive A, layer 1).

Numbers are matched with each configured region as the default, so national formats work for Tunisia and
France, and any number written in international format (`+216…`, `00216…`, `+44…`) is found regardless.
"""

from __future__ import annotations

from collections.abc import Sequence

import phonenumbers

from nona.core.normalize import NormalizedDoc
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span


class PhoneRecognizer:
    def __init__(self, regions: Sequence[str] = ("TN", "FR")) -> None:
        if not regions:
            raise ValueError("at least one region is needed")
        self._regions = tuple(regions)

    @property
    def name(self) -> str:
        return "phonenumbers"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset({EntityType.PHONE})

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        found: dict[tuple[int, int], None] = {}  # ordered set: the same number can match in several regions
        for region in self._regions:
            matcher = phonenumbers.PhoneNumberMatcher(doc.text, region, leniency=phonenumbers.Leniency.VALID)
            for match in matcher:
                found[(match.start, match.end)] = None
        return [to_span(doc, s, e, EntityType.PHONE, 1.0, self.name) for s, e in sorted(found)]
