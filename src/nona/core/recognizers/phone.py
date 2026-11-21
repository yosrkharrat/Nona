"""Phone numbers, validated by libphonenumber (`phonenumbers`), so `22 123 456` and `+33 6 12 34 56 78`
are found but dates, prices and card numbers are not (deep dive A, layer 1).

Numbers are matched with each configured region as the default, so national formats work for Tunisia and
France, and any number written in international format (`+216…`, `00216…`, `+44…`) is found regardless.

Two rules on top of libphonenumber:
- **`+216 (0)71 234 567`** is found. Tunisia has no trunk `0`, but people copy the French `(0)` habit, and
  libphonenumber rejects the form. A miss would be a leak.
- **Business service numbers are skipped**: toll-free (TN `8010…`), shared cost (`8110…`, `8210…`) and premium
  rate (`88…`). They identify a company, not a person, so masking them is over-masking. Set
  `include_service_numbers=True` to mask them anyway.

Fax numbers need no rule: a fax line is an ordinary fixed line, and it is masked like any phone.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import phonenumbers

from nona.core.normalize import NormalizedDoc
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span

SERVICE_TYPES: frozenset[int] = frozenset(
    {
        phonenumbers.PhoneNumberType.TOLL_FREE,
        phonenumbers.PhoneNumberType.SHARED_COST,
        phonenumbers.PhoneNumberType.PREMIUM_RATE,
    }
)
# `+216 (0)71 234 567`, `00216 (0) 71 234 567`, `+33 (0)6 12 34 56 78`: country code, `(0)`, national number.
_TRUNK_ZERO = re.compile(r"(?<![\w+])(?:\+|00)\s?(\d{1,3})[\s.]?\(0\)[\s.-]?((?:\d[\s.-]?){6,13}\d)(?!\d)")


class PhoneRecognizer:
    def __init__(
        self, regions: Sequence[str] = ("TN", "FR"), *, include_service_numbers: bool = False
    ) -> None:
        if not regions:
            raise ValueError("at least one region is needed")
        self._regions = tuple(regions)
        self._skip: frozenset[int] = frozenset() if include_service_numbers else SERVICE_TYPES

    @property
    def name(self) -> str:
        return "phonenumbers"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset({EntityType.PHONE})

    def _keep(self, number: phonenumbers.PhoneNumber) -> bool:
        return phonenumbers.number_type(number) not in self._skip

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        found: dict[tuple[int, int], None] = {}  # ordered set: the same number can match in several regions
        for region in self._regions:
            matcher = phonenumbers.PhoneNumberMatcher(doc.text, region, leniency=phonenumbers.Leniency.VALID)
            for match in matcher:
                if self._keep(match.number):
                    found[(match.start, match.end)] = None
        for m in _TRUNK_ZERO.finditer(doc.text):
            national = re.sub(r"\D", "", m.group(2)).removeprefix("0")
            try:
                number = phonenumbers.parse(f"+{m.group(1)}{national}", None)
            except phonenumbers.NumberParseException:
                continue
            if phonenumbers.is_valid_number(number) and self._keep(number):
                found[(m.start(), m.end())] = None
        return [to_span(doc, s, e, EntityType.PHONE, 1.0, self.name) for s, e in sorted(found)]
