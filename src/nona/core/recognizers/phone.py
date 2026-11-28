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

One context rule (deep dive A, layer 4): **an order cue lowers the score.** Eight-digit order numbers are
often valid Tunisian mobiles (`Ma commande n° 24176705`). When an unbroken run of digits in national format
comes right after an order cue (`commande`, `réf`, `order`, `facture`…), with at most connectors such as `n°`,
`numéro`, `#` or `:` in between on the same line, the span is reported with `CUE_SCORE`, below the resolver's
default threshold. A dot counts only glued to an abbreviation (`réf.`, `cmd.`): a full stop ends the
sentence, and in `…la commande. 22123456 est mon whatsapp` the number is a phone.

Every limit is on purpose, as a leak costs more than over-masking:
- the cue must be adjacent: `envoie la facture au 22 123 456` is a phone number;
- grouped digits (`22 123 456`, `22.123.456`) are how people write phones, never lowered;
- a cue doesn't reach across a line break;
- `suivi`, `colis`, `dossier` and `ticket` aren't cues: `Pour le suivi : 22123456` gives a contact number;
- a number in international format (`+216…`, `00216…`) is never lowered, since nobody writes an order number
  that way.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import phonenumbers

from nona.core.boundaries import BLANK
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
CUE_SCORE = 0.3  # below the resolver's default `min_score` of 0.5, so the span is kept only if asked for
# Matched on the normalized (lower-case) view, against the text that ends right where the number starts.
# `[ \t]`, not `\s`, and `\Z`, not `$`: a cue never reaches across a line break. `colis`, `suivi`, `dossier`
# and `ticket` are left out: they introduce a contact number as often as a reference (`Pour le suivi : …`).
_ORDER_CUE = re.compile(
    r"(?<!\w)(?:commande|cmd\.?|r[ée]f\.?|r[ée]f[ée]rence|facture"
    r"|order|invoice|tracking|booking|طلبية|الطلبية|فاتورة|الفاتورة)"
    rf"(?:[ \t{BLANK}]*(?:n°|n\.|no\.?|nº|num[ée]ro|num\.?|number|nr\.?|#|:|-|رقم))*[ \t{BLANK}]*\Z"
)
_INTERNATIONAL = re.compile(r"\s*(?:\+|00)")

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
    def fingerprint(self) -> str:
        """Its configuration, for the detection cache (`core/cache.py`)."""
        return f"regions={','.join(self._regions)};skip={','.join(map(str, sorted(self._skip)))}"

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
        return [
            to_span(doc, s, e, EntityType.PHONE, _score(doc.text, s, e), self.name) for s, e in sorted(found)
        ]


def order_cue(word: str) -> bool:
    """Whether `word` (normalized) is an order cue on its own (`commande`, `ref`, `طلبية`): the letters glued
    in front of a number that the structured recognizers' second pass leaves in place (`core/pipeline.py`),
    so `commande22123456` stays an order number, as `commande 22123456` is (#93)."""
    return bool(_ORDER_CUE.fullmatch(word))


def _score(text: str, start: int, end: int) -> float:
    """1.0, or `CUE_SCORE` for an unbroken national number right after an order cue.

    Only an unbroken run of digits is lowered: grouping (`22 123 456`, `22.123.456`) is how people write
    phones, not order numbers.
    """
    if _INTERNATIONAL.match(text, start) or not text[start:end].isdigit():
        return 1.0
    return CUE_SCORE if _ORDER_CUE.search(text, max(0, start - 40), start) else 1.0
