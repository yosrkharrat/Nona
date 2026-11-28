"""Where a placeholder or a known value may start and end: one rule for restoring and for propagation.

Restoring writes a value back wherever the model put a placeholder, and forward propagation must find that
value again when the client re-sends the restored text (ADR-0004). If restore accepted a position that
propagation doesn't, the value would leak on the next request: `لPERSON_1` restored to `لكنزة` was not
caught by a plain `\\w` boundary. So both use these boundaries:

- `_` separates words (`file_PERSON_1.pdf`, `PERSON_1_2024`);
- up to two Arabic proclitics may be attached in front (`لPERSON_1`, `وبPERSON_1`), as Arabic writes them;
- what normalization drops (tatweels, Arabic short vowels and marks) is looked past, on the left also
  around the proclitics (`ـلSami`, `لـــالقصوري`, `لِسامي`), on the right too (`Ksouriـ`). What is beyond
  them still decides, so a known `أمين` (Amin) isn't found in `أمينـة` (Amina) or `تَأمين`, #80.
  Multi-word values are found on the normalized view, so restore checks them there too;
- any other letter or digit glued on either side is not a boundary: `PERSON_1abc` and `lPERSON_1` are left
  as they are, so they can't leak either. That includes a placeholder detection leaves glued, which restore
  doesn't write back (`سميرPERSON_1`, `aPHONE_1`, `IP_ADDRESS_1EMAIL_1`): it stays masked, #87.

Placeholders the user typed are found with the loose boundaries (`placeholder_regex`): `عندPERSON_2` still
reserves number 2. No detection may cut through one in its exact shape (`TYPED_PLACEHOLDER`,
`core/resolver.py`); the looser forms (`card 4242…`, `RIB 0812…`) are too often real values.

These strict boundaries are for **names** (`NAME_TYPES`), which a word boundary is all that separates from
ordinary words (`Sami` in `Samir`). Structured values (emails, phones, IBANs…) are distinctive on their own,
so they use loose ones (`LOOSE_LEFT`, `LOOSE_RIGHT`) on both sides: `a@example.comـ`, a value glued to an
Arabic letter, is restored and caught again like any other.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from nona.core.normalize import DROPPED
from nona.core.types import EntityType

NAME_TYPES = frozenset({EntityType.PERSON, EntityType.LOCATION, EntityType.ORGANIZATION, EntityType.CUSTOM})

PROCLITICS = "وفبلك"
_DROPPED = "".join(sorted(DROPPED))
# Not preceded by a letter, a digit or what normalization drops (`_` is fine): `تَأمين` holds no `أمين`.
_NOT_WORD = rf"(?<![^\W_])(?<![{_DROPPED}])"
# A name may start after up to two proclitics, with any number of dropped characters around them (`ـلSami`,
# `لـــالقصوري`, `لِسامي`). Python's look-behind is fixed-width, so the prefix is in the match and the name
# is its group `v` (`resolver.known_spans`).
NAME_START = rf"{_NOT_WORD}[{_DROPPED}]*(?:[{PROCLITICS}][{_DROPPED}]*){{0,2}}"
RIGHT = rf"(?=[{_DROPPED}]*+(?![^\W_]))"  # not followed by a letter or digit, past what normalization drops
LOOSE_LEFT = r"(?<![A-Za-z0-9])"
LOOSE_RIGHT = r"(?!\d)"

_STRICT_END = re.compile(RIGHT)
_LOOSE_START = re.compile(LOOSE_LEFT)
_LOOSE_END = re.compile(LOOSE_RIGHT)


def _type_pattern(etype: EntityType) -> str:
    return etype.value.replace("_", "[ _]")


def placeholder_regex(types: Iterable[EntityType]) -> re.Pattern[str]:
    """`TYPE_n`, case-insensitive, also `TYPE n` and `TYPEn`. Group 1 is the type, group 2 the number, at
    most 6 digits as issued (`int()` refuses more than 4300).

    Loose boundaries here; restoring then holds name placeholders to the strict ones, like forward
    propagation does, so a restored value is always found again.
    """
    names = sorted({_type_pattern(t) for t in types}, key=len, reverse=True)
    if not names:
        return re.compile(r"(?!)")  # matches nothing
    return re.compile(LOOSE_LEFT + "(" + "|".join(names) + r")[ _]?(\d{1,6})" + LOOSE_RIGHT, re.IGNORECASE)


# Any placeholder the user may have typed, of any type.
ANY_PLACEHOLDER = placeholder_regex(EntityType)
# Exactly as issued: upper case, `_`, a short number. `PERSON_1`, never `card 4242424242424242`.
TYPED_PLACEHOLDER = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(sorted((t.value for t in EntityType), key=len, reverse=True))
    + r")_\d{1,6}(?!\d)"
)
# The same, glued to whatever is on its left (`22 123 456PERSON_1`, `a@x.comEMAIL_1`). Group 1 is the type,
# group 2 the number. To blank placeholders out before the structured recognizers' second pass
# (`core/pipeline.py`, #85), which only adds, and to reserve their numbers (`reserved_numbers`).
GLUED_PLACEHOLDER = re.compile(
    "(" + "|".join(sorted((t.value for t in EntityType), key=len, reverse=True)) + r")_(\d{1,6})(?!\d)"
)


def strict_at(text: str, start: int, end: int) -> bool:
    """Whether `text[start:end]` sits at strict boundaries (see above): `NAME_START` before it, walked back
    here, and `RIGHT` after it."""
    i, proclitics = start, 0
    while i > 0 and (text[i - 1] in DROPPED or (text[i - 1] in PROCLITICS and proclitics < 2)):
        proclitics += text[i - 1] in PROCLITICS
        i -= 1
    return not (i > 0 and text[i - 1].isalnum()) and bool(_STRICT_END.match(text, end))


def composes_after(text: str, end: int) -> bool:
    """Whether a combining mark follows `end`, past the marks normalization drops: on the normalized view it
    would compose with the value's last letter (`Ksouri` + U+0301 is `Ksourí`). A tatweel is a base
    character, so a mark after one composes with nothing."""

    def mark(c: str) -> bool:
        return unicodedata.category(c).startswith("M")

    while end < len(text) and text[end] in DROPPED and mark(text[end]):
        end += 1
    return end < len(text) and mark(text[end])


def loose_at(text: str, start: int, end: int) -> bool:
    """Whether `text[start:end]` sits at loose boundaries (see above)."""
    return bool(_LOOSE_START.match(text, start)) and bool(_LOOSE_END.match(text, end))
