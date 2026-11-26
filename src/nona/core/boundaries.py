"""Where a placeholder or a known value may start and end: one rule for restoring and for propagation.

Restoring writes a value back wherever the model put a placeholder, and forward propagation must find that
value again when the client re-sends the restored text (ADR-0004). If restore accepted a position that
propagation doesn't, the value would leak on the next request: `لPERSON_1` restored to `لكنزة` was not
caught by a plain `\\w` boundary. So both use these boundaries:

- `_` separates words (`file_PERSON_1.pdf`, `PERSON_1_2024`);
- up to two Arabic proclitics may be attached in front (`لPERSON_1`, `وبPERSON_1`), as Arabic writes them;
- what normalization drops (tatweels, Arabic short vowels and marks) is looked past: up to two of them on
  the left, also around up to two proclitics (`ـلSami`, `لـالقصوري`, `لِسامي`), any number on the right
  (`Ksouriـ`). What follows them still decides, so a known `أمين` (Amin) isn't found in `أمينـة` (Amina), #80.
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

import itertools
import re
from collections.abc import Iterable

from nona.core.normalize import DROPPED
from nona.core.types import EntityType

NAME_TYPES = frozenset({EntityType.PERSON, EntityType.LOCATION, EntityType.ORGANIZATION, EntityType.CUSTOM})

PROCLITICS = "وفبلك"
TATWEEL = "\u0640"
_NOT_WORD = r"(?<![^\W_])"  # not preceded by a letter or digit (`_` is fine)
_DROPPED = "".join(sorted(DROPPED))
# What may sit between a word boundary and a value: up to two proclitics and up to two dropped characters,
# in any order. Fixed-width look-behinds, one per shape, behind a one-character gate.
_SHAPES = [
    shape
    for n in (1, 2, 3, 4)
    for shape in itertools.product("PD", repeat=n)
    if shape.count("P") <= 2 and shape.count("D") <= 2
]
_PREFIXES = "|".join(
    f"(?<={_NOT_WORD}" + "".join(f"[{PROCLITICS}]" if c == "P" else f"[{_DROPPED}]" for c in s) + ")"
    for s in _SHAPES
)
LEFT = rf"(?:{_NOT_WORD}|(?<=[{PROCLITICS}{_DROPPED}])(?:{_PREFIXES}))"
RIGHT = rf"(?=[{_DROPPED}]*+(?![^\W_]))"  # not followed by a letter or digit, past what normalization drops
LOOSE_LEFT = r"(?<![A-Za-z0-9])"
LOOSE_RIGHT = r"(?!\d)"

_STRICT_START = re.compile(LEFT)
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


def strict_at(text: str, start: int, end: int) -> bool:
    """Whether `text[start:end]` sits at strict boundaries (see above)."""
    return bool(_STRICT_START.match(text, start)) and bool(_STRICT_END.match(text, end))


def loose_at(text: str, start: int, end: int) -> bool:
    """Whether `text[start:end]` sits at loose boundaries (see above)."""
    return bool(_LOOSE_START.match(text, start)) and bool(_LOOSE_END.match(text, end))
