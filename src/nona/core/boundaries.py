"""Where a placeholder or a known value may start and end: one rule for restoring and for propagation.

Restoring writes a value back wherever the model put a placeholder, and forward propagation must find that
value again when the client re-sends the restored text (ADR-0004). If restore accepted a position that
propagation doesn't, the value would leak on the next request: `لPERSON_1` restored to `لكنزة` was not
caught by a plain `\\w` boundary. So both use these boundaries:

- `_` separates words (`file_PERSON_1.pdf`, `PERSON_1_2024`);
- up to two Arabic proclitics may be attached in front (`لPERSON_1`, `وبPERSON_1`), as Arabic writes them;
- a placeholder glued on either side is a boundary (`سمPERSON_1`, `PERSON_1محمد`): no detection cuts
  through a typed one, so the piece next to it gets its own placeholder, glued to it, and restore must put
  that value back (#80). Before the value: `TYPE_n` exactly, any case. After it, Python's fixed-width
  look-behind can't list the types, so any `x_n` shape counts (`file_1محمد`); a looser boundary only makes
  propagation mask more, and restore accepts the same places;
- a tatweel (`ـ`, which normalization drops) is not a letter here: `سميرـ`, `Ksouriـ`;
- any other letter or digit glued on either side is not a boundary: `PERSON_1abc` and `lPERSON_1` are left
  as they are, so they can't leak either.

Placeholders the user typed are found with the loose boundaries (`placeholder_regex`): `عندPERSON_2` still
reserves number 2. No detection may cut through one in its exact shape (`TYPED_PLACEHOLDER`,
`core/resolver.py`); the looser forms (`card 4242…`, `RIB 0812…`) are too often real values.

These strict boundaries are for **names** (`NAME_TYPES`), which a word boundary is all that separates from
ordinary words (`Sami` in `Samir`). Structured values (emails, phones, IBANs…) are distinctive on their own,
so they use loose ones (`LOOSE_LEFT`, `LOOSE_RIGHT`) on both sides: `a@example.comـ`, a value glued to an
Arabic letter, is restored and caught again like any other. A value that starts with a sign (`+216…`,
`(71) …`) needs no boundary on its left at all: `a+216 22 123 456` is restored and caught again (#80).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from nona.core.types import EntityType

NAME_TYPES = frozenset({EntityType.PERSON, EntityType.LOCATION, EntityType.ORGANIZATION, EntityType.CUSTOM})

PROCLITICS = "وفبلك"
_TYPE_NAMES = "|".join(sorted((t.value for t in EntityType), key=len, reverse=True))
# Right after a placeholder (`PERSON_1محمد`): the `(?<=\d)` gate keeps the common case to one check.
_AFTER_PLACEHOLDER = "(?<=\\d)(?:" + "|".join(rf"(?<=[A-Za-z]_\d{{{n}}})" for n in range(1, 7)) + ")"
# Right before one (`سمPERSON_1`), as issued, in any case: propagation also runs on the lowercased view.
_BEFORE_PLACEHOLDER = rf"(?=(?i:{_TYPE_NAMES})_\d{{1,6}}(?!\d))"
_NOT_WORD = r"(?<![^\W_\u0640])"  # not preceded by a letter or digit (`_` and a tatweel are fine)
_PROCLITIC = rf"(?<={_NOT_WORD}[{PROCLITICS}])|(?<={_NOT_WORD}[{PROCLITICS}]{{2}})"
LEFT = rf"(?:{_NOT_WORD}|{_PROCLITIC}|{_AFTER_PLACEHOLDER})"
# Not followed by a letter or digit, unless it is a placeholder.
RIGHT = rf"(?:(?![^\W_\u0640])|{_BEFORE_PLACEHOLDER})"
LOOSE_LEFT = rf"(?:(?<![A-Za-z0-9])|{_AFTER_PLACEHOLDER})"
LOOSE_RIGHT = r"(?!\d)"

_STRICT_START = re.compile(LEFT)
_STRICT_END = re.compile(RIGHT)
_LOOSE_START = re.compile(LOOSE_LEFT)


def _type_pattern(etype: EntityType) -> str:
    return etype.value.replace("_", "[ _]")


def placeholder_regex(types: Iterable[EntityType], *, left: bool = True) -> re.Pattern[str]:
    """`TYPE_n`, case-insensitive, also `TYPE n` and `TYPEn`. Group 1 is the type, group 2 the number.

    Loose boundaries here, none on the left with `left=False`; restoring then checks the left itself, and
    holds name placeholders to the strict boundaries, like forward propagation does, so a restored value is
    always found again.
    """
    return _placeholder_regex(types, LOOSE_LEFT if left else "")


def _placeholder_regex(types: Iterable[EntityType], left: str) -> re.Pattern[str]:
    names = sorted({_type_pattern(t) for t in types}, key=len, reverse=True)
    if not names:
        return re.compile(r"(?!)")  # matches nothing
    return re.compile(left + "(" + "|".join(names) + r")[ _]?(\d+)" + LOOSE_RIGHT, re.IGNORECASE)


# Any placeholder the user may have typed, of any type.
ANY_PLACEHOLDER = placeholder_regex(EntityType)
# Numbers never issued: every placeholder shape, glued or not (`SamiPERSON_1`). Masking can glue an issued
# placeholder to one the user typed, so their boundaries can change (`PERSON_2PERSON_1`); with every shape
# reserved, restore never takes a placeholder the user typed for an issued one (#80).
RESERVED_PLACEHOLDER = _placeholder_regex(EntityType, "")
# Exactly as issued: upper case, `_`, a short number. `PERSON_1`, never `card 4242424242424242`.
TYPED_PLACEHOLDER = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(sorted((t.value for t in EntityType), key=len, reverse=True))
    + r")_\d{1,6}(?!\d)"
)


def self_delimited(value: str) -> bool:
    """Whether a structured value needs no boundary on its left: it starts with a sign (`+216…`)."""
    return bool(value) and not value[0].isalnum()


def loose_start(text: str, start: int) -> bool:
    """Whether a structured value may start at `start` (`LOOSE_LEFT`)."""
    return bool(_LOOSE_START.match(text, start))


def strict_at(text: str, start: int, end: int) -> bool:
    """Whether `text[start:end]` sits at strict boundaries (see above)."""
    return bool(_STRICT_START.match(text, start)) and bool(_STRICT_END.match(text, end))
