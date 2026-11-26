"""Where a placeholder or a known value may start and end: one rule for restoring and for propagation.

Restoring writes a value back wherever the model put a placeholder, and forward propagation must find that
value again when the client re-sends the restored text (ADR-0004). If restore accepted a position that
propagation doesn't, the value would leak on the next request: `لPERSON_1` restored to `لكنزة` was not
caught by a plain `\\w` boundary. So propagation uses these boundaries, and restore writes a value only where
propagation finds it again in the restored text (`pseudonymize.restore`):

- `_` separates words (`file_PERSON_1.pdf`, `PERSON_1_2024`);
- up to two Arabic proclitics may be attached in front (`لPERSON_1`, `وبPERSON_1`), as Arabic writes them,
  also after a tatweel (`ـلPERSON_1`); a tatweel anywhere in a name, or at its edges, is part of it
  (`Ksouriـ`, `لـالقصوري`), since normalization drops it (`resolver.known_spans`);
- any other letter or digit glued on either side is not a boundary: `PERSON_1abc` and `lPERSON_1` are left
  as they are, so they can't leak either. Detection makes sure its own placeholders never end up glued
  (`resolver.resolve`): a name glued to a typed placeholder is masked with it (`سميرPERSON_1`), and a value
  glued to a word takes the word in (`a+216 22 123 456`), #80.

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
from collections.abc import Iterable

from nona.core.types import EntityType

NAME_TYPES = frozenset({EntityType.PERSON, EntityType.LOCATION, EntityType.ORGANIZATION, EntityType.CUSTOM})

PROCLITICS = "وفبلك"
TATWEEL = "\u0640"
_NOT_WORD = r"(?<![^\W_])"  # not preceded by a letter or digit (`_` is fine)
# Up to two proclitics, maybe behind a tatweel (`ـلSami`); a tatweel after them is in the name's pattern.
_PROCLITIC = "|".join(rf"(?<={_NOT_WORD}{t}[{PROCLITICS}]{{{n}}})" for t in ("", TATWEEL) for n in (1, 2))
LEFT = rf"(?:{_NOT_WORD}|{_PROCLITIC})"
RIGHT = r"(?![^\W_])"  # not followed by a letter or digit
LOOSE_LEFT = r"(?<![A-Za-z0-9])"
LOOSE_RIGHT = r"(?!\d)"

_LOOSE_START = re.compile(LOOSE_LEFT)


def _type_pattern(etype: EntityType) -> str:
    return etype.value.replace("_", "[ _]")


def placeholder_regex(types: Iterable[EntityType], *, left: bool = True) -> re.Pattern[str]:
    """`TYPE_n`, case-insensitive, also `TYPE n` and `TYPEn`. Group 1 is the type, group 2 the number, at
    most 6 digits as issued (`int()` refuses more than 4300). With `left=False`, no boundary on the left:
    the caller checks it (`loose_start`).
    """
    names = sorted({_type_pattern(t) for t in types}, key=len, reverse=True)
    if not names:
        return re.compile(r"(?!)")  # matches nothing
    start = LOOSE_LEFT if left else ""
    return re.compile(start + "(" + "|".join(names) + r")[ _]?(\d{1,6})" + LOOSE_RIGHT, re.IGNORECASE)


# Any placeholder the user may have typed, of any type.
ANY_PLACEHOLDER = placeholder_regex(EntityType)
# Exactly as issued: upper case, `_`, a short number. `PERSON_1`, never `card 4242424242424242`.
TYPED_PLACEHOLDER = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(sorted((t.value for t in EntityType), key=len, reverse=True))
    + r")_\d{1,6}(?!\d)"
)


def loose_start(text: str, start: int) -> bool:
    """Whether `text` passes `LOOSE_LEFT` at `start`."""
    return bool(_LOOSE_START.match(text, start))
