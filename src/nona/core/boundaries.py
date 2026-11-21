"""Where a placeholder or a known value may start and end: one rule for restoring and for propagation.

Restoring writes a value back wherever the model put a placeholder, and forward propagation must find that
value again when the client re-sends the restored text (ADR-0004). If restore accepted a position that
propagation doesn't, the value would leak on the next request: `لPERSON_1` restored to `لكنزة` was not
caught by a plain `\\w` boundary. So both use these boundaries:

- `_` separates words (`file_PERSON_1.pdf`, `PERSON_1_2024`);
- up to two Arabic proclitics may be attached in front (`لPERSON_1`, `وبPERSON_1`), as Arabic writes them;
- any other letter or digit glued on either side is not a boundary: `PERSON_1abc` and `lPERSON_1` are left
  as they are, so they can't leak either.

These strict boundaries are for **names** (`NAME_TYPES`), which a word boundary is all that separates from
ordinary words (`Sami` in `Samir`). Structured values (emails, phones, IBANs…) are distinctive on their own,
so they use loose ones (`LOOSE_LEFT`, `LOOSE_RIGHT`) on both sides: `a@example.comـ`, a value glued to an
Arabic letter, is restored and caught again like any other.
"""

from __future__ import annotations

import re

from nona.core.types import EntityType

NAME_TYPES = frozenset({EntityType.PERSON, EntityType.LOCATION, EntityType.ORGANIZATION, EntityType.CUSTOM})

PROCLITICS = "وفبلك"
_NOT_WORD = r"(?<![^\W_])"  # not preceded by a letter or digit (`_` is fine)
LEFT = rf"(?:{_NOT_WORD}|(?<={_NOT_WORD}[{PROCLITICS}])|(?<={_NOT_WORD}[{PROCLITICS}]{{2}}))"
RIGHT = r"(?![^\W_])"  # not followed by a letter or digit
LOOSE_LEFT = r"(?<![A-Za-z0-9])"
LOOSE_RIGHT = r"(?!\d)"

_STRICT_START = re.compile(LEFT)
_STRICT_END = re.compile(RIGHT)


def strict_at(text: str, start: int, end: int) -> bool:
    """Whether `text[start:end]` sits at strict boundaries (see above)."""
    return bool(_STRICT_START.match(text, start)) and bool(_STRICT_END.match(text, end))
