"""Where a placeholder or a known value may start and end: one rule for restoring and for propagation.

Restoring writes a value back wherever the model put a placeholder, and forward propagation must find that
value again when the client re-sends the restored text (ADR-0004). If restore accepted a position that
propagation doesn't, the value would leak on the next request: `لPERSON_1` restored to `لكنزة` was not
caught by a plain `\\w` boundary. So both use these boundaries:

- `_` separates words (`file_PERSON_1.pdf`, `PERSON_1_2024`);
- up to two Arabic proclitics may be attached in front (`لPERSON_1`, `وبPERSON_1`), as Arabic writes them;
- any other letter or digit glued on either side is not a boundary: `PERSON_1abc` and `lPERSON_1` are left
  as they are, so they can't leak either.
"""

from __future__ import annotations

PROCLITICS = "وفبلك"
_NOT_WORD = r"(?<![^\W_])"  # not preceded by a letter or digit (`_` is fine)
LEFT = rf"(?:{_NOT_WORD}|(?<={_NOT_WORD}[{PROCLITICS}])|(?<={_NOT_WORD}[{PROCLITICS}]{{2}}))"
RIGHT = r"(?![^\W_])"  # not followed by a letter or digit
