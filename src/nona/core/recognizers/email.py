"""E-mail addresses: the whole address, without `mailto:` (05 §4)."""

from __future__ import annotations

import re

from nona.core.boundaries import GLUED_PLACEHOLDER
from nona.core.normalize import NormalizedDoc
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span

# Runs on the normalized view, which is already lower-case. The local part allows Unicode letters; the
# domain is ASCII labels and a letter-only TLD, so a sentence's final dot is not part of the address.
# The address ends before anything but a letter or a digit: a `-` or `_` can't end a domain, so in
# `a@x.com- merci` and `a@x.com_x` it was glued to the address, not part of it (#93).
_EMAIL = re.compile(
    r"""
    (?<![\w.%+-])
    [\w%+-]+(?:\.[\w%+-]+)*
    @
    (?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+
    [a-z]{2,63}
    (?![^\W_])
    """,
    re.VERBOSE,
)


class EmailRecognizer:
    @property
    def fingerprint(self) -> str:
        """No configuration: every instance detects the same (detection cache, `core/cache.py`)."""
        return "default"

    @property
    def name(self) -> str:
        return "regex:email"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset({EntityType.EMAIL})

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        spans: list[Span] = []
        for m in _EMAIL.finditer(doc.text):
            end = _before_placeholder(doc, m.start(), m.end())
            spans.append(to_span(doc, m.start(), end, EntityType.EMAIL, 1.0, self.name))
        return spans


def _before_placeholder(doc: NormalizedDoc, start: int, end: int) -> int:
    """Where the address ends when its TLD runs into a placeholder the user typed in its issued shape:
    `a@x.comPERSON_1` is `a@x.com`, so the placeholder keeps its number (#85, #93). `end` when none does,
    or when what is left wouldn't be an address (`a@xPERSON_1.com`): masked whole, then, never leaked."""
    at = doc.text.index("@", start, end)
    o_end = doc.origin[end - 1][1]
    p = GLUED_PLACEHOLDER.search(doc.original, doc.origin[at][1])
    if p is None or p.start() >= o_end:
        return end
    cut = next(i for i in range(at, end) if doc.origin[i][0] >= p.start())
    return cut if _EMAIL.fullmatch(doc.text, start, cut) else end
