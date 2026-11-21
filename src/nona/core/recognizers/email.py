"""E-mail addresses: the whole address, without `mailto:` (05 §4)."""

from __future__ import annotations

import re

from nona.core.normalize import NormalizedDoc
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span

# Runs on the normalized view, which is already lower-case. The local part allows Unicode letters; the
# domain is ASCII labels and a letter-only TLD, so a sentence's final dot is not part of the address.
_EMAIL = re.compile(
    r"""
    (?<![\w.%+-])
    [\w%+-]+(?:\.[\w%+-]+)*
    @
    (?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+
    [a-z]{2,63}
    (?![\w-])
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
        return [
            to_span(doc, m.start(), m.end(), EntityType.EMAIL, 1.0, self.name)
            for m in _EMAIL.finditer(doc.text)
        ]
