"""IPv4 and IPv6 addresses, validated by the standard `ipaddress` module (FR-1)."""

from __future__ import annotations

import ipaddress
import re

from nona.core.normalize import NormalizedDoc
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span

# `_` or a lone `.` around it doesn't hide it (`10.0.0.1_`, `.10.0.0.1`, #93); a letter, a digit or a longer
# dotted number (`1.10.0.0.1`) does.
_IPV4 = re.compile(r"(?<![^\W_])(?<!\d\.)(?:\d{1,3}\.){3}\d{1,3}(?![^\W_]|\.\d)")
# Candidates only; `ipaddress` decides. Times like `10:30:00` are candidates and get rejected.
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])")


def _valid(candidate: str) -> bool:
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return not address.is_unspecified  # `::` and `0.0.0.0` identify nothing


class IpRecognizer:
    @property
    def fingerprint(self) -> str:
        """No configuration: every instance detects the same (detection cache, `core/cache.py`)."""
        return "default"

    @property
    def name(self) -> str:
        return "regex:ip"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset({EntityType.IP_ADDRESS})

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        spans: list[Span] = []
        for pattern in (_IPV4, _IPV6):
            for m in pattern.finditer(doc.text):
                if _valid(m.group()):
                    spans.append(to_span(doc, m.start(), m.end(), EntityType.IP_ADDRESS, 1.0, self.name))
        return sorted(spans, key=Span.sort_key)
