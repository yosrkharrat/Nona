"""The `nona` system: Nona's own pipeline, as a library call, with its default configuration.

In S2 each turn is anonymized on its own. From S3, the whole conversation is anonymized with one shared
mapping (ADR-0004), and from S5 the benchmark goes through the proxy and measures what leaves it (FR-21).
"""

from __future__ import annotations

from collections.abc import Sequence

from nona.bench.dataset import Conversation
from nona.core import Nona


class NonaSystem:
    def __init__(self, nona: Nona | None = None) -> None:
        self._nona = nona or Nona()

    @property
    def name(self) -> str:
        return "nona"

    def payloads(self, conversation: Conversation) -> Sequence[str]:
        return [self._nona.anonymize(turn.text).text for turn in conversation.turns]
