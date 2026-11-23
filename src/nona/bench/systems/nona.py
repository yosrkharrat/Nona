"""The `nona` system: Nona's own pipeline, as a library call, with its default configuration.

The whole conversation is anonymized with one shared mapping, causally (ADR-0004), so turn `i`'s payload is
what every later request sends for it. From S5 the benchmark goes through the proxy and measures what leaves
it (FR-21).
"""

from __future__ import annotations

from collections.abc import Sequence

from nona.bench.dataset import Conversation
from nona.core import Nona


class NonaSystem:
    def __init__(self, nona: Nona | None = None, *, name: str = "nona") -> None:
        self._nona = nona or Nona()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def payloads(self, conversation: Conversation) -> Sequence[str]:
        return list(self._nona.anonymize_messages([turn.text for turn in conversation.turns]).texts)
