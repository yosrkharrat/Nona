"""The `identity` system: sends every turn upstream unchanged.

It is the control for the benchmark itself. It must leak 100% of mentions and over-mask nothing; any other
result means the metrics are wrong, not the system (build plan S1 exit criterion).
"""

from __future__ import annotations

from collections.abc import Sequence

from nona.bench.dataset import Conversation


class IdentitySystem:
    @property
    def name(self) -> str:
        return "identity"

    def payloads(self, conversation: Conversation) -> Sequence[str]:
        return [turn.text for turn in conversation.turns]
