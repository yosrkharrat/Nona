"""Systems under test: anything that turns a conversation into what an upstream LLM would receive.

A system sees a whole conversation and returns, for each turn, the text of that turn **as it is sent
upstream**. In the stateless Chat Completions flow (ADR-0002) the client re-sends the history with every
request, so turn `i` is sent upstream once per later request. Systems must be causal (ADR-0004): the payload
for turn `i` may depend only on turns `0..i`. The benchmark checks each turn's payload against the gold
mentions of that same turn (docs/design/05-dataset.md, section 6).

Every configuration lives in this package, so anyone can rerun it with `nona bench` (04 §F).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from nona.bench.dataset import Conversation
from nona.bench.systems.identity import IdentitySystem


class System(Protocol):
    @property
    def name(self) -> str: ...

    def payloads(self, conversation: Conversation) -> Sequence[str]:
        """One string per turn, in order: what the upstream LLM receives for that turn."""
        ...


SYSTEMS: Mapping[str, Callable[[], System]] = {
    "identity": IdentitySystem,
}

__all__ = ["SYSTEMS", "IdentitySystem", "System"]
