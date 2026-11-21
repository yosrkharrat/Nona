"""Detection cache: recognizer spans per message, so re-sent history isn't detected again (ADR-0004, 03 §4).

In stateless mode every request carries the whole conversation, and the NER model is the slow stage. The
spans the recognizers find in a message depend only on that message's text and on the recognizers
themselves, so they are cached under `sha256(fingerprint + text)`:

- the **fingerprint** covers every recognizer's type and configuration (and, from S4, the model revision),
  so a config or model change misses the cache instead of serving stale spans;
- what depends on the conversation (forward propagation, the denylist, resolving, look-ahead) runs on every
  request, after the cache.

**No text is stored** (deep dive G): the key is a hash, and the value holds offsets, types, scores and
detector names only. A hash of personal data is still sensitive, so the cache lives in memory only, is
bounded (LRU), entries expire after a TTL, and its `repr` shows counts, never keys.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence

from nona import __version__
from nona.core.types import Span


def recognizer_fingerprint(recognizer: object) -> str:
    """What identifies a recognizer's behaviour: its `fingerprint`, or a token unique to this call.

    Built-in recognizers define `fingerprint` from their configuration. It is trusted only when the
    recognizer's own class defines it: a subclass may add settings its parent's fingerprint doesn't cover.
    Otherwise the result is a fresh random token, so the recognizer never shares cache entries with anything
    else (`Nona` computes its fingerprint once, so it still reuses its own entries). Not `id()`: CPython
    reuses the address of a collected object.
    """
    kind = type(recognizer)
    own = kind.__dict__.get("fingerprint")
    value = own.__get__(recognizer, kind) if isinstance(own, property) else own
    name = f"{kind.__module__}.{kind.__qualname__}"
    if isinstance(value, str):
        return f"{name}:{value}"
    return f"{name}:unique={uuid.uuid4().hex}"


def fingerprint(recognizers: Iterable[object]) -> str:
    """The detection config of a recognizer list, in order (order matters: ADR-0002)."""
    parts = [f"nona={__version__}", *(recognizer_fingerprint(r) for r in recognizers)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def cache_key(config: str, text: str) -> str:
    return hashlib.sha256(f"{config}\0{text}".encode()).hexdigest()


class DetectionCache:
    """A bounded, expiring, thread-safe map from `cache_key` to the spans of one message.

    `maxsize=0` disables it. The proxy runs detection in a thread pool, hence the lock. Expired entries are
    swept out on any access at most every quarter of the TTL, not only when their own key is looked up, so a
    hash doesn't outlive its TTL by much while the process is in use.
    """

    def __init__(
        self, maxsize: int = 4096, ttl: float = 3600.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if maxsize < 0:
            raise ValueError("maxsize must be >= 0")
        if ttl <= 0:
            raise ValueError("ttl must be > 0")
        self._maxsize = maxsize
        self._ttl = ttl
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, tuple[Span, ...]]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self._next_sweep = clock() + ttl / 4

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<DetectionCache {len(self)}/{self._maxsize} entries, {self.hits} hits, {self.misses} misses>"

    def _sweep(self, now: float) -> None:
        """Drop every expired entry, at most every quarter of the TTL. Called with the lock held."""
        if now < self._next_sweep:
            return
        self._next_sweep = now + self._ttl / 4
        for key in [k for k, (expires, _) in self._entries.items() if now >= expires]:
            del self._entries[key]

    def get(self, key: str) -> tuple[Span, ...] | None:
        if not self._maxsize:
            return None
        with self._lock:
            self._sweep(self._clock())
            entry = self._entries.get(key)
            if entry is None or self._clock() >= entry[0]:
                if entry is not None:
                    del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry[1]

    def put(self, key: str, spans: Sequence[Span]) -> None:
        if not self._maxsize:
            return
        with self._lock:
            now = self._clock()
            self._sweep(now)
            self._entries[key] = (now + self._ttl, tuple(spans))
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
