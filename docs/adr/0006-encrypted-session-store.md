# ADR-0006: Stateful mode with a pluggable, encrypted session store (memory or Redis)

- Status: **Proposed** (lands in M5)
- Date: 2026-11-20
- Related: [ADR-0002 stateless by default](0002-stateless-by-default.md), [ADR-0004 causal entity linking](0004-causal-entity-linking.md)

## Context

Stateless mode (ADR-0002) stays the default. Two cases still need state:

1. **Clients that don't resend the full history**: sliding-window chat apps, agents that summarize old turns, and
   server-side conversation state such as `previous_response_id` in the Responses API. Without state, `PERSON_1` in
   turn 20 may not refer to the same person as `PERSON_1` in turn 2, and restoration breaks.
2. **More than one replica.** In the Kubernetes sidecar or gateway setup ([03 §8](../design/03-architecture.md#8-deployment)), consecutive requests
   of one conversation can reach different processes. An in-memory vault doesn't work there.

A session mapping holds the original values. It is the most sensitive data Nona could ever store (ADR-0002).
Storing it in a shared service needs a security design, not just a cache.

## Decision

1. **A `SessionStore` protocol** with two implementations:
   - `memory` (default for stateful mode): a dict with a TTL, for a single process.
   - `redis`: opt-in with `sessions.backend: redis`. It works with Redis 7+ and Valkey, which speak the same protocol.

   The protocol lives in `nona.core` as a pure interface. The implementations live in `nona.proxy.sessions`, so the core stays free of I/O (enforced by `tests/test_architecture.py`).
2. **The session id is a bearer secret.** `X-Nona-Session` must be at least 128 bits of randomness (22+ base64url chars).
   Anything shorter is rejected with HTTP 400 `nona_weak_session_id`. Nona never logs or traces the raw id (ADR-0005).
3. **Storage keys don't reveal ids.** Redis key = `nona:s:` + `HMAC-SHA256(server_key, session_id)`.
4. **Values are encrypted.** The mapping is serialized and encrypted with AES-256-GCM (`cryptography`). The key is
   derived per session: `HKDF(master_key, info=session_id)`. The associated data is the storage key, so a blob copied
   to another key fails to decrypt. A Redis dump alone, without the session ids, reveals nothing.
5. **TTL:** sliding 1 h, absolute cap 24 h, both configurable. It is enforced by Redis `EXPIRE` plus an expiry
   timestamp inside the ciphertext.
6. **Concurrency:** under ADR-0004 a mapping is append-only (entities are added, never renumbered). Writes use
   compare-and-set on a version number in a small Lua script. On conflict, Nona reloads, re-links the new mentions and retries (max 3).
7. **Fail closed:** if the store is unreachable, or the master key (`NONA_SESSION_KEY`) is missing or wrong, the
   request is rejected (ADR-0003). Nona never falls back to stateless mode silently, because restoration would then be wrong.
8. **Tests:** the Redis backend's integration tests run against a real Valkey container through `testcontainers`,
   including expiry, CAS conflicts, and swapping blobs between keys.

## Consequences

- ➕ Stateful conversations work across replicas, and the deployment can scale horizontally.
- ➕ A compromised Redis instance, on its own, reveals neither the session ids nor the PII.
- ➖ NFR-1 now says "by default". When this mode is on, encrypted PII sits in Redis memory for up to the TTL. The threat model gets a new row.
- ➖ An attacker with both the master key and a session id can decrypt that session. Key rotation: new sessions use the new key, and old ones expire within 24 h.
- ➖ Another runtime dependency (`redis`, `cryptography`) and a container in the test suite. Both are optional extras: `pip install nona[redis]`.
