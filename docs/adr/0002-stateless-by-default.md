# ADR-0002 — Stateless by default

- Status: **Accepted**
- Date: 2026-11-14

## Context
Placeholders must be consistent across a conversation (FR-7). The obvious design is a server-side vault mapping
session to entity to placeholder. But a vault holding PII is the most sensitive thing Nona could store, and it
needs session ids, TTLs and isolation between users.

## Decision
Exploit the fact that the Chat Completions API is stateless: the client resends the full history on every call.
Nona re-anonymizes the **entire** message list on every request with a fresh, deterministic mapping
(placeholders numbered by order of first appearance) and keeps nothing afterwards.

A detection cache (spans only, in memory, with a TTL) keeps the cost of re-processing low. It's keyed by
`sha256(fingerprint + message text)`, the fingerprint covering the nona version and the detection config
(03 §4, ADR-0004).

A stateful mode (`X-Nona-Session`) is deferred to v0.2, for clients that truncate history.

## Consequences
- ➕ No PII stored between requests, and no cross-user leak is possible by construction.
- ➕ Horizontal scaling is trivial.
- ➖ CPU cost grows with history length. The cache mitigates it.
- ~~➖ If a later message makes an earlier mention ambiguous, earlier placeholders can be renumbered.~~
  Resolved by [ADR-0004](0004-causal-entity-linking.md): linking is causal, so earlier placeholders never change.
