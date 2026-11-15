# ADR-0003 — Fail closed

- Status: **Accepted**
- Date: 2026-11-14

## Context
The detector can crash, time out, or fail to load its model. The proxy then has to either forward the
request unscrubbed (fail open) or refuse it (fail closed).

## Decision
**Fail closed by default** (`on_detector_error: reject`): return HTTP 503 `nona_detector_unavailable` in the OpenAI error
shape, and send nothing upstream. Fail-open exists only as an explicit, loudly logged opt-in for local development
against a local upstream such as Ollama.

## Consequences
- A privacy tool that silently forwards raw data on error would be worse than no tool at all, because it creates false confidence.
- Availability now depends on the detector, so `/healthz` gates readiness and the detector timeout is configurable.
