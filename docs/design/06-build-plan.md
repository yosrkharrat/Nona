# 06: Build plan

> Status: **Active** · 2026-11-20 · Depends on: [01](01-requirements.md)–[05](05-dataset.md), ADR-0001 to ADR-0007
> Budget: about 10 h/week, no deadline. One slice ≈ one to three weeks. v0.1 after roughly 12 weeks.

The design is done. This document says **in what order** it gets built and **how we know each step worked**.

## 1. Strategy: measured vertical slices

We don't build one layer at a time (all of `core/`, then the dataset, then the proxy) and hope it fits at the end.
Each slice ships something that **works end to end and moves a benchmark number**. The benchmark is built first, before
any detector, so every later change shows up as a measured difference instead of an impression.

```
S1 benchmark ─► S2 structured PII ─► S3 linking ─► S4 GLiNER + frozen test set
     ─► S5 proxy + telemetry ─► S6 streaming ─► S7 baselines + perf ─► S8 v0.1 release
     ─► S9 work-platform bridge (Slack first, after v0.1)
```

## 2. How we work

- **Flow:** one issue → one short branch → one PR, self-reviewed (`/code-review`) → green CI → squash merge.
  Conventional Commits, so release-please can write the changelog (ADR-0007).
- **Definition of done** for every PR:
  - tests, with property tests wherever the design names an invariant
  - pyright strict on `core/` and ruff clean
  - docs updated in the same PR
  - an ADR if a real choice was made
  - the slice's benchmark number in the PR description
- **Benchmark discipline:** tune on `dev` only. `test` is frozen at the end of S4 and run **once** for the release table (NFR-11).
- **Dependencies:** added in the slice that needs them. Heavy optional ones go in extras (`nona-proxy[bench]`, `nona-proxy[redis]`).
- **Spikes:** timeboxed to 4 h. Each one ends in an ADR with numbers.
- **Hardware:** a 16 GB laptop. Run one heavy process at a time: GLiNER, Ollama, Presidio/spaCy, or k6.

## 3. Slices

### S0: Go public, harden CI (~4 h)
- Push the repo to GitHub, protect `main`, add issue templates and labels, create one milestone per slice, and mirror this plan as issues.
- PyPI distribution name: **`nona-proxy`**, because `nona` is taken by an inactive 2023 package (§5 Q1).
- Do the cheap parts of ADR-0007 now: SHA-pinned actions, Dependabot (actions and uv), `pip-audit` on `uv.lock`, CodeQL for Python and for the workflows.

### S1: Measure before detecting (~12 h)
**Goal:** the benchmark exists and is proven correct before any detector exists.
- `src/nona/bench/generate.py`: a seeded template generator. Its synthetic values follow 05 §5:
  - TN/FR phone numbers
  - `example.com` emails
  - mod-97-valid IBAN and RIB
  - Luhn-valid test card numbers
  - CIN numbers
  - names from seed lists

  Output goes through the existing `parse_conversation` and `dumps_jsonl` ([dataset.py](../../src/nona/bench/dataset.py)).
- `bench/data/dev.jsonl` v0: about 80 conversations (FR, EN, Arabizi), mostly structured identifiers, plus some names and `no-pii` controls. No test split yet.
- `src/nona/bench/systems/`: a `System` protocol (turns → upstream payload for each turn), plus an `identity` system.
- `src/nona/bench/metrics.py`: leak rate, partial leak rate and over-masking, with the normalization from 05 §6.
- `nona bench --systems identity --split dev`: a language × type table, plus JSON in `results/`.
- **Exit:** the identity system leaks 100%, and the metric unit tests pass.

### S2: Structured detection and the core pipeline (~20 h)
**Goal:** `Nona.anonymize()` / `restore()` work for structured PII.
- `core/normalize.py`: NFKC and Eastern Arabic digits, with an **offset map back to the original text**.
- A `Recognizer` protocol (02 §3) and the FR-1 recognizers: email, phone (`phonenumbers`), IBAN, RIB, card, IP, and CIN with a context word. Each needs ≥ 10 positive and ≥ 10 negative tests.
- Supporting pieces:
  - the overlap resolver, using `Span.sort_key` for determinism
  - the pseudonymizer (`TYPE_n` numbered by first appearance)
  - the non-streaming restorer
  - the allowlist and denylist (FR-5)
- Property tests: `restore(anonymize(x)) == x`, plus determinism.
- A `nona` system adapter.
- **Exit:** the structured leak rate on `dev` drops from 100% to near 0%. This is the first published number, posted in the PR.

### S3: Conversations and linking (~16 h)
**Goal:** ADR-0004 implemented and measured.
- `anonymize_messages()`: causal linking, forward propagation, and canonical tokens (honorifics, Arabizi, spelling variants).
- Gazetteer and context-rule layers (deep dive A, layers 3 and 4), so PERSON detection exists before any ML model.
- A detection cache keyed by `sha256(model rev + config hash + text)`.
- Property test: `anonymize(msgs[:k]) == anonymize(msgs)[:k]`.
- More multi-turn conversations in the dataset (toward ≥ 40 conversations of 4+ turns).
- New metrics: wrong-merge, wrong-split and ambiguity rates, and the leak rate on re-sent assistant turns.
- ADR-0004 moves to Accepted if the numbers support it.

### S4: Names with GLiNER, and the full dataset (~24 h)
**Goal:** multilingual PERSON, LOCATION and ORGANIZATION detection, and a frozen test set.
- **Spike:** `gliner_multi-v2.1` vs `gliner_multi_pii-v1` on `dev`, per language. The result is **ADR-0008 (NER model choice)**, and the licenses go in `THIRD_PARTY.md`.
- An NER recognizer:
  - loaded once and run off the event loop
  - spans from the original text and the de-Arabizi'd view are combined
  - weights come only through `nona models pull` (NFR-3)
  - tests are marked `model` and run in their own CI job with the model cached
- The dataset reaches the 05 §7 targets:
  - Arabic script and Darija
  - about 50 hand-written hard cases
  - LLM drafts checked with a small review script (05 D3)
- **Freeze `test`**: commit `test.sha256` and `DATASHEET.md`.
- Check NFR-7: memory use ≤ 1.5 GB with the model loaded.
- **Exit:** a per-language leak rate on `dev`.

### S5: Non-streaming proxy, instrumented from the start (~20 h)
**Goal:** the official `openai` SDK works through Nona by changing only `base_url` (FR-10).
- FastAPI app:
  - pydantic models for the OpenAI schema, and a payload walker over them
  - a shared httpx client
  - `/v1/models` passthrough and `/healthz` readiness
  - fail-closed 503 (ADR-0003), and errors in the OpenAI shape
  - the inspection API
  - `nona serve` and config loading from `nona.yaml`
- Telemetry from the first commit (ADR-0005):
  - hand-written OTel spans per stage
  - a Prometheus `/metrics` endpoint
  - structlog with a field allowlist
  - the FR-19 leak test over logs, spans and metric labels
- Contract tests with respx, plus an `openai`-SDK test against a mock upstream.
- The benchmark now measures **on the wire**: the `nona` system goes through the proxy to a capturing mock upstream (FR-21).

### S6: Streaming (~14 h)
**Goal:** FR-11. Placeholders split across chunks are restored correctly.
- The stream restorer state machine (deep dive D). Property test: for any random split of the output, `concat(feed(chunks)) + flush() == restore(full)`.
- SSE passthrough, and a metric for the extra time to first token.
- Restoration failure rate in the benchmark.
- An end-to-end test with a small local model on Ollama, marked `e2e` and run locally, not in CI.

### S7: Baselines, utility, performance (~18 h)
**Goal:** the comparison table.
- Presidio (default English, and French spaCy plus patterns) and LLM Guard adapters, in the `bench` extra.
- Utility loss on task 1 (extraction, scored exactly against gold), with bootstrap confidence intervals. The LLM-judge tasks come after v0.1.
- k6 against the mock upstream at 1, 8 and 32 concurrent clients, reporting p50/p95 overhead and requests per second (NFR-15).
- `report.py` builds the README table from **one clean run on the frozen test split**.

### S8: Release v0.1 (~16 h)
- A CPU-only, non-root Dockerfile with the model baked in.
- `deploy/compose/` with the Grafana LGTM stack and a provisioned dashboard.
- A demo GIF (the split-screen page if there's time, FR-18).
- The release workflow from ADR-0007: release-please, PyPI Trusted Publishing, a multi-arch GHCR image, cosign signing, provenance attestations and an SBOM.
- An OpenSSF Scorecard badge.
- The README gets the results, the known failure modes, and how to verify a release.
- Tag **v0.1.0**.

### S9: Work-platform bridge, Slack first (~16 h, after v0.1)
**Goal:** a Slack thread goes to Claude anonymized, and the answer comes back restored, after the asker approves it
(03 §9). It replaces copying a thread into Claude and pasting the answer back.
- First, ADR-0009: the bridge calls `core` in-process, or goes through the proxy (proposal: in-process, 03 §9).
- 01 gets the bridge's requirements and a threat-model section: Slack tokens, where the bridge sees plain text,
  approval before posting.
- The platform adapter protocol, and the Slack adapter: message shortcut and mention, thread reading, ephemeral
  preview with Post / Edit / Discard, Socket Mode. A `slack` extra (`nona-proxy[slack]`).
- The bridge service and the Claude client (official `anthropic` SDK), with fail-closed and PII-free telemetry shared
  with the proxy.
- Tests: a fake platform adapter and a capturing fake model, so CI needs neither Slack nor an API key. A property
  test: whatever the thread, the payload sent to the model holds no gold value the pipeline masks.
- A demo in a personal test workspace with synthetic threads only.
- **Exit:** the leak rate on Slack-style synthetic threads (the dev conversations rendered as threads), measured on
  the payload sent to the model, equals the library's on the same conversations; 0 personal data in the bridge's logs
  and spans (the FR-19 test); and no message is ever posted without an approval click (tested).

### After v0.1, in the order the benchmark suggests
Stateful sessions with Redis (ADR-0006), tool calls (FR-13), ONNX int8, realistic surrogates (ADR-0001 option C), a
Helm chart, a TypeScript client, an MCP server, and fine-tuning GLiNER on Darija and Arabizi. Each one starts only when
a number or a user need justifies it.

## 4. Risks

| Risk | Mitigation |
|---|---|
| The dataset takes longer than the code (the hidden cost) | Generator first, LLM drafts plus a review script, a fixed weekly budget |
| GLiNER is weak on Arabizi | Gazetteer and context layers ship first (S3). Fine-tuning comes after v0.1 |
| Scope creep from the post-v0.1 list | Nothing on that list starts before v0.1 ships, S9 included |
| The bridge touches real workplace data | Development and the demo use a personal workspace with synthetic threads only. Installing it elsewhere is that workspace's call, with its admin's approval |
| Laptop runs out of memory | One heavy process at a time. Small local model for e2e |
| PyPI name taken | Resolved in S0: `nona-proxy` (§5 Q1) |

## 5. Open questions

| # | Question | Proposal |
|---|---|---|
| Q1 | `nona` on PyPI belongs to an unrelated package (v0.0.2, inactive since January 2023) | **Resolved 2026-11-20:** publish as **`nona-proxy`** (free as of 2026-11-20) and keep `import nona` and the `nona` command unchanged. A PEP 541 name request is possible but slow, so it shouldn't block anything |

## 6. Progress

| Slice | Status | Headline number |
|---|---|---|
| S0 | done, except #6 (PyPI publisher) and #7 (old draft), which need the owner | — |
| S1 | done (#2; dev regenerated in #26 with valid phone numbers only) | identity system: 100% leak (110/110 direct-identifier mentions), 0% over-masking, on dev v0 (80 conversations) |
| S2 | done (#25, #27–#32) | structured types (email, phone, IBAN, RIB, card, IP, CIN): **0% leaked** on dev, down from 100%. Over-masking 0.2% of tokens, 2/19 distractors (8-digit order numbers that are valid TN mobiles). Headline 39.1% because names are still undetected (S3, S4) |
| S3 | done (#35–#42, #48) | names: headline **26.1%** (64/245) on dev (120 conversations; not comparable with S2's 80). On the v0 part it is 0%, but inflated: 76 of 98 v0 names are in the gazetteer. On the multi-turn part, whose names are held out of the gazetteer, it is 48.5%, and 70% of PERSON mentions leak. Structured types 0%, over-masking 0/2362 tokens and 0/42 distractors. Linking (ADR-0004, now Accepted): 0 wrong merges (0/57 people), 0 wrong splits (0/24), 0 ambiguity hits (0/46), over the 48.8% of PERSON mentions that get a placeholder. Re-sent assistant turns leak 43.6% (48/110), all names never detected before |
| S3.1 | in progress: #54–#58 done, #59 left | dev extended in #58 (137 conversations, 370 turns; not comparable with S3's 120): headline **25.4%** (72/284). v0 part 0% (its names are listed); multi-turn part 42.1% (72/171). **Held-out PERSON 75.0%** (72/96), the honest name number. Structured types 0%, nothing obfuscated or deleted. Over-masking 0.2% of tokens (6/2754), all on the Arabic controls `كريم` and `آمين` (3 each), left to #50 in S4; every French and Darija control is clean. Linking, PERSON: 0/65 merges, 0/32 splits, 0/61 ambiguity hits, over the 54.0% of mentions with a placeholder, 15 partly leaking ones left out. Re-sent assistant turns leak 38.8% (54/139), none of an entity masked earlier |
| S4 to S8 | not started | — |
