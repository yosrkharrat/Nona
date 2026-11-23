# 01 — Requirements

> Status: **Draft v0.1**, 2026-11-14 · Owner: Yosr Kharrat
> Next step after sign-off: [02 — API & interfaces](02-api.md)

## 1. Problem statement

Teams want to send real conversations (support tickets, WhatsApp messages, CRM notes) to hosted LLMs.
Those messages contain personal data: names, phone numbers, emails, IBANs, ID numbers. Compliance teams
block the project, or developers paste the data in anyway.

Existing scrubbers (Microsoft Presidio, LLM Guard, paid APIs) have three gaps:

1. **One-way redaction.** They replace `Ahmed` with `<PERSON>`, and the app has to live with a degraded answer.
2. **Mostly English defaults.** Their defaults are weak on French, Arabic, Tunisian Darija, and especially
   *Arabizi* (Darija written in Latin script with digits: `3andi rendez-vous m3a Sami`).
3. **No public measurement.** None of them publish how much personal data actually gets through on realistic text.

## 2. Vision

**Nona is a local proxy that sits between an app and any OpenAI-compatible LLM. It replaces personal data
with consistent placeholders before the request leaves the machine, then puts the real values back into
the answer. It ships with a benchmark that reports its own leak rate.**

Integration = change one `base_url`.

What Nona says about itself: it **reduces exposure**. It does **not guarantee anonymity** (see §6).

## 3. Stakeholders & actors

| Actor | Goal |
|---|---|
| **Integrating developer** (primary) | Add privacy to an existing LLM app without a rewrite |
| **End user** of that app | Gets a normal answer, and their data stays out of the provider's logs |
| **Compliance / security reviewer** | Needs evidence: a threat model, measured leak rates, and logs with no PII |
| **Upstream LLM provider** | Treated as *honest but curious*: it must never receive the protected values |
| **Contributor** | Adds a recognizer or language without understanding the whole system |

## 4. Functional requirements

Priority: **M** = must (v0.1), **S** = should (v0.2), **C** = could (later).

### Detection
| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| FR-1 | Detect structured identifiers: email, phone (TN, FR, international), IBAN (mod-97 validated), payment card (Luhn), IPv4/IPv6, Tunisian CIN (with context), Tunisian RIB | M | Each recognizer has ≥10 positive and ≥10 negative unit tests |
| FR-2 | Detect PERSON, LOCATION and ORGANIZATION in French, English, Arabic script, Darija and Arabizi, including code-switched sentences | M | Measured on the held-out test set, reported per language (see FR-20) |
| FR-3 | Detection runs fully offline on CPU | M | Test suite passes with networking disabled |
| FR-4 | Users can add custom recognizers (regex + context words, or a gazetteer file) through configuration | S | A recognizer added in YAML is picked up with no code change |
| FR-5 | Allowlist (never mask, e.g. the company's own name) and denylist (always mask, e.g. a project codename) | M | Covered by unit tests |

### Pseudonymization & restoration
| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| FR-6 | Replace each detected entity with a typed placeholder (`PERSON_1`, `PHONE_2`, …) | M | — |
| FR-7 | **Consistency**: the same real-world entity gets the same placeholder across all messages of a conversation, including variants (`Ahmed Ben Salem`, `Ahmed`, `M. Ben Salem` → one `PERSON_n`). An email that spells the name (`ahmed.bs@gmail.com`) keeps its own `EMAIL_n` and records the person as its `owner` | M | Linking tests (deep dive B), plus a prefix-stability property test on the texts: `anonymize_messages(msgs[:k]).texts == anonymize_messages(msgs).texts[:k]`. Entity metadata (`canonical`, `owner`) and the mapping reflect the whole conversation sent (ADR-0004) |
| FR-8 | **Restoration**: placeholders in the LLM output are replaced with the original values before the client sees them, for values under a reversible policy (FR-9: cards are redacted, so they can't come back) | M | Property test: `restore(anonymize(x)) == x` for all generated `x`, up to linking: a mention comes back as its entity's canonical (longest) form, so `x` itself comes back when no entity has two forms (ADR-0004) |
| FR-9 | Per-type policy: `pseudonymize` (reversible), `redact` (irreversible, e.g. card numbers), or `keep` | S | — |

### Proxy
| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| FR-10 | OpenAI-compatible `POST /v1/chat/completions`, non-streaming | M | The official `openai` Python and JS SDKs work unmodified except for `base_url` |
| FR-11 | Same endpoint with `stream: true` (SSE), with restoration applied to the stream | M | Placeholders split across chunks are restored correctly |
| FR-12 | Forward to a configurable upstream (OpenAI, Mistral, OpenRouter, Ollama, any OpenAI-compatible server) | M | Contract tests against a mock upstream |
| FR-13 | Tool / function calling: anonymize tool results going up, restore tool-call arguments coming down | S | — |
| FR-14 | `GET /v1/models` passthrough | M | — |

### Inspection & tooling
| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| FR-15 | `POST /v1/nona/anonymize` returns the anonymized text plus the entity spans (this powers the demo and debugging) | M | — |
| FR-16 | CLI: `serve`, `scan <file>`, `bench` | M | — |
| FR-17 | Python library API usable without the proxy | M | — |
| FR-18 | Split-screen demo page: what you type vs what the LLM receives, with substitutions highlighted live | S | 10-second GIF in the README |
| FR-19 | Audit events with **no PII**: counts per entity type, detector latency, request id | S | A leak test greps logs, trace spans and metric labels for every gold PII value and finds none |

### Benchmark
| ID | Requirement | Pri | Acceptance criterion |
|---|---|---|---|
| FR-20 | Synthetic labelled dataset: FR / EN / Arabic / Darija / Arabizi / mixed, with a dev/test split. Includes **multi-turn conversations** with conversation-level gold entity ids (ADR-0004) | M | ≥300 messages in v0.1, of which ≥40 conversations of 4+ turns. Test split frozen |
| FR-21 | Leak-rate script measured **on the wire** (what actually leaves the proxy), not just detector recall | M | — |
| FR-22 | Utility-loss evaluation: answer quality with vs without Nona | S | — |
| FR-23 | Baseline adapters for Presidio and LLM Guard, run on the same data | M | Configurations are documented and reproducible |

### Out of scope (v0.x)
- Images, PDFs, audio.
- Embeddings and RAG pipelines (`/v1/embeddings` is a "could").
- Hosted multi-tenant SaaS.
- Hiding quasi-identifiers (e.g. "the only pediatrician in Tozeur"), see §6.
- Any real personal data, even "anonymized" (see §8).

## 5. Non-functional requirements

Targets are **hypotheses to measure**, not promises. The README reports actual numbers.

| ID | Category | Requirement / target |
|---|---|---|
| NFR-1 | **Privacy** | Nona never writes original values to disk by default. Mappings stay in process memory and are dropped at the end of the request (stateless mode, see ADR-0002) |
| NFR-2 | **Privacy** | **Fail closed**: if detection errors or times out, the request is rejected (HTTP 503) and never forwarded unscrubbed (ADR-0003) |
| NFR-3 | **Privacy** | No telemetry and no outbound network except to the configured upstream. Model weights are downloaded once, explicitly. Trace export (OTLP) happens only if the user configures an endpoint (ADR-0005) |
| NFR-4 | **Security** | Binds to `127.0.0.1` by default. The upstream API key is passed through and never logged |
| NFR-5 | **Detection quality** | Recall is prioritized over precision: a leak costs more than over-masking. v0.1 targets on the test split: ≥ 0.98 recall on structured types, ≥ 0.85 on PERSON in mixed-language text |
| NFR-6 | **Latency** | Added overhead p95 ≤ 150 ms for a 300-token message on an Apple M-series CPU. Time-to-first-token overhead ≤ 50 ms when streaming |
| NFR-7 | **Footprint** | ≤ 1.5 GB RSS with the NER model loaded (it must run beside an IDE on a 16 GB laptop) |
| NFR-8 | **Compatibility** | Python 3.11+, macOS and Linux, Docker image ≤ 1.5 GB (CPU-only torch or ONNX) |
| NFR-9 | **Reproducibility** | `nona bench` is deterministic: pinned model revisions, fixed seeds, frozen dataset hash |
| NFR-10 | **Maintainability** | Typed (pyright strict on core), ruff, ≥ 85% line coverage on `core/`, and each recognizer is a self-contained plugin |
| NFR-11 | **Honesty** | The README states the leak rate, the known failure modes and the threat-model limits next to the headline numbers |
| NFR-12 | **Licensing** | Apache-2.0 code. Every model and gazetteer used has a compatible license, recorded in `THIRD_PARTY.md` |
| NFR-13 | **Observability** | Per-stage traces (OpenTelemetry), Prometheus metrics and structured logs, all PII-free by construction and covered by the FR-19 leak test (ADR-0005) |
| NFR-14 | **Supply chain** | Releases are built only in CI, signed (cosign) and attested (SLSA provenance, SBOM). Users can verify a wheel or image without trusting the maintainer (ADR-0007) |
| NFR-15 | **Throughput** | Measured, not assumed: a k6 load test reports p50/p95 overhead and requests per second at 1, 8 and 32 concurrent clients against a mock upstream |

## 6. Threat model (summary)

**Protects against:** the upstream LLM provider (and its logs, training pipeline and subprocessors) seeing
direct identifiers present in the text.

**Does not protect against:**
- **Missed detections.** This is why the leak rate is published.
- **Quasi-identifiers and context.** "My neighbour who runs the pharmacy next to the Sousse train station" can identify someone with no names in it.
- **Inference by the model**, from writing style or rare combinations.
- **A compromised local machine.** Nona holds the mapping in memory.
- **Metadata**: placeholder counts reveal how many people are mentioned.

**Nona-specific risks to design against:**
- *Cross-session mapping leak*: user A's `PERSON_1` restored as user B's name. Mitigation: no shared state in stateless mode; strict session scoping in stateful mode.
- *Placeholder injection*: the text already contains `PERSON_1`, or the model invents `PERSON_7`. Mitigation: escape pre-existing placeholder-like strings, and restore only placeholders issued for this request.
- *Log leakage*: debug logs printing request bodies. Mitigation: a structured logger with a deny-by-default field allowlist, plus a CI test (FR-19).
- *Telemetry leakage*: traces or metrics carrying text, URLs or headers. Mitigation: hand-written spans with typed, allowlisted attributes, export off by default (ADR-0005).
- *Session store compromise* (stateful mode only): Redis holds mappings. Mitigation: AES-GCM per session, keys derived from a master key plus the session id, storage keys are HMACs, TTL (ADR-0006).
- *Supply-chain compromise*: a malicious release sees all traffic. Mitigation: signed, attested CI-only releases and pinned actions (ADR-0007).

## 7. Success criteria for v0.1

1. `pip install` (or `docker run`), change `base_url`, and the chat works with names restored, with or without streaming.
2. A public benchmark table: Nona vs Presidio vs LLM Guard, per language, with leak rate, over-masking rate, latency, and utility loss on at least one task.
3. A demo GIF at the top of the README.
4. A threat model and its limits stated plainly.

## 8. Constraints & assumptions

- **Data provenance**: every test or benchmark message is synthetic, written or generated for this repo. No real personal data is ever used, not even "anonymized".
- Solo developer, about 10 h/week: v0.1 after roughly 12–14 weeks (see the [build plan](06-build-plan.md)).
- Assumption: the clients use the Chat Completions API (stateless, full history sent on every call). This assumption drives ADR-0002.

## 9. Open questions

| # | Question | Proposed answer |
|---|---|---|
| Q1 | **Name.** ~~"Nano" collided with the GNU `nano` editor~~ | **Resolved 2026-11-15:** the project is **Nona**, with CLI `nona` and Python package `nona`. **Updated 2026-11-20:** `nona` is taken on PyPI, so the distribution is published as `nona-proxy`. The command and `import nona` don't change |
| Q2 | Which upstreams are officially supported in v0.1? | OpenAI, OpenRouter, Ollama (local, used for e2e tests) |
| Q3 | Where do the Tunisian first-name, surname and city gazetteers come from, and under what license? | Public sources (Wikipedia/Wikidata, CC0/CC-BY). Verify before shipping |
| Q4 | Should placeholders be `PERSON_1` or realistic fake names? | Decided by benchmark, see ADR-0001 |
