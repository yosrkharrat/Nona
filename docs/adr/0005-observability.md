# ADR-0005: Observability with OpenTelemetry and Prometheus, PII-free by construction

- Status: **Accepted**
- Date: 2026-11-20
- Related: [NFR-3, FR-19](../design/01-requirements.md), [deep dive G](../design/04-deep-dives.md#g-keeping-nona-itself-from-leaking)

## Context

Nona sits in the path of every LLM request. Someone running it needs to answer, without guessing:

- How much latency does Nona add, and which stage (normalize, detect, link, rewrite, upstream) is responsible?
- How often does it fail closed (ADR-0003), and why?
- What does it mask, per entity type and per recognizer, and is the detection cache working?

Observability is also the easiest place for a privacy tool to leak. Traces and logs that record request bodies,
URLs, headers or "helpful" debug attributes would copy the PII Nona just removed into a third system. Generic
auto-instrumentation and LLM-tracing tools are designed to record prompts.

NFR-3 says Nona sends nothing anywhere except to the configured upstream.

## Options

| Option | Verdict |
|---|---|
| Logs only | Not enough to explain where latency comes from. Logs alone don't produce histograms or per-stage timings |
| LLM-tracing SaaS (Langfuse, LangSmith, Helicone) | Rejected. Recording prompts and completions is what these tools are for, which is the opposite of Nona's job |
| Vendor APM SDK (Datadog, Sentry) | Rejected as the core layer, because it ties Nona to one vendor. These tools can still receive OTLP data |
| **OpenTelemetry (traces) + Prometheus (metrics) + structured logs** | **Chosen.** Vendor-neutral standards that companies already run, and every backend accepts them |

## Decision

1. **Traces: OpenTelemetry SDK, instrumented by hand.** There is one span per pipeline stage: `nona.normalize`,
   `nona.detect` (with child spans per recognizer and for `ner.inference`), `nona.link`, `nona.rewrite`, `nona.upstream`
   and `nona.restore`. We do **not** use the FastAPI or httpx auto-instrumentation packages, because they record URLs
   and headers. Manual spans let us decide exactly which attributes exist.
2. **Metrics: a Prometheus `/metrics` endpoint.** Prometheus pulls it, so Nona makes no outbound connection.
   - `nona_requests_total{route, outcome}` with outcome in `ok | fail_closed | upstream_error | client_error`
   - `nona_entities_total{type, source}` with source in `regex | checksum | ner | gazetteer | propagated`
   - `nona_stage_duration_seconds{stage}` and `nona_overhead_seconds` (histograms)
   - `nona_stream_ttft_overhead_seconds` (histogram), `nona_detection_cache_requests_total{result}`
3. **Logs: structlog JSON** with the deny-by-default field allowlist from deep dive G. Each line carries `trace_id` so logs and traces line up.
4. **PII-free by construction:**
   - Every span attribute, metric label and log field comes from a typed allowlist (enums and numbers). The telemetry API does not accept free-form strings.
   - Never recorded: text, entity values, hashes of text, URLs with query strings, headers, the upstream API key, or a raw session id.
   - Label values come from closed enums, so the number of distinct series stays bounded.
5. **Off by default, opt-in export.** `/metrics` is local. Traces go nowhere unless `observability.otlp_endpoint`
   is set, which keeps NFR-3 true. With the default config, the OTel SDK runs with a no-op tracer provider.
6. **The FR-19 leak test covers telemetry too.** The e2e suite runs with an in-memory span exporter and a metrics
   scrape, then searches spans, metric labels and logs for every gold PII value. Any match fails CI.
7. **Demo stack:** `deploy/compose/observability/` runs Nona with the `grafana/otel-lgtm` image (collector,
   Prometheus, Tempo, Loki and Grafana in one container) and a provisioned "Nona overview" dashboard.

## Consequences

- ➕ The latency budget in [03 §5](../design/03-architecture.md#5-concurrency--performance) can be checked against production-like traces, stage by stage.
- ➕ Works with any OTLP backend (Grafana, Honeycomb, Datadog, Jaeger) without code changes.
- ➕ The "telemetry is PII-free" claim is tested in CI, not just stated.
- ➖ Manual instrumentation is more code than auto-instrumentation, and new stages must remember to add spans. A test checks that each stage emits its span.
- ➖ The OTel SDK adds a dependency and a small overhead per span. When tracing is off, the overhead is close to zero.
- ➖ Entity counts per request are metadata (threat model §6). They are aggregated in metrics and only appear per request in a span when tracing is on.
