# Nona

**A local proxy that strips personal data before it reaches an LLM, and puts it back in the answer.**
Built for messy, mixed-language text: French, English, Arabic, Tunisian Darija, Arabizi.

> 🚧 **Status: early development.** Design done, repo skeleton and CI in place (M0). Detection and the proxy are next.

```
You type:        "3andi rendez-vous m3a Sami Trabelsi ghodwa, a3tih 22 123 456"
The LLM sees:    "3andi rendez-vous m3a PERSON_1 ghodwa, a3tih PHONE_1"
You get back:    "...Sami Trabelsi..."   (restored locally)
```

Nona **reduces exposure**. It does not guarantee anonymity. Its leak rate will be measured and published.

## Design docs
1. [Requirements](docs/design/01-requirements.md): functional, non-functional, threat model
2. [API & interfaces](docs/design/02-api.md): the OpenAI-compatible proxy, inspection API, library, CLI, config
3. [High-level design](docs/design/03-architecture.md): components, request lifecycle, state, stack
4. [Deep dives](docs/design/04-deep-dives.md): multilingual detection, entity linking, streaming restoration, tool calls, benchmark method
5. [Benchmark dataset](docs/design/05-dataset.md): schema, labelling guidelines, synthetic value rules, how leaks are counted
6. Decisions: [ADR-0001 placeholder format](docs/adr/0001-placeholder-format.md) · [ADR-0002 stateless by default](docs/adr/0002-stateless-by-default.md) · [ADR-0003 fail closed](docs/adr/0003-fail-closed.md) · [ADR-0004 causal entity linking](docs/adr/0004-causal-entity-linking.md) · [ADR-0005 observability](docs/adr/0005-observability.md) · [ADR-0006 encrypted session store](docs/adr/0006-encrypted-session-store.md) · [ADR-0007 signed releases](docs/adr/0007-release-supply-chain.md)

## Roadmap
- [x] **M0**: repo skeleton, CI (ruff, pyright, pytest), core types
- [ ] **M1**: `core/` library: structured recognizers, GLiNER, linker, placeholders, restore, property tests
- [ ] **M2**: synthetic dataset v0 (300+ msgs, frozen test split) and leak-rate script
- [ ] **M3**: proxy, first non-streaming, then streaming, instrumented from the start: OpenTelemetry traces, Prometheus metrics, PII-free structured logs ([ADR-0005](docs/adr/0005-observability.md))
- [ ] **M4**: Presidio and LLM Guard baselines (resumable bench runner with cached results), results table, k6 load test (p50/p95 overhead, throughput), Grafana dashboard demo, demo GIF, signed release pipeline: PyPI trusted publishing, signed multi-arch GHCR image, SBOM ([ADR-0007](docs/adr/0007-release-supply-chain.md)) → **v0.1 release**
- [ ] **M5**: stateful sessions with an encrypted Redis store ([ADR-0006](docs/adr/0006-encrypted-session-store.md)), tool calls, ONNX int8, Helm chart, TypeScript client, MCP server

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.11+.

```bash
uv sync                      # create .venv with dev tools
uv run pytest --cov          # tests (unit + property-based)
uv run ruff format . && uv run ruff check .
uv run pyright               # strict on src/nona/core
uv run nona --version
```

CI runs the same checks on Python 3.11, 3.12 and 3.13. `nona.core` must stay free of network and process I/O,
which `tests/test_architecture.py` enforces.

## Data policy
All test and benchmark data is synthetic and generated for this repository. No real or client data, not even anonymized.

## License
Apache-2.0
