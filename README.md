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
6. [Build plan](docs/design/06-build-plan.md): measured vertical slices from here to v0.1, definition of done, risks
7. Decisions: [ADR-0001 placeholder format](docs/adr/0001-placeholder-format.md) · [ADR-0002 stateless by default](docs/adr/0002-stateless-by-default.md) · [ADR-0003 fail closed](docs/adr/0003-fail-closed.md) · [ADR-0004 causal entity linking](docs/adr/0004-causal-entity-linking.md) · [ADR-0005 observability](docs/adr/0005-observability.md) · [ADR-0006 encrypted session store](docs/adr/0006-encrypted-session-store.md) · [ADR-0007 signed releases](docs/adr/0007-release-supply-chain.md)

## Roadmap
Built in measured vertical slices: each one works end to end and moves a benchmark number. Details, estimates and
progress are in the [build plan](docs/design/06-build-plan.md).

- [x] **M0**: repo skeleton, CI (ruff, pyright, pytest), core types, strict dataset loader
- [x] **S0**: go public, harden CI (SHA-pinned actions, pip-audit, CodeQL, Dependabot)
- [x] **S1**: benchmark first: synthetic generator, dev split v0, leak and over-masking metrics, `nona bench`
- [x] **S2**: structured detection (email, phone, IBAN, RIB, card, IP, CIN) and `anonymize()` / `restore()`
- [x] **S3**: multi-turn entity linking ([ADR-0004](docs/adr/0004-causal-entity-linking.md), accepted), gazetteer and context rules, linking metrics, detection cache
- [ ] **S4**: names with GLiNER, full dataset, frozen test split
- [ ] **S5**: OpenAI-compatible proxy, instrumented from the start ([ADR-0005](docs/adr/0005-observability.md))
- [ ] **S6**: streaming restoration
- [ ] **S7**: Presidio and LLM Guard baselines, utility loss, k6 load test
- [ ] **S8**: signed release ([ADR-0007](docs/adr/0007-release-supply-chain.md)) → **v0.1**
- Later, if a number or a user asks for it: encrypted Redis sessions ([ADR-0006](docs/adr/0006-encrypted-session-store.md)), tool calls, ONNX int8, Helm chart, TypeScript client, MCP server

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.11+.

```bash
uv sync                      # create .venv with dev tools
uv run pytest --cov          # tests (unit + property-based)
uv run ruff format . && uv run ruff check .
uv run pyright               # strict on src/nona/core
uv run nona --version
```

Benchmark (all data synthetic, see [05](docs/design/05-dataset.md)):

```bash
uv run nona bench --systems identity --split dev     # table on stdout, JSON in results/
uv run python -m nona.bench.dataset check bench/data/dev.jsonl
uv run python -m nona.bench.generate --seed 20261120 --count 80 --prefix dev   # how dev.jsonl v0 was made
```

CI runs the same checks on Python 3.11, 3.12 and 3.13. `nona.core` must stay free of network and process I/O,
which `tests/test_architecture.py` enforces.

## Data policy
All test and benchmark data is synthetic and generated for this repository. No real or client data, not even anonymized.

## License
Apache-2.0
