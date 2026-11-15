# Nona

**A local proxy that strips personal data before it reaches an LLM, and puts it back in the answer.**
Built for messy, mixed-language text: French, English, Arabic, Tunisian Darija, Arabizi.

> 🚧 **Status: design phase.** No code yet. The design is being written in the open.

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
5. Decisions: [ADR-0001 placeholder format](docs/adr/0001-placeholder-format.md) · [ADR-0002 stateless by default](docs/adr/0002-stateless-by-default.md) · [ADR-0003 fail closed](docs/adr/0003-fail-closed.md) · [ADR-0004 causal entity linking](docs/adr/0004-causal-entity-linking.md)

## Roadmap
- [ ] **M0**: repo skeleton, CI (ruff, pyright, pytest), 1 week
- [ ] **M1**: `core/` library: structured recognizers, GLiNER, linker, placeholders, restore, property tests
- [ ] **M2**: synthetic dataset v0 (300+ msgs, frozen test split) and leak-rate script
- [ ] **M3**: proxy, first non-streaming, then streaming
- [ ] **M4**: Presidio and LLM Guard baselines, results table, demo GIF → **v0.1 release**
- [ ] **M5**: Docker, tool calls, stateful sessions, ONNX, TypeScript client, MCP

## Data policy
All test and benchmark data is synthetic and generated for this repository. No real or client data, not even anonymized.

## License
Apache-2.0 (planned)
