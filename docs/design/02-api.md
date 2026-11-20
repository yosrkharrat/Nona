# 02 — API & interfaces

> Status: **Draft v0.1** · Depends on: [01 — Requirements](01-requirements.md)

Nona has four surfaces, in priority order:

1. **OpenAI-compatible HTTP proxy**: the product.
2. **Inspection HTTP API** (`/v1/nona/*`): debugging and the demo.
3. **Python library**: the core, which the proxy is a thin layer over.
4. **CLI**.

Design rule: **the proxy adds nothing the client has to learn.** Every Nona-specific behaviour is opt-in through
headers or config.

---

## 1. Proxy API (OpenAI-compatible)

### Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",   # ← the only change
    api_key=os.environ["OPENAI_API_KEY"],  # passed through to the upstream
)
client.chat.completions.create(model="gpt-4o-mini", messages=[...], stream=True)
```

### Endpoints

| Method | Path | Behaviour |
|---|---|---|
| `POST` | `/v1/chat/completions` | Anonymize → forward → restore. Supports `stream`, `tools`, `tool_choice`, `response_format` |
| `GET` | `/v1/models` | Pure passthrough |
| `GET` | `/healthz` | `200` once the NER model is loaded (readiness) |
| `GET` | `/metrics` | Prometheus metrics, PII-free (optional) |

### What gets anonymized in a request

| Field | Anonymized? | Note |
|---|---|---|
| `messages[].content` (string) | ✅ | All roles, including `system` and earlier `assistant` turns (ADR-0002) |
| `messages[].content[]` parts of type `text` | ✅ | `image_url` parts → rejected with `422` by default (policy `on_image: reject \| passthrough`) |
| `messages[].tool_calls[].function.arguments` | ✅ | Parsed as JSON, and only **string values** are scanned |
| `messages[role=tool].content` | ✅ | Tool results often contain the most PII |
| `messages[].name`, `user` | ✅ | `user` is often an email |
| `tools[].function.description` | ❌ | Written by the developer, not user data |
| `model`, `temperature`, … | ❌ | — |

### What gets restored in a response

`choices[].message.content`, `choices[].message.tool_calls[].function.arguments`, and the equivalent
`delta` fields in streaming chunks. Usage, ids and finish reasons pass through untouched.

### Optional request headers

| Header | Values | Default |
|---|---|---|
| `X-Nona-Policy` | name of a policy profile from config | `default` |
| `X-Nona-Dry-Run` | `true` → returns what *would* be sent upstream, sends nothing | `false` |
| `X-Nona-Session` | random id, ≥ 128 bits → enables stateful mode (v0.2, ADR-0002, ADR-0006) | absent = stateless |

### Response headers (PII-free)

`X-Nona-Request-Id`, `X-Nona-Entities: PERSON=3,PHONE=1`, `X-Nona-Overhead-Ms: 41`

### Errors

Errors use the OpenAI error shape (`{"error": {"message", "type", "code"}}`) so SDKs surface them normally.

| HTTP | `code` | When |
|---|---|---|
| 503 | `nona_detector_unavailable` | Model not loaded or detector crashed → **fail closed**, nothing forwarded |
| 504 | `nona_detector_timeout` | Detection exceeded `detector_timeout_ms` |
| 422 | `nona_unsupported_content` | Image or file content under the `reject` policy |
| 4xx/5xx | upstream code | Upstream errors are relayed **after** restoration (they can echo the prompt) |

---

## 2. Inspection API

### `POST /v1/nona/anonymize`

```jsonc
// request
{ "text": "3andi rendez-vous m3a Sami Trabelsi ghodwa, a3tih 22 123 456",
  "policy": "default", "language_hint": null }

// response
{ "text": "3andi rendez-vous m3a PERSON_1 ghodwa, a3tih PHONE_1",
  "entities": [
    { "placeholder": "PERSON_1", "type": "PERSON", "start": 21, "end": 35,
      "value": "Sami Trabelsi", "score": 0.91, "detector": "gliner+gazetteer" },
    { "placeholder": "PHONE_1",  "type": "PHONE",  "start": 50, "end": 60,
      "value": "22 123 456", "score": 1.0, "detector": "regex:phone_tn" } ],
  "timing_ms": { "normalize": 1, "detect": 38, "link": 2 } }
```

This endpoint returns original values. That is acceptable because it binds to localhost only, and it can be
disabled (`inspection_api: false`).

### `POST /v1/nona/restore`
`{ "text": "...PERSON_1...", "mapping_token": "..." }` → `{ "text": "...Sami Trabelsi..." }`. For library-less clients.

---

## 3. Python library

```python
from nona import Nona

nona = Nona.from_config("nona.yaml")          # loads the models once

result = nona.anonymize("Appelle Ahmed Ben Salem au +216 22 123 456")
result.text          # "Appelle PERSON_1 au PHONE_1"
result.entities      # list[Entity]
result.mapping       # Mapping (opaque; never serialized by default)

nona.restore("PERSON_1 confirme.", result.mapping)   # "Ahmed Ben Salem confirme."

# conversation-level, the same thing the proxy uses
conv = nona.anonymize_messages(messages)             # one shared mapping for all messages
restorer = conv.stream_restorer()                    # for streaming text
for chunk in upstream_stream:
    yield restorer.feed(chunk)
yield restorer.flush()
```

Core types (sketch):

```python
@dataclass(frozen=True)
class Span:      start: int; end: int; type: EntityType; score: float; detector: str
@dataclass(frozen=True)
class Entity:    id: int; type: EntityType; mentions: tuple[Span, ...]; canonical: str
class Recognizer(Protocol):
    name: str; supported_types: frozenset[EntityType]
    def detect(self, doc: NormalizedDoc) -> list[Span]: ...
```

## 4. CLI

> Binary and package name: `nona` (Q1 resolved; check that the PyPI name is free before release).

```
nona serve   [--config nona.yaml] [--port 8787] [--upstream https://api.openai.com/v1]
nona scan    <file|->  [--format text|json|html]        # anonymize a file, print a diff
nona bench   [--systems nona,presidio,llmguard] [--split test] [--out results/]
nona models  pull                                        # explicit model download (NFR-3)
```

## 5. Configuration (`nona.yaml`)

```yaml
upstream:
  base_url: https://api.openai.com/v1
  timeout_s: 120
server:
  host: 127.0.0.1
  port: 8787
  inspection_api: true
detection:
  languages: [fr, en, ar, aeb]            # aeb = Tunisian Arabic (ISO 639-3)
  ner_model: urchade/gliner_multi-v2.1    # pinned revision in lockfile
  ner_threshold: 0.35                     # low on purpose: recall first (NFR-5)
  detector_timeout_ms: 2000
  custom_recognizers:
    - name: client_ref
      type: CUSTOMER_ID
      pattern: '\bCL-\d{6}\b'
  allowlist: ["Acme Corp", "Nona"]
  denylist:  ["Projet Jasmin"]
policies:
  default:
    PERSON: pseudonymize
    EMAIL: pseudonymize
    PHONE: pseudonymize
    CARD: redact                          # irreversible
    LOCATION: pseudonymize
    ORGANIZATION: keep
on_detector_error: reject                 # fail closed (ADR-0003)
logging:
  level: info
  log_bodies: false                       # cannot be enabled with real upstreams
observability:                            # ADR-0005
  metrics: true                           # GET /metrics, local pull, PII-free
  otlp_endpoint: null                     # e.g. http://localhost:4317; null = traces go nowhere (NFR-3)
  sample_ratio: 1.0
sessions:                                 # stateful mode, ADR-0006
  backend: memory                         # memory | redis
  redis_url: null                         # e.g. redis://localhost:6379/0 (Redis 7+ or Valkey)
  ttl_sliding_s: 3600
  ttl_absolute_s: 86400
  # master key comes from NONA_SESSION_KEY, never from this file
```
