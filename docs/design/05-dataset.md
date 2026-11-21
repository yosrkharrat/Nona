# 05: Benchmark dataset specification

> Status: **Draft v0.1, for review**, 2026-11-15 · Depends on: [01 FR-20–23](01-requirements.md), [04 §F](04-deep-dives.md#f-benchmark-methodology), [ADR-0004](../adr/0004-causal-entity-linking.md)
> Implemented by: `src/nona/bench/dataset.py` (loader + validator)

Every number Nona publishes comes from this dataset, so the rules for it are written down before the first message is.

## 1. Principles

1. **Synthetic only.** Every message is written or generated for this repository. No real conversations, not even "anonymized" ones.
2. **Gold labels come from the author, not a model.** Templates insert entities, so their labels are exact. LLM-written messages are checked and corrected by a human before they're accepted.
3. **The test split is frozen.** Its hash is committed. It is never used to tune thresholds, prompts or gazetteers. Tuning happens on `dev`.
4. **Labels follow the guidelines in §4**, so another person would label the same text the same way.

## 2. Files

```
bench/data/
├─ dev.jsonl          # tune here
├─ test.jsonl         # frozen; hash in test.sha256
├─ test.sha256
└─ DATASHEET.md       # how it was made, its limits, its language mix
```

One **conversation** per line. A single message is a conversation with one turn, so there's only one format.

## 3. Schema

```jsonc
{
  "id": "conv-0042",                        // unique across all splits
  "languages": ["aeb-Latn", "fr"],          // BCP 47, see §3.1; 2+ entries = code-switched
  "source": "template",                     // "template" | "llm" | "handwritten"
  "tags": ["arabizi", "multi-turn"],        // difficulty tags, used to break down results (§3.2)
  "entities": [                             // conversation-level gold entities (ADR-0004)
    { "id": "e1", "type": "PERSON", "value": "Ahmed Ben Salem" },
    { "id": "e2", "type": "EMAIL",  "value": "ahmed.bs@example.com", "owner": "e1" },
    { "id": "e3", "type": "PERSON", "value": "Kais Saied", "public": true }
  ],
  "turns": [
    {
      "role": "user",                       // "system" | "user" | "assistant" | "tool"
      "text": "Rappelle Ahmed demain.",
      "mentions": [
        { "text": "Ahmed", "entity": "e1", "start": 9, "end": 14 }
      ],
      "distractors": [                      // optional: look like PII, are not (for over-masking analysis)
        { "text": "demain", "note": "example only" }
      ]
    }
  ]
}
```

| Field | Rule |
|---|---|
| `entities[].type` | One of `EntityType` (`src/nona/core/types.py`) |
| `entities[].value` | Canonical form: the longest full form in the conversation |
| `entities[].owner` | Optional. Links an email or phone to the person it belongs to (deep dive B, step 6) |
| `entities[].public` | Optional, default `false`. Public figures are labelled but **excluded from the headline leak rate** |
| `mentions[].start/end` | Python string indices (Unicode code points, **not** UTF-16), half-open `[start, end)` |
| `mentions[].text` | Must equal `turn.text[start:end]`. The validator checks this |
| `mentions[].ambiguous` | Optional. `true` when even a human can't tell which entity is meant (`Ahmed` with two Ahmeds). It counts for leaks and is skipped by the linking metrics |
| `assistant` turns | Represent the **restored** answer as the client sends it back in the history. ADR-0004 makes their leak rate a separate metric |

**Authoring form.** Writing offsets by hand is error-prone. A mention may omit `start`/`end` and give only `text`, plus `occurrence` (0-based) when the text appears more than once in the turn. The loader computes the offsets, and the canonical writer always stores them. Committed files are always in canonical form.

### 3.1 Language tags

| Tag | Meaning |
|---|---|
| `fr`, `en` | French, English |
| `ar` | Modern Standard Arabic, Arabic script |
| `aeb-Arab` | Tunisian Darija, Arabic script |
| `aeb-Latn` | Tunisian Darija in Latin script with digits (Arabizi: `3andi`, `m3a`, `7aja`) |

### 3.2 Difficulty tags (open list; the validator warns on unknown tags)

`arabizi`, `code-switching`, `multi-turn`, `common-word-name` (Amal, Nour, Salam…), `clitic` (لسامي), `eastern-digits` (٩٨٧), `spelling-variant` (Mohamed/Med), `honorific`, `ambiguous-reference`, `no-pii` (control: nothing to find).

## 4. Labelling guidelines

| Type | Label | Don't label |
|---|---|---|
| PERSON | Full names, first names alone, surnames alone, nicknames that identify someone | Honorifics and kinship words outside the span: `M.`, `Dr`, `si`, `khalti`, `3amm`. Names used as common words (`amal` = hope) |
| Arabic clitics | Only the name: `لسامي` → `سامي` | The clitic `ل`, `ب`, `و`, `ف` |
| PHONE | The whole number, including `+216`/`00216`, inner spaces, dots or dashes | Surrounding words (`tel:`, `nomro`) |
| EMAIL | The whole address | `mailto:` |
| IBAN, RIB, CARD | The whole number, including inner spaces | Labels like `IBAN:` |
| NATIONAL_ID | 8-digit CIN **only when the context says it's an ID** | Prices, order numbers, dates |
| LOCATION | City or finer: city, neighbourhood, street, address | Countries and regions (`Tunisie`, `Sahel`) |
| ORGANIZATION | Specific companies, clinics, schools | Generic words (`la banque`, `l'hôpital`) |

**Headline leak rate** covers *direct identifiers* only: PERSON (non-public), EMAIL, PHONE, IBAN, RIB, CARD, NATIONAL_ID, IP_ADDRESS. LOCATION and ORGANIZATION are reported separately, so they can't inflate or hide the headline number.

## 5. Synthetic value rules

Fake values must never be the real data of a real person.

| Type | Rule |
|---|---|
| Names | Common first names combined with common surnames, generated at random. No public figures except entities marked `public` |
| Emails | Reserved domains only: `example.com`, `example.org`, `example.net` (RFC 2606). Detection doesn't depend on the domain |
| Phones | Random numbers that libphonenumber accepts, drawn from the whole numbering plan: Tunisian mobile (`2x`, `4x`, `5x`, `9x`, `3x`) and fixed lines (`7x`, `3x`, also used for fax), French `06`/`07`. Written every common way: `XX XXX XXX`, dots, dashes, pairs, `+216`, `(+216)`, `00 216`, `216`, and the `+216 (0)` habit. Business service numbers (toll-free, shared cost, premium) appear only as distractors. A collision with a real number is possible but meaningless, since no name is attached to it outside this file |
| IBAN / RIB | Valid checksums (mod-97) with random digits |
| Cards | Luhn-valid numbers from published **test** card ranges only |
| CIN | Random 8 digits |

## 6. How leaks are counted (input to the leak-rate script)

The benchmark captures the **payload sent upstream** for each turn (04 §F). For each gold mention:

- **Normalization before comparing:** NFKC, casefold, Eastern Arabic digits → ASCII digits, collapse whitespace. For PHONE, IBAN, RIB, CARD and NATIONAL_ID, compare **digits and letters only** (`22.123.456` ≡ `22123456`).
- **Leaked:** the normalized mention text appears in the normalized upstream payload of the same turn, at word boundaries (`Sami` does not match inside `Samir`).
- **Partially leaked:** not leaked, but a token of 3+ characters from the mention (not an honorific) appears at word boundaries. For example, the surname got through.
- **Over-masking:** a placeholder in the upstream payload whose range doesn't overlap any gold mention. Reported overall and on `distractors`.

**Details fixed by the implementation** (`src/nona/bench/metrics.py`, unit and property tests in `tests/bench/test_metrics.py`):

- **Arabic proclitics** don't break the word boundary: `سامي` counts as leaked inside `لسامي` (up to two of `و ف ب ل ك`).
- **Phones** also leak through their national number: `+216 22 123 456` counts as leaked if `22123456` reaches the upstream, and `+33 6 12 34 56 78` if `06 12 34 56 78` does.
- **Partial leaks ignore tokens that identify no one:** honorifics, the name particles `ben`/`bin`/`bou`, phone country codes, the domain of an e-mail address, and the country-and-check head of an IBAN (`TN59`).
- **Over-masking is measured black-box.** The original turn and the payload are split into tokens and aligned (`difflib`). A word token outside every gold mention that was replaced or removed counts as over-masked. Rate = over-masked tokens / non-PII word tokens. Distractors are reported apart: the share of distractors that were touched. This works the same for Nona, for baselines that don't report offsets, and for the on-the-wire capture (FR-21).
- **Scopes:** the headline covers direct identifiers in `user`, `system` and `tool` turns. Restored `assistant` turns that the client re-sends get their own leak rate (ADR-0004).

The `identity` system (sends everything unchanged) must score 100% leaked and 0% over-masked. A property test also checks that a system replacing exactly the gold mentions scores 0% and 0%, on any seed of the generator.

**Known limit:** a word-boundary substring check can count a coincidental match as a leak, e.g. a gold name `Nour` while the payload legitimately contains the word *nour*. This makes the leak rate slightly pessimistic, which is the safe direction. Cases are listed in the report rather than silently removed.

## 7. Size and balance (v0.1)

- ≥ 300 turns total, ≥ 40 conversations of 4+ turns (FR-20).
- Each of `fr`, `en`, `ar`, `aeb-Arab`, `aeb-Latn` appears in ≥ 15% of conversations, and ≥ 30% of conversations are code-switched.
- ≥ 10% of conversations are `no-pii` controls.
- About 50 hand-written hard cases, tagged.
- Split: about 40% dev, 60% test, stratified by language and source.

### Current `dev` (S3)

| Part | Conversations | How it's made |
|---|---|---|
| v0 (`dev-0001`…`0080`) | 80, mostly single-turn | `generate --seed 1 --count 80` (S1). Committed before #34 changed phone formats, so today's generator gives slightly different values for it; the committed file is the source of truth |
| multi-turn (`dev-0081`…`0120`) | 40, 4–5 turns each | `generate --set multi-turn --seed 2 --count 40 --start 81` (S3, #41) |

The multi-turn set exists for linking (ADR-0004): a short name before the full one, two people sharing a first name with `ambiguous: true` references, honorifics, assistant turns that repeat restored names, lowercase Arabizi names, cities (LOCATION), name-like decoys (`amal`, `sana`, `3amel`, `3ali`), and no-PII controls. Most of its people come from the generator's `tn2` pool, which holds names that are **not** in the gazetteer (a test enforces it). That way the benchmark measures detection rather than list overlap: on v0 the gazetteer reaches 0% leaked because 76 of the generator's 98 v0 names are common enough to be listed, while on the multi-turn set 70% of names still leak.

## 8. Open questions for review

| # | Question | Proposal |
|---|---|---|
| D1 | Should public figures be labelled at all? | Yes, labelled with `public: true` and excluded from the headline. It keeps the data honest and lets the over-masking analysis see them |
| D2 | LOCATION at city level: is that too strict? | Keep city or finer. Report it separately (§4), so the choice doesn't affect the headline |
| D3 | Who checks LLM-written messages? | You, with a small review script that shows each turn with its mentions highlighted |
