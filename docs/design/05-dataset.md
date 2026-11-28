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

- **Normalization before comparing:** drop invisible format characters (Unicode Cf: zero-width spaces and joiners, soft hyphens) and the Arabic tatweel (`ـ`, which only stretches a letter), NFKC, casefold, Eastern Arabic digits → ASCII digits, collapse whitespace. For PHONE, IBAN, RIB, CARD and NATIONAL_ID, compare **digits and letters only** (`22.123.456` ≡ `22123456`).
- **Leaked:** the normalized mention text appears in the normalized upstream payload of the same turn, at word boundaries (`Sami` does not match inside `Samir`).
- **Obfuscated** (PERSON): not leaked as written, but its letters appear in order with only separators between them, at word boundaries (`S a m i`, `S.a.m.i`). A model reads it back, so the headline counts it as a leak, and the report shows its share.
- **Partially leaked:** not leaked, but a token of 3+ characters from the mention (not an honorific) appears at word boundaries. For example, the surname got through.
- **Over-masking:** a placeholder in the upstream payload whose range doesn't overlap any gold mention. Reported overall and on `distractors`. Tokens are compared case-insensitively: lowercasing a word doesn't mask it.
- **Deleted:** a mention that neither leaked nor carries a placeholder. Deleting every name would give a perfect leak rate, so this rate sits next to it.
- **Causal payloads:** turn `i`'s payload is computed from turns `0..i` only (the runner calls the system once per prefix), so a system can't use a later turn to mask an earlier one (ADR-0004).

**Details fixed by the implementation** (`src/nona/bench/metrics.py`, unit and property tests in `tests/bench/test_metrics.py`):

- **Arabic proclitics** don't break the word boundary: `سامي` counts as leaked inside `لسامي` (up to two of `و ف ب ل ك`).
- **Phones** also leak through their national number: `+216 22 123 456` counts as leaked if `22123456` reaches the upstream, and `+33 6 12 34 56 78` if `06 12 34 56 78` does.
- **Partial leaks ignore tokens that identify no one:** honorifics, the name particles `ben`/`bin`/`bou`, phone country codes, the domain of an e-mail address, and the country-and-check head of an IBAN (`TN59`).
- **Over-masking is measured black-box.** The original turn and the payload are split into tokens and aligned (`difflib`). A word token outside every gold mention that was replaced or removed counts as over-masked. Rate = over-masked tokens / non-PII word tokens. Distractors are reported apart: the share of distractors that were touched. This works the same for Nona, for baselines that don't report offsets, and for the on-the-wire capture (FR-21).
- **Scopes:** the headline covers direct identifiers in `user`, `system` and `tool` turns. Restored `assistant` turns that the client re-sends get their own leak rate (ADR-0004). A **held-out** PERSON leak rate covers only people with no name in the gazetteer, so it measures detection rather than list overlap.
- **Linking (ADR-0004), also black-box.** The same alignment tells which payload text replaced each gold mention: its *placeholder* (`Mme Trabelsi` → `Mme PERSON_1` gives `PERSON_1`; a longer replaced stretch is cut into runs of mention and non-mention words and paired in order, so `M. Sami` → `Monsieur PERSON_1` and `Sami Ben Ali` → `PERSON_1 PERSON_2` both work). A mention gets none when it leaked in full, was deleted, or shares its replacement with another mention in a way that can't be split. A partial leak (`Ahmed PERSON_1`) keeps its placeholder but is reported apart (*partly leaking*) and left out of merge, split and ambiguity: its placeholder covers part of the mention only, and counting it would credit linking for a detection gap. Within each conversation and type, over non-public mentions that have a placeholder, skipping `ambiguous: true`:
  - **wrong merge** = entities that share a placeholder with another gold entity / entities with a placeholder;
  - **wrong split** = entities under 2+ placeholders / entities with 2+ masked mentions;
  - **ambiguity** = mentions after an entity's first masked one whose placeholder is neither that entity's first placeholder nor one any other entity uses (an extra "person") / such later mentions. Every ambiguity hit is also part of a wrong split.

  The report also gives the share of in-scope mentions that got a placeholder, since leaked mentions don't count here. A redaction system (`[REDACTED]` everywhere) scores 0% split and merges every entity that has company; a fresh placeholder per mention splits every entity. Property tests check both, and that a system replacing the gold mentions by entity scores 0% on all three.

The `identity` system (sends everything unchanged) must score 100% leaked and 0% over-masked. A property test also checks that a system replacing exactly the gold mentions scores 0% and 0%, on any seed of the generator.

**Known limits:** homoglyphs (a Cyrillic `a` inside a Latin name) aren't caught: NFKC doesn't map them, and folding confusables would also merge genuine Cyrillic or Greek text. And a word-boundary substring check can count a coincidental match as a leak, e.g. a gold name `Nour` while the payload legitimately contains the word *nour*. This makes the leak rate slightly pessimistic, which is the safe direction. Cases are listed in the report rather than silently removed.

## 7. Size and balance (v0.1)

- ≥ 300 turns total, ≥ 40 conversations of 4+ turns (FR-20).
- Each of `fr`, `en`, `ar`, `aeb-Arab`, `aeb-Latn` appears in ≥ 15% of conversations, and ≥ 30% of conversations are code-switched.
- ≥ 10% of conversations are `no-pii` controls.
- About 50 hand-written hard cases, tagged.
- Split: about 40% dev, 60% test, stratified by language and source.

### Current splits (S4, #74)

Counts from `python -m nona.bench.dataset stats bench/data/dev.jsonl bench/data/test.jsonl`. A language counts
for every conversation it appears in, so a code-switched conversation counts once per language;
`tests/bench/test_splits.py` checks the targets above on both splits.

| | Target | `dev` | `test` |
|---|---|---|---|
| Conversations | | 275 | 353 (56% of 628) |
| Turns | ≥ 300 in all | 609 | 614 |
| Conversations of 4+ turns | ≥ 40 in all | 88 | 78 |
| `fr` | ≥ 15% | 114 (41.5%) | 120 (34.0%) |
| `en` | ≥ 15% | 62 (22.5%) | 77 (21.8%) |
| `ar` | ≥ 15% | 46 (16.7%) | 73 (20.7%) |
| `aeb-Arab` | ≥ 15% | 54 (19.6%) | 76 (21.5%) |
| `aeb-Latn` | ≥ 15% | 82 (29.8%) | 113 (32.0%) |
| Code-switched | ≥ 30% | 83 (30.2%) | 106 (30.0%) |
| `no-pii` | ≥ 10% | 60 (21.8%) | 86 (24.4%) |
| Hand-written hard cases | about 50 | 21 | 30 |
| LLM drafts | | 12 | 18 |

How each part is made (every generated part is pinned by `test_generated_parts_are_pinned`):

| Split | Part | Conversations | How it's made |
|---|---|---|---|
| `dev` | v0 (`dev-0001`…`0080`) | 80, mostly single-turn | `generate --seed 20261120 --count 80 --prefix dev` (S1, regenerated in #34) |
| `dev` | multi-turn (`dev-0081`…`0137`) | 57, 4–5 turns each | `generate --set multi-turn --seed 2 --count 57 --prefix dev --start 81`: 19 templates, three times each (S3 #41, extended in #58) |
| `dev` | S4 (`dev-0138`…`0242`) | 105 | `generate --set dev-s4 --seed 4 --count 105 --prefix dev --start 138`: 35 templates, three times each (#74) |
| `test` | templates (`test-0001`…`0305`) | 305 | `generate --set test --seed 20261128 --count 305 --prefix test`: 61 templates, five times each (#74) |
| `dev` | hard cases (`dev-hard-01`…`21`) | 21 | Hand-written (#74), see below |
| `dev` | LLM drafts (`dev-llm-01`…`12`) | 12 | Drafted by an LLM around supplied fake entities, then reviewed (#74) |
| `test` | hard cases (`test-hard-01`…`30`) | 30 | Hand-written (#74) |
| `test` | LLM drafts (`test-llm-01`…`18`) | 18 | Drafted by an LLM, then reviewed (#74) |

**The S4 set and the test split (#74).** The S4 templates bring Arabic script (`ar`, `aeb-Arab`) and
code-switching up to the targets: Darija in Arabic script with clitics glued to names (`ل`, `و`, `ب`), Arabic
cities (LOCATION), phones and CIN numbers in Eastern Arabic digits (`٢٢ ١٢٣ ٤٥٦`), Arabic and Arabizi mixed
with French and English in one turn, and no-PII controls built on the common words of #50 used as words
(`حساب جاري`, `سنة سعيدة`, `ألف مبروك`, `Ramdhan karim`, `amine`, `farah`, `chams`, `kmar`) and English or
French ones (`car jack`, `grace period`, `bill`, `olivier`, `rose`). The `test` split has **its own templates**
(no turn text in common with `dev`, a test checks it), its own seed, and **its own held-out names**: the
`tn3` and `ar3` pools, which are in no other pool, not in the gazetteer and nowhere in `dev`. So tuning on
`dev` can't learn the test's names or sentences. Its hash is in `bench/data/test.sha256`, and a test fails if
the file changes without it. The owner validated its Darija items on 2026-11-28, and the split has been
frozen since #75: `nona bench` refuses to score it without `--published` (the file `test.sha256` names, even
edited, or a byte-for-byte copy; a guard against accidents, see `nona.cli.is_frozen`), and
[`DATASHEET.md`](../../bench/data/DATASHEET.md) gives how it was made and its limits.

**Hard cases and LLM drafts (#74).** The 51 hard cases exercise the #50 lists both ways. Common-word names
*as names*: after a cue (`kallamt Amal`, `9olt l sana`, `صاحبي كريم`, `قلت لنور`, `khalti Hayet`, `3amm
Salah`, `الحاجّة رحمة`), and with no cue at all (`سلّملي على كريم`, `ken tchouf amal`, `حضرت أمال`), which
the owner's rule leaves unmasked in Arabic script and lower-case Arabizi, so they are expected leaks. The same
words *as words*: fixed expressions (`3id mabrouk`, `رحمة الله`, `نور عينيا`, `fama amal`, `صالح للاستعمال`)
and the decoys the cue lists mask on purpose (`rajel karim`, `sana jet`, `chouft amal`, `weld 3aziz`: they
count as over-masking by design). Then Eastern Arabic digits next to look-alikes (a CIN and an order number
in one sentence, prices, dates), clitics (`ل`, `و`, `لـ` with a tatweel), Arabizi spellings (`Mo7amed`,
`7amza`, `Med`, `Mou3ez`, `Ra7ma`, `Sa3id`), one person written in two scripts (`Sami` / `سامي`), a street
named after a public figure (`12 avenue Habib Bourguiba`, a LOCATION), `Sidi Bou Said`, and a public figure
(`public: true`) next to a private `Amal`. The 30 LLM drafts are realistic requests (reply to, summarise,
translate or correct a WhatsApp message or an e-mail) written around supplied fake entities; `dev` drafts
use `tn2`/`ar2` names and `test` drafts `tn3`/`ar3` ones. Every Darija item is to be validated by a native
speaker with `python -m nona.bench.review` (D3) before the split is frozen; all were accepted on 2026-11-28.

**History of `dev` (S3.1).** The multi-turn set exists for linking (ADR-0004): a short name before the full one, with the short form detected (`Mme Islem`, or a listed first name) so absorption is measured; two people sharing a first name with `ambiguous: true` references (held-out names, and a `-listed` copy); honorifics; assistant turns that repeat restored names; lowercase Arabizi names; cities (LOCATION); Arabic-script conversations (`ar`, `aeb-Arab`) with clitics glued to names (`لـ`, `وـ`); and no-PII controls with name-like words: `amal`, `sana`, `3amel`, `3ali`, French words that fold to names (`marié`, `claire`, `pierre`, `Grâce`), Darija expressions (`Mabrouk 3lik`, `ya sahbi`, `lazem saber`), a brand after a cue (`m3a Ooredoo`), and Arabic expressions (`رمضان كريم`, `اللهم آمين`, `أمل`).

Most of its people come from the generator's `tn2` and `ar2` pools, which hold names that are **not** in the gazetteer (a test enforces it), so the benchmark measures detection rather than list overlap. The report's **held-out** PERSON rate keeps only those people: 75.0% of their mentions leak (72/96), while on v0, whose names are all at least partly listed, PERSON leaks 0%. The street in `mt-fr-moved-city` was an unlabelled LOCATION; the turn was reworded in #58.

## 8. Open questions for review

| # | Question | Proposal |
|---|---|---|
| D1 | Should public figures be labelled at all? | Yes, labelled with `public: true` and excluded from the headline. It keeps the data honest and lets the over-masking analysis see them |
| D2 | LOCATION at city level: is that too strict? | Keep city or finer. Report it separately (§4), so the choice doesn't affect the headline |
| D3 | Who checks LLM-written messages? | You, with a small review script that shows each turn with its mentions highlighted. Done in #74: `python -m nona.bench.review bench/data/dev.jsonl --source llm` prints each conversation with its labels, and `--decisions review.json` asks accept / fix / drop for each and saves the answers |
