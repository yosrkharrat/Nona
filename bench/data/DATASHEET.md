# Datasheet: the Nona benchmark dataset

Answers to the questions of *Datasheets for Datasets* (Gebru et al.) for `bench/data/`. The rules the data
follows are in [05-dataset.md](../../docs/design/05-dataset.md); this file says what was actually built,
and what it can't tell you.

**In one paragraph.** 628 synthetic chat conversations (1,223 turns) in French, English, Modern Standard
Arabic and Tunisian Darija (in Arabic script and in Latin script, "Arabizi"), often mixed in one message,
with every piece of personal data labelled by hand or by construction. It measures how much personal data a
pseudonymizing proxy lets through to an LLM. Nothing in it comes from a real conversation or a real person.
The `test` split is frozen: its hash is committed, and it is run once, for the published numbers.

## Motivation

- **Why it exists.** No public benchmark measures PII leakage in Tunisian chat as it is really written:
  Darija in two scripts, code-switched with French and English, with names that are also everyday words
  (`Amal` = hope, `Karim` = generous, `Mabrouk` = congratulations). Nona's leak rate (its headline number,
  NFR-11) needs one, and a fair comparison with Presidio and LLM Guard needs the same one.
- **Who made it.** The Nona project, for this repository, between 2026-11-15 and 2026-11-28 (S1 to S4 of
  the [build plan](../../docs/design/06-build-plan.md)). Templates, hard cases and drafts were written by
  Claude, an AI coding assistant, working in the repository at the owner's direction. The repository owner,
  a native speaker of Tunisian Darija, chose the cases and reviewed every Darija, Arabizi and Arabic-script
  item. No funding.

## Composition

| | `dev` | `test` |
|---|---|---|
| Conversations | 275 | 353 |
| Turns (user / assistant) | 609 (420 / 189) | 614 (447 / 167) |
| Conversations of 4+ turns | 88 | 78 |
| Gold entities | 448 | 488 |
| Gold mentions | 761 | 748 |
| Distractors (look like PII, are not) | 153 | 182 |
| `no-pii` controls | 60 (21.8%) | 86 (24.4%) |
| Hand-written / LLM-drafted / templated | 21 / 12 / 242 | 30 / 18 / 305 |

- **One instance** is a conversation: a list of turns (`user` or `assistant`; the schema also allows
  `system` and `tool`, which this release doesn't use), the entities it contains, and each mention's offsets.
  The schema is in [05 §3](../../docs/design/05-dataset.md#3-schema).
- **Entity types and mentions** (`dev` / `test`): PERSON 449 / 407, PHONE 156 / 146, EMAIL 60 / 48,
  LOCATION 30 / 55, IBAN 19 / 26, RIB 20 / 16, NATIONAL_ID (Tunisian CIN) 11 / 24, CARD 10 / 16,
  IP_ADDRESS 6 / 10. There is no ORGANIZATION entity in this release.
- **Languages.** `fr`, `en`, `ar` (MSA), `aeb-Arab` (Darija in Arabic script), `aeb-Latn` (Arabizi). Each
  is in at least 15% of each split's conversations, and 30% of each are code-switched; the exact shares are in
  [05 §7](../../docs/design/05-dataset.md#current-splits-s4-74), and `tests/bench/test_splits.py` checks
  the targets.
- **Difficulty tags** (`dev` / `test`): `multi-turn` 101 / 89, `code-switching` 83 / 106, `arabizi` 82 / 113,
  `common-word-name` 47 / 93, `honorific` 32 / 26, `clitic` 23 / 24, `eastern-digits` 14 / 25,
  `ambiguous-reference` 9 / 10, `spelling-variant` 4 / 5. Results can be broken down by them.
- **Public figures.** One entity in `test` is marked `public: true` (a public figure named next to a private
  person); it is labelled but left out of the headline. `dev` has none.
- **Splits.** `dev` is for tuning. `test` (56% of conversations) has its own templates (no turn text in
  common with `dev`), its own seed and its own held-out name pools (`tn3`, `ar3`), so tuning on `dev` can't
  learn its sentences or its names. Ids are unique across both files.
- **Is anything confidential or about a real person?** No. See *Collection*.

## Collection

Three sources, all written for this repository:

1. **Templates (547 conversations).** Sentences written by hand with typed slots, filled by a seeded
   generator (`src/nona/bench/generate.py`). Every templated part is reproduced byte for byte by the commands
   in 05 §7, and a test pins each one. Fake values follow [05 §5](../../docs/design/05-dataset.md#5-synthetic-value-rules):
   - names: common Tunisian, French and English first names and surnames, combined at random;
   - e-mails on the reserved `example.com/.org/.net` domains only;
   - phones that libphonenumber accepts, random within the Tunisian and French numbering plans, written in
     every usual format, also in Eastern Arabic digits;
   - IBAN and RIB with valid checksums and random digits;
   - cards from the published test ranges only (they can't be charged);
   - CIN: 8 random digits.
2. **Hand-written hard cases (51).** Written one by one, not from a template, for what templates do badly: common-word names with
   and without a cue, the same words used as words, Eastern Arabic digits next to look-alike numbers, clitics
   glued to names (`لسامي`), Arabizi spellings (`Mo7amed`, `Mou3ez`), one person written in two scripts, a
   street named after a public figure.
3. **LLM drafts (30).** Realistic requests (reply to, summarise, translate or correct a WhatsApp message or an
   e-mail) drafted by an LLM (Claude, with no external API call) around fake entities it was given, then
   checked by a human
   ([05 D3](../../docs/design/05-dataset.md#8-open-questions-for-review)).

Names in `dev`'s later parts and in all of `test` come from pools that are **not** in Nona's gazetteer
(`tn2`/`ar2` for `dev`, `tn3`/`ar3` for `test`; tests enforce it). A real person may happen to share a
generated name or number; no real person's data was used, and no name is tied to a number outside this file.

## Labelling

- **Templates** are labelled by construction: the generator knows where it put each value.
- **Hand-written cases and LLM drafts** are labelled by hand against the guidelines in
  [05 §4](../../docs/design/05-dataset.md#4-labelling-guidelines). The loader checks that every mention's
  text equals the turn text at its offsets.
- **Gold follows the guidelines, not Nona.** A bare `كريم` that refers to a person is a PERSON, even where
  Nona's rule deliberately leaves it unmasked. Some decoys (`rajel karim`, `sana jet`) are masked by Nona on
  purpose and count as over-masking.
- **Review.** Every Darija, Arabizi and Arabic-script hand-written item and draft was reviewed by the owner
  with `python -m nona.bench.review` on 2026-11-28 and accepted as written.

## Uses

- **Intended:** measuring what a pseudonymizing proxy sends upstream (leak, partial leak, over-masking,
  linking), per language and type, with `nona bench`. The method is in
  [05 §6](../../docs/design/05-dataset.md#6-how-leaks-are-counted-input-to-the-leak-rate-script).
- **Rules for `test`:** never tune thresholds, lists, prompts or models on it, and never look at its
  failures to fix them. `nona bench` refuses to score it without `--published`: the file that the
  `test.sha256` beside it names, even edited, and a byte-for-byte copy elsewhere in a checkout. It guards
  against accidents; an edited copy in another folder gets through. A bug found there gets its own `dev`
  case before it is fixed (example: #100).
- **Not suitable for:** training a model (too small, and templated); measuring how often PII appears in real
  chat (the mix of types and languages was chosen to hit targets, not sampled); other Arabic dialects
  (Moroccan, Algerian, Levantine…), which it doesn't contain.

## Limits

Read the numbers with these in mind:

- **Synthetic.** Templates repeat (each `test` template appears 5 times with different values, each `dev`
  S4 template 3 times), so the data is more regular than real chat, and errors cluster by template.
- **One reviewer.** The Darija items were checked by one native speaker. Darija varies by region and age;
  another speaker could find some phrasings unnatural.
- **Built by the system's authors.** The same people and assistant who wrote Nona wrote the templates and hard cases. Held-out
  name pools and a separate `test` template set limit that bias but don't remove it.
- **Small counts for some types.** IP_ADDRESS (6 / 10 mentions), CARD (10 / 16) and CIN (11 / 24) give rates
  with wide uncertainty. Per-language cells in the report can rest on a handful of mentions.
- **Pessimistic leak check.** A gold name found at word boundaries counts as leaked even when the payload
  holds the same word as a word (05 §6, *Known limits*). Homoglyph spellings aren't covered.
- **No ORGANIZATION, almost no public figures.** Those parts of the metrics are untested here.
- **Text only.** No OCR noise, voice transcripts, attachments or very long documents.
- **`test` was scored once before the freeze** (#74, 2026-11-28), to check that it loads and scores.
  Nothing was tuned on it; the one bug it showed is tracked in #100 and needs a `dev` case first.

## Distribution and maintenance

- **Where:** in this repository, under the [Apache-2.0](../../LICENSE) licence like the code.
- **Integrity:** `bench/data/test.sha256` holds the hash of `test.jsonl`; `shasum -a 256 -c
  bench/data/test.sha256` checks it from the repository root, and `tests/bench/test_generate.py` fails in CI
  if the file and the hash disagree.
- **Changes.** `dev` can grow at any time; the report prints each split's hash, so numbers from different
  versions aren't compared by mistake. `test` changes only to fix a labelling error, in a commit that changes
  `test.sha256`, says why, and invalidates every number published before it. An error can be reported as a
  GitHub issue with the label `area:bench`.
