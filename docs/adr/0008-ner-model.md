# ADR-0008: NER model choice

- Status: **Proposed** (S4 spike, #71). Becomes Accepted when the recognizer (#72) meets the conditions below. Part 2 (quality) addressed conditions 1 and 2; 3 and 4 are part 3.
- Date: 2026-11-23
- Related: [deep dive A](../design/04-deep-dives.md#a-detection-in-mixed-french--arabic--darija--arabizi) (layer 2), [ADR-0004](0004-causal-entity-linking.md), [ADR-0007](0007-release-supply-chain.md), NFR-3, NFR-6, NFR-7, NFR-12

## Context

The gazetteer and context rules (S3) only find names that are in the lists: on `dev`, 75.0% of held-out PERSON mentions leak. A model has to find the rest. The build plan names two candidates, both GLiNER on an mDeBERTa-v3 base (289M parameters, about 1.16 GB of weights):

| | `urchade/gliner_multi-v2.1` | `urchade/gliner_multi_pii-v1` |
|---|---|---|
| Training languages | multilingual | en, fr, de, es, pt, it: **no Arabic** |
| Weights | `model.safetensors` | `pytorch_model.bin` only (pickle) |
| License | Apache-2.0 | Apache-2.0 |
| Pinned revision | `443d26d654e0324125a96bebd8e796c14ff2efe6` | `1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d` |

## Method

[`bench/spike_gliner.py`](../../bench/spike_gliner.py) (and [`spike_gliner_variants.py`](../../bench/spike_gliner_variants.py) for the label and trimming variants under *Conditions*) adds one model as a recognizer next to Nona's recognizers and scores it with the benchmark harness, so these are the numbers `nona bench` prints. Labels `person`, `location`, `organization`; the model reads the **original** text (the normalized view is case-folded, and capitals are its best cue). Each model runs in its own process (16 GB machine). The model is called once per distinct message, and the threshold sweep filters those predictions. `gliner` 0.2.30, torch 2.15.0 and transformers 5.22.2 on CPU (resolved with `--exclude-newer 2026-11-09`: no release younger than two weeks, a rule #72 keeps for the `ner` extra), on an M-series Mac. Another session's render jobs were loading the CPU during the first runs, so latency is given for both conditions.

`dev` is 137 conversations (370 turns). The per-language held-out samples are small (6 to 45 mentions) and LOCATION/ORGANIZATION has 6: a difference of one or two mentions is noise.

## Results

Gazetteer + model, against the current pipeline (no model). PERSON linking rates, over mentions with a placeholder.

| Configuration | Direct leak | Held-out PERSON leak | LOC/ORG leak | Over-masked tokens | Distractors | Wrong split | Ambiguity | Re-sent turns leak |
|---|---|---|---|---|---|---|---|---|
| No model (today) | 25.4% | 75.0% | 83.3% | 0.2% | 8.7% | 0% | 0% | 38.8% |
| **v2.1 @ 0.5** | **2.1%** | **6.2%** | 33.3% | **6.3%** | **14.5%** | 29.5% | 19.4% | 3.6% |
| v2.1 @ 0.4 | 1.4% | 4.2% | 33.3% | 7.7% | 24.6% | 29.5% | 19.4% | 3.6% |
| pii-v1 @ 0.5 | 3.5% | 9.4% | 0.0% | 10.0% | 17.4% | 24.2% | 17.2% | 0.0% |
| pii-v1 @ 0.4 | 2.5% | 6.2% | 0.0% | 11.3% | 21.7% | 24.2% | 17.7% | 0.0% |

No configuration makes a wrong merge. Held-out PERSON by language, at 0.5 (leaked / mentions):

| | fr | en | aeb-Latn | aeb-Latn+fr | aeb-Arab | ar |
|---|---|---|---|---|---|---|
| No model | 30/45 | 12/15 | 6/6 | 12/12 | 9/9 | 3/9 |
| v2.1 | 0/45 | 0/15 | 1/6 | 3/12 | 2/9 | 0/9 |
| pii-v1 | 0/45 | 0/15 | 3/6 | 2/12 | 1/9 | 3/9 |

| | v2.1 | pii-v1 |
|---|---|---|
| Latency per message, p50 / p95 (load average ≈ 3) | 44 / 50 ms | 45 / 52 ms |
| Latency per message, p50 / p95 (render job running, load ≈ 7) | 289 / 550 ms | 378 / 615 ms |
| RSS steady after loading | 1.64 GB | 1.94 GB |
| RSS peak while loading (varies between runs) | 2.29–2.74 GB | 2.16–2.75 GB |

The longest `dev` message is 25 words, so the latencies are for short messages.

The model alone (structured recognizers, no gazetteer) leaks 3.5% (v2.1) and 4.6% (pii-v1) at 0.5: the gazetteer still adds recall.

## Decision

**`urchade/gliner_multi-v2.1`, pinned at `443d26d6`, threshold 0.5, labels `person`, `location`, `organization`, combined with the gazetteer and context rules.**

- **Arabic.** v2.1 leaks 0 of 9 held-out Arabic names; pii-v1 leaks 3 of 9, consistent with its training languages. Arabic and Darija are why Nona exists.
- **Over-masking.** At equal threshold, v2.1 replaces 6.3% of non-PII tokens, pii-v1 10.0%.
- **Supply chain.** v2.1 ships safetensors. pii-v1 ships only a pickle, which can run code when it's loaded. Nona sits in the path of every prompt, so this ADR sets a rule: **model weights are loaded from safetensors only**. #72 enforces it in the loader.
- **0.5 over 0.4.** 0.4 catches two more held-out names (4.2% vs 6.2%) but touches 24.6% of distractors instead of 14.5%. Revisit once #72 has cut over-masking.
- pii-v1 did better on re-sent assistant turns (0% vs 3.6%) and LOCATION/ORGANIZATION (0 of 6 vs 2 of 6). Both are small samples; #72 reports them again.

## Conditions for Accepted (work for #72)

The model brings the leak rate from 25.4% to about 2%, but it can't ship as measured here:

1. **Linking splits rise from 0% to about 30%.** Causes found on `dev`:
   - **Type confusion**, the largest share: the model calls a person a place in one turn and a person in the next (`Bilel Gharbi` → `LOCATION_1`, later `PERSON_1`); many Tunisian surnames are also place words. Linking is per type, so the person splits. Keeping only `person` isn't the fix: splits fall only to 22%, and the names typed as places leak (held-out 10.4%). #72 needs a type rule for name-like spans (link across PERSON/LOCATION, or prefer PERSON on a token set already known as a person).
   - **ADR-0004's two-entity rule**, now reached: `Jaouadi`, then `Mootez`, then `Mootez Jaouadi` gives an ambiguous placeholder, which hindsight would merge. ADR-0004 accepted this as rare; with a model it isn't. If it stays frequent after the type fix, ADR-0004 is reopened, as it says.
   - Arabic bare first names after a full name (`سامي` after `سامي الجلاصي`) sometimes get a new placeholder. Not diagnosed yet.
2. **Over-masking** comes from span edges and Arabizi. The model includes honorifics (`Mme`, `Dr`, `السيدة`; trimming them takes tokens from 6.3% to 5.4%) and tags pronouns (`I`, `Il`, `Je`) and Darija words (`3al`, `nomrou`, `behi`, `rabbi`). The de-Arabizi'd view (deep dive A) hasn't been tried; it may help here or hurt.
3. **NFR-7 (≤ 1.5 GB).** Steady RSS is 1.64 GB in fp32, and loading peaks at 2.3–2.7 GB. Two thirds of the parameters are the multilingual embedding table (192M). In a one-off check (not in the spike scripts), dynamic int8 quantization of the linear layers made RSS larger (2.56 GB: it keeps a copy), and an fp16 embedding breaks inference. The likely path is ONNX with int8 weights (already the v0.2 lever in 03 §5), or trimming the embedding vocabulary to the scripts we need. Loading from safetensors with memory mapping should cut the peak.
4. **NFR-6 (added overhead p95 ≤ 150 ms for a 300-token message).** On short messages (at most 25 words) the model takes 50 ms at p95 on a lightly loaded machine, but 550 ms while a render job loads the CPU. Nothing here measures a 300-token message. #72 measures one, on an idle machine and under load, through the whole pipeline.
5. **Long messages.** GLiNER reads at most `max_len` = 384 words and ignores the rest. Every `dev` message is shorter, so these numbers don't cover long prompts: a name in the signature of a pasted 600-word email would leak. #72 splits long texts into overlapping windows and tests it.

Pinned too: the tokenizer from `microsoft/mdeberta-v3-base` (MIT) at `a0484667b22365f84929a935b5e50a51f71f159d`. GLiNER loads it by name, which offline resolves through the cache's `refs/main`; the spike checks that this is the pinned revision, and #72 loads it from the pinned snapshot path instead. It needs `sentencepiece` and `protobuf`. transformers 5 warns about a "Mistral regex" when loading it; tokenization looks right, and #72 checks it with a test.

## Implementation (#72, part 1)

- [`core/recognizers/ner.py`](../../src/nona/core/recognizers/ner.py): `NerRecognizer` takes any `NerModel` (identity, `max_words`, `predict`), so `nona.core` never imports torch and tests use a fake. It reads the original text, cuts long texts into windows of at most `max_words` model words overlapping by 32 (the model ignores everything past 384 words: a name in a 600-word email's signature leaked without windows, and is found with them), trims leading honorifics from names, and keeps the model's score. Its fingerprint carries the model id and revision.
- [`nona/models.py`](../../src/nona/models.py) and `nona models pull`: the GLiNER and mDeBERTa revisions and every file's SHA-256 are pinned in code. Files are staged and installed only when every hash matches, so a failed download or hash check installs nothing (a disk error while moving verified files into place can leave some updated; pulling again finishes), and staging left by a killed pull is cleaned up. Files are stored exactly as published. Only safetensors are fetched; pickle weights in either folder are refused, by `pull` and on load. Every load re-checks the small files (configs, tokenizer) against their pinned hashes, and the weights by size. Loading is offline: once the libraries import, the Hub and transformers are switched to offline mode for the rest of the process (they read it at import, so it can't be scoped; a later pull must run in another process), and a real-model test blocks sockets while loading. It never writes to the models folder: both folders are rebuilt in a temporary folder holding only the pinned files (small ones written from the bytes just verified, the weights symlinked, or copied where symlinks need privileges), with GLiNER's `model_name` pointing at that encoder copy. So nothing resolves by Hub name, a file dropped next to the pinned ones (a `tokenizer.json`) is never seen, the small files can't change between check and load, and a read-only folder works. The weights are read through the link, checked by size on load and by hash in `pull`. Where symlinks need privileges (Windows, not a supported platform under NFR-8), every load copies the 1.1 GB weights into the temp folder. If importing the libraries fails, the process is put back online.
- Packaging: the optional `ner` extra (`gliner`, `torch`, `protobuf`, `huggingface-hub`), CPU-only torch wheels on Linux (NFR-8), and the dependency age rule as `exclude-newer` in `pyproject.toml`. The audit job covers the extras (the Linux `+cpu` torch build is audited under its public version).
- `nona bench --systems nona-ner` runs it; `pytest -m model` tests the real model (deselected by default). With pip, CPU-only torch is installed from its own index with `--index-url`, not `--extra-index-url`, which would let pip take a package from either index. It is **not** in the default pipeline.

On `dev` (`nona-ner`): direct leak 2.1%, held-out PERSON 6.2%, over-masked tokens 5.4% (honorific trimming, from 6.3%), PERSON wrong split 26.2%, ambiguity 18.6%. A real-model test records the Arabizi case behind conditions 1 and 2: in `3andi rendez-vous m3a Mootez Jaouadi ghodwa` the model calls the name a place at 0.38, and with a final period takes `Mootez Jaouadi ghodwa` as one location at 0.59.

## Implementation (#72, part 2: quality)

Conditions 1 and 2, measured on `dev` with `nona-ner` (same data, model and threshold as part 1):

| | Direct leak | Held-out PERSON | Re-sent turns | Over-masked tokens | Distractors | PERSON wrong split | Ambiguity |
|---|---|---|---|---|---|---|---|
| Part 1 | 2.1% (6/284) | 6.2% (6/96) | 3.6% | 5.4% (148/2754) | 14.5% (10/69) | 26.2% (16/61) | 18.6% (29/156) |
| **Part 2** | **1.4%** (4/284) | **4.2%** (4/96) | 3.6% | **1.0%** (28/2754) | 13.0% (9/69) | **4.9%** (3/61) | **3.8%** (6/156) |

Still no wrong merge (0/95). The default `nona` system (no model) gives byte-identical results.

What did it, in [`ner.py`](../../src/nona/core/recognizers/ner.py) (one message) and [`pipeline.py`](../../src/nona/core/pipeline.py) (the conversation):

- **The de-Arabizi'd view** (condition 2), tried as asked: the model reads the text with each Arabizi digit replaced by one letter (`3andi` → `aandi`), so offsets don't move. On its own it took leaks from 2.1% to 1.8%, over-masked tokens from 5.4% to 5.0% and wrong splits from 26.2% to 21.3%. Capitalizing listed lowercase names as well didn't help. Combining spans from the original and the view (the build plan's wording) caught one more held-out name (leak 1.1%) and changed nothing else, for a second model call on every Arabizi message: left to part 3, which has the latency numbers.
- **Span cleaning** (condition 2): model spans are cut at words that are never names: honorifics, also behind an Arabic clitic (`بالسيدة`), the gazetteer's stopwords and pronouns (`I`, `Il`, `Je`), a list of Darija words in Latin and Arabic script (`NEVER_NAMES` in `gazetteer_data.py`: `nomrou`, `Behi`, `باهي`, `نومرو`), and Arabizi words with digits that aren't listed names (`9alli`, `y3aychek`). A listed name behind Arabic clitics loses them (`وسامي`). This is most of the over-masking drop, and the Arabic splits: `نومرو زكرياء` had made the person absorb "number".
- **Types** (condition 1). The model's LOCATION spans on `dev` were 7 places, 16 persons and 36 Arabizi chunks; its ORGANIZATION spans included 6 persons, which policy `KEEP` would have sent in clear. A LOCATION or ORGANIZATION with a listed first name (not a common word, not a listed city, not starting with or following a place or organization word: `rue`, `Clinique`, `شارع`…) is a PERSON; a LOCATION after a person cue (`m3a`, `with`, a greeting, an honorific) too; a lowercase LOCATION that isn't a listed city is dropped. In the conversation: a model LOCATION that fits a known person, or follows one with `and` between, is a PERSON, and an ORGANIZATION whose tokens all belong to known people too (`Clinique Ahmed Ben Salem` stays an organization). A propagated name beats any other span with the same extent, so a name keeps the placeholder, and the type, it was first given. A model span that is a known form behind clitics (`لزكرياء`) is aligned with it.
- **Cue-backed low scores** (deep dive A, layer 4). In Arabizi the model often scores a name under 0.5 (`Hazem 9alli`: 0.46). From 0.2 up, a capitalized span backed by a person cue before it or a speech verb after it (#50's `9alli`, `قالي`) is kept at the threshold score, and joins the capitalized piece right after it. The real-model test that was a strict xfail in part 1, `3andi rendez-vous m3a Mootez Jaouadi ghodwa`, now passes, with and without the final period.

Left, and why:

- **Splits: 3 of 61, all ADR-0004's two-entity case.** One template, three times: `Dr Zghal`, then `Chirine`, then `Chirine Zghal`, which fits both earlier entities and gets its own placeholder. Under ADR-0004's freeze no linking rule can bring these to zero: both placeholders were sent before the full name. Whether to reopen ADR-0004 (for example, writing the full name as its parts' placeholders, `PERSON_2 PERSON_1`, which adds no new person) is the owner's decision.
- **Leaks: 4 held-out mentions**, all bare first names the model doesn't find even at 0.2: `غيث` twice in Arabic (an unlisted Arabic word is never spread from its full name, by design since #55), a lowercase Arabizi `bacem`, and `Chiheb 9alli` after two Chihebs were named. Re-sent assistant turns leak 5 of 139, the same bare names.
- **Distractors: 9 of 69.** `m3a Ooredoo` (masked before too, now as a PERSON: `m3a` is a person cue), and `رمضان كريم`, `اللهم آمين`, which the gazetteer masks; #50 (Arabic common-word names need context) covers those.
- Words that are also given names stay off `NEVER_NAMES`, as a span is cut at them whatever its score: `Rani` ("I am") and `ken` ("if") are masked when the model takes them (5 of the 28 tokens), while `Wala`, `Bahi` or `يسر` (Yosr) are never cut out of a name.
- `NEVER_NAMES` and the speech verbs are a first list, to be checked by a native speaker as #50's were. Everything here was tuned on `dev`: #75 checks it once on the frozen `test`.

## Consequences

- ➕ The leak rate drops by an order of magnitude on every language, including Arabic script.
- ➕ One model for PERSON, LOCATION and ORGANIZATION, with zero-shot labels.
- ➖ The heaviest dependency so far (torch, transformers), so it becomes an optional extra (`nona-proxy[ner]`), and weights come only through `nona models pull` (NFR-3).
- ➖ Over-masking and linking get worse before #72 fixes them, so the model isn't in the default pipeline until then.
- ➖ The spike tuned on `dev`. The threshold and any filters #72 adds are checked once on the frozen `test` set (#75).
