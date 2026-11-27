# ADR-0008: NER model choice

- Status: **Accepted** (2026-11-27, #72). Proposed after the S4 spike (#71), with the conditions below. Part 2 (quality) addressed conditions 1 and 2. Part 3 measured conditions 3 and 4, and the owner amended NFR-6 and NFR-7 (see *Decision on NFR-6 and NFR-7*).
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
- **A fix from the security review of PR #79: an Arabizi word is cut, never unmasked.** The first version dropped every unlisted Arabizi word it cut at. Names are written with digits too, so a confident `Mo3ez`, `Mo7sen`, `Na3ima Ka3bi`, `Sa3ida Bel7aj` or `3li` was sent in clear where part 1 masked it, and so was `Mo7sen` after a known `Mohsen Trabelsi`. The real model does return these as PERSON (`Mo3ez` 0.83, `Sa3ida Bel7aj` 0.97). The cut stays, since the type, place and cue rules work on its pieces, but each Arabizi word cut out of a span is masked again, typed like the model's span. From a low-score span, only a word that looks like a name (a capital or a leading digit) and that a cue backs is masked again (`Mo7sen 9alli`, `m3a Na3ima Ka3bi` at 0.29–0.31). Only the closed lists ever unmask a word: honorifics, stopwords, `NEVER_NAMES`. Removing the open rule outright was tried first. It made spans coarser, and the rules built on the pieces then leaked other names (`Sami ya5dem Sfax` kept as an organization, `sa7a Si Mootaz`, `Mootaz ya5dem Sfax` dropped whole). A property test runs the whole pipeline: every word of a confident PERSON span stays masked unless a closed list says otherwise. During the fix, a differential fuzz against the version before it (random texts over a fixed vocabulary of names, Darija words, cities and cues, with random predictions) checked that no word it masked is now in clear. Each case it found was fixed. The fuzz is not part of the test suite.
- **The price** is over-masking and placeholders split per word: a Darija word the model takes into a name becomes its own placeholder unless it is in `NEVER_NAMES` (`Sami y7ebha Ali` → `PERSON_1 PERSON_2 PERSON_3`), and that placeholder then masks the word in later messages too, and a confident `Sa3ida Bel7aj` takes two placeholders. The words are masked one by one, and never joined to another span, because a longer span could partly overlap another recognizer's name, and the resolver keeps only the longer span (since #89 it keeps the rest of a name span as its own pieces, but a split name is still worse than one placeholder). A word masked again from a span that holds a person is a person. On `dev` the re-masking first added 13 over-masked tokens (`y5allik`, `ab3athli`, `3malt`, `3al`); adding those Darija words to `NEVER_NAMES` (with `m3a`, `3andi`, `3aslema`, `sbe7`, `3asr`, the speech verbs, and a few common nouns) brought it back to 28. Leaks, splits and merges on `dev` are unchanged.
- **Types** (condition 1). The model's LOCATION spans on `dev` were 7 places, 16 persons and 36 Arabizi chunks; its ORGANIZATION spans included 6 persons, which policy `KEEP` would have sent in clear. A LOCATION or ORGANIZATION with a listed first name (not a common word, not a listed city, not starting with or following a place or organization word: `rue`, `Clinique`, `شارع`…) is a PERSON; a LOCATION after a person cue (`m3a`, `with`, a greeting, an honorific) too; a lowercase LOCATION that isn't a listed city is dropped. In the conversation: a model LOCATION that fits a known person, or follows one with `and` between, is a PERSON, and an ORGANIZATION whose tokens all belong to known people too (`Clinique Ahmed Ben Salem` stays an organization). A propagated name beats any other span with the same extent, so a name keeps the placeholder, and the type, it was first given. A model span that is a known form behind clitics (`لزكرياء`) is aligned with it.
- **Cue-backed low scores** (deep dive A, layer 4). In Arabizi the model often scores a name under 0.5 (`Hazem 9alli`: 0.46). From 0.2 up, a capitalized span backed by a person cue before it or a speech verb after it (#50's `9alli`, `قالي`) is kept at the threshold score, and joins the capitalized piece right after it. The real-model test that was a strict xfail in part 1, `3andi rendez-vous m3a Mootez Jaouadi ghodwa`, now passes, with and without the final period.

Left, and why:

- **Splits: 3 of 61, all ADR-0004's two-entity case.** One template, three times: `Dr Zghal`, then `Chirine`, then `Chirine Zghal`, which fits both earlier entities and gets its own placeholder. Under ADR-0004's freeze no linking rule can bring these to zero: both placeholders were sent before the full name. Whether to reopen ADR-0004 (for example, writing the full name as its parts' placeholders, `PERSON_2 PERSON_1`, which adds no new person) is the owner's decision.
- **Leaks: 4 held-out mentions**, all bare first names the model doesn't find even at 0.2: `غيث` twice in Arabic (an unlisted Arabic word is never spread from its full name, by design since #55), a lowercase Arabizi `bacem`, and `Chiheb 9alli` after two Chihebs were named. Re-sent assistant turns leak 5 of 139, the same bare names.
- **Distractors: 9 of 69.** `m3a Ooredoo` (masked before too, now as a PERSON: `m3a` is a person cue), and `رمضان كريم`, `اللهم آمين`, which the gazetteer masks; #50 (Arabic common-word names need context) covers those.
- Words that are also given names stay off `NEVER_NAMES`, as a span is cut at them whatever its score: `Rani` ("I am") and `ken` ("if") are masked when the model takes them (5 of the 28 tokens), while `Wala`, `Bahi` or `يسر` (Yosr) are never cut out of a name.
- **Lowercase name parts next to a found name can leak from a low-score span** (#83): a low span must look like a name, so `m3a Mootez ben salah` masks only `Mootez`. As before #79.
- `NEVER_NAMES` and the speech verbs are a first list, to be checked by a native speaker as #50's were. A Darija word missing from it is over-masked, never leaked. Everything here was tuned on `dev`: #75 checks it once on the frozen `test`.

## Implementation (#72, part 3: resources)

Conditions 3 and 4, measured with [`bench/resources.py`](../../bench/resources.py) on the M-series Mac of the spike (10 cores, torch on 4 threads), one model process at a time. GB are 10⁹ bytes. Three synthetic messages of about 300 tokens, counted with the model's own tokenizer: French (299), Arabizi and French (290), Arabic (304).

**NFR-7 (≤ 1.5 GB) is met**: 1.22–1.33 GB of RSS on the measured workload, and a 0.50–0.51 GB footprint even with every weight page read, which is what NFR-7 counts since #72 (see the caveat below). The loader read the 1.16 GB of fp32 weights into memory after building a randomly initialized model of the same size: loading peaked at 2.9 GB, and RSS stayed at 1.97 GB after `dev`, over the limit. It now asks GLiNER to build the model on torch's `meta` device and put the safetensors file's tensors in place (`low_cpu_mem_usage=True`): the weights stay **mapped from the file**, and only the pages inference reads are resident. Two thirds of the parameters are the 250k-row multilingual embedding table, and a workload only reads the rows of the tokens it sees. `dev` results are byte-identical.

| | Load | RSS after loading | Peak RSS | RSS after `dev` + the 3 long messages | Physical footprint |
|---|---|---|---|---|---|
| Part 2 (weights read into memory) | 8–17 s | 1.73 GB | 2.89 GB | 1.97 GB | 1.61 GB |
| **Part 3 (weights mapped)**, 4 runs | **2 s** | **0.60 GB** | **1.22–1.33 GB** | **1.22–1.33 GB** | **0.50 GB** |
| Part 3, every weight page read (upper bound) | | | 1.97–2.07 GB | 1.97–2.07 GB | 0.51 GB |

**The caveat:** RSS grows with the vocabulary a process has seen, up to the upper bound, which reads the whole embedding table, as a process that had seen tokens of every script would. Everything above the 0.5 GB footprint is **clean file pages**: shared with the page cache, never swapped, dropped by the kernel under memory pressure and read again from disk if needed. NFR-7 was written in RSS, which counts them, so it held only for the workload measured; the owner amended it to count what the kernel can't reclaim (the footprint).

- A `model` test loads the model in a fresh process, anonymizes long messages in four languages, and checks the peak stays ≤ 1.5 GB (1.19 GB; 2.88 GB with the old loader) and that GLiNER didn't fall back to reading the weights, which it does with only a warning.
- Only a symlinked weights file is mapped. Where the loader has to copy it into its temporary folder (no symlinks), it reads it as before, since that folder is deleted after loading.
- **The mapped file is read for the life of the process**, so it must not change: written in place, the process would run weights nobody checked; truncated, or replaced over a network filesystem, the next page read would crash it. `pull` replaces files and never writes into them, but the loader doesn't rely on that: it records the file's device, inode, size and mtime and checks them again before every model call (one `stat`). Any change, or a missing file, fails closed (`ModelError`, so `DetectorError`) on that call and every later one, until the process restarts, also after a `touch` that left the bytes alone or a network mount that dropped for a moment (nothing is hashed: a new mtime counts as a write); that includes a file `pull` replaced, which would be safe on a local disk, but the process should restart to use the new file anyway. A `stat` that fails otherwise (a full file table, a network error) fails only that call. ctime isn't compared: `chmod`, an xattr or a backup tool's hard link move it without touching the bytes. Not caught: a write that keeps the size and sets mtime back, which takes write access to the models folder, the same access that could swap same-size weights before the next start (loading checks their size, `pull` their hash); and a file truncated during a call (about 200 ms), which can still crash the process.

**NFR-6 as first written (added overhead p95 ≤ 150 ms for 300 tokens) is not met with the model.** Everything Nona adds before a request is sent (normalize, recognizers, resolve, link, rewrite), with a fresh detection cache per run, p50 / p95 over 50 runs:

| Message | Rules only, quiet | With the model, quiet (load ≈ 3) | Rules only, 6 busy processes | With the model, 6 busy processes (load ≈ 10) |
|---|---|---|---|---|
| French, 299 tokens | 3.5 / 4.1 ms | 198 / **214 ms** | 4.1 / 8.3 ms | 687 / **772 ms** |
| Arabizi + French, 290 | 2.4 / 2.5 ms | 196 / **230 ms** | 3.0 / 6.0 ms | 615 / **816 ms** |
| Arabic, 304 | 3.2 / 3.3 ms | 217 / **245 ms** | 3.9 / 7.5 ms | 628 / **837 ms** |

The model is nearly all of it: one call takes 195–207 ms at p50, in the encoder's matrix products (addmm and attention bmm, 62% of CPU time). It is compute-bound: 4 threads is the fastest setting (1: 393 ms, 2: 322 ms, 4 to 8: 281–284 ms on a 410-token draft). 03's budget assumed 80–120 ms for this step.

What was tried, on the same messages and `dev`:

| Option | One model call, p50 / p95 | RSS | `dev` | Verdict |
|---|---|---|---|---|
| torch fp32, mapped (shipped) | 195–207 / 215–227 ms | 1.22–1.33 GB | as part 2 | NFR-7 ✓ (caveat), NFR-6 ✗ |
| torch bf16 (`dtype`, cast on read) | 1586 ms (French) | 1.17 GB, peak 2.31 GB | scores lower | 5.7× slower on this CPU |
| torch dynamic int8 (`quantize`, qnnpack) | 460 ms (French) | 1.97 GB, peak 2.95 GB | no entity at all | broken |
| ONNX Runtime int8 (gliner's export) | 121–130 / 141–154 ms | 1.63 GB | direct leak 1.4% → 21.8%, held-out PERSON 4.2% → 64.6% | broken |
| ONNX Runtime fp32 (gliner's export) | 117–137 / 146–152 ms | 2.43–2.59 GB | byte-identical | NFR-6 at the limit, NFR-7 ✗ |

The ONNX runs were at load ≈ 5. ONNX Runtime keeps its own copy of the weights (disabling prepacking and the memory arena didn't change that, and made it slower), exporting peaks at 4.8 GB, and it adds `onnxruntime` and a derived 1.1 GB file to the supply chain (ADR-0007). Under load, none of these comes near 150 ms.

**No second model call on the original text** (left open by part 2): it would add one more model call, about 200 ms, on every Arabizi message, for one more held-out name on `dev`.

### Decision on NFR-6 and NFR-7 (owner, 2026-11-27)

- **NFR-6 is amended for the model path: ≤ 300 ms at p95 for 300 tokens, on a machine with no other heavy job; ≤ 150 ms without the model, as before.** The shipped torch fp32 path is at 214–245 ms. The other options cost more than they give: ONNX Runtime fp32 is at the old limit only on a quiet machine (146–152 ms), breaks NFR-7 (2.4–2.6 GB), and adds `onnxruntime`, an export step and a derived 1.1 GB file; a smaller encoder reopens this ADR with a new spike, and risks the Darija, Arabizi and Arabic recall the model is here for (held-out PERSON 75% → 4.2%). About 250 ms in front of an LLM call that takes seconds is the price of that recall. Under a heavy CPU job the overhead reaches 770–840 ms; NFR-6 doesn't cover that case, and the README will say so (NFR-11).
- **NFR-7 counts the memory the OS can't reclaim: the physical footprint on macOS, 0.50–0.51 GB.** NFR-7 is there so Nona runs beside an IDE on a 16 GB laptop. Mapped weights are clean file pages: the kernel drops them under memory pressure and reads them again from disk, so they never push the machine into swap. In RSS, NFR-7 held only for the workload measured (up to 2.07 GB if every weight page is read). RSS is still reported next to the footprint, and the `model` test still checks that peak RSS stays ≤ 1.5 GB on its fixed workload: stricter than NFR-7, it fails if the weights stop being mapped.

## Consequences

- ➕ The leak rate drops by an order of magnitude on every language, including Arabic script.
- ➕ One model for PERSON, LOCATION and ORGANIZATION, with zero-shot labels.
- ➖ The heaviest dependency so far (torch, transformers), so it becomes an optional extra (`nona-proxy[ner]`), and weights come only through `nona models pull` (NFR-3).
- ➖ Over-masking and linking get worse before #72 fixes them, so the model isn't in the default pipeline until then.
- ➖ The spike tuned on `dev`. The threshold and any filters #72 adds are checked once on the frozen `test` set (#75).
