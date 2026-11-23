# ADR-0008: NER model choice

- Status: **Proposed** (S4 spike, #71). Becomes Accepted when the recognizer (#72) meets the conditions below.
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

[`bench/spike_gliner.py`](../../bench/spike_gliner.py) (and [`spike_gliner_variants.py`](../../bench/spike_gliner_variants.py) for the label and trimming variants under *Conditions*) adds one model as a recognizer next to Nona's recognizers and scores it with the benchmark harness, so these are the numbers `nona bench` prints. Labels `person`, `location`, `organization`; the model reads the **original** text (the normalized view is case-folded, and capitals are its best cue). Each model runs in its own process (16 GB machine). The model is called once per distinct message, and the threshold sweep filters those predictions. `gliner` 0.2.30, torch 2.15.0 and transformers 5.22.2 on CPU (resolved with `--exclude-newer 2026-11-09`, ADR-0007), an M-series Mac with another heavy process running, so latencies are an upper bound.

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
| Latency per message, p50 / p95 | 289 / 550 ms | 378 / 615 ms |
| RSS: peak while loading / steady after | 2.29 GB / 1.64 GB | 2.16 GB / not measured |

The model alone (structured recognizers, no gazetteer) leaks 3.5% (v2.1) and 4.6% (pii-v1) at 0.5: the gazetteer still adds recall.

## Decision

**`urchade/gliner_multi-v2.1`, pinned at `443d26d6`, threshold 0.5, labels `person`, `location`, `organization`, combined with the gazetteer and context rules.**

- **Arabic.** v2.1 leaks 0 of 9 held-out Arabic names; pii-v1 leaks 3 of 9, consistent with its training languages. Arabic and Darija are why Nona exists.
- **Over-masking.** At equal threshold, v2.1 replaces 6.3% of non-PII tokens, pii-v1 10.0%.
- **Supply chain.** v2.1 ships safetensors. pii-v1 ships only a pickle, which runs code on load; ADR-0007 doesn't let that into the path of every prompt.
- **0.5 over 0.4.** 0.4 catches two more held-out names (4.2% vs 6.2%) but touches 24.6% of distractors instead of 14.5%. Revisit once #72 has cut over-masking.
- pii-v1 did better on re-sent assistant turns (0% vs 3.6%) and LOCATION/ORGANIZATION (0 of 6 vs 2 of 6). Both are small samples; #72 reports them again.

## Conditions for Accepted (work for #72)

The model brings the leak rate from 25.4% to about 2%, but it can't ship as measured here:

1. **Linking splits rise from 0% to about 30%.** Causes found on `dev`:
   - **Type confusion**, the largest share: the model calls a person a place in one turn and a person in the next (`Bilel Gharbi` → `LOCATION_1`, later `PERSON_1`); many Tunisian surnames are also place words. Linking is per type, so the person splits. Keeping only `person` isn't the fix: splits fall only to 22%, and the names typed as places leak (held-out 10.4%). #72 needs a type rule for name-like spans (link across PERSON/LOCATION, or prefer PERSON on a token set already known as a person).
   - **ADR-0004's two-entity rule**, now reached: `Jaouadi`, then `Mootez`, then `Mootez Jaouadi` gives an ambiguous placeholder, which hindsight would merge. ADR-0004 accepted this as rare; with a model it isn't. If it stays frequent after the type fix, ADR-0004 is reopened, as it says.
   - Arabic bare first names after a full name (`سامي` after `سامي الجلاصي`) sometimes get a new placeholder. Not diagnosed yet.
2. **Over-masking** comes from span edges and Arabizi. The model includes honorifics (`Mme`, `Dr`, `السيدة`; trimming them takes tokens from 6.3% to 5.4%) and tags pronouns (`I`, `Il`, `Je`) and Darija words (`3al`, `nomrou`, `behi`, `rabbi`). The de-Arabizi'd view (deep dive A) hasn't been tried; it may help here or hurt.
3. **NFR-7 (≤ 1.5 GB).** Steady RSS is 1.64 GB in fp32, and loading peaks at 2.29 GB. Two thirds of the parameters are the multilingual embedding table (192M). Dynamic int8 quantization of the linear layers made RSS larger, and an fp16 embedding breaks inference. The likely path is ONNX with int8 weights (already the v0.2 lever in 03 §5), or trimming the embedding vocabulary to the scripts we need. Loading from safetensors with memory mapping should cut the peak.
4. **NFR-6.** 289 ms p50 is above the 80–120 ms in 03 §5's budget, measured on short messages with another heavy process running. Re-measure on an idle machine; the ONNX path addresses both.

Pinned too: the tokenizer from `microsoft/mdeberta-v3-base` (MIT) at `a0484667b22365f84929a935b5e50a51f71f159d`, which GLiNER loads by name. It needs `sentencepiece` and `protobuf`. transformers 5 warns about a "Mistral regex" when loading it; tokenization looks right, and #72 checks it with a test.

## Consequences

- ➕ The leak rate drops by an order of magnitude on every language, including Arabic script.
- ➕ One model for PERSON, LOCATION and ORGANIZATION, with zero-shot labels.
- ➖ The heaviest dependency so far (torch, transformers), so it becomes an optional extra (`nona-proxy[ner]`), and weights come only through `nona models pull` (NFR-3).
- ➖ Over-masking and linking get worse before #72 fixes them, so the model isn't in the default pipeline until then.
- ➖ The spike tuned on `dev`. The threshold and any filters #72 adds are checked once on the frozen `test` set (#75).
