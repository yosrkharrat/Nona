# Third-party components

Nona's code is Apache-2.0. Every model, dataset and list it uses must have a compatible license (NFR-12).
The PyPI package never contains model weights: `nona models pull` downloads them at the pinned revision. The
release container image has the pinned model baked in ([ADR-0007](docs/adr/0007-release-supply-chain.md)),
so it redistributes the weights under their license, with this file and the model's notices included.

## Models

| Component | Used for | License | Pinned revision |
|---|---|---|---|
| [`urchade/gliner_multi-v2.1`](https://huggingface.co/urchade/gliner_multi-v2.1) | PERSON, LOCATION and ORGANIZATION detection ([ADR-0008](docs/adr/0008-ner-model.md)) | Apache-2.0 | `443d26d654e0324125a96bebd8e796c14ff2efe6` |
| [`microsoft/mdeberta-v3-base`](https://huggingface.co/microsoft/mdeberta-v3-base) | Tokenizer and config of the GLiNER encoder (no weights: GLiNER's file holds them) | MIT | `a0484667b22365f84929a935b5e50a51f71f159d` |

Evaluated and not chosen: [`urchade/gliner_multi_pii-v1`](https://huggingface.co/urchade/gliner_multi_pii-v1) (Apache-2.0).

## Libraries

Runtime dependencies, including the ones the NER model needs directly. Everything else is in `uv.lock`.

| Package | Used for | License |
|---|---|---|
| [`phonenumbers`](https://pypi.org/project/phonenumbers/) | Phone validation and formatting | Apache-2.0 |
| [`gliner`](https://pypi.org/project/gliner/) (optional `ner` extra, from #72) | Running the NER model | Apache-2.0 |
| [`torch`](https://pypi.org/project/torch/) (via `gliner`) | Model runtime | BSD-3-Clause, with bundled parts under Apache-2.0, MIT, BSD-2-Clause, BSL-1.0, ISC and Zlib |
| [`transformers`](https://pypi.org/project/transformers/) (via `gliner`) | Tokenizer and encoder | Apache-2.0 |
| [`sentencepiece`](https://pypi.org/project/sentencepiece/) (via `gliner`) | The mDeBERTa tokenizer | Apache-2.0 |
| [`protobuf`](https://pypi.org/project/protobuf/) (to be declared in the `ner` extra: `gliner` doesn't pull it in) | Reading the mDeBERTa tokenizer with transformers 5 | BSD-3-Clause |

## Data

The gazetteer lists (`src/nona/core/recognizers/gazetteer_data.py`) and the benchmark data (`bench/data`) are written for this project and are synthetic: no scraped or third-party data.
