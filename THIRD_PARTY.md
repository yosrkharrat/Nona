# Third-party components

Nona's code is Apache-2.0. Every model, dataset and list it uses must have a compatible license (NFR-12).
Models are never bundled: `nona models pull` downloads them at the pinned revision.

## Models

| Component | Used for | License | Pinned revision |
|---|---|---|---|
| [`urchade/gliner_multi-v2.1`](https://huggingface.co/urchade/gliner_multi-v2.1) | PERSON, LOCATION and ORGANIZATION detection ([ADR-0008](docs/adr/0008-ner-model.md)) | Apache-2.0 | `443d26d654e0324125a96bebd8e796c14ff2efe6` |
| [`microsoft/mdeberta-v3-base`](https://huggingface.co/microsoft/mdeberta-v3-base) | Tokenizer and config of the GLiNER encoder (no weights: GLiNER's file holds them) | MIT | `a0484667b22365f84929a935b5e50a51f71f159d` |

Evaluated and not chosen: [`urchade/gliner_multi_pii-v1`](https://huggingface.co/urchade/gliner_multi_pii-v1) (Apache-2.0).

## Libraries

Direct runtime dependencies. Their own dependencies are in `uv.lock`.

| Package | Used for | License |
|---|---|---|
| [`phonenumbers`](https://pypi.org/project/phonenumbers/) | Phone validation and formatting | Apache-2.0 |
| [`gliner`](https://pypi.org/project/gliner/) (optional `ner` extra, from #72) | Running the NER model | Apache-2.0 |

## Data

The gazetteer lists (`src/nona/core/recognizers/gazetteer_data.py`) and the benchmark data (`bench/data`) are written for this project and are synthetic: no scraped or third-party data.
