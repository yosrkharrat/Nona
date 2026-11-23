"""S4 spike (#71): GLiNER candidates on `dev`, one model per process (16 GB machine, NFR-7).

Not product code. It adds a GLiNER recognizer next to Nona's recognizers and scores every configuration with
the benchmark harness, so the numbers are the ones `nona bench` would print. The model is called once per
distinct message at the lowest threshold; each configuration filters those predictions, so a threshold sweep
costs one pass of the model.

    uv run --frozen --exclude-newer 2026-11-09 --with 'gliner==0.2.30' --with protobuf \
        python bench/spike_gliner.py --model urchade/gliner_multi-v2.1 \
        --revision 443d26d654e0324125a96bebd8e796c14ff2efe6 --out <dir>

Writes `<dir>/<model>.json` (every configuration) and prints a compact table.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nona.bench.metrics import LEAKS, Rate, Report
from nona.bench.runner import run, summary
from nona.core import EntityType, Nona
from nona.core.normalize import NormalizedDoc
from nona.core.recognizers import GazetteerRecognizer, default_recognizers
from nona.core.types import Span

LABELS = {
    "person": EntityType.PERSON,
    "location": EntityType.LOCATION,
    "organization": EntityType.ORGANIZATION,
}
FLOOR = 0.3  # lowest threshold of the sweep: the model is called once at this one
THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
DATA = Path(__file__).resolve().parent / "data" / "dev.jsonl"


# GLiNER loads its encoder's tokenizer and config by name; pinned too (ADR-0008, THIRD_PARTY.md).
TOKENIZER = "microsoft/mdeberta-v3-base"
TOKENIZER_REVISION = "a0484667b22365f84929a935b5e50a51f71f159d"


def peak_rss_mb() -> float:
    """The highest RSS so far (`ru_maxrss`): it never goes down, so after loading it is the load peak."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KiB on Linux


def current_rss_mb() -> float:
    """The resident size now, from `ps` (macOS and Linux): what NFR-7 is about once the model is loaded."""
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])  # noqa: S603, S607
    return int(out) / 1024


def snapshot(model: str, revision: str) -> Path:
    """The pinned snapshot folder. It was pulled with only the weights format we chose, so the Hub client
    calls it incomplete and won't resolve it offline: load the folder directly."""
    from huggingface_hub.constants import HF_HUB_CACHE

    local = Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}" / "snapshots" / revision
    if not (local / "gliner_config.json").is_file():
        raise SystemExit(f"model not pulled: {local}")
    # Offline, the encoder's tokenizer resolves through the cache's `refs/main`: check it is the pinned one.
    ref = Path(HF_HUB_CACHE) / f"models--{TOKENIZER.replace('/', '--')}" / "refs" / "main"
    found = ref.read_text().strip() if ref.is_file() else "missing"
    if found != TOKENIZER_REVISION:
        raise SystemExit(f"{TOKENIZER} resolves to {found}, not the pinned {TOKENIZER_REVISION}")
    return local


class CachedModel:
    """Predictions per distinct text, at the floor threshold, with the time each first call took."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._memo: dict[str, list[tuple[int, int, str, float]]] = {}
        self.latencies_ms: list[float] = []

    def predict(self, text: str) -> list[tuple[int, int, str, float]]:
        hit = self._memo.get(text)
        if hit is None:
            t0 = time.perf_counter()
            found = self.model.predict_entities(text, list(LABELS), threshold=FLOOR, flat_ner=True)
            self.latencies_ms.append((time.perf_counter() - t0) * 1000)
            hit = [(e["start"], e["end"], e["label"], float(e["score"])) for e in found]
            self._memo[text] = hit
        return hit

    @property
    def texts(self) -> list[str]:
        return list(self._memo)


@dataclass(frozen=True)
class GlinerRecognizer:
    model: CachedModel
    threshold: float
    tag: str

    @property
    def name(self) -> str:
        return "gliner"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset(LABELS.values())

    @property
    def fingerprint(self) -> str:
        return f"gliner:{self.tag}:{self.threshold}"

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        # The original text: the normalized view is case-folded, and capitals are the model's best cue.
        # The resolver drops spans under 0.5, so a kept prediction scores at least that; look-ahead seeds
        # (>= 0.8) still need a confident model score.
        return [
            Span(s, e, LABELS[label], max(score, 0.5), "gliner")
            for s, e, label, score in self.model.predict(doc.original)
            if score >= self.threshold and e > s
        ]


class SpikeSystem:
    def __init__(self, name: str, nona: Nona) -> None:
        self._name, self._nona = name, nona

    @property
    def name(self) -> str:
        return self._name

    def payloads(self, conversation: Any) -> list[str]:
        return list(self._nona.anonymize_messages([t.text for t in conversation.turns]).texts)


def held_out_by_language(report: Report) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for m in report.mentions:
        if m.type is EntityType.PERSON and not m.public and not m.listed and m.role != "assistant":
            cell = out.setdefault(m.language, [0, 0])
            cell[0] += m.outcome in LEAKS
            cell[1] += 1
    return out


def fmt(rate: Rate) -> str:
    return "  n/a" if rate.value is None else f"{rate.value:5.1%}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from gliner import GLiNER

    local = snapshot(args.model, args.revision)
    before = current_rss_mb()
    t0 = time.perf_counter()
    model = CachedModel(GLiNER.from_pretrained(str(local), local_files_only=True))
    load_s = time.perf_counter() - t0
    gc.collect()
    steady = current_rss_mb()
    load_peak = peak_rss_mb()
    tag = args.model.rsplit("/", 1)[-1]

    structured = [r for r in default_recognizers() if not isinstance(r, GazetteerRecognizer)]
    configs: dict[str, Nona] = {"baseline (gazetteer, no model)": Nona()}
    for t in THRESHOLDS:
        configs[f"gazetteer + {tag} @{t}"] = Nona([*default_recognizers(), GlinerRecognizer(model, t, tag)])
        configs[f"{tag} alone @{t}"] = Nona([*structured, GlinerRecognizer(model, t, tag)])

    results: dict[str, Any] = {}
    print(f"{'configuration':38} direct heldPER  LOC/ORG  overtok distr  merge split ambig  assist")
    for name, nona in configs.items():
        r = run(SpikeSystem(name, nona), "dev", DATA)
        rep = r.report
        person = rep.linking((EntityType.PERSON,))
        results[name] = {**summary(r), "held_out_by_language": held_out_by_language(rep)}
        print(
            f"{name:38} {fmt(rep.headline)} {fmt(rep.held_out)}  {fmt(rep.location_organization)}  "
            f"{fmt(rep.masked_tokens)} {fmt(rep.masked_distractors)}  {fmt(person.wrong_merge)} "
            f"{fmt(person.wrong_split)} {fmt(person.ambiguity)}  {fmt(rep.assistant)}"
        )

    lat = sorted(model.latencies_ms)
    perf = {
        "model": args.model,
        "revision": args.revision,
        "load_seconds": round(load_s, 1),
        "rss_mb_before_load": round(before),
        "rss_mb_peak_while_loading": round(load_peak),
        "rss_mb_steady_after_load": round(steady),
        "rss_mb_after_run": round(current_rss_mb()),
        "messages": len(lat),
        # GLiNER truncates past `max_len` words (384): these numbers only cover messages shorter than that.
        "longest_message_words": max(len(t.split()) for t in model.texts),
        "max_len": getattr(getattr(model.model, "config", None), "max_len", None),
        "latency_ms_p50": round(statistics.median(lat), 1),
        "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1),
    }
    print(json.dumps(perf, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{tag}.json").write_text(json.dumps({"perf": perf, "configs": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
