"""NFR-6 and NFR-7 for the NER model (#72 part 3, ADR-0008 conditions 3 and 4). Not product code.

Memory (NFR-7: RSS ≤ 1.5 GB with the model loaded). One process per run, so peaks don't mix:

    uv run --frozen --extra ner python bench/resources.py memory
    uv run --frozen --extra ner python bench/resources.py memory --touch-all   # upper bound, see below

It loads the model, then runs every `dev` message and the three long messages below through the whole
pipeline, and prints RSS after loading, after the workload, and the peak. The weights stay mapped from the
safetensors file (`nona.models`), so only the pages inference reads are resident: RSS grows with the
vocabulary a workload touches. `--touch-all` reads the whole embedding table once, the bound for a process
that has seen every one of the 250k tokens. The macOS physical footprint (what Activity Monitor shows,
without clean file pages, which the kernel drops under memory pressure) is printed too.

Latency (NFR-6: added overhead p95 ≤ 150 ms for a 300-token message):

    uv run --frozen --extra ner python bench/resources.py latency [--load N] [--runs 50]

Each run anonymizes one long message with a fresh `Nona` (an empty detection cache: a repeated message would
be a cache hit), with and without the model. That is everything Nona adds to a request before it is sent
(normalize, recognizers, resolve, link, rewrite); the proxy's HTTP hop comes in S5. `--load N` starts N busy
processes first, standing in for another job on the machine (ADR-0008 measured 289/550 ms under a render job).
It also times one model call on the Arabizi message's original text: the cost of the second call that
ADR-0008 left to this part.

"300 tokens" are counted with the model's own tokenizer (mDeBERTa); an LLM's tokenizer gives a similar count
on French and fewer on Arabic. Synthetic data only.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

LONG = {
    "fr": (
        "Bonjour Maître Ben Ammar,\n\n"
        "Suite à notre échange téléphonique de mardi, je vous transmets le récapitulatif du dossier de ma "
        "mère, Mme Najet Hammami, née le 14 mars 1958 à Monastir. Elle a signé le compromis de vente de "
        "l'appartement de la rue de Marseille le 3 octobre, en présence de son voisin, M. Habib Sellami, qui "
        "avait servi de témoin. L'acheteur, Karim Bouzid, devait verser l'acompte avant le 15 novembre, mais "
        "rien n'est arrivé sur le compte de ma mère (RIB 08 006 0123456789012 34) à ce jour.\n\n"
        "De notre côté, la Banque de Tunisie nous demande une attestation avant de débloquer le prêt relais, "
        "et le notaire, Me Lotfi Gharbi, attend votre avis pour fixer la date de la signature définitive.\n\n"
        "Pourriez-vous nous dire si nous pouvons annuler le compromis, ou s'il faut d'abord envoyer une mise "
        "en demeure ? Vous pouvez me répondre à l'adresse sami.hammami@example.com ou m'appeler le soir au "
        "22 345 678.\n\n"
        "Bien cordialement,\n"
        "Sami Hammami"
    ),
    "aeb-Latn+fr": (
        "Salut Yassine, ça va ? Bon écoute, 3andi mochkla m3a l'proprio mta3 dar fi La Marsa, Si Mounir "
        "Jlassi, w ma n3rafch chnowa na3mel. L'contrat mta3 l'kra ykammel fi 31 décembre, w houwa 9alli "
        "lbara7 elli lazemni nkhalli dar 9bal 15 décembre khater bech ybi3ha l wa7ed sa7bou, Anis Mejri. "
        "Ena 3andi bail mte3 3 snin, signé 9oddem l'notaire Me Rim Chaouachi, w khallast dima fi wa9t, "
        "par virement 3al compte mte3ou. Omi, Saida Ferchichi, temchi tsakken m3aya mel 3am elli fet, w "
        "ma najmouch nal9aw dar o5ra fi 15 youm, surtout m3a l'prix mta3 tawa.\n\n"
        "Ken t3raf 7ad ynajem "
        "y3awenni, ab3athli nomrou, walla 9olou ykalmni 3al 55 123 987.\n\n"
        "Behi, y3aychek, w sallemli 3la Amira w les enfants."
    ),
    "ar": (
        "السلام عليكم أستاذة سلمى، أرجو أن تكوني بخير. أكتب إليك بخصوص ملف والدي، السيد الهادي الطرابلسي، "
        "المقيم في نهج الحرية بصفاقس. منذ شهر أكتوبر وهو ينتظر رد الصندوق الوطني للتأمين على المرض بخصوص "
        "استرجاع مصاريف العملية التي أجراها في مصحة الياسمين على يد الدكتور منير القاسمي. قدمنا كل الوثائق "
        "المطلوبة: الوصفات، الفواتير، وشهادة الطبيب، وأرسلها أخي كريم بالبريد المضمون يوم 12 أكتوبر.\n\n"
        "اتصلت بالمكتب الجهوي أكثر من خمس مرات على الرقم 74 456 789، وفي كل مرة تقول لي الموظفة، "
        "السيدة نادية بن يوسف، إن الملف في طور الدراسة. والدي عمره 72 سنة ولا يستطيع أن يتنقل بسهولة، "
        "وأمي فاطمة الزهراء تعبت من الانتظار.\n\n"
        "هل يمكنك أن توضحي لنا الخطوات اللازمة، وهل هناك آجال قانونية للرد؟ يمكنك مراسلتي على "
        "hedi.family@example.com أو الاتصال بي على 98 765 432 بعد الساعة الخامسة مساء.\n\n"
        "مع فائق الاحترام،\nياسين الطرابلسي"  # noqa: RUF001 - Arabic text
    ),
}


def _rss_mb() -> float:
    ps = shutil.which("ps") or "ps"
    out = subprocess.run([ps, "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)  # noqa: S603
    return int(out.stdout.strip()) / 1024


def _peak_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 20) if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux


def _footprint_mb() -> float | None:
    if sys.platform != "darwin":
        return None
    vmmap = shutil.which("vmmap") or "vmmap"
    out = subprocess.run([vmmap, "--summary", str(os.getpid())], capture_output=True, text=True).stdout  # noqa: S603
    for line in out.splitlines():
        if line.startswith("Physical footprint:"):
            value = line.split(":", 1)[1].strip()
            unit = {"K": 1 / 1024, "M": 1, "G": 1024}[value[-1]]
            return float(value[:-1]) * unit
    return None


def _messages(dataset: Path) -> list[str]:
    from nona.bench.dataset import load_jsonl

    return [turn.text for conv in load_jsonl(dataset) for turn in conv.turns]


def _load():
    from nona.models import load_ner_model

    return load_ner_model()


def memory(args: argparse.Namespace) -> dict[str, object]:
    from nona.core import Nona
    from nona.core.recognizers import NerRecognizer, default_recognizers

    before = _rss_mb()
    start = time.perf_counter()
    model = _load()
    load_s = time.perf_counter() - start
    gc.collect()
    out: dict[str, object] = {
        "rss_before_load_mb": round(before),
        "load_s": round(load_s, 1),
        "rss_after_load_mb": round(_rss_mb()),
        "peak_after_load_mb": round(_peak_mb()),
    }
    nona = Nona([*default_recognizers(), NerRecognizer(model)])
    texts = _messages(args.dataset) + list(LONG.values())
    for text in texts:
        nona.anonymize(text)
    if args.touch_all:
        import torch

        with torch.no_grad():
            for p in model._model.model.parameters():
                float(p.float().sum())
    gc.collect()
    out |= {
        "messages": len(texts),
        "touch_all": args.touch_all,
        "rss_after_workload_mb": round(_rss_mb()),
        "peak_mb": round(_peak_mb()),
        "footprint_mb": _footprint_mb(),
    }
    return out


def _burn() -> None:
    while True:
        pass


def _percentiles(samples: list[float]) -> dict[str, float]:
    q = statistics.quantiles(samples, n=20, method="inclusive")
    return {"p50_ms": round(statistics.median(samples), 1), "p95_ms": round(q[18], 1)}


def latency(args: argparse.Namespace) -> dict[str, object]:
    from nona.core import Nona
    from nona.core.recognizers import NerRecognizer, default_recognizers
    from nona.core.recognizers.ner import LABELS

    model = _load()
    tokenizer = model._model.data_processor.transformer_tokenizer
    rules = default_recognizers()
    ner = NerRecognizer(model)
    burners = [multiprocessing.Process(target=_burn, daemon=True) for _ in range(args.load)]
    for b in burners:
        b.start()
    try:
        time.sleep(2 if burners else 0)
        out: dict[str, object] = {"load_processes": args.load, "runs": args.runs, "cpus": os.cpu_count()}
        for lang, text in LONG.items():
            row: dict[str, object] = {
                "words": len(text.split()),
                "model_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
            }
            for name, recognizers in (("no_model", rules), ("with_model", [*rules, ner])):
                for _ in range(3):  # warm-up
                    Nona(recognizers).anonymize(text)
                samples = []
                for _ in range(args.runs):
                    nona = Nona(recognizers)
                    start = time.perf_counter()
                    nona.anonymize(text)
                    samples.append((time.perf_counter() - start) * 1000)
                row[name] = _percentiles(samples)
            calls = []
            for _ in range(args.runs):
                start = time.perf_counter()
                model.predict(text, list(LABELS), 0.2)
                calls.append((time.perf_counter() - start) * 1000)
            row["one_model_call"] = _percentiles(calls)
            out[lang] = row
        return out
    finally:
        for b in burners:
            b.terminate()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    mem = sub.add_parser("memory")
    mem.add_argument("--dataset", type=Path, default=Path("bench/data/dev.jsonl"))
    mem.add_argument("--touch-all", action="store_true")
    lat = sub.add_parser("latency")
    lat.add_argument("--runs", type=int, default=50)
    lat.add_argument("--load", type=int, default=0, help="busy processes to start first")
    args = parser.parse_args(argv)
    result = memory(args) if args.command == "memory" else latency(args)
    print(json.dumps({"command": args.command, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
