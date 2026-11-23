"""S4 spike (#71), ADR-0008 conditions: v2.1 with only the `person` label, and with honorifics trimmed.

Same setup as `spike_gliner.py`, no arguments:

    uv run --frozen --exclude-newer 2026-11-09 --with 'gliner==0.2.30' --with protobuf \
        python bench/spike_gliner_variants.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spike_gliner as sg

from nona.bench.runner import run
from nona.core import EntityType, Nona
from nona.core.canonical import canonical_tokens
from nona.core.normalize import NormalizedDoc
from nona.core.recognizers import default_recognizers
from nona.core.types import Span

MODEL = "urchade/gliner_multi-v2.1"
REVISION = "443d26d654e0324125a96bebd8e796c14ff2efe6"
WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Variant:
    model: sg.CachedModel
    threshold: float
    person_only: bool
    trim: bool

    @property
    def name(self) -> str:
        return "gliner"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset(sg.LABELS.values())

    @property
    def fingerprint(self) -> str:
        return f"gliner-variant:{self.threshold}:{self.person_only}:{self.trim}"

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        out = []
        for start, end, label, score in self.model.predict(doc.original):
            if score < self.threshold or (self.person_only and label != "person"):
                continue
            s = start
            if self.trim:  # drop leading words that carry no name token (honorifics, titles)
                for w in WORD.finditer(doc.original[start:end]):
                    if canonical_tokens(w.group()):
                        break
                    s = start + w.end()
                while s < end and doc.original[s].isspace():
                    s += 1
            if s < end:
                out.append(Span(s, end, sg.LABELS[label], max(score, 0.5), "gliner"))
        return out


def main() -> int:
    from gliner import GLiNER

    model = sg.CachedModel(GLiNER.from_pretrained(str(sg.snapshot(MODEL, REVISION)), local_files_only=True))
    f = sg.fmt
    print(f"{'variant':34} direct heldPER  LOC/ORG  overtok distr  merge split ambig  assist")
    for t in (0.4, 0.5):
        for person_only, trim in ((False, False), (True, False), (False, True), (True, True)):
            name = f"@{t} person_only={person_only} trim={trim}"
            nona = Nona([*default_recognizers(), Variant(model, t, person_only, trim)])
            rep = run(sg.SpikeSystem(name, nona), "dev", sg.DATA).report
            p = rep.linking((EntityType.PERSON,))
            print(
                f"{name:34} {f(rep.headline)} {f(rep.held_out)}  {f(rep.location_organization)}  "
                f"{f(rep.masked_tokens)} {f(rep.masked_distractors)}  {f(p.wrong_merge)} {f(p.wrong_split)} "
                f"{f(p.ambiguity)}  {f(rep.assistant)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
