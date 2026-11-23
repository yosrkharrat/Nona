"""NER recognizer: PERSON, LOCATION and ORGANIZATION from a GLiNER model (deep dive A layer 2, ADR-0008).

The model is **injected**: anything with the `NerModel` shape. `nona.models` loads the pinned GLiNER weights
and adapts them; tests pass a fake. So `nona.core` stays free of torch and file I/O, and loads without the
`ner` extra.

What this layer adds on top of the model:

- **The original text**, not the normalized view: normalizing folds case, and capitals are the model's best
  cue. Offsets are then original offsets already.
- **Windows.** A model reads a bounded number of words (GLiNER: `max_len`, 384) and silently ignores the
  rest, so a name in the signature of a long email would leak. Long texts are cut into windows of at most
  `max_words` words that overlap by `overlap_words`, so a name cut by one window is whole in the next. Words
  are counted the way GLiNER splits them (a run of word characters, or any other single character), so a
  window never exceeds what the model reads.
- **Leading honorifics are trimmed** from PERSON spans (`Mme Islem` → `Islem`): the model often includes
  them, and they identify no one (ADR-0008).
- **A threshold** (ADR-0008: 0.5). A span keeps the model's score, so the resolver's `min_score` still
  applies, and look-ahead only spreads words from confident spans (0.8, ADR-0004).

A model error propagates; the pipeline turns it into `DetectorError` (fail closed, ADR-0003).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol

from nona.core.canonical import is_honorific
from nona.core.normalize import NormalizedDoc
from nona.core.types import EntityType, Span

# The labels asked of the model, and the type each one gives (ADR-0008).
LABELS: Mapping[str, EntityType] = {
    "person": EntityType.PERSON,
    "location": EntityType.LOCATION,
    "organization": EntityType.ORGANIZATION,
}

# GLiNER's whitespace splitter: a word is a run of word characters (joined by `-` or `_`), or one other
# non-space character. Windows count words this way so they fit what the model reads.
_MODEL_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")
_WORD = re.compile(r"\S+")


class NerPrediction(NamedTuple):
    start: int  # character offsets in the text given to `predict`
    end: int
    label: str
    score: float


class NerModel(Protocol):
    @property
    def identity(self) -> str:
        """Model id and revision: part of the detection-cache key, so a new model never reuses spans."""
        ...

    @property
    def max_words(self) -> int:
        """How many words (counted like `_MODEL_WORD`) the model reads in one call."""
        ...

    def predict(self, text: str, labels: Sequence[str], threshold: float) -> Sequence[NerPrediction]: ...


class NerRecognizer:
    """Spans from an NER model, windowed over long texts (module docstring)."""

    def __init__(self, model: NerModel, *, threshold: float = 0.5, overlap_words: int = 32) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if not 0 <= overlap_words < model.max_words // 2:
            raise ValueError("overlap_words must be at least 0 and under half the model's max_words")
        self._model = model
        self._threshold = threshold
        self._overlap = overlap_words

    @property
    def name(self) -> str:
        return "ner"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset(LABELS.values())

    @property
    def fingerprint(self) -> str:
        labels = ",".join(f"{k}={v.value}" for k, v in LABELS.items())
        return (
            f"model={self._model.identity};threshold={self._threshold};labels={labels};"
            f"max_words={self._model.max_words};overlap={self._overlap}"
        )

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        text = doc.original
        best: dict[tuple[int, int, EntityType], float] = {}
        for w_start, w_end in windows(text, self._model.max_words, self._overlap):
            for p in self._model.predict(text[w_start:w_end], list(LABELS), self._threshold):
                etype = LABELS.get(p.label)
                if etype is None or p.score < self._threshold or not 0 <= p.start < p.end <= w_end - w_start:
                    continue
                start, end = w_start + p.start, w_start + p.end
                if etype is EntityType.PERSON:
                    start = _skip_honorifics(text, start, end)
                if start >= end:
                    continue
                key = (start, end, etype)
                best[key] = max(best.get(key, 0.0), min(p.score, 1.0))
        return [Span(s, e, t, score, self.name) for (s, e, t), score in sorted(best.items())]


def windows(text: str, max_words: int, overlap: int) -> list[tuple[int, int]]:
    """Character ranges of at most `max_words` model words, overlapping by `overlap` words, covering `text`.

    Cut between words, never inside one. A text that fits is one window.
    """
    if max_words < 1 or not 0 <= overlap < max_words:
        raise ValueError("need max_words >= 1 and 0 <= overlap < max_words")
    words = [m.span() for m in _MODEL_WORD.finditer(text)]
    if len(words) <= max_words:
        return [(0, len(text))] if words else []
    out: list[tuple[int, int]] = []
    step = max_words - overlap
    first = 0
    while True:
        last = min(first + max_words, len(words)) - 1
        out.append((words[first][0], words[last][1]))
        if last == len(words) - 1:
            return out
        first += step


def _skip_honorifics(text: str, start: int, end: int) -> int:
    """Move `start` past leading honorifics (`Mme`, `Dr.`) and the spaces after them."""
    for m in _WORD.finditer(text, start, end):
        if not is_honorific(m.group()):
            return m.start()
    return end
