"""Normalizer: the text recognizers read, with an exact map back to the original (03 §2, deep dive A).

Detection runs on the normalized view. Replacement happens on the original text, so every normalized span
must map back to a range of the original. Each original character keeps its own identity; nothing is
merged across a word boundary, so the map stays exact.

What normalization does, per *cluster* (a base character and the combining marks that follow it):
1. NFKC: full-width letters → ASCII, ligatures `ﬁ` → `fi`, composed accents `e◌́` → `é`.
2. Case folding: `Sami` → `sami`, `ß` → `ss`.
3. Eastern Arabic and Persian digits to ASCII: `٩٨` → `98`, so one phone regex covers both scripts.
4. Arabic tatweel and short-vowel diacritics are dropped: `ســامي` → `سامي`, `سَامِي` → `سامي`.

Not done here: the Arabizi digit map (`3` → `ع`). It would turn phone numbers into letters, so it is a
separate view for the name recognizers (S4), combined with this one at detection time.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_TATWEEL = "ـ"
# Arabic harakat, tanwin, shadda, sukun, superscript alef and Quranic marks.
_ARABIC_MARKS = frozenset(
    [chr(c) for c in range(0x0610, 0x061B)]
    + [chr(c) for c in range(0x064B, 0x0660)]
    + ["ٰ"]
    + [chr(c) for c in range(0x06D6, 0x06EE)]
)
_DROP = str.maketrans(dict.fromkeys([_TATWEEL, *_ARABIC_MARKS]))


def _is_mark(c: str) -> bool:
    return unicodedata.category(c).startswith("M")


def _clusters(text: str) -> list[tuple[int, int]]:
    """[start, end) of each base character plus the combining marks attached to it."""
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(text) + 1):
        if i == len(text) or not _is_mark(text[i]):
            out.append((start, i))
            start = i
    return [c for c in out if c[1] > c[0]]


def normalize_cluster(cluster: str) -> str:
    """Normalize one cluster. Exposed so tests can check the whole-text result is built from it."""
    folded = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", cluster).casefold())
    return folded.translate(_DIGITS).translate(_DROP)


@dataclass(frozen=True, slots=True)
class NormalizedDoc:
    """The original text, its normalized view, and where each normalized character came from.

    `origin[i]` is the original range `[start, end)` that produced normalized character `i`. Several
    normalized characters can share one origin (`ﬁ` → `f`, `i`), and an original character can produce
    none (tatweel).
    """

    original: str
    text: str
    origin: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.origin) != len(self.text):
            raise ValueError("origin must have one entry per normalized character")

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Map a normalized range `[start, end)` to the smallest original range that contains it."""
        if not 0 <= start < end <= len(self.text):
            raise ValueError(f"invalid normalized range [{start}, {end}) for length {len(self.text)}")
        return self.origin[start][0], self.origin[end - 1][1]

    def original_text(self, start: int, end: int) -> str:
        """The original substring behind a normalized range."""
        o_start, o_end = self.to_original(start, end)
        return self.original[o_start:o_end]


def normalize(text: str) -> NormalizedDoc:
    parts: list[str] = []
    origin: list[tuple[int, int]] = []
    for start, end in _clusters(text):
        out = normalize_cluster(text[start:end])
        parts.append(out)
        origin.extend([(start, end)] * len(out))
    return NormalizedDoc(text, "".join(parts), tuple(origin))
