"""Names and places from seed lists and context cues, before any ML model (deep dive A, layers 3 and 4).

**Layer 3, gazetteer.** Each word is reduced to its canonical token (`core/canonical.py`), so one list entry
covers its spellings: `Mohamed`/`Mohammed`/`Med`, Arabizi `3ali`, Arabic `محمد`. Names from the lists
(`gazetteer_data.py`) start a *name run*: the following words joined by a space or a hyphen, as long as they
are names too, a particle with a name after it (`Ben Salem`), or a capitalized word (`Ahmed Karoui`).

**Layer 4, context cues.** A cue right before a capitalized word starts a run even when the word isn't in the
lists, and lifts the score of a run that is:
- *strong*: honorifics and kinship words (`M.`, `Mme`, `Dr`, `sidi`, `khalti`, `3amm`, `السيد`) and
  name phrases (`je m'appelle`, `my name is`, `esmou`, `m3a`);
- *weak*: words that are honorifics in one language and ordinary words in another (`si` is "if" in French,
  `me` in English), and greetings (`Bonjour`, `Merci`, `Hi`, `3aslema`).
The cue itself is never part of the span.

**Scores** (the resolver keeps spans at 0.5 and above by default):

| Run | Score |
|---|---|
| after a strong cue | 0.95 |
| capitalized, two words or more, with a list name | 0.9 |
| capitalized single list name | 0.85 |
| after a weak cue, capitalized | 0.8 |
| Arabic script, list name | 0.8 |
| lowercase, two words or more, with a list name | 0.7 |
| lowercase list first name (Arabizi writes names lowercase) | 0.6 |
| common-word name (`Amal`, `Nour`), capitalized mid-sentence | 0.6 |
| common-word name at the start of a sentence (masked at the default threshold: a leak costs more) | 0.5 |
| lowercase surname alone, lowercase common-word name, Arabic common-word name | 0.3 |

**Arabic clitics.** A word that misses the lists is tried again without a leading `و ف ب ل ك`:
`لسامي` → `سامي`, and the span covers the name only.

**Places.** Cities from the lists are LOCATION spans (0.8) when they start with a capital letter (Latin
script). Cities that are also ordinary words (`Nice`) aren't matched at the start of a sentence. A person run
never takes a word that is part of a city, so `Sidi Bouzid` stays a place.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from nona.core.canonical import HONORIFICS, PARTICLES, canonical_tokens
from nona.core.normalize import NormalizedDoc, normalize
from nona.core.recognizers import gazetteer_data as data
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span

MAX_RUN = 4  # words in one name


def _keys(names: tuple[str, ...]) -> frozenset[str]:
    return frozenset(k for name in names for k in canonical_tokens(name))


FIRST = _keys(data.FIRST_NAMES_TN + data.FIRST_NAMES_FR_EN) | _keys(
    tuple(n for n in data.NAMES_AR if " " not in n and not n.startswith("ال"))
)
LAST = _keys(data.SURNAMES_TN) | _keys(tuple(n for n in data.NAMES_AR if " " in n or n.startswith("ال")))
COMMON = _keys(data.COMMON_WORDS)
NAMES = FIRST | LAST

# Honorifics that are also ordinary words in another language: weak cues only.
WEAK_HONORIFICS = frozenset({"si", "me", "m", "pr", "sir", "miss"})
STRONG_HONORIFICS = HONORIFICS - WEAK_HONORIFICS
# Matched on the normalized text that ends where the run starts.
_STRONG_PHRASE = re.compile(
    r"(?<!\w)(?:je m[\'\u2019]appelle|m[\'\u2019]appelle|my name is|name is"
    r"|esmou|ismou|ism-ou|esmi|ismi|esm|m3a|ena|اسمي|اسمه)\s*[:,]?\s*$"
)
_WEAK_PHRASE = re.compile(
    r"(?<!\w)(?:bonjour|bonsoir|salut|merci|cher|chère|hi|hello|dear|thanks|thank you|3aslema|aslema|ahla"
    r"|ya|مرحبا|شكرا)\s*[:,!]?\s*$"
)
# Capitalized words that start sentences or name things, but never people.
STOPWORDS = frozenset(
    {
        "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "le", "la", "les", "un", "une", "des",
        "ce", "cette", "mon", "ma", "mes", "son", "sa", "ses", "et", "ou", "mais", "donc", "pour", "avec",
        "i", "the", "a", "an", "it", "he", "she", "we", "they", "you", "my", "your", "his", "her",
        "and", "or",
        "but", "please", "ok", "oui", "non", "yes", "no", "bonjour", "merci", "hello", "hi", "thanks",
        "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday", "janvier", "février", "mars", "avril", "mai",
        "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre", "january", "february",
        "march", "april", "june", "july", "august", "september", "october", "november", "december",
        "sms", "cin", "rib", "iban", "tél", "tel", "fax", "dt", "tnd", "email", "mail", "pdf", "id",
    }
)  # fmt: skip
_CLITICS = "وفبلك"
_GAP = re.compile(r"[ \t]+|[ \t]*-[ \t]*")
_HONORIFIC_GAP = re.compile(r"\.?[ \t]+|\.")  # `M. Ben Salem`, `Dr Gharbi`, `Mme.Ksouri`
_WORD = re.compile(r"[^\W_]+")
_SENTENCE_END = frozenset('.!?:\n"«“')


@functools.lru_cache(maxsize=8192)
def _key(word: str) -> str | None:
    tokens = canonical_tokens(word)
    return next(iter(tokens)) if len(tokens) == 1 else None


@dataclass(frozen=True, slots=True)
class _Word:
    start: int  # normalized offsets
    end: int
    text: str  # normalized
    key: str | None
    cap: bool | None  # None for scripts without case
    initial: bool  # first word of a sentence
    clitic: int = 0  # leading Arabic clitic letters to leave out of the span

    @property
    def name(self) -> bool:
        return self.key in NAMES

    @property
    def honorific(self) -> bool:
        return self.text in HONORIFICS


def _words(doc: NormalizedDoc) -> list[_Word]:
    out: list[_Word] = []
    for m in _WORD.finditer(doc.text):
        text = m.group()
        if text.isdigit():
            continue
        o_start, o_end = doc.to_original(m.start(), m.end())
        first = next((c for c in doc.original[o_start:o_end] if c.isalpha()), "")  # `3ali` → `a`
        cap = first.isupper() if first.lower() != first.upper() else None
        before = doc.original[:o_start].rstrip(" \t")
        initial = not before or before[-1] in _SENTENCE_END
        key = _key(doc.original[o_start:o_end])
        clitic = 0
        if key not in NAMES and cap is None and len(text) > 2 and text[0] in _CLITICS:
            stripped = _key(text[1:])
            if stripped in NAMES:
                key, clitic = stripped, 1
        out.append(_Word(m.start(), m.end(), text, key, cap, initial, clitic))
    return out


class GazetteerRecognizer:
    @property
    def name(self) -> str:
        return "gazetteer"

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset({EntityType.PERSON, EntityType.LOCATION})

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        places = _cities(doc)
        taken = [(s, e) for s, e, _ in places]
        words = [w for w in _words(doc) if not any(s < w.end and w.start < e for s, e in taken)]
        spans = [to_span(doc, s, e, EntityType.LOCATION, score, self.name) for s, e, score in places]
        i = 0
        while i < len(words):
            run, cue = self._run(doc, words, i)
            if not run:
                i += 1
                continue
            score = _score(run, cue)
            start = run[0].start + run[0].clitic
            spans.append(to_span(doc, start, run[-1].end, EntityType.PERSON, score, self.name))
            i = words.index(run[-1]) + 1
        return spans

    def _run(self, doc: NormalizedDoc, words: list[_Word], i: int) -> tuple[list[_Word], str]:
        """The name run starting at word `i` (empty if none), and the cue before it: strong, weak or ""."""
        cue = _cue(doc, words, i)
        first = words[i]
        if first.honorific or (first.text in STOPWORDS and not first.name):
            return [], ""
        starts = first.name or _particle(doc, words, i) or (first.cap and cue != "" and len(first.text) > 1)
        if not starts:
            return [], ""
        run = [first]
        j = i
        while (
            len(run) < MAX_RUN
            and j + 1 < len(words)
            and _GAP.fullmatch(doc.text[words[j].end : words[j + 1].start])
        ):
            nxt = words[j + 1]
            if nxt.honorific or (nxt.text in STOPWORDS and not nxt.name):
                break
            if nxt.name or _particle(doc, words, j + 1) or (nxt.cap and run[-1].cap):
                run.append(nxt)
                j += 1
            else:
                break
        if run[-1].text in PARTICLES and len(run) > 1:  # a trailing `Ben` with nothing after it
            run.pop()
        return run, cue


def _particle(doc: NormalizedDoc, words: list[_Word], i: int) -> bool:
    """A name particle (`ben`, `el`, `بن`) followed by a name or a capitalized word."""
    if words[i].text not in PARTICLES or i + 1 >= len(words):
        return False
    nxt = words[i + 1]
    adjacent = _GAP.fullmatch(doc.text[words[i].end : nxt.start]) is not None
    glued = _key(words[i].text + " " + nxt.text)
    return adjacent and (nxt.name or glued in LAST or bool(nxt.cap))


def _cue(doc: NormalizedDoc, words: list[_Word], i: int) -> str:
    if (
        i > 0
        and words[i - 1].honorific
        and _HONORIFIC_GAP.fullmatch(doc.text[words[i - 1].end : words[i].start])
    ):
        honorific = words[i - 1].text
        if honorific == "m" and not doc.text[words[i - 1].end :].startswith("."):
            return ""  # `M` without its dot is too often something else
        return "strong" if honorific in STRONG_HONORIFICS else "weak"
    before = doc.text[max(0, words[i].start - 30) : words[i].start]
    if _STRONG_PHRASE.search(before):
        return "strong"
    if _WEAK_PHRASE.search(before):
        return "weak"
    return ""


def _score(run: list[_Word], cue: str) -> float:
    listed = [w for w in run if w.name]
    common_only = bool(listed) and all(w.key in COMMON for w in listed) and len(run) == 1
    if cue == "strong":
        return 0.95
    caps = [w.cap for w in run]
    if None in caps:  # Arabic script
        return 0.3 if common_only else 0.8
    if all(caps):
        if cue == "weak":
            return 0.8
        if len(run) > 1 and listed:
            return 0.9
        if common_only:
            return 0.5 if run[0].initial else 0.6
        return 0.85 if listed else 0.3
    if len(run) > 1 and listed:
        return 0.7
    if listed and not common_only and any(w.key in FIRST for w in listed):
        return 0.6
    return 0.3


@functools.lru_cache(maxsize=1)
def _city_patterns() -> list[tuple[re.Pattern[str], bool]]:
    common = {normalize(c).text for c in data.COMMON_CITIES}
    out: list[tuple[re.Pattern[str], bool]] = []
    for city in sorted(set(data.CITIES), key=len, reverse=True):
        words = normalize(city).text.split()
        pattern = re.compile(r"(?<!\w)" + r"[ \t-]+".join(map(re.escape, words)) + r"(?!\w)")
        out.append((pattern, " ".join(words) in common))
    return out


def _cities(doc: NormalizedDoc) -> list[tuple[int, int, float]]:
    """Non-overlapping city matches, longest first: (normalized start, end, score)."""
    found: list[tuple[int, int, float]] = []
    for pattern, is_common in _city_patterns():
        for m in pattern.finditer(doc.text):
            if any(m.start() < e and s < m.end() for s, e, _ in found):
                continue
            o_start, _ = doc.to_original(m.start(), m.end())
            first = doc.original[o_start]
            if first.lower() != first.upper() and not first.isupper():
                continue  # Latin script needs its capital
            before = doc.original[:o_start].rstrip(" \t")
            if is_common and (not before or before[-1] in _SENTENCE_END):
                continue
            found.append((m.start(), m.end(), 0.8))
    return sorted(found)
