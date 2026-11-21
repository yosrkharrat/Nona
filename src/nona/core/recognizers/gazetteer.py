"""Names and places from seed lists and context cues, before any ML model (deep dive A, layers 3 and 4).

**Layer 3, gazetteer.** Each word is reduced to its canonical token (`core/canonical.py`), so one list entry
covers its spellings: `Mohamed`/`Mohammed`/`Med`, Arabizi `mo7amed`, Arabic `محمد`. Names from the lists
(`gazetteer_data.py`) start a *name run*: the following words joined by a space or a bare hyphen (`Ben-Salem`;
a spaced ` - ` separates items, not name parts), as long as they are names too, a particle with a name after
it (`Ben Salem`, `Abdel Kader`), or one capitalized word that isn't in the lists (`Ahmed Karoui`). A run may
start with a particle without that counting as the unlisted word. In Arabic script, which has no capitals, the
unlisted word must look like a family name: the article, then a word ending in ي (`محمد الزغلامي`), so
`السيدة سعاد وصلت` doesn't take the verb. `M'barek` is one word.

**Layer 4, context cues.** A cue right before a word lifts the score of a run:
- *strong*: honorifics and kinship words (`M.`, `Mme`, `Dr`, `sidi`, `khalti`, `3amm`, `السيد`, `السيدة`) and
  name phrases (`je m'appelle`, `my name is`, `esmou`, `m3a`). A strong cue starts a run on a capitalized
  word, or on any word in Arabic script, even one that isn't in the lists; such a run may hold two unlisted
  words (`je m'appelle Kenza Haddad`).
- *weak*: words that are honorifics in one language and ordinary words in another (`si` is "if" in French,
  `me` in English), and greetings (`Bonjour`, `Merci`, `Hi`, `3aslema`). A weak cue only lifts a listed
  name: `Merci Beaucoup` and `Hello World` stay unmasked.
The cue itself is never part of the span, and a cue doesn't reach across a line break.

**Scores** (the resolver keeps spans at 0.5 and above by default):

| Run | Score |
|---|---|
| after a strong cue | 0.95 |
| capitalized, two words or more, with a listed name | 0.9 |
| capitalized single listed name | 0.85 |
| after a weak cue, capitalized, with a listed name | 0.8 |
| Arabic script, listed name | 0.8 |
| lowercase, two words or more, with a listed name that isn't a common word | 0.7 |
| lowercase listed first name, spelled as is (Arabizi writes names lowercase) | 0.6 |
| common-word name (`Amal`, `Nour`), capitalized mid-sentence | 0.6 |
| common-word name at the start of a sentence (masked at the default threshold: a leak costs more) | 0.5 |
| anything else: a lowercase common word (`3amel`, doing), a lowercase surname alone, ... | 0.3 |

**Arabic.** Prepositions such as `على` ("on", which folds to the same letters as `علي`, Ali) are never names.
A word that misses the lists is tried again without up to two leading clitics `و ف ب ل ك` (`لسامي`, `ولسامي`;
`للطرابلسي` is ل + ال), and the span leaves the clitics out.

**Places.** Cities from the lists are LOCATION spans (0.8) when they start with a capital letter (Latin
script). Latin spellings match with or without accents (`Gabès` = `Gabes`), and Arabic ones after up to two
clitics (`لصفاقس`, `للمنستير`). Cities that are also ordinary words (`Nice`) aren't matched at the start of
a sentence. A person run never takes a word that is part of a city, so `Sidi Bouzid` stays a place.

Linear in the text length: the proxy re-runs whole conversations, so nothing here looks back further than a
few characters.
"""

from __future__ import annotations

import functools
import re
import unicodedata
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

# Honorifics that are also ordinary words in another language: weak cues only. `M.` is strong once its dot
# is confirmed.
WEAK_HONORIFICS = frozenset({"si", "me", "m", "pr", "sir", "miss"})
STRONG_HONORIFICS = HONORIFICS - WEAK_HONORIFICS
# Matched on the normalized text that ends where the run starts. `[ \t]`, not `\s`: no cue across lines.
_STRONG_PHRASE = re.compile(
    r"(?<!\w)(?:je m['\u2019]appelle|m['\u2019]appelle|my name is|name is"
    r"|esmou|ismou|ism-ou|esmi|ismi|esm|m3a|ena|اسمي|اسمه)[ \t]*[:,]?[ \t]*$"
)
_WEAK_PHRASE = re.compile(
    r"(?<!\w)(?:bonjour|bonsoir|salut|merci|cher|chère|hi|hello|dear|thanks|thank you|3aslema|aslema|ahla"
    r"|ya|مرحبا|شكرا)[ \t]*[:,!]?[ \t]*$"
)
# Words that start sentences or name things, but never people.
STOPWORDS = frozenset(
    {
        "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "le", "la", "les", "un", "une", "des",
        "ce", "cette", "mon", "ma", "mes", "son", "sa", "ses", "et", "ou", "mais", "donc", "pour", "avec",
        "i", "the", "a", "an", "it", "he", "she", "we", "they", "you", "my", "your", "his", "her", "and",
        "or", "but", "please", "ok", "oui", "non", "yes", "no", "bonjour", "merci", "hello", "hi", "thanks",
        "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday", "janvier", "février", "mars", "avril", "mai",
        "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre", "january", "february",
        "march", "april", "june", "july", "august", "september", "october", "november", "december",
        "today", "tomorrow", "yesterday", "morning", "evening", "aujourd", "demain", "hier", "matin", "soir",
        "sms", "cin", "rib", "iban", "tél", "tel", "fax", "dt", "tnd", "email", "mail", "pdf", "id",
    }
)  # fmt: skip
_CLITICS = "وفبلك"
_GAP = re.compile(r"[ \t]+|-")
_HONORIFIC_GAP = re.compile(r"\.?[ \t]+|\.")  # `M. Ben Salem`, `Dr Gharbi`, `Mme.Ksouri`
_WORD = re.compile(r"(?<![^\W_])[Mm]['\u2019][^\W_]+|[^\W_]+")  # `M'barek` stays one word
_SENTENCE_END = frozenset('.!?:\n"«“')


@functools.lru_cache(maxsize=8192)
def _key(word: str) -> str | None:
    tokens = canonical_tokens(word)
    return next(iter(tokens)) if len(tokens) == 1 else None


def _initial(original: str, start: int) -> bool:
    """Whether the text at `start` begins a sentence. Looks back over spaces only, never the whole prefix."""
    i = start - 1
    while i >= 0 and original[i] in " \t":
        i -= 1
    return i < 0 or original[i] in _SENTENCE_END


@dataclass(frozen=True, slots=True)
class _Word:
    start: int  # normalized offsets
    end: int
    text: str  # normalized
    key: str | None
    cap: bool | None  # None for scripts without case
    initial: bool  # first word of a sentence
    honorific: bool
    clitic: int = 0  # leading Arabic clitic letters to leave out of the span

    @property
    def name(self) -> bool:
        return self.key in NAMES and self.text not in STOPWORDS

    @property
    def stop(self) -> bool:
        return (
            self.honorific or (self.text in STOPWORDS and self.key not in NAMES) or self.text in ARABIC_STOP
        )


# Arabic prepositions and pronouns, compared before folding: `على` ("on") would fold to `علي`, Ali.
ARABIC_STOP = frozenset({"على", "الى", "إلى", "في", "من", "عن", "مع", "هذا", "هذه", "هو", "هي"})


def _strip_clitics(text: str, key: str | None) -> tuple[str | None, int]:
    """A listed name behind up to two Arabic clitics: `لسامي`, `ولسامي`, `للطرابلسي` (ل + ال). Returns the
    name's key and how many letters to leave out of the span, or the word unchanged."""
    for n in (1, 2):
        if len(text) - n < 2 or any(c not in _CLITICS for c in text[:n]):
            break
        rest = text[n:]
        alef = "\u0627"
        for candidate in (rest, alef + rest) if rest.startswith("ل") else (rest,):  # لل → ل + ال
            stripped = _key(candidate)
            if stripped in NAMES:
                return stripped, n
    return key, 0


def _words(doc: NormalizedDoc) -> list[_Word]:
    out: list[_Word] = []
    for m in _WORD.finditer(doc.text):
        text = m.group()
        if text.isdigit():
            continue
        o_start, o_end = doc.to_original(m.start(), m.end())
        surface = doc.original[o_start:o_end]
        first = next((c for c in surface if c.isalpha()), "")  # `3ali` → `a`
        cap = first.isupper() if first.lower() != first.upper() else None
        key = _key(surface)
        honorific = not canonical_tokens(surface) and not text.isdigit()  # canonical folding: `السيدة` too
        clitic = 0
        if key not in NAMES and cap is None:
            key, clitic = _strip_clitics(text, key)
        out.append(
            _Word(
                m.start(),
                m.end(),
                text,
                key,
                cap,
                _initial(doc.original, o_start),
                honorific,
                clitic,
            )
        )
    return out


class GazetteerRecognizer:
    @property
    def fingerprint(self) -> str:
        """No configuration: every instance detects the same (detection cache, `core/cache.py`)."""
        return "default"

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
            end, cue = _run(doc, words, i)
            if end == i:
                i += 1
                continue
            run = words[i:end]
            start = run[0].start + run[0].clitic
            spans.append(to_span(doc, start, run[-1].end, EntityType.PERSON, _score(run, cue), self.name))
            i = end
        return spans


def _run(doc: NormalizedDoc, words: list[_Word], i: int) -> tuple[int, str]:
    """The end (exclusive) of the name run starting at word `i` (`i` if none), and the cue before it."""
    cue = _cue(doc, words, i)
    first = words[i]
    if first.stop:
        return i, ""
    cued = cue == "strong" and (first.cap or first.cap is None) and len(first.text) > 1
    if not (first.name or _particle(doc, words, i) or cued):
        return i, ""
    unlisted = 0 if first.name or first.text in PARTICLES else 1  # `Abdel Kader …`: a particle is no name
    max_unlisted = 2 if cue == "strong" else 1
    j = i + 1
    while j < len(words) and j - i < MAX_RUN and _GAP.fullmatch(doc.text[words[j - 1].end : words[j].start]):
        nxt = words[j]
        if nxt.stop:
            break
        if nxt.name or _particle(doc, words, j):
            j += 1
        elif unlisted < max_unlisted and (
            (nxt.cap and words[j - 1].cap)
            or (nxt.cap is None and words[j - 1].cap is None and _nisba(nxt.text))
        ):
            unlisted += 1
            j += 1
        else:
            break
    if j - i > 1 and words[j - 1].text in PARTICLES:  # a trailing `Ben` with nothing after it
        j -= 1
    return j, cue


def _nisba(text: str) -> bool:
    """An Arabic family name of the `الزغلامي`, `الطرابلسي` kind: the article, then 3+ letters ending in ي."""
    return text.startswith("ال") and text.endswith("ي") and len(text) >= 6 and text not in ARABIC_STOP


def _particle(doc: NormalizedDoc, words: list[_Word], i: int) -> bool:
    """A name particle (`ben`, `el`, `بن`) followed by a name or a capitalized word."""
    if words[i].text not in PARTICLES or i + 1 >= len(words):
        return False
    nxt = words[i + 1]
    adjacent = _GAP.fullmatch(doc.text[words[i].end : nxt.start]) is not None
    glued = _key(words[i].text + " " + nxt.text)
    return adjacent and (nxt.name or glued in LAST or bool(nxt.cap))


def _cue(doc: NormalizedDoc, words: list[_Word], i: int) -> str:
    prev = words[i - 1] if i > 0 else None
    if prev is not None and prev.honorific and _HONORIFIC_GAP.fullmatch(doc.text[prev.end : words[i].start]):
        if prev.text == "m":
            # `M.` is a strong cue; `M` without its dot is too often something else (`M Bricolage`).
            return "strong" if doc.text.startswith(".", prev.end) else ""
        return "strong" if prev.text in STRONG_HONORIFICS or prev.cap is None else "weak"
    before = doc.text[max(0, words[i].start - 30) : words[i].start]
    if _STRONG_PHRASE.search(before):
        return "strong"
    if _WEAK_PHRASE.search(before):
        return "weak"
    return ""


def _score(run: list[_Word], cue: str) -> float:
    listed = [w for w in run if w.name]
    common_only = bool(listed) and all(w.key in COMMON for w in listed)
    if cue == "strong":
        return 0.95
    if not listed:
        return 0.3  # an unlisted word after a weak cue, or a particle alone
    caps = [w.cap for w in run]
    if None in caps:  # Arabic script
        return 0.3 if common_only and len(run) == 1 else 0.8
    if all(caps):
        if cue == "weak":
            return 0.8
        if len(run) > 1:
            return 0.9
        if common_only:
            return 0.5 if run[0].initial else 0.6
        return 0.85
    if common_only:
        return 0.3
    if len(run) > 1:
        return 0.7
    return 0.6 if any(w.key in FIRST for w in listed) else 0.3


# Latin letters and their accented forms: `Gabès`, `Gabes` and `GABES` are one city.
_ACCENTS = {"a": "aàâäá", "e": "eéèêë", "i": "iîïí", "o": "oôöó", "u": "uùûüú", "c": "cç", "y": "yÿ"}
_ARABIC_LEFT = "(?:(?<![^\\W_])|(?<=(?<![^\\W_])[وفبلك])|(?<=(?<![^\\W_])[وفبلك]{2}))"


def _letter(c: str) -> str:
    base = unicodedata.normalize("NFD", c)[0]
    return f"[{_ACCENTS[base]}]" if base in _ACCENTS else re.escape(c)


@functools.lru_cache(maxsize=1)
def _city_patterns() -> list[tuple[re.Pattern[str], bool]]:
    common = {_unaccent(normalize(c).text) for c in data.COMMON_CITIES}
    out: list[tuple[re.Pattern[str], bool]] = []
    for city in sorted(set(data.CITIES), key=len, reverse=True):
        words = normalize(city).text.split()
        body = r"[ \t-]+".join("".join(map(_letter, w)) for w in words)
        is_common = _unaccent(" ".join(words)) in common
        if any("\u0600" <= c <= "\u06ff" for c in city):
            # Arabic: up to two clitics in front (`لصفاقس`), and `للمنستير` = ل + المنستير
            out.append((re.compile(_ARABIC_LEFT + body + r"(?!\w)"), is_common))
            if words[0].startswith("ال"):
                rest = r"[ \t-]+".join([re.escape(words[0][1:]), *map(re.escape, words[1:])])
                after_lam = r"(?:(?<=(?<![^\W_])ل)|(?<=(?<![^\W_])[وف]ل))"
                out.append((re.compile(after_lam + rest + r"(?!\w)"), is_common))
        else:
            out.append((re.compile(r"(?<!\w)" + body + r"(?!\w)"), is_common))
    return out


def _unaccent(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


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
            if is_common and _initial(doc.original, o_start):
                continue
            found.append((m.start(), m.end(), 0.8))
    return sorted(found)
