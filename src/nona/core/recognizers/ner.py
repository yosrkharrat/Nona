"""NER recognizer: PERSON, LOCATION and ORGANIZATION from a GLiNER model (deep dive A layer 2, ADR-0008).

The model is **injected**: anything with the `NerModel` shape. `nona.models` loads the pinned GLiNER weights
and adapts them; tests pass a fake. So `nona.core` stays free of torch and file I/O, and loads without the
`ner` extra.

What this layer adds on top of the model:

- **The original text**, not the normalized view: normalizing folds case, and capitals are the model's best
  cue. Offsets are then original offsets already. One change is made, offset for offset (the de-Arabizi'd
  view, below).
- **Windows.** A model reads a bounded number of words (GLiNER: `max_len`, 384) and silently ignores the
  rest, so a name in the signature of a long email would leak. Long texts are cut into windows of at most
  `max_words` words that overlap by `overlap_words`, so a name cut by one window is whole in the next. Words
  are counted the way GLiNER splits them (a run of word characters, or any other single character), so a
  window never exceeds what the model reads.
- **The de-Arabizi'd view.** The model reads the text with Arabizi digits replaced by a letter, one for one
  (`3andi` → `aandi`, `9alli` → `qalli`), so offsets don't move. On `dev` it lowers leaks, over-masking and
  wrong splits together (ADR-0008). Only words that mix Latin letters with the Arabizi digits 2, 3, 5, 7, 9,
  each next to a letter, change: `15h`, `hamza23`, `B2B` and phone numbers stay as written. Combining
  the model's spans on both texts caught one more held-out name on `dev` for a second model call on every
  Arabizi message; ADR-0008 records it, and part 3 decides with the latency numbers.
- **Spans are cleaned** (ADR-0008, condition 2). A span is cut at words that are never names, from closed
  lists: honorifics (`Mme Islem` → `Islem`, `بالسيدة` too), the gazetteer's stopwords and pronouns (`I` in
  `I'll`), and Darija words (`NEVER_NAMES`: `nomrou`, `Behi`, `9alli`). An unlisted word stays masked,
  over-masked rather than leaked (fail closed, ADR-0003): names are written with Arabizi digits too (`Mo3ez`,
  `Na3ima Ka3bi`), so a word with digits is no sign of a non-name. Each piece left with a word of two letters
  or more is a span; a trailing particle is dropped. A listed name behind Arabic clitics loses them (`وسامي` →
  `سامي`), as in the gazetteer.
- **Types** (ADR-0008, condition 1). The model often calls a person a place, and an Arabizi chunk a place.
  A LOCATION or ORGANIZATION span with a listed first name that isn't a common word, and isn't a listed city,
  is a PERSON (`Bilel Gharbi`); so is a LOCATION right after a person cue (`m3a Hazem Drira`, `with`, a
  greeting, an honorific). A LOCATION in lowercase Latin letters is dropped unless it is a listed city:
  `nomrou`, `ykhdmou 24/24`, `jardin` were the model's, and no gold place was. The conversation-level rule
  (a span that fits a known person) is in `core/pipeline.py`.
- **A threshold** (ADR-0008: 0.5). A span keeps the model's score, so the resolver's `min_score` still
  applies, and look-ahead only spreads words from confident spans (0.8, ADR-0004).
- **Cue-backed low scores** (deep dive A, layer 4: a context cue raises a weak candidate). In Arabizi the
  model often finds a name but scores it under 0.5 (`Hazem 9alli`: 0.46). A span from `cued_threshold` (0.2)
  up is kept whole, as a PERSON scored at the threshold, when it looks like a name (a capitalized word, a word
  starting with an Arabizi digit like `3li`, or Arabic) and a cue backs it: a person cue before it, or a
  speech verb right after it (`9alli`, `قالي`, `said`, from #50's list). A cued span takes in the low-score
  span right after it, spaces only between, or a confident one whose words all look like names: in `m3a Mootez
  Jaouadi` the model scores `Mootez` 0.25 and `Jaouadi` 0.42, one name.

A model error propagates; the pipeline turns it into `DetectorError` (fail closed, ADR-0003).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol

from nona.core.boundaries import PROCLITICS
from nona.core.canonical import ABD_PREFIXES, PARTICLES, canonical_tokens, is_honorific
from nona.core.normalize import NormalizedDoc, normalize
from nona.core.recognizers import gazetteer_data
from nona.core.recognizers.gazetteer import (
    ARABIC_STOP,
    FIRST,
    NAMES,
    STOPWORDS,
    STRONG_PHRASE,
    WEAK_PHRASE,
    cities,
    common,
    strip_clitics,
    unaccent,
)
from nona.core.types import EntityType, Span

NER = "ner"  # the detector name of model spans: `core/pipeline.py` corrects their types

# The labels asked of the model, and the type each one gives (ADR-0008).
LABELS: Mapping[str, EntityType] = {
    "person": EntityType.PERSON,
    "location": EntityType.LOCATION,
    "organization": EntityType.ORGANIZATION,
}

# GLiNER's whitespace splitter: a word is a run of word characters (joined by `-` or `_`), or one other
# non-space character. Windows count words this way so they fit what the model reads.
_MODEL_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")
# A word inside a span: letters and digits with their marks, joined by an apostrophe or a hyphen. Dots and
# commas separate, so a glued `Dr.Sami` gives `Dr` then `Sami`.
_MARKS = "\u0610-\u061a\u0640\u064b-\u065f\u0670"
_SPAN_WORD = re.compile(rf"[\w{_MARKS}]+(?:['\u2019-][\w{_MARKS}]+)*")
_ARABIC_MARK = re.compile(rf"[{_MARKS}]")

# The de-Arabizi'd view: each Arabizi digit becomes one letter, so offsets are unchanged.
_ARABIZI_LETTER = {"2": "a", "3": "a", "5": "k", "7": "h", "9": "q"}
_ALNUM_RUN = re.compile(r"[^\W_]+")

_NEVER_NAMES = frozenset(normalize(w).text for w in gazetteer_data.NEVER_NAMES)
_CITIES = frozenset(unaccent(normalize(c).text) for c in gazetteer_data.CITIES)
_NAME_PARTICLES = PARTICLES | ABD_PREFIXES
# Streets, districts and organizations are named after people (`rue Mohamed Ali`, `Clinique Ahmed Ben Salem`,
# `شارع الحبيب بورقيبة`): a span that starts with one of these words, or comes right after one, keeps the
# model's type.
_HEADS = frozenset(
    normalize(w).text
    for w in (
        "rue", "avenue", "av", "boulevard", "bd", "route", "impasse", "place", "cité", "cite", "street", "st",
        "road", "clinique", "pharmacie", "hôpital", "hopital", "cabinet", "société", "societe", "ste", "sarl",
        "banque", "bank", "école", "ecole", "lycée", "lycee", "université", "universite", "institut",
        "centre", "center", "hôtel", "hotel", "café", "cafe", "restaurant", "clinic", "pharmacy", "hospital",
        "school", "company",
        "شارع", "نهج", "طريق", "حي", "ساحة", "مستشفى", "مصحة", "صيدلية", "شركة", "بنك", "مدرسة", "معهد",
        "جامعة", "مقهى", "نزل",
    )
)  # fmt: skip
# Speech and contact verbs right after a name (#50): `Hazem 9alli`, `Aymen kallamni`, `سامي قالي`.
_SPEECH_AFTER = re.compile(
    r"[ \t]+(?:"
    + "|".join(re.escape(normalize(v).text) for v in gazetteer_data.SPEECH_VERBS)
    + r"|said|a dit|m['\u2019]a dit)(?!\w)"
)
# What makes the next name a person: the gazetteer's strong and weak phrases (`je m'appelle`, `m3a`,
# greetings), and `with`, `avec`, `مع` ("with": a meeting is with someone). Matched on the folded text.
_WITH = re.compile(r"(?<!\w)(?:with|avec|مع|ma3)[ \t]*\Z")
_GAP = re.compile(r"[ \t]+")
RULES_VERSION = 5  # part of the fingerprint: bump it when cleaning or typing changes, so cached spans go


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

    def __init__(
        self, model: NerModel, *, threshold: float = 0.5, cued_threshold: float = 0.2, overlap_words: int = 32
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if not 0.0 < cued_threshold <= threshold:
            raise ValueError("cued_threshold must be in (0, threshold]")
        if not 0 <= overlap_words < model.max_words // 2:
            raise ValueError("overlap_words must be at least 0 and under half the model's max_words")
        self._model = model
        self._threshold = threshold
        self._cued = cued_threshold
        self._overlap = overlap_words

    @property
    def name(self) -> str:
        return NER

    @property
    def supported_types(self) -> frozenset[EntityType]:
        return frozenset(LABELS.values())

    @property
    def fingerprint(self) -> str:
        labels = ",".join(f"{k}={v.value}" for k, v in LABELS.items())
        return (
            f"model={self._model.identity};threshold={self._threshold};cued={self._cued};labels={labels};"
            f"max_words={self._model.max_words};overlap={self._overlap};rules={RULES_VERSION}"
        )

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        text = doc.original
        view = model_view(text)
        places: list[tuple[int, int]] | None = None  # listed cities, original offsets: computed when needed
        best: dict[tuple[int, int, EntityType], float] = {}
        cued: set[tuple[int, int, EntityType]] = set()
        pieces: set[tuple[int, int, EntityType]] = set()  # uncued low spans, kept only once joined
        high: set[tuple[int, int, EntityType]] = set()  # spans at the threshold or above: always kept
        for w_start, w_end in windows(text, self._model.max_words, self._overlap):
            for p in self._model.predict(view[w_start:w_end], list(LABELS), self._cued):
                etype = LABELS.get(p.label)
                if etype is None or p.score < self._cued or not 0 <= p.start < p.end <= w_end - w_start:
                    continue
                start, end = w_start + p.start, w_start + p.end
                if etype is not EntityType.PERSON and places is None:
                    places = [doc.to_original(s, e) for s, e, _ in cities(doc)]
                low = p.score < self._threshold
                for piece in clean(text, start, end, person=etype is EntityType.PERSON):
                    t = retype(text, *piece, etype, places or [], cue_at=start)
                    if t is None or (
                        low and (t is EntityType.ORGANIZATION or _place(text, *piece, places or []))
                    ):
                        continue
                    # A place span keeps its honorifics; once it is a person, they are trimmed (`Mme Islem`).
                    parts = clean(text, *piece) if t is EntityType.PERSON and etype is not t else [piece]
                    for a, b in parts:
                        if low:  # kept whole if it looks like a name and a cue backs it, or joined
                            key, score = (a, b, EntityType.PERSON), self._threshold
                            backed = _name_shaped(text, a, b) and (
                                _person_cue(text, a) or _person_cue(text, start) or _speech_after(text, b)
                            )
                            (cued if backed else pieces).add(key)
                        else:
                            key, score = (a, b, t), min(p.score, 1.0)
                            high.add(key)
                        best[key] = max(best.get(key, 0.0), score)
        joined = _join_cued(text, best, cued, pieces - cued - high, high)
        return [Span(s, e, t, score, self.name) for (s, e, t), score in joined]


def _join_cued(
    text: str,
    best: dict[tuple[int, int, EntityType], float],
    cued: set[tuple[int, int, EntityType]],
    pieces: set[tuple[int, int, EntityType]],
    high: set[tuple[int, int, EntityType]],
) -> list[tuple[tuple[int, int, EntityType], float]]:
    """Spans sorted, with each cued span joined to the PERSON span right after it, spaces only between: a cue
    introduces one name, which the model may score in pieces (`m3a Mootez Jaouadi`: 0.25 and 0.42; `m3a
    Mootez ben salah`). `pieces` (low, uncued) are only kept when joined that way. A confident span joins only
    when every word looks like a name, so a cued name doesn't take in `Ali y7eb Mariem`."""
    joined: list[tuple[tuple[int, int, EntityType], float]] = []
    for key, score in sorted(best.items()):
        if joined:
            (a, b, t), prev = joined[-1]
            if (
                t is EntityType.PERSON
                and (a, b, t) in cued
                and key[2] is EntityType.PERSON
                and key[0] > b
                and _GAP.fullmatch(text, b, key[0])
                and (key not in high or _name_shaped(text, key[0], key[1], every=True))
            ):
                joined[-1] = ((a, key[1], EntityType.PERSON), max(prev, score))
                cued.add(joined[-1][0])
                continue
        if key not in pieces:
            joined.append((key, score))
    return joined


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


def model_view(text: str) -> str:
    """The text as the model reads it: Arabizi digits replaced by a letter, offset for offset (module
    docstring). `3andi m3a Sami` → `aandi maa Sami`."""
    return _ALNUM_RUN.sub(lambda m: _de_arabizi(m.group()) if arabizi(m.group()) else m.group(), text)


def _de_arabizi(word: str) -> str:
    return "".join(_ARABIZI_LETTER.get(c, c) for c in word)


def arabizi(word: str) -> bool:
    """A word mixing Latin letters with Arabizi digits (2 3 5 7 9), each digit next to a letter, with at least
    two letters, not all capitals: `3andi`, `9alli`, `y5allik`. Not `15h`, `3h`, `hamza23`, `4G`, `B2B`."""
    letters = 0
    for i, c in enumerate(word):
        if c.isdigit():
            if c not in _ARABIZI_LETTER:
                return False
            if not ((i > 0 and _latin(word[i - 1])) or (i + 1 < len(word) and _latin(word[i + 1]))):
                return False
        elif _latin(c):
            letters += 1
        else:
            return False
    return letters >= 2 and letters < len(word) and not word.isupper()


def _latin(c: str) -> bool:
    return c.isascii() and c.isalpha()


@functools.lru_cache(maxsize=8192)
def never_name(word: str, *, honorific: bool = True) -> bool:
    """Whether a written word can't be (part of) a name (module docstring): an honorific, also behind Arabic
    clitics (`بالسيدة`), or a stopword or Darija word that isn't a listed name. Closed lists only: any other
    word may be a name (`Mo3ez`). With `honorific=False`, honorifics don't count: a place may hold one
    (`rue Dr Ali Belhouane`)."""
    folded = normalize(word).text
    if _honorific(word):
        return honorific
    tokens = canonical_tokens(word)
    listed = bool(tokens) and tokens <= NAMES
    if word in ARABIC_STOP:
        return True
    return (folded in STOPWORDS or folded in _NEVER_NAMES) and not listed


def _honorific(word: str) -> bool:
    """An honorific, also behind up to two Arabic clitics (`بالسيدة`)."""
    folded = normalize(word).text
    return is_honorific(word) or (folded[:1] in PROCLITICS and _clitic_honorific(folded))


def _clitic_honorific(folded: str) -> bool:
    """`بالسيدة` (ب + السيدة), `وللسيد`: up to two clitics before an Arabic honorific."""
    return any(
        all(c in PROCLITICS for c in folded[:n]) and is_honorific(folded[n:])
        for n in (1, 2)
        if len(folded) > n
    )


def clean(text: str, start: int, end: int, *, person: bool = True) -> list[tuple[int, int]]:
    """The pieces of a model span that may be names (module docstring): cut at never-name words, without a
    trailing particle or clitics before a listed Arabic name, each with a word of two letters or more.
    Honorifics cut only a PERSON span (`person`)."""
    words = [(m.start(), m.end()) for m in _SPAN_WORD.finditer(text, start, end)]
    pieces: list[list[tuple[int, int]]] = [[]]
    for a, b in words:
        if never_name(text[a:b], honorific=person):
            pieces.append([])
        else:
            pieces[-1].append((a, b))
    out: list[tuple[int, int]] = []
    for piece in pieces:
        while piece and normalize(text[piece[-1][0] : piece[-1][1]]).text in _NAME_PARTICLES:
            piece.pop()
        if not piece or not any(sum(c.isalpha() for c in text[a:b]) >= 2 for a, b in piece):
            continue
        a = piece[0][0]
        first = normalize(text[a : piece[0][1]]).text
        _, clitics = strip_clitics(first, None)
        if clitics and not canonical_tokens(text[a : piece[0][1]]) <= NAMES:
            a = _past_letters(text, a, clitics)
        out.append((a, piece[-1][1]))
    return out


def _past_letters(text: str, i: int, n: int) -> int:
    """The offset after `n` letters from `i`, skipping the Arabic marks that normalizing drops."""
    while n:
        i += 1
        n -= 1
        while i < len(text) and _ARABIC_MARK.match(text[i]):
            i += 1
    return i


def retype(
    text: str,
    start: int,
    end: int,
    etype: EntityType,
    places: Sequence[tuple[int, int]],
    cue_at: int | None = None,
) -> EntityType | None:
    """The type a cleaned model span gets, or `None` to drop it (module docstring). `cue_at` is where the
    model's span started, before cleaning: a cue is looked for there and before the piece itself."""
    if etype is EntityType.PERSON:
        return etype
    surface = text[start:end]
    cue_at = start if cue_at is None else cue_at
    if _place(text, start, end, places, cue_at):
        return etype
    if _listed_first_name(surface):
        return EntityType.PERSON
    first = _SPAN_WORD.match(surface)
    leading = first is not None and _honorific(first.group())  # `Mme Islem`, typed a place
    cued = _person_cue(text, cue_at) or _person_cue(text, start) or leading
    if etype is EntityType.LOCATION and cued:
        return EntityType.PERSON
    if etype is EntityType.LOCATION and any(_latin(c) for c in surface) and surface == surface.lower():
        return None
    return etype


def _place(
    text: str, start: int, end: int, places: Sequence[tuple[int, int]], cue_at: int | None = None
) -> bool:
    """Whether a span is a place or an organization whatever names it holds: a listed city, or a head word
    (`rue`, `Clinique`, `شارع`) as its first word or right before it (`place_or_org`)."""
    surface = text[start:end]
    if any(a < end and start < b for a, b in places) or unaccent(normalize(surface).text) in _CITIES:
        return True
    return place_or_org(text, start, end) or (cue_at is not None and place_or_org(text, cue_at, end))


def place_or_org(text: str, start: int, end: int) -> bool:
    """A span that starts with a place or organization head word, or right after one: `rue Mohamed Ali`,
    `Clinique Ahmed Ben Salem`, `à la Clinique` + `Ahmed Ben Salem`. Such a span is named after someone, and
    keeps the model's type (`core/pipeline.py` too)."""
    first = _SPAN_WORD.match(text, start, end)
    if first is not None and normalize(first.group()).text in _HEADS:
        return True
    before = normalize(text[max(0, start - 30) : start]).text.split()
    return bool(before) and before[-1].rstrip(".") in _HEADS and text[start - 1 : start] in " \t"


def _listed_first_name(surface: str) -> bool:
    return any(
        (tokens := canonical_tokens(m.group()))
        and tokens <= FIRST
        and not any(common(t, _cased(m.group())) for t in tokens)
        for m in _SPAN_WORD.finditer(surface)
    )


def _name_shaped(text: str, start: int, end: int, *, every: bool = False) -> bool:
    """What a name looks like when the model is unsure: a word with letters that starts with a capital, with
    an Arabizi digit (`3li`, `7amdi`), or is in a script without capitals (Arabic). `Hazem y7ki` passes;
    `bureau 3` and `montaser` don't. With `every`, each word with letters must."""
    words = [w for w in _SPAN_WORD.findall(text, start, end) if any(c.isalpha() for c in w)]
    shaped = (not _cased(w) or w[0].isupper() or (w[0] in _ARABIZI_LETTER and arabizi(w)) for w in words)
    return all(shaped) if every else any(shaped)


def _speech_after(text: str, end: int) -> bool:
    """A speech or contact verb right after a name, on the same line (#50): `Hazem 9alli`, `سامي قالي`."""
    return _SPEECH_AFTER.match(normalize(text[end : end + 20]).text) is not None


def _cased(word: str) -> bool:
    return any(c.lower() != c.upper() for c in word)


def _person_cue(text: str, start: int) -> bool:
    """A person cue right before `start`, on the same line: an honorific, or a gazetteer cue phrase."""
    before = normalize(text[max(0, start - 30) : start]).text
    words = before.split()
    if words and _honorific(words[-1]) and before[-1:] in (" ", "\t"):
        return True
    return bool(STRONG_PHRASE.search(before) or WEAK_PHRASE.search(before) or _WITH.search(before))
