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
  name phrases (`je m'appelle`, `my name is`, `esmou`). A strong cue starts a run on a capitalized word, or on
  any word in Arabic script, even one that isn't in the lists; such a run may hold two unlisted words
  (`je m'appelle Kenza Haddad`). `M.` between a number and a unit is not a cue (`2 M. Dinars`, million).
  Darija kinship and relation words are strong too (`khouya`, `okhti`, `sa7bi`, `jari`, `mart`, `weld`,
  `خويا`, `صاحبي`, #50).
- *speech*: Darija speech and contact verbs (`9olt l`, `kallamt`, `b3atht l`, `sa2alt`, `7kit m3a`, `chouft`,
  `قلت ل`, `كلمت`, #50). They lift a listed name, even lowercase or a common word (`kallamt amal`, 0.9), but
  never start a run on an unlisted word: `kallamt Ooredoo` stays unmasked, and an unlisted name is left to
  the NER model. A verb agreeing with a person right after a listed name does the same (`amal 9alli`,
  `karim ja`, `نور قالت`), unless the word carries an article (`el nour ja`, `النور جا`).
- *weak*: words that are honorifics in one language and ordinary words in another (`si` is "if" in French,
  `me` in English, `3am` "year" in Darija), Darija function words (`m3a` "with", `ena` "I": `m3a Ooredoo`,
  `Ena Tounsi`), and greetings (`Bonjour`, `Merci`, `Hi`, `3aslema`). A weak cue only lifts a listed name:
  `Merci Beaucoup` and `Hello World` stay unmasked.
The cue itself is never part of the span, and a cue doesn't reach across a line break. Some cues also read as
ordinary words, and the owner kept them knowing it (#50): `rajel karim` (a generous man), `weld 3aziz`,
`chouft amal` (I saw hope), `sana jet` (the year came) are masked.

**Never a name** (#50): fixed expressions in which a common-word name is the ordinary word: `mabrouk 3lik`,
`alf mabrouk`, `3id mabrouk`, `ramdhan karim`, `rabbi karim`, `sana sa3ida`, `kol sana`, `3andi amal`,
`amal kbir`, `nour 3iniya`, `رحمة الله`, and their Arabic forms. Only a strong cue overrides them
(`Mme Mabrouk !`).

**Scores** (the resolver keeps spans at 0.5 and above by default):

| Run | Score |
|---|---|
| after a strong cue | 0.95 |
| a listed name after a speech verb, or before a verb agreeing with a person (any case) | 0.9 |
| capitalized, two words or more, with a listed name | 0.9 |
| capitalized single listed name | 0.85 |
| after a weak cue, capitalized, with a listed name | 0.8 |
| Arabic script, listed name (a common-word one only after a cue: `مرحبا نور`) | 0.8 |
| lowercase or partly capitalized, 2+ words, a listed name that isn't a common word (see below) | 0.7 |
| lowercase Tunisian first name (Arabizi writes names lowercase: `sami`, `mo7amed`) | 0.6 |
| common-word name (`Amal`, `Nour`), capitalized mid-sentence | 0.6 |
| common-word name at the start of a sentence (masked at the default threshold: a leak costs more) | 0.5 |
| anything else: a lowercase common word (`3amel`), surname alone, French or English first name | 0.3 |

In lowercase, French and English first names are ordinary words (`claire`, `pierre`, and `marié`, which folds
to `marie`). A run that is partly capitalized (`Nicolas ben Salem`) is scored on all its listed names.

**Arabic.** Prepositions such as `على` ("on", which folds to the same letters as `علي`, Ali) are never names.
A word that misses the lists is tried again without up to two leading clitics `و ف ب ل ك` (`لسامي`, `ولسامي`;
`للطرابلسي` is ل + ال), and the span leaves the clitics out. In the ل + ال case, the span keeps the
second ل (`لطرابلسي`), so it doesn't link to `الطرابلسي`: linking it would make restore write `لالطرابلسي`.

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

from nona.core.canonical import ABD_PREFIXES, HONORIFICS, PARTICLES, canonical_tokens
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
FIRST_TN = _keys(data.FIRST_NAMES_TN)
LAST = _keys(data.SURNAMES_TN) | _keys(tuple(n for n in data.NAMES_AR if " " in n or n.startswith("ال")))
# An Arabic name shares its Latin form's key (`كريم` → `karim`), so one list covers both scripts. The owner
# decided Arabic script follows the Arabizi rule (#50): a bare `كريم`, like a lowercase `karim`, is masked
# only with a cue. Accepted cost: a bare Arabic common-word name with nothing around it leaks.
COMMON = _keys(data.COMMON_WORDS)
# The Arabic forms the list names itself (`نور`, `أمال`): only these count as ordinary words when a model
# already flagged the span (`ner.py` retypes a listed first name to PERSON). The rule above is about finding
# a bare word without a model.
COMMON_AR = _keys(tuple(w for w in data.COMMON_WORDS if any("\u0600" <= c <= "\u06ff" for c in w)))


def common(key: str | None, cased: bool) -> bool:
    """Whether a listed name is also an ordinary word, for detection without a model: the same in every
    script since #50. `cased` stays for the callers that ask per script."""
    del cased
    return key in COMMON


def common_as_written(key: str | None, cased: bool) -> bool:
    """Whether a listed name is an ordinary word in the script it is written in (`Karim` is, `كريم` isn't):
    for a span a model already flagged."""
    return key in (COMMON if cased else COMMON_AR)


NAMES = FIRST | LAST
# What a lowercase run may rest on: Arabizi writes Tunisian names in lowercase, while a lowercase French or
# English first name (`claire`, `pierre`, `rose`) is an ordinary word.
TUNISIAN = FIRST_TN | LAST | _keys(data.NAMES_AR)

# Honorifics that are also ordinary words in another language: weak cues only. `M.` is strong once its dot
# is confirmed.
WEAK_HONORIFICS = frozenset({"si", "me", "m", "pr", "sir", "miss", "3am"})  # `3am sa3id`: happy year
STRONG_HONORIFICS = HONORIFICS - WEAK_HONORIFICS
# Matched on the normalized text that ends where the run starts. `[ \t]`, not `\s`, and `\Z`, not `$` (which
# matches before a final newline): no cue across lines.
_NAME_PHRASE = re.compile(
    r"(?<!\w)(?:je m['\u2019]appelle|m['\u2019]appelle|my name is|name is"
    r"|esmou|ismou|ism-ou|esmi|ismi|esm|اسمي|اسمه)[ \t]*[:,]?[ \t]*\Z"
)
# Arabic letters people spell either way: the normalized view keeps them, only canonical keys fold them.
# Alef with or without hamza or madda, ta marbuta or heh, alef maqsura or yeh.
_ALEFS = "\u0627\u0623\u0625\u0622"
_TA_HEH = "\u0629\u0647"
_YEHS = "\u0649\u064a"
_AR_VARIANTS = {c: f"[{group}]" for group in (_ALEFS, _TA_HEH, _YEHS) for c in group}


def _alternation(words: tuple[str, ...]) -> str:
    """Cue words or phrases as one regex alternation on the normalized view, longest first: spaces match runs
    of spaces or tabs (never a line break), Arabic letters their spelling variants."""

    def one(word: str) -> str:
        return "".join(
            r"[ \t]+" if c == " " else _AR_VARIANTS.get(c, re.escape(c)) for c in normalize(word).text
        )

    return "|".join(one(w) for w in sorted(set(words), key=len, reverse=True))


STRONG_PHRASE = re.compile(
    rf"(?<![^\W_])(?:{_alternation(data.KIN_CUES)})[ \t]*[:,]?[ \t]*\Z|" + _NAME_PHRASE.pattern
)
SPEECH_PHRASE = re.compile(
    rf"(?<![^\W_])(?:{_alternation(data.SPEECH_CUES + data.KIN_CUES_AR)})[ \t]*[:,]?[ \t]*\Z"
)
# A verb agreeing with a person right after the name, in the same clause, and the article that rules it out.
_VERB_AFTER = re.compile(rf"[ \t]+(?:{_alternation(data.VERBS_AFTER_NAME)})(?![^\W_])")
_ARTICLE_BEFORE = re.compile(r"(?<![^\W_])(?:el|al|l)[ \t'\u2019]*\Z")
WEAK_PHRASE = re.compile(
    r"(?<!\w)(?:bonjour|bonsoir|salut|merci|cher|chère|hi|hello|dear|thanks|thank you|3aslema|aslema|ahla"
    r"|ya|m3a|ena|مرحبا|شكرا)[ \t]*[:,!]?[ \t]*\Z"
)
# After `Mabrouk`, these make it "congratulations", not a name.
_BLESSING_AFTER = re.compile(
    r"[ \t]*,?[ \t]*(?:!|(?:3lik|3likom|3lih|3liha|3likoum|alik|alikom|3alik|3alikom)(?!\w))"
)
# What `M.` means after a number (`2 M. Dinars`, two million dinars): a unit, not an honorific.
_AMOUNT_UNITS = frozenset(
    {"dinars", "dinar", "dt", "tnd", "d", "euros", "euro", "eur", "dollars", "usd", "de"}
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
        "grâce", "sms", "cin", "rib", "iban", "tél", "tel", "fax", "dt", "tnd", "email", "mail", "pdf", "id",
    }
)  # fmt: skip
_CLITICS = "وفبلك"
_RUN_PARTICLES = PARTICLES | ABD_PREFIXES  # a run may start with `Abdel` even where it isn't glued
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
            self.honorific
            or (self.text in STOPWORDS and self.key not in NAMES)
            or self.text in ARABIC_STOP
            or (self.cap is not True and _fold_ar(self.text) in _CONTEXT_WORDS)
        )


def _fold_ar(text: str) -> str:
    """The Arabic spelling variants `_AR_VARIANTS` accepts, folded to one form."""
    return text.translate(_AR_FOLD)


_AR_FOLD = str.maketrans({c: group[0] for group in (_ALEFS, _TA_HEH, _YEHS) for c in group})
# The one-word cues and verbs of #50 never join a name run unless capitalized: `sa7bi` (my friend) folds to
# the name Sahbi, but in `sa7bi karim` it is the cue. `Sahbi Ben Salah` stays a name.
_CONTEXT_WORDS = frozenset(
    _fold_ar(normalize(w).text)
    for w in data.KIN_CUES + data.KIN_CUES_AR + data.SPEECH_CUES + data.VERBS_AFTER_NAME
    if " " not in w
)


# Arabic prepositions and pronouns, compared before folding: `على` ("on") would fold to `علي`, Ali.
ARABIC_STOP = frozenset({"على", "الى", "إلى", "في", "من", "عن", "مع", "هذا", "هذه", "هو", "هي"})


def strip_clitics(text: str, key: str | None) -> tuple[str | None, int]:
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
            key, clitic = strip_clitics(text, key)
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
        places = cities(doc)
        taken = bytearray(len(doc.text) + 1)  # 1 where a city is: overlap checks in O(word length)
        for s, e, _ in places:
            taken[s:e] = b"\x01" * (e - s)
        words = [w for w in _words(doc) if not any(taken[w.start : w.end])]
        never = _never_names(doc.text)
        spans = [to_span(doc, s, e, EntityType.LOCATION, score, self.name) for s, e, score in places]
        i = 0
        while i < len(words):
            if _cue_word(doc, words, i):
                i += 1  # `Sa7bi karim`, `Sa7bi, karim` at the start of a message: the cue, then the name
                continue
            end, cue = _run(doc, words, i, never)
            if end == i:
                i += 1
                continue
            run = words[i:end]
            if cue != "strong" and run[0].start + run[0].clitic in never and len(run) > 1:
                i += 1  # `3idek mabrouk Sami`: the blessing stays out, the name after it is tried on its own
                continue
            if cue != "strong" and (
                (len(run) == 1 and run[0].start + run[0].clitic in never)
                or (run[0].key == "mabrouk" and _BLESSING_AFTER.match(doc.text, run[0].end))
            ):
                # `Mabrouk !`, `ramdhan karim`: the ordinary word; a name after a strong cue (`Mme Mabrouk`)
                i = end
                continue
            if cue in ("", "weak") and _verb_after(doc, run):
                cue = "speech"
            start = run[0].start + run[0].clitic
            spans.append(to_span(doc, start, run[-1].end, EntityType.PERSON, _score(run, cue), self.name))
            i = end
        return spans


def _run(doc: NormalizedDoc, words: list[_Word], i: int, never: set[int]) -> tuple[int, str]:
    """The end (exclusive) of the name run starting at word `i` (`i` if none), and the cue before it. A run
    stops before the name word of a fixed expression (`never`): `Nour Amal kbir` is `Nour`; and an unlisted
    word that only opens one (`Ramdhan` in `Khouya, Ramdhan Karim`) is no name."""
    cue = _cue(doc, words, i)
    first = words[i]
    if first.stop:
        return i, ""
    cued = cue == "strong" and (first.cap or first.cap is None) and len(first.text) > 1
    if not (first.name or _particle(doc, words, i) or cued):
        return i, ""
    unlisted = 0 if first.name or first.text in _RUN_PARTICLES else 1  # `Abdel Kader …`: no name yet
    max_unlisted = 2 if cue == "strong" else 1
    j = i + 1
    while j < len(words) and j - i < MAX_RUN and _GAP.fullmatch(doc.text[words[j - 1].end : words[j].start]):
        nxt = words[j]
        if nxt.stop:
            break
        if nxt.start + nxt.clitic in never:
            if j == i + 1 and not first.name:
                return i, ""
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
    if j - i > 1 and words[j - 1].text in _RUN_PARTICLES:  # a trailing `Ben` with nothing after it
        j -= 1
    return j, cue


# Adjectives shaped like a family name (`الثاني`, "the second"): never taken as one.
NISBA_ADJECTIVES = frozenset(
    {
        "الثاني", "الثالث", "الشخصي", "الماضي", "الحالي", "العربي", "التالي", "الرسمي", "الوطني", "الدولي",
        "الجامعي", "الصحي", "المالي", "الاداري", "الإداري", "التجاري", "الاساسي", "الأساسي", "الحقيقي",
        "النهائي", "الدراسي", "الشهري", "السنوي", "اليومي", "الاسبوعي", "الأسبوعي", "المحلي", "الجهوي",
        "الفني", "التقني", "الطبي", "القانوني", "الاجتماعي", "الاقتصادي", "السياسي", "الامني", "الأمني",
        "الماضى", "الثانوي", "الابتدائي", "الإبتدائي", "العادي", "الضروري", "الكافي", "الباقي", "الجاي",
    }
)  # fmt: skip


def _nisba(text: str) -> bool:
    """An Arabic family name of the `الزغلامي`, `الطرابلسي` kind: the article, then 3+ letters ending in ي."""
    return (
        text.startswith("ال")
        and text.endswith("ي")
        and len(text) >= 6
        and text not in ARABIC_STOP
        and text not in NISBA_ADJECTIVES
    )


def _particle(doc: NormalizedDoc, words: list[_Word], i: int) -> bool:
    """A name particle (`ben`, `el`, `بن`) followed by a name or a capitalized word."""
    if words[i].text not in _RUN_PARTICLES or i + 1 >= len(words):
        return False
    nxt = words[i + 1]
    adjacent = _GAP.fullmatch(doc.text[words[i].end : nxt.start]) is not None
    glued = _key(words[i].text + " " + nxt.text)
    return adjacent and (nxt.name or glued in LAST or bool(nxt.cap))


def _cue(doc: NormalizedDoc, words: list[_Word], i: int) -> str:
    prev = words[i - 1] if i > 0 else None
    if prev is not None and prev.honorific and _HONORIFIC_GAP.fullmatch(doc.text[prev.end : words[i].start]):
        if prev.text == "m":
            # `M.` is a strong cue; `M` without its dot is too often something else (`M Bricolage`), and
            # between a number and a unit it means million (`2 M. Dinars`).
            j = prev.start - 1
            while j >= 0 and doc.text[j] in " \t":
                j -= 1
            if j >= 0 and doc.text[j].isdigit() and words[i].text in _AMOUNT_UNITS:
                return ""
            return "strong" if doc.text.startswith(".", prev.end) else ""
        return "strong" if prev.text in STRONG_HONORIFICS or prev.cap is None else "weak"
    before = doc.text[max(0, words[i].start - 30) : words[i].start]
    if STRONG_PHRASE.search(before):
        return "strong"
    if SPEECH_PHRASE.search(before):
        return "speech"
    if WEAK_PHRASE.search(before):
        return "weak"
    return ""


def _score(run: list[_Word], cue: str) -> float:
    listed = [w for w in run if w.name]
    common_only = bool(listed) and all(common(w.key, w.cap is not None) for w in listed)
    if cue == "strong":
        return 0.95
    if not listed:
        return 0.3  # an unlisted word after a weak cue, or a particle alone
    if cue == "speech":
        lowercase = all(w.cap is False for w in run)
        # in lowercase, only a Tunisian name: `chouft pierre`, `kallamt grace` are ordinary words
        return 0.3 if lowercase and not any(w.key in TUNISIAN for w in listed) else 0.9
    caps = [w.cap for w in run]
    if None in caps:  # Arabic script
        return 0.3 if common_only and len(run) == 1 and not cue else 0.8
    if all(caps):
        if cue == "weak":
            return 0.8
        if len(run) > 1:
            return 0.9
        if common_only:
            return 0.5 if run[0].initial else 0.6
        return 0.85
    # Partly capitalized (`Nicolas ben Salem`): the capitalized names vouch for the run
    if any(caps):
        if common_only:
            return 0.3
        return 0.7 if len(run) > 1 else 0.3
    # All lowercase: only Tunisian names count (see `TUNISIAN`)
    tunisian = [w for w in listed if w.key in TUNISIAN]
    if not tunisian or all(common(w.key, True) for w in tunisian):
        return 0.3
    if len(run) > 1:
        return 0.7
    return 0.6 if any(w.key in FIRST_TN for w in tunisian) else 0.3


_CUE_GAP = re.compile(r"[ \t]*[:,][ \t]*|[ \t]+|-")


def _cue_word(doc: NormalizedDoc, words: list[_Word], i: int) -> bool:
    """A capitalized one-word cue that folds to a listed name, right before a listed name: `Sa7bi karim`,
    `SA7BI, karim` (a phone capitalizes the first word). `Sahbi Ben Salah` stays a name: `Ben` is unlisted."""
    if i + 1 >= len(words) or words[i].cap is not True or _fold_ar(words[i].text) not in _CONTEXT_WORDS:
        return False
    nxt = words[i + 1]
    return nxt.name and _CUE_GAP.fullmatch(doc.text[words[i].end : nxt.start]) is not None


def _verb_after(doc: NormalizedDoc, run: list[_Word]) -> bool:
    """A verb agreeing with a person right after the run (`amal 9alli`), and no article on its first word
    (`el nour ja`, `النور جا`: the light came)."""
    first = run[0]
    if (
        first.text in ("el", "al")
        or first.text.startswith("ال")
        or _ARTICLE_BEFORE.search(doc.text, max(0, first.start - 6), first.start)
    ):
        return False
    return _VERB_AFTER.match(doc.text, run[-1].end) is not None


def _never_pattern(before: str, name: str, after: str) -> re.Pattern[str]:
    left = rf"(?<![^\W_])(?:{_alternation((before,))})[ \t]+" if before else r"(?<![^\W_])"
    right = rf"(?=[ \t]+(?:{_alternation((after,))})(?![^\W_]))" if after else r"(?![^\W_])"
    return re.compile(left + "(?P<n>" + _alternation((name,)) + ")" + right)


@functools.cache
def _never_patterns() -> tuple[re.Pattern[str], ...]:
    return tuple(_never_pattern(b, n, a) for b, n, a in data.NEVER_NAME_EXPRESSIONS)


def _never_names(text: str) -> set[int]:
    """Where a common-word name is the ordinary word of a fixed expression (normalized offsets): `ramdhan
    karim`, `3andi amal`. One linear pass per expression."""
    return {m.start("n") for p in _never_patterns() for m in p.finditer(text)}


# Latin letters and their accented forms: `Gabès`, `Gabes` and `GABES` are one city.
_ACCENTS = {"a": "aàâäá", "e": "eéèêë", "i": "iîïí", "o": "oôöó", "u": "uùûüú", "c": "cç", "y": "yÿ"}
_ARABIC_LEFT = "(?:(?<![^\\W_])|(?<=(?<![^\\W_])[وفبلك])|(?<=(?<![^\\W_])[وفبلك]{2}))"


def _letter(c: str) -> str:
    base = unicodedata.normalize("NFD", c)[0]
    return f"[{_ACCENTS[base]}]" if base in _ACCENTS else re.escape(c)


@functools.lru_cache(maxsize=1)
def _city_patterns() -> list[tuple[re.Pattern[str], bool]]:
    common = {unaccent(normalize(c).text) for c in data.COMMON_CITIES}
    out: list[tuple[re.Pattern[str], bool]] = []
    for city in sorted(set(data.CITIES), key=len, reverse=True):
        words = normalize(city).text.split()
        body = r"[ \t-]+".join("".join(map(_letter, w)) for w in words)
        is_common = unaccent(" ".join(words)) in common
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


def unaccent(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def cities(doc: NormalizedDoc) -> list[tuple[int, int, float]]:
    """Non-overlapping city matches, longest first: (normalized start, end, score). Linear in the text length
    times the number of patterns: each match is checked against a per-character mask, not earlier matches."""
    found: list[tuple[int, int, float]] = []
    taken = bytearray(len(doc.text) + 1)
    for pattern, is_common in _city_patterns():
        for m in pattern.finditer(doc.text):
            if any(taken[m.start() : m.end()]):
                continue
            o_start, _ = doc.to_original(m.start(), m.end())
            first = doc.original[o_start]
            if first.lower() != first.upper() and not first.isupper():
                continue  # Latin script needs its capital
            if is_common and _initial(doc.original, o_start):
                continue
            found.append((m.start(), m.end(), 0.8))
            taken[m.start() : m.end()] = b"\x01" * (m.end() - m.start())
    return sorted(found)
