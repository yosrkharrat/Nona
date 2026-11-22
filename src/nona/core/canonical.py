"""Canonical token sets for PERSON mentions: what entity linking compares (deep dive B, step 1; ADR-0004).

`Ahmed Ben Salem`, `M. Ben Salem`, `ben-salem` and `BENSALEM` must be comparable, so each mention becomes a
set of canonical tokens:

1. **Normalize** like the detectors do (NFKC, case folding, Eastern Arabic digits, no Arabic diacritics),
   then drop Latin accents (`Hélène` → `helene`) and unify Arabic letter variants (hamza forms of alef to
   bare alef, teh marbuta to heh, alef maqsura to yeh).
2. **Drop honorifics and kinship cues** (`M.`, `Mme`, `Dr`, `si`, `sidi`, `khalti`, `3amm`, `السيد`…). They
   say how someone is addressed, not who they are.
3. **Glue name particles to the next word**: `ben salem` → `bensalem`, `el abed` → `elabed`, and the same
   for `بن`. A particle is part of the surname, and people write it attached, spaced or hyphenated.
   `Abdel` and `Abd` glue before one of the names of God: `Abdel Kader`, `Abd El Kader` and `Abdelkader`
   are all `abdelkader`, while `Abdel Trabelsi` stays {abdel, trabelsi}. A
   word-initial `M'` is part of the name, not the honorific: `M'hamed` → `mhamed`.
4. **Arabizi digits** inside a word stand for Arabic letters: `3` (ع) and `2` (ء) are dropped, `7` → `h`,
   `9` → `k`, `5` → `kh`. So `3ali` → `ali` and `m7ammed` → `mhammed`.
5. **Spelling variants** map to one form, from a small table: `Mohammed`, `Muhammad`, `Med` → `mohamed`, and
   common Arabic-script names map to their Latin form (`سامي` → `sami`), so linking works across scripts.

The result is a `frozenset`, as linking only asks whether one set contains the other. An empty set means the
mention has nothing to link on (an honorific alone); the linker treats it as its own entity.
"""

from __future__ import annotations

import functools
import itertools
import re
import unicodedata
from collections.abc import Mapping

from nona.core.normalize import normalize

# Compared after normalization, with a trailing dot removed (`m.`, `dr.`).
HONORIFICS: frozenset[str] = frozenset(
    {
        # French and English
        "m", "mr", "mrs", "ms", "mme", "mlle", "miss", "monsieur", "madame", "mademoiselle", "dr", "docteur",
        "doctor", "pr", "prof", "professeur", "me", "maitre", "sir",
        # Tunisian, in Latin and Arabizi
        "si", "sidi", "lalla", "khalti", "khali", "3amm", "3ammi", "3am", "3ami", "hajj", "haj", "hajja",
        "el-hajj", "cheikh", "sheikh",
        # Arabic script
        "سي", "سيدي", "السيد", "السيده", "سيده", "الانسه", "الدكتور", "الدكتوره", "دكتور", "د", "الاستاذ",
        "الاستاذه", "استاذ", "خالتي", "خالي", "عمي", "الحاج", "الحاجه", "للا",
    }
)  # fmt: skip

PARTICLES: frozenset[str] = frozenset(
    {
        "ben",
        "bin",
        "ibn",
        "bent",
        "bint",
        "bou",
        "abou",
        "abu",
        "el",
        "al",
        "ould",
        "بن",
        "ابن",
        "بنت",
        "بو",
        "ابو",
    }
)

# Variant → canonical form. Latin keys are folded and unaccented; Arabic keys use the letters of step 1.
VARIANTS: Mapping[str, str] = {
    **dict.fromkeys(
        ["mohammed", "mohamad", "mohammad", "muhammad", "muhammed", "mouhamed", "med", "mhd"], "mohamed"
    ),
    **dict.fromkeys(["ahmad", "ahmet"], "ahmed"),
    **dict.fromkeys(["youssouf", "yousef", "yusuf", "youcef", "yousuf"], "youssef"),
    **dict.fromkeys(["mahmud", "mahmood"], "mahmoud"),
    **dict.fromkeys(["abdullah", "abdellah", "abdalla"], "abdallah"),
    **dict.fromkeys(["fatima", "fatema"], "fatma"),
    **dict.fromkeys(["khedija", "khadidja"], "khadija"),
    **dict.fromkeys(["mariam", "meriem", "maryam", "meryem", "myriam"], "mariem"),
    **dict.fromkeys(["samy"], "sami"),
    **dict.fromkeys(["hichem"], "hicham"),
    **dict.fromkeys(["walid"], "oualid"),
    **dict.fromkeys(["noor"], "nour"),
    **dict.fromkeys(["moncef"], "monsef"),
    # Tunisian spellings of one name, both in the gazetteer lists
    # Not `Rim`/`Rym`: `rim` (gazelle) is a common word and `Rym` isn't, and merged keys share that status.
    **dict.fromkeys(["cyrine"], "syrine"),
    **dict.fromkeys(["yacine", "yasine"], "yassine"),
    **dict.fromkeys(["sofien", "sofiane"], "sofiene"),
    **dict.fromkeys(["jlassi"], "jelassi"),
    **dict.fromkeys(["laabidi"], "labidi"),
    **dict.fromkeys(["chedli"], "chedly"),
    **dict.fromkeys(["ouassim"], "wassim"),
    **dict.fromkeys(["qais", "kays"], "kais"),
    **dict.fromkeys(["mahdi"], "mehdi"),
    **dict.fromkeys(["amel"], "amal"),
    # Arabic script → Latin
    "محمد": "mohamed",
    "احمد": "ahmed",
    "سامي": "sami",
    "علي": "ali",
    "يوسف": "youssef",
    "محمود": "mahmoud",
    "فاطمه": "fatma",
    "خديجه": "khadija",
    "مريم": "mariem",
    "امال": "amal",
    "نور": "nour",
    "ليلي": "leila",
    "سلمي": "salma",
    "هشام": "hicham",
    "وليد": "oualid",
    "عبدالله": "abdallah",
    "منصف": "monsef",
    "رحمه": "rahma",
    "سالم": "salem",
    "بنسالم": "bensalem",
    "عمر": "omar",
    "حمزه": "hamza",
    "كريم": "karim",
    "مهدي": "mehdi",
    "امين": "amine",
    "ياسين": "yassine",
    "خالد": "khaled",
    "نبيل": "nabil",
    "طارق": "tarek",
    "صالح": "salah",
    "سليم": "selim",
    "عايشه": "aicha",
    "زينب": "zeineb",
    "ايناس": "ines",
    "ريم": "rym",
    "اسماء": "asma",
    "هدي": "houda",
    "ساره": "sarra",
    "مني": "mouna",
    "ناديه": "nadia",
    "امل": "amal",
    "ايوب": "ayoub",
    "الطرابلسي": "trabelsi",
    "الغربي": "gharbi",
    "الجلاصي": "jelassi",
    "المنصوري": "mansouri",
    "بنعلي": "benali",
    "بنعمر": "benamor",
    "الخليفي": "khelifi",
    "الماجري": "mejri",
}  # fmt: skip

_APOSTROPHES = "'\u2019"  # straight and typographic
_WORD = re.compile(rf"[\w{_APOSTROPHES}-]+")
_ARABIC_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي"})
_ARABIZI = {"3": "", "2": "", "7": "h", "9": "k", "5": "kh"}


def _fold(text: str) -> str:
    """Detector normalization, then no Latin accents and unified Arabic letters."""
    decomposed = unicodedata.normalize("NFD", normalize(text).text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped).translate(_ARABIC_LETTERS)


def _de_arabizi(word: str) -> str:
    """`3ali` → `ali`, `m7ammed` → `mhammed`. Only for words that mix letters and digits."""
    if word.isdigit() or not any(c.isdigit() for c in word):
        return word
    return "".join(_ARABIZI.get(c, c) for c in word if not (c.isdigit() and c not in _ARABIZI))


_M_ELISION = re.compile(rf"(?<![\w{_APOSTROPHES}])m[{_APOSTROPHES}](?=\w)")  # `M'hamed`, `M'barek`


def _words(surface: str) -> list[str]:
    """Folded words, split on spaces, hyphens and apostrophes, honorifics and bare numbers removed.

    A word-initial `M'` is part of the name, not the honorific `M`: `M'hamed` → `mhamed`, `M'barek` →
    `mbarek`.
    """
    out: list[str] = []
    for raw in _WORD.findall(_M_ELISION.sub("m", _fold(surface))):
        if raw.strip("-" + _APOSTROPHES) in HONORIFICS:
            continue
        out.extend(p for p in re.split(rf"[-{_APOSTROPHES}]", raw) if p)
    return [w for w in out if w not in HONORIFICS and not w.isdigit()]


# `Abd` ("servant of") + one of the names of God makes one first name: `Abdel Kader`, `Abd El Aziz`. `Abdel`
# alone is also a first name (`Abdel Trabelsi`), so it only glues before one of these.
ABD_PREFIXES: frozenset[str] = frozenset({"abd", "abdel", "abdal", "abdul", "abdou"})
DIVINE_NAMES: frozenset[str] = frozenset(
    {
        "kader", "qader", "kadir", "aziz", "karim", "rahman", "rahim", "jalil", "majid", "wahed", "wahab",
        "latif", "hamid", "hakim", "malek", "razak", "razzak", "basset", "monem", "fattah", "ghani", "jabbar",
        "kabir", "halim", "hafidh", "hafid", "salem", "razzek", "moula", "naceur", "nasser", "wahid", "allah",
        "lah", "ali",
    }
)  # fmt: skip


def _abd_glue(words: list[str], i: int) -> bool:
    """Whether word `i` is an `Abd` prefix that belongs to the next name (`abdel kader`, `abd el aziz`)."""
    if words[i] not in ABD_PREFIXES or i + 1 >= len(words):
        return False
    nxt = words[i + 1]
    if nxt in {"el", "al"} and i + 2 < len(words):
        nxt = words[i + 2]
    return nxt in DIVINE_NAMES


@functools.lru_cache(maxsize=8192)
def _glued(surface: str) -> tuple[tuple[str, str, str], ...]:
    """`(token, particles, word)` per token of a mention: `ben salem` → `("bensalem", "ben", "salem")`.

    Cached: linking asks for the same surface forms after every message (owners, `canonical_tokens`)."""
    words = [w for w in map(_de_arabizi, _words(surface)) if w and w not in HONORIFICS]
    out: list[tuple[str, str, str]] = []
    glue = ""
    for i, word in enumerate(words):
        if (word in PARTICLES and i + 1 < len(words)) or _abd_glue(words, i):
            glue += word
            continue
        token = VARIANTS.get(glue + word, glue + word)
        if token not in HONORIFICS:
            out.append((token, glue, word))
        glue = ""
    return tuple(out)


def canonical_tokens(surface: str) -> frozenset[str]:
    """The canonical token set of one PERSON mention."""
    return frozenset(token for token, _, _ in _glued(surface))


def glued_parts(surface: str) -> dict[str, tuple[str, str]]:
    """Tokens glued from a particle and a word: `Ben Salem` → {"bensalem": ("ben", "salem")}.

    Only these may be written without their particle in a handle (`ahmed.salem`). A name that merely starts
    like a particle (`Elyes`, `Boukadida`, `Benjamin`) is not split.
    """
    return {token: (glue, word) for token, glue, word in _glued(surface) if glue}


# A word as written: letters with their combining marks (NFD accents, Arabic harakat and tatweel), and
# apostrophe elisions kept whole (`O'Brien`). Hyphens separate words (`Jean-Pierre`, `Ben-Salem`).
_MARKS = "\u0300-\u036f\u0610-\u061a\u0640\u064b-\u065f\u0670"
_NAME_WORD = re.compile(rf"[\w{_MARKS}]+(?:[{_APOSTROPHES}][\w{_MARKS}]+)*")


def name_words(surface: str) -> list[str]:
    """Words of a PERSON mention that name the same person on their own, as written (ADR-0004, look-ahead).

    `Chirine` and `Boukadida` in `Mme Chirine Boukadida`. A word qualifies when its own token set is inside
    the mention's, so honorifics, particles and the half of a glued surname (`Salem` in `Ben Salem` is not
    `bensalem`) don't, nor half of an elision (`Brien` in `O'Brien`). Skipped: words with fewer than 3
    letters, all-lowercase Latin words (a lowercase `amal` is as likely the word "hope"), and anything with a
    digit or an underscore (a typed placeholder such as `PERSON_1`).
    """
    whole = canonical_tokens(surface)
    out: list[str] = []
    for word in _NAME_WORD.findall(surface):
        letters = sum(c.isalpha() for c in word)
        if letters < 3 or word.islower() or word == surface.strip():
            continue
        if "_" in word or any(c.isdigit() for c in word):
            continue
        tokens = canonical_tokens(word)
        if tokens and tokens <= whole and word not in out:
            out.append(word)
    return out


def compatible(a: frozenset[str], b: frozenset[str]) -> bool:
    """Two non-empty token sets that one contains the other (ADR-0004, step 4)."""
    return bool(a) and bool(b) and (a <= b or b <= a)


# --------------------------------------------------------------------------- handles (deep dive B, step 6)

_HANDLE_SEPARATORS = re.compile(r"[._\-]+")
_MIN_FULL = 3  # a handle must spell out at least one name part this long, not only initials
_MAX_PIECES = 6  # the superset search tries every subset of pieces: bounded, so no input can blow it up
_MAX_LETTERS = 64
_DIGITS = "0123456789"

Splits = dict[str, tuple[str, str]]  # glued token → (particles, word), from `glued_parts`


def _handle_pieces(handle: str) -> tuple[str, ...]:
    """Letters of each piece of a local part or handle.

    A `+tag` is dropped (`ahmed+devis`). Runs of 2+ digits at either end are a year or a counter (`sami92`,
    `92sami`), not Arabizi letters; a single leading digit is (`3ali`, `7amza`). Other digits are read as
    Arabizi or dropped.
    """
    local = handle.lstrip("@").split("+", 1)[0]
    out: list[str] = []
    for piece in _HANDLE_SEPARATORS.split(local):
        folded = _fold(piece).rstrip(_DIGITS)
        head = len(folded) - len(folded.lstrip(_DIGITS))
        if head >= 2:
            folded = folded[head:]
        letters = "".join(c for c in _de_arabizi(folded) if c.isalpha())
        if letters:
            out.append(letters)
    return tuple(out)


@functools.lru_cache(maxsize=4096)
def _units(token: str, split: tuple[str, str] | None) -> tuple[frozenset[str], frozenset[str]]:
    """How a handle may write one canonical token: in full, and as initials.

    Full: the token and its spelling variants (`med` for `mohamed`); for a glued surname, also the word after
    the particle (`salem` for `bensalem`). Initials: the first letter; for a glued surname, also the initials
    of its parts (`bs`) and of the word (`s`).
    """
    full = {token, *(variant for variant, canonical in VARIANTS.items() if canonical == token)}
    initials = {token[0]}
    if split is not None:
        glue, word = split
        full.add(word)
        initials |= {glue[0] + word[0], word[0]}
    return frozenset(full), frozenset(initials)


def _spells(letters: str, units: list[tuple[frozenset[str], frozenset[str]]], every_token: bool) -> bool:
    """`letters` split entirely into units, each token at most once (all of them if `every_token`), with at
    least one name part written in full."""
    everything = (1 << len(units)) - 1

    @functools.cache
    def fits(pos: int, used: int, spelled: bool) -> bool:
        if pos == len(letters):
            return spelled and (used == everything or not every_token)
        for i, (full, initials) in enumerate(units):
            if used & (1 << i):
                continue
            for unit in full:
                if letters.startswith(unit, pos) and fits(
                    pos + len(unit), used | (1 << i), spelled or len(unit) >= _MIN_FULL
                ):
                    return True
            for unit in initials:
                if letters.startswith(unit, pos) and fits(pos + len(unit), used | (1 << i), spelled):
                    return True
        return False

    return fits(0, 0, False)


_FrozenSplits = tuple[tuple[str, tuple[str, str]], ...]


def _freeze(tokens: frozenset[str], splits: Splits | None) -> _FrozenSplits:
    """The splits that matter for these tokens, hashable, so results can be cached."""
    return tuple(sorted((t, s) for t, s in (splits or {}).items() if t in tokens))


def _prepare(
    handle: str, tokens: frozenset[str], splits: _FrozenSplits
) -> tuple[tuple[str, ...], list[tuple[frozenset[str], frozenset[str]]]] | None:
    pieces = _handle_pieces(handle)
    ordered = sorted(tokens)
    if not pieces or not ordered or len(ordered) > 8 or sum(map(len, pieces)) > _MAX_LETTERS:
        return None
    by_token = dict(splits)
    return pieces, [_units(t, by_token.get(t)) for t in ordered]


def handle_matches(
    handle: str, tokens: frozenset[str], splits: Splits | None = None, *, superset: bool = True
) -> bool:
    """Whether a handle spells a person's tokens or initials: `ahmed.bs`, `abensalem`, `a.bensalem` for
    {ahmed, bensalem} (with `splits` from `glued_parts`, which say `bensalem` is `ben` + `salem`).

    The handle's letters must split entirely into units, each token used at most once, in any order, with at
    least one name part written in full (3+ letters). Separators, a `+tag` and digit runs that aren't Arabizi
    letters are ignored (`sami92`).

    With `superset`, a handle may also say more than the name known so far, as linking allows: separated
    pieces that aren't the person's can be left out, if the other pieces cover **every** token. So
    `wassim.ferchichi` fits someone known as `Wassim`, while `sami.jlassi` doesn't fit `Sami Trabelsi`. At
    most 6 pieces are searched that way.

    Not cached across calls: the cache would keep email handles from every conversation in process memory.
    The linker calls it once per (email, person version) instead (`core/linking.py`).
    """
    prepared = _prepare(handle, tokens, _freeze(tokens, splits))
    if prepared is None:
        return False
    pieces, units = prepared
    if _spells("".join(pieces), units, every_token=False):
        return True
    if not superset or len(pieces) > _MAX_PIECES:
        return False
    for kept in itertools.product((True, False), repeat=len(pieces)):
        if any(kept) and not all(kept):
            letters = "".join(p for p, keep in zip(pieces, kept, strict=True) if keep)
            if _spells(letters, units, every_token=True):
                return True
    return False


def handle_names(handle: str, tokens: frozenset[str], splits: Splits | None = None) -> bool:
    """Whether one separated piece of the handle is a name part of this person, written in full."""
    prepared = _prepare(handle, tokens, _freeze(tokens, splits))
    if prepared is None:
        return False
    pieces, units = prepared
    return any(piece in full and len(piece) >= _MIN_FULL for piece in pieces for full, _ in units)
