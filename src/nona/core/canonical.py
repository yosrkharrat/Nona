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
4. **Arabizi digits** inside a word stand for Arabic letters: `3` (ع) and `2` (ء) are dropped, `7` → `h`,
   `9` → `k`, `5` → `kh`. So `3ali` → `ali` and `m7ammed` → `mhammed`.
5. **Spelling variants** map to one form, from a small table: `Mohammed`, `Muhammad`, `Med` → `mohamed`, and
   common Arabic-script names map to their Latin form (`سامي` → `sami`), so linking works across scripts.

The result is a `frozenset`, as linking only asks whether one set contains the other. An empty set means the
mention has nothing to link on (an honorific alone); the linker treats it as its own entity.
"""

from __future__ import annotations

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


def _words(surface: str) -> list[str]:
    """Folded words, split on spaces, hyphens and apostrophes, honorifics and bare numbers removed."""
    out: list[str] = []
    for raw in _WORD.findall(_fold(surface)):
        if raw.strip("-" + _APOSTROPHES) in HONORIFICS:
            continue
        out.extend(p for p in re.split(rf"[-{_APOSTROPHES}]", raw) if p)
    return [w for w in out if w not in HONORIFICS and not w.isdigit()]


def canonical_tokens(surface: str) -> frozenset[str]:
    """The canonical token set of one PERSON mention."""
    words = [w for w in map(_de_arabizi, _words(surface)) if w and w not in HONORIFICS]
    tokens: list[str] = []
    glue = ""
    for i, word in enumerate(words):
        if word in PARTICLES and i + 1 < len(words):
            glue += word
            continue
        tokens.append(glue + word)
        glue = ""
    return frozenset(VARIANTS.get(t, t) for t in tokens) - HONORIFICS


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
