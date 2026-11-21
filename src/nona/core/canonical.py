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


# --------------------------------------------------------------------------- handles (deep dive B, step 6)

_HANDLE_SEPARATORS = re.compile(r"[._+\-]+")
_MIN_FULL = 3  # a handle must spell out at least one name part this long, not only initials


def _handle_pieces(handle: str) -> list[str]:
    """Letters of each piece of a local part or handle: Arabizi digits read, other digits dropped."""
    out: list[str] = []
    for piece in _HANDLE_SEPARATORS.split(handle.lstrip("@")):
        # Trailing digits are a year or a counter (`sami92`), not Arabizi letters (`9` would read as `k`).
        letters = "".join(c for c in _de_arabizi(_fold(piece).rstrip("0123456789")) if c.isalpha())
        if letters:
            out.append(letters)
    return out


def _units(token: str) -> tuple[frozenset[str], frozenset[str]]:
    """How a handle may write one canonical token: in full, and as initials.

    Full: the token, its spelling variants (`med` for `mohamed`) and, for a glued surname, the part after the
    particle (`salem` for `bensalem`). Initials: the first letter, and for a glued surname the initials of its
    parts (`bs`) or of the part after the particle (`s`).
    """
    full = {token, *(variant for variant, canonical in VARIANTS.items() if canonical == token)}
    initials = {token[0]}
    for particle in PARTICLES:
        rest = token.removeprefix(particle)
        if rest != token and len(rest) >= _MIN_FULL:
            full.add(rest)
            initials |= {particle[0] + rest[0], rest[0]}
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


def handle_matches(handle: str, tokens: frozenset[str]) -> bool:
    """Whether a handle spells a person's tokens or initials: `ahmed.bs`, `abensalem`, `a.bensalem` for
    {ahmed, bensalem}.

    The handle's letters must split entirely into units, each token used at most once, in any order, with at
    least one name part written in full (3+ letters). Separators and digits that aren't Arabizi letters are
    ignored (`sami92`).

    A handle may also say more than the name known so far, as linking allows a superset: separated pieces
    that aren't the person's can be left out, if the other pieces cover **every** token. So
    `wassim.ferchichi` fits someone known as `Wassim`, while `sami.jlassi` doesn't fit `Sami Trabelsi`.
    """
    pieces = _handle_pieces(handle)
    ordered = sorted(tokens)
    if not pieces or not ordered or len(ordered) > 8:
        return False
    units = [_units(t) for t in ordered]
    if _spells("".join(pieces), units, every_token=False):
        return True
    for kept in itertools.product((True, False), repeat=len(pieces)):
        if any(kept) and not all(kept):
            letters = "".join(p for p, keep in zip(pieces, kept, strict=True) if keep)
            if _spells(letters, units, every_token=True):
                return True
    return False
