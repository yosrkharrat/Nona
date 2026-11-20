"""Leak and over-masking metrics (docs/design/05-dataset.md, section 6; 04 §F).

Everything here compares a turn's gold labels with the text that system sent upstream for that turn. It
never looks inside a system, so baselines and Nona are measured the same way, and in S5 the same code
measures what actually goes over the wire (FR-21).
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from nona.bench.dataset import Conversation
from nona.core import EntityType

# The headline leak rate covers direct identifiers only (05 §4). LOCATION and ORGANIZATION are reported apart.
DIRECT_IDENTIFIERS = frozenset(
    {
        EntityType.PERSON, EntityType.EMAIL, EntityType.PHONE, EntityType.IBAN, EntityType.RIB,
        EntityType.CARD, EntityType.NATIONAL_ID, EntityType.IP_ADDRESS,
    }
)  # fmt: skip
# Compared on digits and letters only, so `22.123.456` and `22123456` are the same value (05 §6).
COMPACT_TYPES = frozenset(
    {EntityType.PHONE, EntityType.IBAN, EntityType.RIB, EntityType.CARD, EntityType.NATIONAL_ID}
)
# Tokens that don't identify anyone on their own, so seeing them alone is not a partial leak:
# honorifics and kinship words (deep dive A), name particles, and phone country codes.
NON_IDENTIFYING = frozenset(
    {
        "mme", "mlle", "mrs", "miss", "si", "sidi", "khalti", "3amm", "ben", "bin", "bent", "ibn", "bou",
        "abou", "des", "van", "von", "216", "00216", "0033",
    }
)  # fmt: skip
PROCLITICS = frozenset("وفبلك")  # Arabic letters that attach to the next word: لسامي = ل + سامي (05 §4)

_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_WORD = re.compile(r"\w+")
_TOKEN = re.compile(r"\w+|[^\w\s]")
_COUNTRY_CODES = ("216", "33")  # Tunisia, France: the phone formats the dataset uses (05 §5)
_IBAN_HEAD = re.compile(
    r"[a-z]{2}\d{2}"
)  # `tn59`: country and check digits, shared by every IBAN of a country


# --------------------------------------------------------------------------- normalization


def normalize(text: str) -> str:
    """NFKC, casefold, Eastern Arabic digits to ASCII, collapse whitespace."""
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_EASTERN_DIGITS)
    return " ".join(folded.split())


def compact(text: str) -> str:
    """Digits and letters only, after `normalize`."""
    return "".join(c for c in normalize(text) if c.isalnum())


def _national(phone: str) -> str:
    """The national significant number: `+216 22 123 456` → `22123456`, `+33 6 12…` and `06 12…` → `612…`."""
    digits = compact(phone)
    if digits.startswith("00"):
        digits = digits[2:]
    for country in _COUNTRY_CODES:
        if digits.startswith(country) and len(digits) - len(country) >= 8:
            digits = digits[len(country) :]
            break
    return digits.removeprefix("0")


def _value_needles(text: str, etype: EntityType) -> set[str]:
    """Compact forms of a structured value that count as the whole value leaking."""
    needles = {compact(text)}
    if etype is EntityType.PHONE:
        needles.add(_national(text))  # the number without its country code still reaches the person
    return {n for n in needles if n}


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _left_boundary(haystack: str, i: int, needle: str) -> bool:
    if not _is_word_char(needle[0]):
        return True
    for attached in range(3):  # the needle may carry 0, 1 or 2 Arabic proclitics: لسامي, وبسامي
        j = i - attached
        if j < 0:
            return False
        if j == 0 or not _is_word_char(haystack[j - 1]):
            return all(c in PROCLITICS for c in haystack[j:i])
    return False


def contains_word(haystack: str, needle: str) -> bool:
    """`needle` occurs in `haystack` at word boundaries: `sami` is not found in `samir`, but is in `لسامي`."""
    if not needle:
        return False
    i = haystack.find(needle)
    while i != -1:
        end = i + len(needle)
        right_ok = not _is_word_char(needle[-1]) or end == len(haystack) or not _is_word_char(haystack[end])
        if right_ok and _left_boundary(haystack, i, needle):
            return True
        i = haystack.find(needle, i + 1)
    return False


# --------------------------------------------------------------------------- leaks


class Outcome(StrEnum):
    LEAKED = "leaked"  # the value reached the upstream
    PARTIAL = "partial"  # not the value, but an identifying part of it (e.g. the surname)
    PROTECTED = "protected"


def identifying_tokens(text: str, etype: EntityType) -> list[str]:
    """Parts of a value that would count as a partial leak: 3+ characters and identifying on their own."""
    if etype is EntityType.EMAIL:
        text = text.split("@", 1)[0]  # `example.com` identifies no one
    tokens = [t for t in _WORD.findall(normalize(text)) if len(t) >= 3 and t not in NON_IDENTIFYING]
    if etype is EntityType.IBAN:
        tokens = [t for t in tokens if not _IBAN_HEAD.fullmatch(t)]
    return tokens


def classify(mention_text: str, etype: EntityType, payload: str) -> Outcome:
    """Did this gold mention reach the upstream, in full, in part, or not at all? (05 §6)"""
    norm_payload = normalize(payload)
    if etype in COMPACT_TYPES:
        compact_payload = compact(payload)
        if any(needle in compact_payload for needle in _value_needles(mention_text, etype)):
            return Outcome.LEAKED
    elif contains_word(norm_payload, normalize(mention_text)):
        return Outcome.LEAKED
    if any(contains_word(norm_payload, t) for t in identifying_tokens(mention_text, etype)):
        return Outcome.PARTIAL
    return Outcome.PROTECTED


# --------------------------------------------------------------------------- over-masking


def changed_ranges(original: str, payload: str) -> list[tuple[int, int]]:
    """Ranges of `original` that the system replaced or removed, found by aligning tokens.

    Black-box on purpose: it works for any system, including baselines that don't report offsets and the
    on-the-wire capture in S5. A placeholder like `PERSON_1` is one token, so it aligns cleanly.
    """
    a = list(_TOKEN.finditer(original))
    b = _TOKEN.findall(payload)
    matcher = difflib.SequenceMatcher(None, [m.group() for m in a], b, autojunk=False)
    return [
        (a[i1].start(), a[i2 - 1].end())
        for tag, i1, i2, _, _ in matcher.get_opcodes()
        if tag in ("replace", "delete")
    ]


def _overlaps(start: int, end: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start < r_end and r_start < end for r_start, r_end in ranges)


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class Rate:
    hits: int
    total: int

    @property
    def value(self) -> float | None:
        return self.hits / self.total if self.total else None

    def __str__(self) -> str:
        value = self.value
        return "—" if value is None else f"{100 * value:.1f}% ({self.hits}/{self.total})"

    def __add__(self, other: Rate) -> Rate:
        return Rate(self.hits + other.hits, self.total + other.total)


@dataclass(frozen=True, slots=True)
class MentionResult:
    conversation: str
    turn: int
    role: str
    language: str  # the conversation's languages joined with "+", e.g. "aeb-Latn+fr"
    type: EntityType
    public: bool
    text: str
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class ConversationResult:
    mentions: tuple[MentionResult, ...]
    masked_tokens: Rate  # non-PII word tokens the system replaced
    masked_distractors: Rate  # distractors the system replaced


def evaluate(conversation: Conversation, payloads: Sequence[str]) -> ConversationResult:
    if len(payloads) != len(conversation.turns):
        raise ValueError(
            f"{conversation.id}: system returned {len(payloads)} payloads for {len(conversation.turns)} turns"
        )
    language = "+".join(conversation.languages)
    mentions: list[MentionResult] = []
    tokens = Rate(0, 0)
    distractors = Rate(0, 0)
    for i, (turn, payload) in enumerate(zip(conversation.turns, payloads, strict=True)):
        for m in turn.mentions:
            entity = conversation.entity(m.entity)
            outcome = classify(m.text, entity.type, payload)
            mentions.append(
                MentionResult(
                    conversation.id, i, turn.role, language, entity.type, entity.public, m.text, outcome
                )
            )
        changed = changed_ranges(turn.text, payload)
        gold = [(m.start, m.end) for m in turn.mentions]
        for w in _WORD.finditer(turn.text):
            if not _overlaps(w.start(), w.end(), gold):
                tokens += Rate(int(_overlaps(w.start(), w.end(), changed)), 1)
        for d in turn.distractors:
            distractors += Rate(int(_overlaps(d.start, d.end, changed)), 1)
    return ConversationResult(tuple(mentions), tokens, distractors)


@dataclass(frozen=True, slots=True)
class Report:
    """Everything one system scored on one split. Rates are over mentions (05 §6)."""

    mentions: tuple[MentionResult, ...]
    masked_tokens: Rate
    masked_distractors: Rate

    @classmethod
    def from_results(cls, results: Iterable[ConversationResult]) -> Report:
        mentions: list[MentionResult] = []
        tokens = Rate(0, 0)
        distractors = Rate(0, 0)
        for r in results:
            mentions.extend(r.mentions)
            tokens += r.masked_tokens
            distractors += r.masked_distractors
        return cls(tuple(mentions), tokens, distractors)

    def select(
        self,
        *,
        types: Iterable[EntityType] = DIRECT_IDENTIFIERS,
        assistant: bool = False,
        language: str | None = None,
    ) -> list[MentionResult]:
        """Mentions in scope: given types, public figures excluded, user/system/tool or assistant turns."""
        wanted = frozenset(types)
        return [
            m
            for m in self.mentions
            if m.type in wanted
            and not m.public
            and (m.role == "assistant") == assistant
            and (language is None or m.language == language)
        ]

    @staticmethod
    def rate(mentions: Sequence[MentionResult], outcome: Outcome = Outcome.LEAKED) -> Rate:
        return Rate(sum(m.outcome is outcome for m in mentions), len(mentions))

    @property
    def headline(self) -> Rate:
        """Leak rate on direct identifiers in user, system and tool turns. Public figures excluded."""
        return self.rate(self.select())

    @property
    def partial(self) -> Rate:
        return self.rate(self.select(), Outcome.PARTIAL)

    @property
    def assistant(self) -> Rate:
        """Leak rate on restored assistant turns that the client sends back (ADR-0004)."""
        return self.rate(self.select(assistant=True))

    @property
    def location_organization(self) -> Rate:
        return self.rate(self.select(types=(EntityType.LOCATION, EntityType.ORGANIZATION)))

    @property
    def languages(self) -> list[str]:
        return sorted({m.language for m in self.mentions})

    @property
    def types(self) -> list[EntityType]:
        present = {m.type for m in self.mentions}
        return [t for t in EntityType if t in present]
