"""Leak, over-masking and linking metrics (docs/design/05-dataset.md, section 6; 04 §F; ADR-0004).

Everything here compares a turn's gold labels with the text that system sent upstream for that turn. It
never looks inside a system, so baselines and Nona are measured the same way, and in S5 the same code
measures what actually goes over the wire (FR-21).
"""

from __future__ import annotations

import difflib
import itertools
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from nona.bench.dataset import Conversation
from nona.core import EntityType
from nona.core.canonical import canonical_tokens
from nona.core.recognizers.gazetteer import NAMES as GAZETTEER_NAMES

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
    """Drop format characters (Unicode Cf: zero-width spaces and joiners, soft hyphens), NFKC, casefold,
    Eastern Arabic digits to ASCII, collapse whitespace.

    Format characters are invisible, so `Sa\u200bmi` reads as `Sami` to a model: a system that inserts them
    hasn't protected anything. Homoglyphs (a Cyrillic a, U+0430, in `Sami`) are not caught: NFKC doesn't map
    them, and folding confusables would also merge real Cyrillic or Greek text. 05 §6 documents that limit.
    """
    visible = "".join(c for c in text if unicodedata.category(c) != "Cf")
    folded = unicodedata.normalize("NFKC", visible).casefold().translate(_EASTERN_DIGITS)
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
    OBFUSCATED = "obfuscated"  # a name spelled out with separators (`S a m i`, `S.a.m.i`): still a leak
    PARTIAL = "partial"  # not the value, but an identifying part of it (e.g. the surname)
    PROTECTED = "protected"


# What the leak rate counts: a name a model can read back, however it is spelled.
LEAKS = frozenset({Outcome.LEAKED, Outcome.OBFUSCATED})


def _spread_out(name: str) -> re.Pattern[str]:
    """A name's letters with any non-letters between them, at word boundaries: `S a m i`, `S.a.m.i`."""
    letters = [c for c in name if c.isalnum()]
    return re.compile(r"(?<!\w)" + r"[\W_]*".join(map(re.escape, letters)) + r"(?!\w)")


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
    if etype is EntityType.PERSON and _spread_out(normalize(mention_text)).search(norm_payload):
        return Outcome.OBFUSCATED
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
    b = [t.casefold() for t in _TOKEN.findall(payload)]
    # Case-insensitive: a system that only lowercases a word hasn't masked it (05 §6).
    matcher = difflib.SequenceMatcher(None, [m.group().casefold() for m in a], b, autojunk=False)
    return [
        (a[i1].start(), a[i2 - 1].end())
        for tag, i1, i2, _, _ in matcher.get_opcodes()
        if tag in ("replace", "delete")
    ]


def _overlaps(start: int, end: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start < r_end and r_start < end for r_start, r_end in ranges)


# --------------------------------------------------------------------------- placeholders


def placeholders(original: str, payload: str, spans: Sequence[tuple[int, int]]) -> list[str | None]:
    """What replaced each span of `original` in `payload`, read from the text alone (04 §F).

    Same token alignment as `changed_ranges`. Inside each replaced stretch:
    - as many tokens on both sides: paired one to one (`Sami Kais` → `PERSON_1 PERSON_2`);
    - otherwise the stretch is cut into runs (each span, and the words between them). One payload token per
      run, or one per span, pairs them in order (`M. Sami` → `Monsieur PERSON_1`, `Sami Ben Ali` →
      `PERSON_1 PERSON_2`). With a single span, the payload tokens left after one per neighbouring word go
      to the span, at least one (`Ahmed Ben Salem` → `PERSON_1`).
    - anything else covering two spans can't be attributed: both get `None`.

    `None` also when nothing replaced the span (it leaked in full, or was deleted). A partly replaced span
    keeps the replaced part, so a partial leak can still carry a placeholder: `Mme Trabelsi` →
    `Mme PERSON_1` gives `PERSON_1`.
    """
    a = list(_TOKEN.finditer(original))
    b = _TOKEN.findall(payload)
    owner: list[int | None] = []
    for tok in a:
        hit = [k for k, (start, end) in enumerate(spans) if tok.start() < end and start < tok.end()]
        owner.append(hit[0] if hit else None)
    parts: list[list[str]] = [[] for _ in spans]
    lost = [False] * len(spans)
    matcher = difflib.SequenceMatcher(None, [m.group() for m in a], b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        replacement = b[j1:j2]
        if i2 - i1 == j2 - j1:
            for i, token in zip(range(i1, i2), replacement, strict=True):
                if (k := owner[i]) is not None:
                    parts[k].append(token)
            continue
        runs = [k for k, _ in itertools.groupby(owner[i1:i2])]
        named = [k for k in runs if k is not None]
        if len(replacement) == len(runs):
            for k, token in zip(runs, replacement, strict=True):
                if k is not None:
                    parts[k].append(token)
        elif len(replacement) == len(named):
            for k, token in zip(named, replacement, strict=True):
                parts[k].append(token)
        elif len(named) == 1:
            k = named[0]
            inside = [i for i in range(i1, i2) if owner[i] == k]
            before, after = inside[0] - i1, i2 - 1 - inside[-1]
            width = max(1, len(replacement) - before - after)
            first = min(before, len(replacement) - width)
            parts[k].extend(replacement[first : first + width])
        else:
            for k in named:
                lost[k] = True
    return [None if lost[k] or not parts[k] else " ".join(parts[k]) for k in range(len(spans))]


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
    entity: str = ""
    ambiguous: bool = False
    placeholder: str | None = None  # what replaced the mention in the payload, if anything (`placeholders`)
    listed: bool = False  # PERSON whose gold value has a gazetteer name: detection there is list overlap


@dataclass(frozen=True, slots=True)
class Linking:
    """How consistently a system's placeholders follow the gold entities of each conversation (ADR-0004).

    Over non-public mentions, skipping `ambiguous: true` ones (05 §3). Only mentions that carry a placeholder
    take part: a leaked mention is a detection failure, counted by the leak rate. Entities are compared within
    one conversation and one type.
    """

    placed: Rate  # in-scope mentions that carry a placeholder
    wrong_merge: Rate  # entities sharing a placeholder with another entity / entities with a placeholder
    wrong_split: Rate  # entities under 2+ placeholders / entities with 2+ placed mentions
    ambiguity: Rate  # later mentions under a placeholder of their own / later placed mentions
    partly_leaking: Rate = Rate(0, 0)  # placed mentions that also leak a part (`Chirine PERSON_1`) / placed

    def __add__(self, other: Linking) -> Linking:
        return Linking(
            self.placed + other.placed,
            self.wrong_merge + other.wrong_merge,
            self.wrong_split + other.wrong_split,
            self.ambiguity + other.ambiguity,
            self.partly_leaking + other.partly_leaking,
        )


NO_LINKING = Linking(Rate(0, 0), Rate(0, 0), Rate(0, 0), Rate(0, 0), Rate(0, 0))


def link(mentions: Sequence[MentionResult]) -> Linking:
    """Linking rates for the mentions of one conversation and one type, given in reading order.

    - **Wrong merge:** an entity shares a placeholder with another gold entity (`Ahmed Ben Salem` and
      `Ahmed Karoui` both `PERSON_1`). Costs utility, leaks nothing.
    - **Wrong split:** an entity appears under two or more placeholders (`Ahmed` → `PERSON_1`, then
      `Ahmed Ben Salem` → `PERSON_2`).
    - **Ambiguity:** a mention after the entity's first placed one gets a placeholder that is neither the
      entity's first placeholder nor one any other entity uses: an extra "person" for the model. ADR-0004
      gives an ambiguous mention its own placeholder; gold-ambiguous mentions are skipped, so this counts the
      mentions a human can resolve and the system couldn't. Every hit is also part of a wrong split.

    A placed mention that also leaks part of the name (`Chirine PERSON_1`: the surname propagated, the first
    name in clear) is counted in `partly_leaking` and left out of merge, split and ambiguity: its placeholder
    covers only part of the mention, so it says little about linking, and counting it would credit the
    linker for a detection gap.
    """
    scoped = [m for m in mentions if not m.public and not m.ambiguous]
    by_entity: dict[str, list[str]] = defaultdict(list)
    partly = 0
    for m in scoped:
        if m.placeholder is None:
            continue
        if m.outcome is Outcome.PARTIAL:
            partly += 1
            continue
        by_entity[m.entity].append(m.placeholder)
    first = {e: labels[0] for e, labels in by_entity.items()}
    owners: dict[str, set[str]] = defaultdict(set)
    for e, labels in by_entity.items():
        for label in labels:
            owners[label].add(e)
    merged = sum(any(len(owners[label]) > 1 for label in labels) for labels in by_entity.values())
    multi = [labels for labels in by_entity.values() if len(labels) > 1]
    split = sum(len(set(labels)) > 1 for labels in multi)
    later = Rate(0, 0)
    for e, labels in by_entity.items():
        claimed = {label for label, users in owners.items() if users - {e}}
        for label in labels[1:]:
            later += Rate(int(label != first[e] and label not in claimed), 1)
    linked = sum(len(labels) for labels in by_entity.values())
    placed = linked + partly
    return Linking(
        Rate(placed, len(scoped)),
        Rate(merged, len(by_entity)),
        Rate(split, len(multi)),
        later,
        Rate(partly, placed),
    )


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
        labels = placeholders(turn.text, payload, [(m.start, m.end) for m in turn.mentions])
        for m, label in zip(turn.mentions, labels, strict=True):
            entity = conversation.entity(m.entity)
            outcome = classify(m.text, entity.type, payload)
            listed = entity.type is EntityType.PERSON and bool(
                canonical_tokens(entity.value) & GAZETTEER_NAMES
            )
            mentions.append(
                MentionResult(
                    conversation.id, i, turn.role, language, entity.type, entity.public, m.text, outcome,
                    m.entity, m.ambiguous, label, listed,
                )
            )  # fmt: skip
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
    def rate(mentions: Sequence[MentionResult], outcomes: Iterable[Outcome] = LEAKS) -> Rate:
        wanted = frozenset(outcomes)
        return Rate(sum(m.outcome in wanted for m in mentions), len(mentions))

    @property
    def headline(self) -> Rate:
        """Leak rate on direct identifiers in user, system and tool turns. Public figures excluded. Obfuscated
        names count as leaks."""
        return self.rate(self.select())

    @property
    def partial(self) -> Rate:
        return self.rate(self.select(), (Outcome.PARTIAL,))

    @property
    def obfuscated(self) -> Rate:
        """Names spelled out with separators (part of the headline)."""
        return self.rate(self.select(), (Outcome.OBFUSCATED,))

    @property
    def deleted(self) -> Rate:
        """Mentions that neither leaked nor carry a placeholder: removed, or replaced by something the
        alignment couldn't attribute. Deleting everything would give a perfect leak rate, so it shows here."""
        scope = self.select()
        return Rate(sum(m.outcome is Outcome.PROTECTED and m.placeholder is None for m in scope), len(scope))

    @property
    def held_out(self) -> Rate:
        """PERSON leak rate on names that aren't in the gazetteer: detection, not list overlap (05 §4)."""
        return self.rate([m for m in self.select(types=(EntityType.PERSON,)) if not m.listed])

    @property
    def assistant(self) -> Rate:
        """Leak rate on restored assistant turns that the client sends back (ADR-0004)."""
        return self.rate(self.select(assistant=True))

    @property
    def location_organization(self) -> Rate:
        return self.rate(self.select(types=(EntityType.LOCATION, EntityType.ORGANIZATION)))

    def linking(self, types: Iterable[EntityType] | None = None) -> Linking:
        """Linking rates over every role, per conversation and type, then pooled. All types by default."""
        wanted = frozenset(EntityType if types is None else types)
        groups: dict[tuple[str, EntityType], list[MentionResult]] = defaultdict(list)
        for m in self.mentions:
            if m.type in wanted:
                groups[m.conversation, m.type].append(m)
        total = NO_LINKING
        for group in groups.values():
            total += link(group)
        return total

    @property
    def languages(self) -> list[str]:
        return sorted({m.language for m in self.mentions})

    @property
    def types(self) -> list[EntityType]:
        present = {m.type for m in self.mentions}
        return [t for t in EntityType if t in present]
