"""Span resolver: from overlapping candidates to one clean, deterministic set of spans (03 §2).

Recognizers report a union of candidates, overlaps included (the RIB inside a Tunisian IBAN, a phone and a
CIN on the same 8 digits). The resolver:

1. drops spans below `min_score`, and spans whose text is on the allowlist (FR-5);
2. adds every denylist term as a `CUSTOM` span (FR-5), and every known surface form from earlier messages of
   the conversation (forward propagation, ADR-0004);
3. cuts every name span (PERSON, LOCATION, ORGANIZATION, CUSTOM) at a placeholder the user typed
   (`Sami Trabelsi PERSON_1` → `Sami Trabelsi`; `Sami PERSON_1 Trabelsi` → `Sami` and `Trabelsi`): a detection
   that swallowed one would replace it, and its text would end up in the mapping. A structured value is never
   cut: `PERSON_1@example.com` is one email;
4. drops a recognizer span with exactly the extent of a propagated name: the name keeps the placeholder, and
   the type, it was first given (ADR-0004, ADR-0008). Denylist terms aren't dropped: their policy wins;
5. keeps non-overlapping spans greedily: longest first, then type priority, then score, then `Span.sort_key`.

The result never depends on the order recognizers ran in (ADR-0002: placeholders are numbered by first
appearance, so any instability would renumber them).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Sequence

from nona.core.boundaries import LEFT, LOOSE_LEFT, LOOSE_RIGHT, NAME_TYPES, RIGHT, TATWEEL, TYPED_PLACEHOLDER
from nona.core.normalize import NormalizedDoc, normalize
from nona.core.recognizers.span import to_span
from nona.core.types import EntityType, Span

# Higher wins when two spans have the same length. Explicit and checksum-validated types first, then
# context-gated ones, then the rest. A CIN beats a phone on the same 8 digits because its context word
# says what it is.
PRIORITY: dict[EntityType, int] = {
    EntityType.CUSTOM: 100,
    EntityType.IBAN: 90,
    EntityType.RIB: 85,
    EntityType.CARD: 80,
    EntityType.NATIONAL_ID: 70,
    EntityType.EMAIL: 60,
    EntityType.IP_ADDRESS: 55,
    EntityType.PHONE: 50,
    EntityType.PERSON: 40,
    EntityType.ORGANIZATION: 30,
    EntityType.LOCATION: 20,
}


PROPAGATION = "propagation"
DENYLIST = "denylist"


def _fold(term: str) -> str:
    return " ".join(term.casefold().split())


def denylist_spans(doc: NormalizedDoc, terms: Iterable[str]) -> list[Span]:
    """Every occurrence of a denylist term, matched on the normalized view at word boundaries."""
    spans: list[Span] = []
    for term in terms:
        needle = _fold(term)
        if not needle:
            continue
        pattern = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, needle.split())) + r"(?!\w)")
        for m in pattern.finditer(doc.text):
            spans.append(to_span(doc, m.start(), m.end(), EntityType.CUSTOM, 1.0, DENYLIST))
    return spans


def known_spans(
    doc: NormalizedDoc, known: Iterable[tuple[EntityType, str]], detector: str = PROPAGATION
) -> list[Span]:
    """Forward propagation (ADR-0004, step 2): each occurrence of a surface form linked in an earlier message.

    Multi-word forms match on the normalized view (case, spacing and digits don't matter). Single-word forms
    match the original text exactly, at word boundaries, so a known `Amal` doesn't mask *amal* ("hope").
    Boundaries are the ones restoring uses (`core/boundaries.py`): `_` separates, and Arabic proclitics may
    be attached (`لكنزة`), so a restored value is always caught again. A single-word name may hold tatweels
    anywhere (`Ksouriـ`, `لـالقصوري`), since normalization drops them. Overlapping occurrences are all found
    (`a@x.com@x.com` holds `x.com@x.com` twice); the resolver keeps the longest.
    """
    spans: list[Span] = []
    for etype, surface in dict.fromkeys(known):
        pattern, on_normalized = _known_pattern(surface, etype in NAME_TYPES)
        if on_normalized:
            spans.extend(to_span(doc, s, e, etype, 1.0, detector) for s, e in _occurrences(pattern, doc.text))
        else:
            spans.extend(Span(s, e, etype, 1.0, detector) for s, e in _occurrences(pattern, doc.original))
    return spans


def _occurrences(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    """Every match, overlapping ones too (the next search starts one character after the last match began),
    without the tatweels at its edges, so the user's `لـ` stays when the name is restored."""
    out: list[tuple[int, int]] = []
    m = pattern.search(text)
    while m:
        s, e = m.span()
        while s < e and text[s] == TATWEEL:
            s += 1
        while e > s and text[e - 1] == TATWEEL:
            e -= 1
        out.append((s, e))
        m = pattern.search(text, m.start() + 1)
    return out


@functools.lru_cache(maxsize=4096)
def _known_pattern(surface: str, name: bool) -> tuple[re.Pattern[str], bool]:
    """The pattern for one known form, and whether it runs on the normalized view. Names use the strict
    boundaries, structured values the loose ones (`core/boundaries.py`). Cached, because the proxy re-runs the
    whole conversation on every request."""
    left, right = (LEFT, RIGHT) if name else (LOOSE_LEFT, LOOSE_RIGHT)
    if not name and surface.strip() and not surface.strip()[0].isalnum():
        left = ""  # a value that starts with a sign (`+216…`) needs no boundary on its left (#80)
    words = normalize(surface).text.split()
    if len(words) > 1:
        return re.compile(left + r"\s+".join(map(re.escape, words)) + right), True
    if not words:
        return re.compile(r"(?!)"), False  # matches nothing
    if not name:
        return re.compile(left + re.escape(surface.strip()) + right), False
    body = f"{TATWEEL}*".join(map(re.escape, surface.strip().replace(TATWEEL, "")))
    return re.compile(f"{left}{TATWEEL}*{body}{TATWEEL}*{right}"), False


def _rank(span: Span) -> tuple[int, int, float, tuple[int, int, str, float, str]]:
    return (-len(span), -PRIORITY[span.type], -span.score, span.sort_key())


def placeholder_holes(original: str) -> list[tuple[int, int]]:
    """Where the user typed a placeholder in its exact shape (`TYPED_PLACEHOLDER`)."""
    return [m.span() for m in TYPED_PLACEHOLDER.finditer(original)]


def cut_at_placeholders(
    span: Span, original: str, holes: Sequence[tuple[int, int]] | None = None
) -> list[Span]:
    """The pieces of a span on either side of every typed placeholder it overlaps, spaces trimmed; the span
    itself when it overlaps none. `holes` from `placeholder_holes`, computed here if not given.

    A piece glued to a typed placeholder (`سميرPERSON_1`, `PERSON_1محمد`, `M.٢PERSON_1`) is masked together
    with it: its own placeholder would be glued to the typed one, where restore can't write it back (#80).
    The model then doesn't see that typed placeholder, which only odd text loses."""
    holes = placeholder_holes(original) if holes is None else holes
    inside = [(a, b) for a, b in holes if a < span.end and span.start < b]
    glued = [(a, b) for a, b in holes if a <= span.end and span.start <= b]
    if not glued:
        return [span]
    return _join_glued(_pieces(span, original, inside), glued, original)


def _pieces(span: Span, original: str, inside: Sequence[tuple[int, int]]) -> list[Span]:
    if not inside:
        return [span]
    bounds = [span.start, *(x for h in inside for x in h), span.end]
    pieces: list[Span] = []
    for a, b in zip(bounds[0::2], bounds[1::2], strict=True):
        a, b = max(a, span.start), min(b, span.end)
        while a < b and original[a].isspace():
            a += 1
        while b > a and original[b - 1].isspace():
            b -= 1
        if a < b:
            pieces.append(Span(a, b, span.type, span.score, span.detector))
    return pieces


def _word_char(c: str) -> bool:
    return c.isalnum() or c == TATWEEL


def _join_glued(pieces: list[Span], holes: Sequence[tuple[int, int]], original: str) -> list[Span]:
    """Each piece with the typed placeholders it touches and the letters glued to those (`PERSON_1a`), and
    pieces joined through one."""
    out: list[Span] = []
    for p in pieces:
        a, b = p.start, p.end
        for h_start, h_end in holes:
            if h_end == a:
                a = h_start
                while a > 0 and _word_char(original[a - 1]):
                    a -= 1
            if h_start == b:
                b = h_end
                while b < len(original) and _word_char(original[b]):
                    b += 1
        if out and out[-1].end >= a:
            a = out.pop().start
        out.append(p if (a, b) == (p.start, p.end) else Span(a, b, p.type, p.score, p.detector))
    return out


def resolve(
    spans: Iterable[Span],
    original: str,
    *,
    min_score: float = 0.5,
    allowlist: Sequence[str] = (),
) -> list[Span]:
    """Non-overlapping spans, sorted by position. `original` is the text the spans index into."""
    allowed = {_fold(term) for term in allowlist}
    spans = list(spans)
    holes = placeholder_holes(original)
    if holes:
        # Only names: cutting `PERSON_1@example.com` would leave `@example.com`, which can't round-trip.
        spans = [
            piece
            for s in spans
            for piece in (cut_at_placeholders(s, original, holes) if s.type in NAME_TYPES else [s])
        ]
    candidates = [s for s in spans if s.score >= min_score and _fold(s.text_in(original)) not in allowed]
    known = {(s.start, s.end) for s in candidates if s.detector == PROPAGATION and s.type in NAME_TYPES}
    if known:
        candidates = [
            s for s in candidates if s.detector in (PROPAGATION, DENYLIST) or (s.start, s.end) not in known
        ]
    kept: list[Span] = []
    for span in sorted(candidates, key=_rank):
        if not any(span.overlaps(k) for k in kept):
            kept.append(span)
    return sorted(kept, key=Span.sort_key)


_ASCII_WORD = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def _ascii_word(c: str) -> bool:
    """Whether `c` is, or normalizes to, an ASCII letter or digit (`٢` → `2`, `ﬁ` → `fi`): `LOOSE_LEFT` fails
    after it on the original text or on the normalized view."""
    return c in _ASCII_WORD or any(x in _ASCII_WORD for x in normalize(c).text)


def take_glued_words(spans: list[Span], original: str) -> list[Span]:
    """A structured value glued to a word on its left (`a+216 22 123 456`, a phone starting with its sign)
    takes the word in: its placeholder would be glued to the word, where restore can't write it back (#80).
    `spans` resolved, in order."""
    out: list[Span] = []
    for s in spans:
        a = s.start
        if s.type not in NAME_TYPES:
            floor = out[-1].end if out else 0
            while a > floor and _ascii_word(original[a - 1]):
                a -= 1
        out.append(s if a == s.start else Span(a, s.end, s.type, s.score, s.detector))
    return out
