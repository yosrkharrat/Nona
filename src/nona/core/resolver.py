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
   A name span that loses to spans it only partly overlaps keeps its **whole words** that touch none of them
   and no propagated name, as pieces that compete like any other span (#89): the model's confident
   `Zorgati Sami` against the gazetteer's longer `Sami Mo3ez Sami` would otherwise send `Zorgati` in clear.
   A word is a run of letters, digits, combining marks and what normalizing drops (the tatweel). A word
   the winner cuts stays as the winner leaves it, as before #89 (`و` in `وكريم`, `ـــ` in `عليـــ`): a
   piece is never glued to a placeholder, which restoring would never write back (#87). A piece never takes
   two letters of a propagated name that is still pending or kept, which keeps its placeholder and type
   (step 4: `بصفاقس`); a propagated name that loses is cut into pieces like any other. A piece needs two
   letters or digits, and goes through the allowlist. A denylist term cut this way masks its pieces too,
   even over a propagated name (step 4): half of a term the user hid is not left in clear. A span inside
   another is dropped whole.

The result never depends on the order recognizers ran in (ADR-0002: placeholders are numbered by first
appearance, so any instability would renumber them).
"""

from __future__ import annotations

import functools
import heapq
import itertools
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

from nona.core.boundaries import (
    LOOSE_LEFT,
    LOOSE_RIGHT,
    NAME_START,
    NAME_TYPES,
    RIGHT,
    TYPED_PLACEHOLDER,
)
from nona.core.normalize import DROPPED, NormalizedDoc, normalize
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
    be attached (`لكنزة`), so a restored value is always caught again. A single-word name may also hold what
    normalization drops between its letters (`القصـوري`, `القصُوري`).
    """
    spans: list[Span] = []
    for etype, surface in dict.fromkeys(known):
        pattern, on_normalized = _known_pattern(surface, etype in NAME_TYPES)
        # Group `v`: a name's match holds the proclitics before it (`NAME_START`).
        if on_normalized:
            spans.extend(
                to_span(doc, m.start("v"), m.end("v"), etype, 1.0, detector)
                for m in pattern.finditer(doc.text)
            )
        else:
            spans.extend(
                Span(m.start("v"), m.end("v"), etype, 1.0, detector) for m in pattern.finditer(doc.original)
            )
    return spans


@functools.lru_cache(maxsize=4096)
def _known_pattern(surface: str, name: bool) -> tuple[re.Pattern[str], bool]:
    """The pattern for one known form, and whether it runs on the normalized view. Names use the strict
    boundaries, structured values the loose ones (`core/boundaries.py`). Cached, because the proxy re-runs the
    whole conversation on every request."""
    left, right = (NAME_START, RIGHT) if name else (LOOSE_LEFT, LOOSE_RIGHT)
    words = normalize(surface).text.split()
    if len(words) > 1:
        body = r"\s+".join(map(re.escape, words))
        return re.compile(f"{left}(?P<v>{body}){right}"), True
    if not words:
        return re.compile(r"(?!)"), False  # matches nothing
    if not name:
        return re.compile(f"{left}(?P<v>{re.escape(surface.strip())}){right}"), False
    # What normalization drops may sit anywhere in the name (`القصـوري`, `القصُوري`).
    dropped = f"[{''.join(sorted(DROPPED))}]*"
    body = dropped.join(map(re.escape, "".join(c for c in surface.strip() if c not in DROPPED)))
    return re.compile(f"{left}(?P<v>{body}){right}"), False


def _rank(span: Span) -> tuple[int, int, float, tuple[int, int, str, float, str]]:
    return (-len(span), -PRIORITY[span.type], -span.score, span.sort_key())


def _letter(c: str) -> bool:
    """A character of a word: a letter or digit, a combining mark (`José` in NFD), or what normalizing drops
    (`DROPPED`: the tatweel)."""
    return c.isalnum() or c in DROPPED or unicodedata.category(c).startswith("M")


def _letters(text: str) -> int:
    return sum(c.isalnum() and c not in DROPPED for c in text)


def _outside(span: Span, clash: Sequence[Span], names: Sequence[Span], original: str) -> list[Span]:
    """The pieces of `span` (module docstring, step 5): runs of its whole words that overlap none of `clash`
    and share no two letters with one of `names`, with two letters or digits at least. Never `span` itself,
    as one of `clash` overlaps it."""
    words: list[tuple[int, int]] = []
    i = span.start
    while i < span.end:
        if _letter(original[i]):
            j = i
            while j < span.end and _letter(original[j]):
                j += 1
            words.append((i, j))
            i = j
        else:
            i += 1

    def free(a: int, b: int) -> bool:
        return not any(o.start < b and a < o.end for o in clash) and not any(
            _letters(original[max(a, n.start) : min(b, n.end)]) >= 2 for n in names
        )

    runs: list[list[tuple[int, int]]] = []
    previous_free = False
    for a, b in words:
        if not free(a, b):
            previous_free = False
            continue
        if previous_free and free(runs[-1][-1][1], a):
            runs[-1].append((a, b))
        else:
            runs.append([(a, b)])
        previous_free = True
    pieces: list[Span] = []
    for run in runs:
        a, b = run[0][0], run[-1][1]
        if _letters(original[a:b]) >= 2:
            pieces.append(Span(a, b, span.type, span.score, span.detector))
    return pieces


def placeholder_holes(original: str) -> list[tuple[int, int]]:
    """Where the user typed a placeholder in its exact shape (`TYPED_PLACEHOLDER`)."""
    return [m.span() for m in TYPED_PLACEHOLDER.finditer(original)]


def cut_at_placeholders(
    span: Span, original: str, holes: Sequence[tuple[int, int]] | None = None
) -> list[Span]:
    """The pieces of a span on either side of every typed placeholder it overlaps, spaces trimmed; the span
    itself when it overlaps none. `holes` from `placeholder_holes`, computed here if not given."""
    holes = placeholder_holes(original) if holes is None else holes
    inside = [(a, b) for a, b in holes if a < span.end and span.start < b]
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
    # Propagated names still pending or kept: a piece never takes their words (step 5).
    alive = Counter(s for s in candidates if s.detector == PROPAGATION and s.type in NAME_TYPES)
    known = {(s.start, s.end) for s in alive}
    if known:
        candidates = [
            s for s in candidates if s.detector in (PROPAGATION, DENYLIST) or (s.start, s.end) not in known
        ]
    kept: list[Span] = []
    order = itertools.count()  # ties in `_rank` are identical spans: never compare two `Span`s
    pending = [(_rank(s), next(order), s) for s in candidates]
    heapq.heapify(pending)
    while pending:
        span = heapq.heappop(pending)[2]
        clash = [k for k in kept if span.overlaps(k)]
        if not clash:
            kept.append(span)
            continue
        if alive[span]:  # a propagated name that lost: it no longer holds its words
            alive[span] -= 1
        if span.type in NAME_TYPES:  # pieces are shorter than the span: the loop ends
            # A denylist term's policy wins over a propagated name (step 4), so only `clash` cuts it.
            names = [] if span.detector == DENYLIST else list(+alive)
            for piece in _outside(span, clash, names, original):
                if _fold(piece.text_in(original)) not in allowed:
                    heapq.heappush(pending, (_rank(piece), next(order), piece))
                    if piece.detector == PROPAGATION:
                        alive[piece] += 1
    return sorted(kept, key=Span.sort_key)
