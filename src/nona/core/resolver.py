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
   Then the names that lost a partial overlap fill the gaps (#89): each keeps, as pieces, the words of the
   text that lie wholly inside it and touch nothing kept, nor a propagated name's piece still pending. The
   model's confident `Zorgati Sami` against the gazetteer's longer `Sami Mo3ez Sami` would otherwise send
   `Zorgati` in clear. Pieces never displace a span the first pass kept, so the result only adds pieces to
   what it was before #89 (a piece of `Pharmacie Ahmed, Sami` can't swallow the person `Ahmed`). Pieces
   compete among themselves like the spans did, and a piece that loses is cut again. A word is a run of
   letters, digits, combining marks and what normalizing drops (the tatweel). A word cut by another span,
   or by the losing span's own edge, stays as before (`و` in `وكريم`, `ـــ` in `عليـــ`): a piece is never
   glued to a placeholder, which restoring would never write back (#87). A piece needs two letters or
   digits, and goes through the allowlist. A denylist term's pieces ignore propagated names: its policy
   wins (step 4). A span inside another is dropped whole. A piece kept comes out with `PIECE_DETECTOR`:
   the linker masks it on its own and never links or propagates it, since it holds whatever words the span
   lost (`de` from `de Sami`).

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
from nona.core.linking import PIECE_DETECTOR
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


@functools.lru_cache(maxsize=4096)
def word_char(c: str) -> bool:
    """A character of a word, for the pieces of step 5: a letter or digit, a combining mark (`José` in NFD),
    what normalizing drops (`DROPPED`: the tatweel), or a symbol that normalizes to letters (`№` → `No`,
    `™` → `TM`): restoring checks a multi-word value's edges on the normalized view, where a piece glued to
    one would touch a letter and never be written back."""
    return (
        c.isalnum()
        or c in DROPPED
        or unicodedata.category(c).startswith("M")
        or any(x.isalnum() for x in unicodedata.normalize("NFKC", c))
    )


def letters(text: str) -> int:
    """How many letters and digits `text` holds, not counting what normalizing drops."""
    return sum(c.isalnum() and c not in DROPPED for c in text)


def _outside(span: Span, blockers: Sequence[Span], original: str) -> list[Span]:
    """The pieces of `span` (module docstring, step 5): runs of the text's words that lie wholly inside
    `span` and overlap none of `blockers`, with two letters or digits at least. Never `span` itself, as one
    of `blockers` overlaps it."""
    i = span.start
    while i > 0 and word_char(original[i - 1]):  # the word `span` starts in, if it starts inside one
        i -= 1
    words: list[tuple[int, int]] = []
    while i < span.end:
        if not word_char(original[i]):
            i += 1
            continue
        j = i
        while j < len(original) and word_char(original[j]):
            j += 1
        if letters(original[i:j]):
            words.append((i, j))
        i = j

    def free(a: int, b: int) -> bool:
        return not any(o.start < b and a < o.end for o in blockers)

    runs: list[list[tuple[int, int]]] = []
    previous_free = False
    for a, b in words:
        if a < span.start or b > span.end or not free(a, b):  # cut at the span's edge, or taken
            previous_free = False
            continue
        if previous_free and free(runs[-1][-1][1], a):
            runs[-1].append((a, b))
        else:
            runs.append([(a, b)])
        previous_free = True
    return [
        Span(run[0][0], run[-1][1], span.type, span.score, span.detector)
        for run in runs
        if letters(original[run[0][0] : run[-1][1]]) >= 2
    ]


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


def _candidates(spans: Iterable[Span], original: str, min_score: float, allowed: set[str]) -> list[Span]:
    """Steps 1 to 4 (module docstring)."""
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
    return candidates


def resolve(
    spans: Iterable[Span],
    original: str,
    *,
    min_score: float = 0.5,
    allowlist: Sequence[str] = (),
    pieces: bool = True,
) -> list[Span]:
    """Non-overlapping spans, sorted by position. `original` is the text the spans index into. With
    `pieces=False`, step 5 keeps only its first pass: `fill_gaps` adds the pieces later (`core/pipeline.py`
    does, after its look-ahead)."""
    allowed = {_fold(term) for term in allowlist}
    candidates = _candidates(spans, original, min_score, allowed)
    kept: list[Span] = []
    for span in sorted(candidates, key=_rank):
        if not any(span.overlaps(k) for k in kept):
            kept.append(span)
    return _fill(kept, candidates, original, allowed) if pieces else sorted(kept, key=Span.sort_key)


def fill_gaps(
    kept: Sequence[Span],
    spans: Iterable[Span],
    original: str,
    *,
    min_score: float = 0.5,
    allowlist: Sequence[str] = (),
) -> list[Span]:
    """`kept` (resolved with `pieces=False`) and the pieces of the name spans among `spans` that lost to it
    (step 5, second pass), sorted by position."""
    allowed = {_fold(term) for term in allowlist}
    return _fill(list(kept), _candidates(spans, original, min_score, allowed), original, allowed)


def _fill(kept: list[Span], candidates: Sequence[Span], original: str, allowed: set[str]) -> list[Span]:
    held = set(kept)
    lost = [
        c for c in candidates if c.type in NAME_TYPES and c not in held and any(c.overlaps(k) for k in kept)
    ]
    if not lost:
        return sorted(kept, key=Span.sort_key)

    # Step 5, second pass (#89): pieces of the names that lost only fill what nothing kept covers, so every
    # span the first pass kept stays as it is. Propagated pieces first: no other piece takes their words.
    order = itertools.count()  # ties in `_rank` are identical spans: never compare two `Span`s
    pending: list[tuple[tuple[int, int, float, tuple[int, int, str, float, str]], int, Span]] = []
    alive: Counter[Span] = Counter()  # propagated pieces still pending

    def cut(span: Span) -> None:
        blockers = [k for k in kept if k.overlaps(span)]
        if span.detector != DENYLIST:  # a denylist term's policy wins over a propagated name (step 4)
            blockers += [n for n, c in alive.items() if c and n.overlaps(span)]
        for piece in _outside(span, blockers, original):
            if _fold(piece.text_in(original)) not in allowed:
                heapq.heappush(pending, (_rank(piece), next(order), piece))
                if piece.detector == PROPAGATION:
                    alive[piece] += 1

    for span in sorted(lost, key=lambda s: (s.detector != PROPAGATION, _rank(s))):
        cut(span)
    first = len(kept)
    while pending:
        piece = heapq.heappop(pending)[2]
        if alive[piece]:
            alive[piece] -= 1
        if any(piece.overlaps(k) for k in kept):  # another piece took part of it: pieces are shorter
            cut(piece)
        else:
            kept.append(piece)
    # Kept pieces are marked: each is masked on its own, and never links or propagates (`core/linking.py`).
    pieces = [Span(p.start, p.end, p.type, p.score, PIECE_DETECTOR) for p in kept[first:]]
    return sorted([*kept[:first], *pieces], key=Span.sort_key)
