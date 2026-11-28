"""`Nona`: the library entry point (02 §3, FR-17). Normalize → detect → resolve → pseudonymize; restore.

`anonymize_messages` is the conversation-level path the proxy uses (ADR-0004): messages are processed in
order, every surface form linked in an earlier message is searched for again (forward propagation), and
placeholders issued for a message never change whatever comes after it. Inside a message, the name words of
each detected PERSON are searched for too (look-ahead), so `Chirine. Chirine Boukadida…` masks both.

Look-ahead only spreads words that a confident detection vouches for:
- a recognizer span scoring at least `LOOK_AHEAD_MIN_SCORE`. A weak one can't spread its words, even once
  it comes back as a propagated form scoring 1.0 in a later message;
- from a gazetteer run without a strong cue, only listed names: in `Merci Sami Bonne journée`, the run takes
  `Bonne` as a surname, and spreading it would mask every `Bonne` of the conversation;
- a propagated form spreads only the words its person was vouched for when first found (its seeds,
  `core/linking.py`).

A model's span types and edges are corrected with what the conversation knows (ADR-0008, condition 1; the
message-local rules are in `core/recognizers/ner.py`):
- a LOCATION or ORGANIZATION span from the NER model that fits a known person becomes a PERSON. A
  LOCATION fits when its name tokens contain a known person's or are contained in them (`Bacem Hachicha`
  after `Bacem`), or when it follows a person with only `and` between them (`m3a Hazem Boukadida w Hazem
  Drira`). An ORGANIZATION fits only when every token belongs to known people, so `Clinique Ahmed Ben
  Salem` stays an organization, as does a span that starts with a place or organization word or follows
  one (`rue Mohamed Ali`). A listed city never changes type;
- a model span that is a known form with Arabic clitics in front (`لزكرياء` after `زكرياء`) loses them, as
  propagation does, so it links to that form;
- the resolver keeps a propagated form over a recognizer span with the same extent and another type
  (`core/resolver.py`), so a name keeps the placeholder it was given.

Fail closed (ADR-0003): if any recognizer raises, `anonymize` raises `DetectorError` and returns nothing.
A partly masked text is never produced, so a caller can't forward it by mistake.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from collections.abc import Mapping as AbcMapping

from nona.core.boundaries import PROCLITICS
from nona.core.cache import DetectionCache, cache_key, fingerprint
from nona.core.canonical import canonical_tokens, compatible, name_words
from nona.core.normalize import DROPPED, NormalizedDoc, normalize
from nona.core.pseudonymize import (
    DEFAULT_POLICIES,
    Anonymized,
    AnonymizedConversation,
    Mapping,
    Policy,
    Pseudonymizer,
    fold_surface,
    restore,
)
from nona.core.recognizers import Recognizer, default_recognizers
from nona.core.recognizers.gazetteer import NAMES, common
from nona.core.recognizers.ner import NER, place_or_org
from nona.core.resolver import (
    cut_at_placeholders,
    denylist_spans,
    fill_gaps,
    known_spans,
    placeholder_holes,
    resolve,
)
from nona.core.types import EntityType, Span

# Look-ahead only takes the words of a confident PERSON span, so one weak false positive can't spread its
# words over the rest of the conversation (they become known forms).
LOOK_AHEAD_MIN_SCORE = 0.8
GAZETTEER = "gazetteer"
STRONG_CUE_SCORE = 0.95  # what the gazetteer gives a run after a strong cue (`M.`, `je m'appelle`)
# Between two names of one list of people: `Sami w Ahmed`, `Sami et Ahmed`, `سامي و أحمد`.
_AND = re.compile(r"[ \t]+(?:w|wa|et|and|و)[ \t]+")
_RETYPED = frozenset({EntityType.LOCATION, EntityType.ORGANIZATION})


def _cased(word: str) -> bool:
    """Whether the word's script has capitals (Latin, Cyrillic…), as opposed to Arabic."""
    return any(c.lower() != c.upper() for c in word)


def _as_people(
    text: str, recognized: Sequence[Span], known: Sequence[tuple[EntityType, str]], propagated: Sequence[Span]
) -> list[Span]:
    """Model LOCATION and ORGANIZATION spans that fit a known person, as PERSON (module docstring)."""
    people = [canonical_tokens(surface) for etype, surface in known if etype is EntityType.PERSON]
    people += [canonical_tokens(s.text_in(text)) for s in propagated if s.type is EntityType.PERSON]
    people += [canonical_tokens(s.text_in(text)) for s in recognized if s.type is EntityType.PERSON]
    people = [t for t in people if t]
    if not people:
        return list(recognized)
    every = frozenset[str]().union(*people)
    places = [s for s in recognized if s.detector == GAZETTEER and s.type is EntityType.LOCATION]
    persons = sorted((s for s in recognized if s.type is EntityType.PERSON), key=lambda s: s.end)
    out: list[Span] = []
    for s in recognized:
        if (
            s.type in _RETYPED
            and s.detector == NER
            and not any(s.overlaps(p) for p in places)
            and not place_or_org(text, s.start, s.end)
        ):
            tokens = canonical_tokens(s.text_in(text))
            if s.type is EntityType.ORGANIZATION:
                fits = bool(tokens) and tokens <= every
            else:
                fits = any(compatible(tokens, t) for t in people) or any(
                    _AND.fullmatch(text, p.end, s.start) for p in persons if p.end < s.start
                )
            if fits:
                s = Span(s.start, s.end, EntityType.PERSON, s.score, s.detector)
        out.append(s)
    return out


def _behind_clitics(text: str, recognized: Sequence[Span], propagated: Sequence[Span]) -> list[Span]:
    """A recognizer name span that is a propagated form behind one or two Arabic clitics, maybe with what
    normalization drops around them (`لـ`, `لِ`), starts at the form (module docstring), so `لزكرياء` links to
    `زكرياء` and restores with its clitic in place."""
    starts = {(p.start, p.end) for p in propagated}
    out: list[Span] = []
    for s in recognized:
        i, proclitics = s.start, 0
        while s.detector != GAZETTEER and i < s.end and (text[i] in DROPPED or text[i] in PROCLITICS):
            proclitics += text[i] in PROCLITICS
            i += 1
            if 1 <= proclitics <= 2 and (i, s.end) in starts:
                s = Span(i, s.end, s.type, s.score, s.detector)
                break
            if proclitics > 2:
                break
        out.append(s)
    return out


class DetectorError(RuntimeError):
    """A recognizer failed. Nothing may be sent upstream (ADR-0003); the proxy turns this into HTTP 503."""

    def __init__(self, detector: str) -> None:
        # The message names the detector only. The cause stays attached for debugging, but the text being
        # analysed is never copied into the message (deep dive G).
        super().__init__(f"recognizer {detector!r} failed")
        self.detector = detector


class Nona:
    def __init__(
        self,
        recognizers: Sequence[Recognizer] | None = None,
        *,
        policies: AbcMapping[EntityType, Policy] | None = None,
        allowlist: Sequence[str] = (),
        denylist: Sequence[str] = (),
        min_score: float = 0.5,
        cache: DetectionCache | None = None,
    ) -> None:
        """`cache` holds recognizer spans per message (`core/cache.py`). Each `Nona` gets its own unless one
        is passed; several instances may share one, as the key covers their recognizers' configuration."""
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        folded_allow = {" ".join(t.casefold().split()) for t in allowlist}
        clash = sorted(t for t in denylist if " ".join(t.casefold().split()) in folded_allow)
        if clash:
            raise ValueError(f"terms both allowed and denied: {clash}")
        self._recognizers = list(default_recognizers() if recognizers is None else recognizers)
        self._policies = dict(DEFAULT_POLICIES if policies is None else policies)
        self._allowlist = tuple(allowlist)
        self._denylist = tuple(denylist)
        self._min_score = min_score
        self._cache = DetectionCache() if cache is None else cache
        self._fingerprint = fingerprint(self._recognizers)

    def detect(self, text: str) -> list[Span]:
        """Resolved spans over `text`, before any policy is applied."""
        return self._detect(text, ())[0]

    def _detect(
        self,
        text: str,
        known: Iterable[tuple[EntityType, str]],
        seeds: AbcMapping[str, frozenset[str]] | None = None,
    ) -> tuple[list[Span], dict[Span, frozenset[str]]]:
        """Resolved spans, and the seeds of each PERSON span (see the module docstring). `seeds` are the
        earlier messages' (`Pseudonymizer.seeds`)."""
        doc = normalize(text)
        known = list(known)
        propagated = known_spans(doc, known)
        recognized = self._recognize(doc, text)
        recognized = _behind_clitics(text, _as_people(text, recognized, known, propagated), propagated)
        candidates: list[Span] = denylist_spans(doc, self._denylist) + propagated
        # A span kept as is (an ORGANIZATION, by default) must not hide a value already masked earlier:
        # `Clinique Ahmed Ben Salem` after `Ahmed Ben Salem` was issued would send the name in clear.
        candidates.extend(
            s
            for s in recognized
            if self._policies.get(s.type) is not Policy.KEEP or not any(s.overlaps(p) for p in propagated)
        )
        spans = resolve(candidates, text, min_score=self._min_score, allowlist=self._allowlist, pieces=False)
        words, span_seeds = self._look_ahead_words(text, spans, recognized, seeds or {})
        # It fills gaps only, never retyping a detection, except a span kept as is (an ORGANIZATION): that
        # one gives way, so `… de la Clinique Ahmed Ben Salem` doesn't send a name found in this message.
        keep = {s for s in spans if self._policies.get(s.type) is Policy.KEEP}
        extra = [
            c
            for c in known_spans(doc, words, detector="look-ahead")
            if not any(c.overlaps(s) for s in spans if s not in keep)
        ]
        if not extra:
            return self._fill_gaps(spans, candidates, text), span_seeds
        hiding = {k for k in keep if any(k.overlaps(c) for c in extra)}
        kept = [s for s in spans if s not in hiding]
        final = resolve(
            [*kept, *extra], text, min_score=self._min_score, allowlist=self._allowlist, pieces=False
        )
        kept_spans = set(final)
        seeds_kept = {s: t for s, t in span_seeds.items() if s in kept_spans}
        return self._fill_gaps(final, [*candidates, *extra], text), seeds_kept

    def _fill_gaps(self, spans: list[Span], candidates: Sequence[Span], text: str) -> list[Span]:
        """The pieces of the names that lost a partial overlap, in what nothing covers once the look-ahead
        has run (`core/resolver.py`, step 5, #89): they never change a span kept, nor what the look-ahead
        finds. A piece carries no seeds, so it never spreads."""
        return fill_gaps(spans, candidates, text, min_score=self._min_score, allowlist=self._allowlist)

    @staticmethod
    def _look_ahead_words(
        text: str,
        spans: Sequence[Span],
        recognized: Sequence[Span],
        seeds: AbcMapping[str, frozenset[str]],
    ) -> tuple[list[tuple[EntityType, str]], dict[Span, frozenset[str]]]:
        """Look-ahead (ADR-0004, step 4): the forms to search for in this message, and each PERSON seeds.

        A confident detection gives its full form and its name words: a bare first name before the full name
        is caught too. Names that are also ordinary words (`Mabrouk`, congratulations; `Amal`, hope) are never
        spread by look-ahead: the gazetteer's context rules decide on each occurrence (a capitalized one is
        masked anyway, at 0.5 at the start of a sentence). In a script without capitals (Arabic), a word has
        no capital to vouch for it, so only listed names spread: `محمد الثاني` must not make every `الثاني`
        ("the second") a name.
        """

        def spreadable(word: str) -> bool:
            tokens = canonical_tokens(word)
            cased = _cased(word)
            return not any(common(t, cased) for t in tokens) and (cased or tokens <= NAMES)

        # The resolver may keep a propagated form over the recognizer span at the same place: the recognizer
        # still vouches for it. Recognizer spans are cut at typed placeholders as the resolver cuts them.
        confident: dict[tuple[int, int], list[Span]] = {}
        holes = placeholder_holes(text)
        for r in recognized:
            if r.type is EntityType.PERSON and r.score >= LOOK_AHEAD_MIN_SCORE:
                for piece in cut_at_placeholders(r, text, holes):
                    confident.setdefault((piece.start, piece.end), []).append(piece)

        words: list[tuple[EntityType, str]] = []
        span_seeds: dict[Span, frozenset[str]] = {}
        for s in spans:
            if s.type is not EntityType.PERSON:
                continue
            surface = s.text_in(text)
            parts = [w for w in name_words(surface) if spreadable(w)]
            vouched: list[str] = []
            for r in confident.get((s.start, s.end), []):
                listed_only = r.detector == GAZETTEER and r.score < STRONG_CUE_SCORE
                vouched += [w for w in parts if not listed_only or canonical_tokens(w) <= NAMES]
            if confident.get((s.start, s.end)) and spreadable(surface):
                words.append((EntityType.PERSON, surface))
            if s.detector == "propagation":
                inherited = seeds.get(fold_surface(surface), frozenset[str]())
                vouched += [w for w in parts if inherited and canonical_tokens(w) <= inherited]
            vouched = list(dict.fromkeys(vouched))
            words.extend((EntityType.PERSON, w) for w in vouched)
            tokens = frozenset(t for w in vouched for t in canonical_tokens(w))
            if tokens:
                span_seeds[s] = tokens
        return words, span_seeds

    def _recognize(self, doc: NormalizedDoc, text: str) -> tuple[Span, ...]:
        """Every recognizer's spans for one message, from the cache when this text was seen with this config.

        Fail closed: if a recognizer raises, nothing is cached and `DetectorError` propagates.
        """
        key = cache_key(self._fingerprint, text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        found: list[Span] = []
        for recognizer in self._recognizers:
            try:
                found.extend(recognizer.detect(doc))
            except Exception as exc:
                raise DetectorError(recognizer.name) from exc
        self._cache.put(key, found)
        return tuple(found)

    @property
    def cache(self) -> DetectionCache:
        return self._cache

    def anonymize(self, text: str) -> Anonymized:
        return self.anonymize_messages([text]).messages[0]

    def anonymize_messages(self, messages: Sequence[str]) -> AnonymizedConversation:
        """Anonymize a conversation with one shared mapping. Causal: message `i` depends only on `0..i`."""
        pseudonymizer = Pseudonymizer(self._policies)
        for text in messages:
            spans, seeds = self._detect(text, pseudonymizer.known_surfaces(), pseudonymizer.seeds())
            pseudonymizer.add(text, spans, seeds)
        return pseudonymizer.result()

    @staticmethod
    def restore(text: str, mapping: Mapping) -> str:
        return restore(text, mapping)
