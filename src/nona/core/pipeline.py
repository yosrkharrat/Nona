"""`Nona`: the library entry point (02 §3, FR-17). Normalize → detect → resolve → pseudonymize; restore.

`anonymize_messages` is the conversation-level path the proxy uses (ADR-0004): messages are processed in
order, every surface form linked in an earlier message is searched for again (forward propagation), and
placeholders issued for a message never change whatever comes after it. Inside a message, the name words of
each detected PERSON are searched for too (look-ahead), so `Chirine. Chirine Boukadida…` masks both.

Fail closed (ADR-0003): if any recognizer raises, `anonymize` raises `DetectorError` and returns nothing.
A partly masked text is never produced, so a caller can't forward it by mistake.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections.abc import Mapping as AbcMapping

from nona.core.canonical import canonical_tokens, name_words
from nona.core.normalize import normalize
from nona.core.pseudonymize import (
    DEFAULT_POLICIES,
    Anonymized,
    AnonymizedConversation,
    Mapping,
    Policy,
    Pseudonymizer,
    restore,
)
from nona.core.recognizers import Recognizer, default_recognizers
from nona.core.recognizers.gazetteer import COMMON
from nona.core.resolver import denylist_spans, known_spans, resolve
from nona.core.types import EntityType, Span


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
    ) -> None:
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

    def detect(self, text: str) -> list[Span]:
        """Resolved spans over `text`, before any policy is applied."""
        return self._detect(text, ())

    def _detect(self, text: str, known: Iterable[tuple[EntityType, str]]) -> list[Span]:
        doc = normalize(text)
        candidates: list[Span] = denylist_spans(doc, self._denylist) + known_spans(doc, known)
        for recognizer in self._recognizers:
            try:
                candidates.extend(recognizer.detect(doc))
            except Exception as exc:
                raise DetectorError(recognizer.name) from exc
        spans = resolve(candidates, text, min_score=self._min_score, allowlist=self._allowlist)
        # Look-ahead (ADR-0004, step 4): a bare first name before the full name in the same message.
        # Names that are also ordinary words (`Mabrouk`, congratulations; `Amal`, hope) are left to the
        # gazetteer's context rules: `Mabrouk ! Mabrouk Jlassi a eu son bac` keeps its first `Mabrouk`.
        words = [
            (EntityType.PERSON, word)
            for s in spans
            if s.type is EntityType.PERSON
            for word in name_words(s.text_in(text))
            if not canonical_tokens(word) & COMMON
        ]
        if not words:
            return spans
        candidates.extend(known_spans(doc, words, detector="look-ahead"))
        return resolve(candidates, text, min_score=self._min_score, allowlist=self._allowlist)

    def anonymize(self, text: str) -> Anonymized:
        return self.anonymize_messages([text]).messages[0]

    def anonymize_messages(self, messages: Sequence[str]) -> AnonymizedConversation:
        """Anonymize a conversation with one shared mapping. Causal: message `i` depends only on `0..i`."""
        pseudonymizer = Pseudonymizer(self._policies)
        for text in messages:
            pseudonymizer.add(text, self._detect(text, pseudonymizer.known_surfaces()))
        return pseudonymizer.result()

    @staticmethod
    def restore(text: str, mapping: Mapping) -> str:
        return restore(text, mapping)
