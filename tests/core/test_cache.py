"""Detection cache (ADR-0004, 03 §4): recognizer spans per message, keyed by config and text, never text."""

from __future__ import annotations

import pytest
from hypothesis import given, settings

from nona.core import EntityType, Nona, NormalizedDoc, Span
from nona.core.cache import DetectionCache, cache_key, fingerprint, recognizer_fingerprint
from nona.core.pipeline import DetectorError
from nona.core.recognizers import PhoneRecognizer, default_recognizers
from tests.core.test_linking import conversation, nona


def _span(start: int = 0) -> Span:
    return Span(start, start + 4, EntityType.PERSON, 0.9, "t")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Counting:
    """Finds `Sami` and counts how many texts it was asked about."""

    name = "counting"
    supported_types = frozenset({EntityType.PERSON})

    def __init__(self) -> None:
        self.calls: list[str] = []

    def detect(self, doc: NormalizedDoc) -> list[Span]:
        self.calls.append(doc.original)
        i = doc.original.find("Sami")
        return [Span(i, i + 4, EntityType.PERSON, 0.9, self.name)] if i >= 0 else []


# --------------------------------------------------------------------------- the cache itself


def test_get_returns_what_was_put_and_counts() -> None:
    cache = DetectionCache()
    assert cache.get("k") is None
    cache.put("k", [_span()])
    assert cache.get("k") == (_span(),)
    assert (cache.hits, cache.misses, len(cache)) == (1, 1, 1)


def test_least_recently_used_entry_is_evicted() -> None:
    cache = DetectionCache(maxsize=2)
    cache.put("a", [])
    cache.put("b", [])
    assert cache.get("a") == ()  # `a` is now the most recent
    cache.put("c", [])
    assert cache.get("b") is None
    assert cache.get("a") == ()
    assert cache.get("c") == ()


def test_entries_expire_after_the_ttl() -> None:
    clock = Clock()
    cache = DetectionCache(ttl=10, clock=clock)
    cache.put("k", [_span()])
    clock.now = 9.9
    assert cache.get("k") is not None
    clock.now = 10.0
    assert cache.get("k") is None
    assert len(cache) == 0  # an expired entry is dropped, not kept


def test_expired_entries_are_swept_out_on_any_access() -> None:
    clock = Clock()
    cache = DetectionCache(ttl=8, clock=clock)
    cache.put("old", [_span()])
    clock.now = 9
    assert cache.get("other") is None
    assert len(cache) == 0  # `old` is gone though nobody asked for it


def test_maxsize_zero_disables_it() -> None:
    cache = DetectionCache(maxsize=0)
    cache.put("k", [_span()])
    assert cache.get("k") is None
    assert len(cache) == 0
    assert cache.misses == 0  # a disabled cache reports no activity


@pytest.mark.parametrize(("kwargs", "message"), [({"maxsize": -1}, "maxsize"), ({"ttl": 0}, "ttl")])
def test_invalid_settings_are_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DetectionCache(**kwargs)  # type: ignore[arg-type]


def test_key_depends_on_config_and_text_and_is_not_the_text() -> None:
    key = cache_key("config", "Sami Trabelsi")
    assert key == cache_key("config", "Sami Trabelsi")
    assert key != cache_key("other", "Sami Trabelsi")
    assert key != cache_key("config", "Sami Trabelsi.")
    assert "Sami" not in key
    assert len(key) == 64


def test_it_stores_spans_only_and_never_shows_keys() -> None:
    text = "Rappelle Sami Trabelsi au 22 123 456"
    n = Nona()
    n.anonymize(text)
    entries = n.cache._entries  # checking what is kept in memory
    assert len(entries) == 1
    ((key, (_, spans)),) = entries.items()
    assert spans
    assert all(type(s) is Span for s in spans)
    for value in ("Sami", "Trabelsi", "22 123 456", text):
        assert value not in key
        assert all(value not in s.detector for s in spans)
    assert key not in repr(n.cache)
    assert "Sami" not in repr(n.cache)


# --------------------------------------------------------------------------- fingerprints


def test_same_configuration_same_fingerprint() -> None:
    assert fingerprint(default_recognizers()) == fingerprint(default_recognizers())


def test_a_configuration_change_changes_the_fingerprint() -> None:
    assert recognizer_fingerprint(PhoneRecognizer()) != recognizer_fingerprint(
        PhoneRecognizer(include_service_numbers=True)
    )
    assert recognizer_fingerprint(PhoneRecognizer()) != recognizer_fingerprint(PhoneRecognizer(["TN"]))
    assert fingerprint(default_recognizers()) != fingerprint(default_recognizers()[::-1])  # order matters


def test_a_recognizer_without_a_fingerprint_shares_with_no_one() -> None:
    a = Counting()
    assert recognizer_fingerprint(a) != recognizer_fingerprint(a)  # not even with itself, in another Nona
    shared = DetectionCache()
    Nona([a], cache=shared).anonymize("Sami")
    Nona([a], cache=shared).anonymize("Sami")
    assert shared.hits == 0
    n = Nona([a], cache=shared)
    n.anonymize("Sami")
    n.anonymize("Sami")
    assert shared.hits == 1  # one Nona still reuses its own entries


def test_a_subclass_does_not_inherit_its_parents_fingerprint() -> None:
    class Configured(PhoneRecognizer):
        def __init__(self, extra: str) -> None:
            super().__init__()
            self.extra = extra

    assert recognizer_fingerprint(Configured("a")) != recognizer_fingerprint(Configured("a"))
    assert recognizer_fingerprint(PhoneRecognizer()) == recognizer_fingerprint(PhoneRecognizer())


# --------------------------------------------------------------------------- in the pipeline


def test_re_sent_history_is_not_detected_again() -> None:
    counting = Counting()
    n = Nona([counting])
    history = ["Rappelle Sami demain.", "Il a confirmé."]
    first = n.anonymize_messages(history)
    second = n.anonymize_messages([*history, "Merci Sami."])
    assert counting.calls == [*history, "Merci Sami."]  # each text detected once
    assert second.texts[:2] == first.texts


def test_a_config_change_misses_the_cache() -> None:
    shared = DetectionCache()
    text = "appelle le 80 100 200"  # a toll-free service number: masked only when asked
    default = Nona([PhoneRecognizer()], cache=shared)
    services = Nona([PhoneRecognizer(include_service_numbers=True)], cache=shared)
    assert default.anonymize(text).text == text
    assert services.anonymize(text).text == "appelle le PHONE_1"
    assert shared.hits == 0
    assert len(shared) == 2


def test_same_config_shares_a_cache_across_instances() -> None:
    shared = DetectionCache()
    Nona(cache=shared).anonymize("Rappelle Sami Trabelsi")
    Nona(cache=shared).anonymize("Rappelle Sami Trabelsi")
    assert shared.hits == 1


def test_settings_after_detection_still_apply_to_cached_spans() -> None:
    shared = DetectionCache()
    text = "Rappelle Sami Trabelsi"
    assert Nona(cache=shared).anonymize(text).text == "Rappelle PERSON_1"
    assert Nona(cache=shared, allowlist=["Sami Trabelsi"]).anonymize(text).text == text
    assert Nona(cache=shared, min_score=0.99).anonymize(text).text == text
    assert shared.hits == 2


def test_a_failing_recognizer_caches_nothing() -> None:
    class Flaky:
        name = "flaky"
        supported_types = frozenset({EntityType.PERSON})
        fail = True

        def detect(self, doc: NormalizedDoc) -> list[Span]:
            if self.fail:
                raise RuntimeError("boom")
            return []

    flaky = Flaky()
    n = Nona([flaky])
    with pytest.raises(DetectorError):
        n.anonymize("Sami")
    assert len(n.cache) == 0
    flaky.fail = False
    assert n.anonymize("Sami").text == "Sami"


@settings(max_examples=100, deadline=None)
@given(conversation)
def test_the_cache_never_changes_a_result(msgs: list[str]) -> None:
    cached = nona(blind_to=["relancé"])
    uncached = nona(blind_to=["relancé"], cache=DetectionCache(maxsize=0))
    assert cached.anonymize_messages(msgs).texts == uncached.anonymize_messages(msgs).texts
    again = cached.anonymize_messages(msgs)  # now from the cache
    assert again.texts == uncached.anonymize_messages(msgs).texts
    assert again.mapping == uncached.anonymize_messages(msgs).mapping
