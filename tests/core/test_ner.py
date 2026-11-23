"""NER recognizer with a fake model: offsets, threshold, honorifics, windows over long texts (ADR-0008)."""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nona.core import DetectorError, EntityType, Nona, normalize
from nona.core.recognizers import NerPrediction, NerRecognizer
from nona.core.recognizers.ner import _MODEL_WORD, windows

Predict = Callable[[str], Sequence[NerPrediction]]


class FakeModel:
    """Finds each listed name wherever it appears in what it's given, and records what it was given."""

    def __init__(
        self,
        names: dict[str, tuple[str, float]] | None = None,
        *,
        max_words: int = 384,
        identity: str = "fake@1",
        predict: Predict | None = None,
    ) -> None:
        self._names = names or {}
        self._max_words = max_words
        self._identity = identity
        self._predict = predict
        self.calls: list[str] = []

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def max_words(self) -> int:
        return self._max_words

    def predict(self, text: str, labels: Sequence[str], threshold: float) -> list[NerPrediction]:
        assert len(_MODEL_WORD.findall(text)) <= self._max_words, "a window is longer than the model reads"
        self.calls.append(text)
        if self._predict is not None:
            return list(self._predict(text))
        return [
            NerPrediction(m.start(), m.end(), label, score)
            for name, (label, score) in self._names.items()
            for m in re.finditer(re.escape(name), text)
            if score >= threshold
        ]


def detect(model: FakeModel, text: str, **kwargs: float) -> list[tuple[str, EntityType, float]]:
    spans = NerRecognizer(model, **kwargs).detect(normalize(text))  # type: ignore[arg-type]
    return [(text[s.start : s.end], s.type, s.score) for s in spans]


def test_spans_are_in_original_offsets_and_keep_the_model_score() -> None:
    # The normalized view folds case and strips the tatweel, so its offsets differ from the original's.
    text = "RDV avec Hélène Trabelsi à Sfax, et سامـي"
    model = FakeModel(
        {"Hélène Trabelsi": ("person", 0.91), "Sfax": ("location", 0.7), "سامـي": ("person", 0.6)}
    )
    assert detect(model, text) == [
        ("Hélène Trabelsi", EntityType.PERSON, 0.91),
        ("Sfax", EntityType.LOCATION, 0.7),
        ("سامـي", EntityType.PERSON, 0.6),
    ]
    assert model.calls == [text], "the model reads the original text, capitals included"


def test_threshold_and_unknown_labels() -> None:
    model = FakeModel({"Sami": ("person", 0.49), "Ines": ("person", 0.5), "Tunisie": ("country", 0.99)})
    assert detect(model, "Sami, Ines, Tunisie") == [("Ines", EntityType.PERSON, 0.5)]
    assert detect(model, "Sami, Ines, Tunisie", threshold=0.4) == [
        ("Sami", EntityType.PERSON, 0.49),
        ("Ines", EntityType.PERSON, 0.5),
    ]


def test_predictions_outside_the_window_are_ignored() -> None:
    model = FakeModel(predict=lambda text: [NerPrediction(0, len(text) + 5, "person", 0.9)])
    assert detect(model, "Sami") == []


@pytest.mark.parametrize(
    ("text", "predicted", "found"),
    [
        ("appelle Mme Islem demain", "Mme Islem", "Islem"),
        ("le Dr. Sami Haddad", "Dr. Sami Haddad", "Sami Haddad"),
        ("السيدة كنزة", "السيدة كنزة", "كنزة"),
        ("M. Ben Salem", "M. Ben Salem", "Ben Salem"),
        ("Mme  Dr Islem", "Mme  Dr Islem", "Islem"),
    ],
)
def test_leading_honorifics_are_trimmed_from_person_spans(text: str, predicted: str, found: str) -> None:
    model = FakeModel({predicted: ("person", 0.9)})
    assert detect(model, text) == [(found, EntityType.PERSON, 0.9)]


def test_an_honorific_alone_is_dropped_and_other_types_are_not_trimmed() -> None:
    model = FakeModel({"Mme": ("person", 0.9), "Dr Ali Belhouane": ("location", 0.8)})
    assert detect(model, "Mme a dit: rue Dr Ali Belhouane") == [
        ("Dr Ali Belhouane", EntityType.LOCATION, 0.8)
    ]


def test_a_name_past_the_model_window_is_found_in_a_later_window() -> None:
    filler = " ".join(f"mot{i}" for i in range(1000))
    text = f"{filler} signé Kenza Boukadida."
    model = FakeModel({"Kenza Boukadida": ("person", 0.9)}, max_words=100)
    assert detect(model, text) == [("Kenza Boukadida", EntityType.PERSON, 0.9)]
    assert len(model.calls) > 1


def test_a_name_cut_by_a_window_is_whole_in_the_overlap_and_reported_once() -> None:
    words = [f"w{i}" for i in range(30)]
    words[18:20] = ["Kenza", "Boukadida"]  # words 18-19: the first window (0-19 at most 20) holds both
    text = " ".join(words)
    model = FakeModel({"Kenza Boukadida": ("person", 0.9)}, max_words=20)
    assert detect(model, text, overlap_words=5) == [("Kenza Boukadida", EntityType.PERSON, 0.9)]
    words[18:20], words[19:21] = ["w18", "w19"], ["Kenza", "Boukadida"]  # now cut by the first window's end
    text = " ".join(words)
    assert detect(model, text, overlap_words=5) == [("Kenza Boukadida", EntityType.PERSON, 0.9)]


@given(
    st.lists(st.sampled_from(["mot", "Sami", ",", "l'école", "3andi", "سامي", "22-33", "\n"]), max_size=300),
    st.integers(min_value=2, max_value=60),
    st.data(),
)
def test_windows_cover_every_word_with_the_given_overlap(
    words: list[str], max_words: int, data: st.DataObject
) -> None:
    overlap = data.draw(st.integers(min_value=0, max_value=max_words - 1))
    text = " ".join(words)
    spans = [m.span() for m in _MODEL_WORD.finditer(text)]
    ws = windows(text, max_words, overlap)
    if not spans:
        assert ws == []
        return
    covered: list[list[int]] = []
    for start, end in ws:
        inside = [i for i, (s, e) in enumerate(spans) if s >= start and e <= end]
        assert 0 < len(inside) <= max_words
        assert len(_MODEL_WORD.findall(text[start:end])) == len(inside), "windows cut between words"
        covered.append(inside)
    assert sorted({i for c in covered for i in c}) == list(range(len(spans)))
    for a, b in itertools.pairwise(covered):
        assert len(set(a) & set(b)) == overlap


def test_a_text_that_fits_is_one_call_and_an_empty_one_none() -> None:
    model = FakeModel()
    NerRecognizer(model).detect(normalize("Sami et Ines"))
    NerRecognizer(model).detect(normalize("   "))
    assert model.calls == ["Sami et Ines"]


def test_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="threshold"):
        NerRecognizer(FakeModel(), threshold=0)
    with pytest.raises(ValueError, match="overlap"):
        NerRecognizer(FakeModel(max_words=10), overlap_words=5)


def test_the_fingerprint_names_the_model_and_the_settings() -> None:
    a = NerRecognizer(FakeModel(identity="m@1")).fingerprint
    assert "m@1" in a
    assert a != NerRecognizer(FakeModel(identity="m@2")).fingerprint
    assert a != NerRecognizer(FakeModel(identity="m@1"), threshold=0.6).fingerprint


def test_a_new_model_revision_misses_the_detection_cache() -> None:
    old = FakeModel({"Chiheb": ("person", 0.9)}, identity="m@1")
    new = FakeModel({"Chiheb": ("person", 0.9)}, identity="m@2")
    shared = Nona([NerRecognizer(old)])
    shared.anonymize("bonjour Chiheb")
    Nona([NerRecognizer(new)], cache=shared.cache).anonymize("bonjour Chiheb")
    assert new.calls == ["bonjour Chiheb"]


def test_a_model_error_fails_closed() -> None:
    def broken(text: str) -> list[NerPrediction]:
        raise RuntimeError("out of memory")

    with pytest.raises(DetectorError, match="'ner'"):
        Nona([NerRecognizer(FakeModel(predict=broken))]).anonymize("bonjour Chiheb")


def test_in_the_pipeline_a_held_out_name_is_masked_and_linked() -> None:
    model = FakeModel({"Chiheb Drira": ("person", 0.92), "Chiheb": ("person", 0.85)})
    from nona.core.recognizers import default_recognizers

    conv = Nona([*default_recognizers(), NerRecognizer(model)]).anonymize_messages(
        ["Rappelle Chiheb demain.", "Chiheb Drira a envoyé le devis."]
    )
    assert conv.texts == ("Rappelle PERSON_1 demain.", "PERSON_1 a envoyé le devis.")
    assert conv.mapping.value("PERSON_1") == "Chiheb Drira"
