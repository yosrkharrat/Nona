"""The real NER model (ADR-0008). Skipped unless the `ner` extra is installed and `nona models pull` was run.

Run with `uv run --extra ner pytest -m model`. CI runs them in their own job with the model cached (#73).
"""

from __future__ import annotations

import importlib.util
from unittest import mock

import pytest

from nona.core import EntityType, Nona, normalize
from nona.core.recognizers import NerRecognizer, default_recognizers
from nona.models import GLINER, ModelError, check

pytestmark = pytest.mark.model


def _pulled() -> bool:
    try:
        check()
    except ModelError:
        return False
    return True


if importlib.util.find_spec("gliner") is None or not _pulled():
    pytest.skip("needs the `ner` extra and `nona models pull`", allow_module_level=True)

from nona.models import load_ner_model  # noqa: E402


def _refuse(*args: object, **kwargs: object) -> None:
    raise OSError("the network was used while loading the model (NFR-3)")


@pytest.fixture(scope="module")
def recognizer() -> NerRecognizer:
    # Loading must never reach the network: any connection attempt fails the whole module.
    with mock.patch("socket.socket.connect", _refuse), mock.patch("socket.create_connection", _refuse):
        return NerRecognizer(load_ner_model())


def test_the_pinned_model_loads_offline(recognizer: NerRecognizer) -> None:
    assert f"{GLINER.repo}@{GLINER.revision}" in recognizer.fingerprint
    assert "max_words=384" in recognizer.fingerprint


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Rendez-vous demain avec Chiheb Drira au bureau.", "Chiheb Drira"),
        pytest.param(
            "3andi rendez-vous m3a Mootez Jaouadi ghodwa",
            "Mootez Jaouadi",
            # In Arabizi the model calls the name a place, under the threshold (0.38), and with a final period
            # takes `Mootez Jaouadi ghodwa` as one location (0.59): ADR-0008 conditions 1 and 2, #72 part 2.
            marks=pytest.mark.xfail(reason="Arabizi: name typed as a location, low score", strict=True),
        ),
        ("موعد مع إسلام الشتيوي غدا", "إسلام الشتيوي"),
    ],
)
def test_held_out_names_are_found(recognizer: NerRecognizer, text: str, name: str) -> None:
    spans = recognizer.detect(normalize(text))
    assert any(text[s.start : s.end] == name and s.type is EntityType.PERSON for s in spans), spans


def test_a_name_after_the_model_window_is_found(recognizer: NerRecognizer) -> None:
    body = " ".join(["Le projet avance bien, la livraison est prévue pour la semaine prochaine."] * 60)
    text = f"{body}\n\nCordialement,\nKenza Boukadida"
    assert len(text.split()) > 600
    spans = recognizer.detect(normalize(text))
    assert any(text[s.start : s.end] == "Kenza Boukadida" for s in spans), spans


def test_mme_is_not_part_of_the_name(recognizer: NerRecognizer) -> None:
    text = "J'ai appelé Mme Islem Ghariani ce matin."
    spans = recognizer.detect(normalize(text))
    assert all(not text[s.start : s.end].startswith("Mme") for s in spans), spans


def test_in_the_pipeline(recognizer: NerRecognizer) -> None:
    conv = Nona([*default_recognizers(), recognizer]).anonymize_messages(
        ["Rappelle Chiheb Drira demain.", "Chiheb Drira a confirmé."]
    )
    assert conv.texts == ("Rappelle PERSON_1 demain.", "PERSON_1 a confirmé.")
