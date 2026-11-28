"""The real NER model (ADR-0008). Skipped unless the `ner` extra is installed and `nona models pull` was run.

Run with `uv run --extra ner pytest -m model`. CI runs them in their own job with the model cached (#73).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from unittest import mock

import pytest

from nona.core import EntityType, Nona, normalize
from nona.core.recognizers import NerRecognizer, default_recognizers
from nona.models import GLINER, ModelError, check

pytestmark = pytest.mark.model


def _missing() -> str | None:
    """Why the model can't run here, or `None`."""
    if importlib.util.find_spec("gliner") is None:
        return "the `ner` extra isn't installed"
    try:
        check()
    except ModelError as exc:
        return str(exc)
    return None


# CI's model job sets NONA_MODEL_TESTS=required with `-m model`, so a missing model fails the job instead of
# skipping every test (#73). Only with `-m model`: it fails at import, before other runs deselect these.
if (why := _missing()) is not None:
    if os.environ.get("NONA_MODEL_TESTS") == "required":
        pytest.fail(f"NONA_MODEL_TESTS=required, but {why}", pytrace=False)
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
        # Arabizi: on the original text the model calls the name a place under the threshold (0.38), and
        # with a final period takes `Mootez Jaouadi ghodwa` as one location (0.59). The de-Arabizi'd view,
        # the `m3a` cue joining the pieces (0.25 and 0.42) and span cleaning make it one PERSON (#72 part 2).
        ("3andi rendez-vous m3a Mootez Jaouadi ghodwa", "Mootez Jaouadi"),
        ("3andi rendez-vous m3a Mootez Jaouadi ghodwa.", "Mootez Jaouadi"),
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


# NFR-7 counts only memory the OS can't reclaim (ADR-0008). This checks peak RSS ≤ 1.5 GB on a fixed workload,
# which is stricter and portable: it fails if the weights stop being mapped from their file. A fresh process,
# so the peak is the load's and this workload's, not the test session's. Long messages in each script touch
# more of the embedding table than short ones.
_MEMORY = textwrap.dedent(
    """
    import json, resource, sys, warnings
    from nona.core import Nona
    from nona.core.recognizers import NerRecognizer, default_recognizers
    from nona.models import load_ner_model

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = load_ner_model()
    nona = Nona([*default_recognizers(), NerRecognizer(model)])
    texts = [
        "Rendez-vous demain avec Chiheb Drira au bureau, puis appelle Mme Islem Ghariani. " * 12,
        "3andi rendez-vous m3a Mootez Jaouadi ghodwa, 9alli Hazem elli el contrat wsel. " * 12,
        "موعد مع إسلام الشتيوي غدا في صفاقس، وقال لي السيد منير القاسمي إن الملف جاهز. " * 12,
        "Call Kenza Boukadida in London about the Acme Corp contract before Friday. " * 12,
    ]
    for text in texts:
        nona.anonymize(text)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024  # bytes on macOS, KiB on Linux
    print(json.dumps({"peak_bytes": peak_bytes, "warnings": [str(w.message) for w in caught]}))
    """
)


def test_the_model_fits_in_nfr7_memory() -> None:
    command = [sys.executable, "-c", _MEMORY]  # this interpreter and a fixed script
    done = subprocess.run(command, capture_output=True, text=True, timeout=600)  # noqa: S603
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    # gliner falls back to reading every weight into memory, with only a warning, when it can't map them.
    assert not [w for w in result["warnings"] if "low_cpu_mem_usage" in w], result["warnings"]
    assert result["peak_bytes"] <= 1.5e9, result
