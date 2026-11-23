"""`nona.models`: pinned pull with hash checks, offline load checks, the CLI (ADR-0008, NFR-3). No network."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from nona import models
from nona.cli import main
from nona.models import GLINER, ModelError, ModelSet, PinnedFile, PinnedRepo, check, load_ner_model, pull

CONFIG = json.dumps({"model_name": "microsoft/mdeberta-v3-base", "max_len": 384}).encode()
FILES = {
    ("org/gliner", "gliner_config.json"): CONFIG,
    ("org/gliner", "model.safetensors"): b"weights" * 100,
    ("org/encoder", "config.json"): b"{}",
    ("org/encoder", "spm.model"): b"tokenizer",
}


def _pinned(repo: str, revision: str) -> PinnedRepo:
    files = tuple(
        PinnedFile(name, hashlib.sha256(data).hexdigest(), len(data))
        for (r, name), data in FILES.items()
        if r == repo
    )
    return PinnedRepo(repo, revision, files)


FAKE = ModelSet(_pinned("org/gliner", "a" * 40), _pinned("org/encoder", "b" * 40))


class FakeHub:
    def __init__(self, tamper: str | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._tamper = tamper

    def __call__(self, repo: str, filename: str, revision: str, folder: Path) -> Path:
        self.calls.append((repo, filename, revision))
        data = FILES[(repo, filename)]
        if filename == self._tamper:
            data = data + b"!"
        path = folder / filename
        path.write_bytes(data)
        return path


def test_pull_fetches_every_pinned_file_at_its_revision_and_links_the_encoder(tmp_path: Path) -> None:
    hub = FakeHub()
    folder = pull(tmp_path, download=hub, models=FAKE)
    assert folder == tmp_path / "gliner@aaaaaaaaaaaa"
    assert sorted(hub.calls) == sorted((r, n, ("a" if r == "org/gliner" else "b") * 40) for r, n in FILES)
    config = json.loads((folder / "gliner_config.json").read_text())
    assert config["model_name"] == str((tmp_path / "encoder@bbbbbbbbbbbb").resolve())
    assert config["max_len"] == 384
    assert (folder / "gliner_config.pinned.json").read_bytes() == CONFIG, "the pinned original is kept"
    assert check(tmp_path, models=FAKE) == folder


def test_pull_is_idempotent(tmp_path: Path) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    again = FakeHub()
    pull(tmp_path, download=again, models=FAKE)
    assert again.calls == []


def test_a_file_with_the_wrong_hash_is_refused_and_not_installed(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match=r"model\.safetensors.*Nothing was installed"):
        pull(tmp_path, download=FakeHub(tamper="model.safetensors"), models=FAKE)
    assert not (tmp_path / "gliner@aaaaaaaaaaaa" / "model.safetensors").exists()
    assert not list((tmp_path / "gliner@aaaaaaaaaaaa").glob(".pull-*")), "no temporary download left behind"


def test_a_corrupted_file_is_fetched_again(tmp_path: Path) -> None:
    folder = pull(tmp_path, download=FakeHub(), models=FAKE)
    (folder / "model.safetensors").write_bytes(b"x" * len(FILES[("org/gliner", "model.safetensors")]))
    hub = FakeHub()
    pull(tmp_path, download=hub, models=FAKE)
    assert hub.calls == [("org/gliner", "model.safetensors", "a" * 40)]


def test_check_says_how_to_pull_when_files_are_missing(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="nona models pull"):
        check(tmp_path, models=FAKE)


def test_pickle_weights_next_to_the_model_are_refused(tmp_path: Path) -> None:
    folder = pull(tmp_path, download=FakeHub(), models=FAKE)
    (folder / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(ModelError, match="safetensors only"):
        check(tmp_path, models=FAKE)


def test_load_needs_the_ner_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    monkeypatch.setitem(sys.modules, "gliner", None)  # makes `import gliner` fail
    with pytest.raises(ModelError, match=r"nona-proxy\[ner\]"):
        load_ner_model(tmp_path, models=FAKE)


def test_load_relinks_the_encoder_when_the_folder_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pull(tmp_path / "old", download=FakeHub(), models=FAKE)
    (tmp_path / "old").rename(tmp_path / "new")
    monkeypatch.setitem(sys.modules, "gliner", None)
    with pytest.raises(ModelError):
        load_ner_model(tmp_path / "new", models=FAKE)
    config = json.loads((tmp_path / "new" / "gliner@aaaaaaaaaaaa" / "gliner_config.json").read_text())
    assert config["model_name"] == str((tmp_path / "new" / "encoder@bbbbbbbbbbbb").resolve())


def test_the_real_pins_are_safetensors_only_and_full_hashes() -> None:
    names = [f.name for f in GLINER.files]
    assert "model.safetensors" in names
    assert not any(n.endswith((".bin", ".pt", ".pth", ".pkl", ".ckpt")) for n in names)
    for repo in (models.GLINER, models.ENCODER):
        assert len(repo.revision) == 40
        assert all(len(f.sha256) == 64 and f.size > 0 for f in repo.files)


def test_models_dir_follows_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NONA_MODELS_DIR", str(tmp_path))
    assert models.models_dir() == tmp_path
    monkeypatch.delenv("NONA_MODELS_DIR")
    assert models.models_dir() == Path.home() / ".cache" / "nona" / "models"


def test_cli_models_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Path] = []

    def fake_pull(root: Path) -> Path:
        seen.append(root)
        return root / "gliner"

    monkeypatch.setattr(models, "pull", fake_pull)
    assert main(["models", "pull", "--dir", str(tmp_path)]) == 0
    assert seen == [tmp_path]
    assert f"ok: {tmp_path / 'gliner'}" in capsys.readouterr().out


def test_cli_models_pull_reports_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing(root: Path) -> Path:
        raise ModelError("sha256 mismatch")

    monkeypatch.setattr(models, "pull", failing)
    assert main(["models", "pull", "--dir", str(tmp_path)]) == 1
    assert "error: sha256 mismatch" in capsys.readouterr().err


def test_bench_with_the_ner_system_and_no_model_says_how_to_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NONA_MODELS_DIR", str(tmp_path))
    assert main(["bench", "--systems", "nona-ner", "--out", str(tmp_path / "out")]) == 1
    assert "nona models pull" in capsys.readouterr().err
