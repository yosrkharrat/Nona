"""`nona.models`: pinned pull with hash checks, offline load checks, the CLI (ADR-0008, NFR-3). No network."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import ClassVar

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
    def __init__(self, tamper: str | None = None, fail: str | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._tamper = tamper
        self._fail = fail

    def __call__(self, repo: str, filename: str, revision: str, folder: Path) -> Path:
        self.calls.append((repo, filename, revision))
        if filename == self._fail:
            raise ConnectionError("no route to huggingface.co")
        data = FILES[(repo, filename)]
        if filename == self._tamper:
            data = data + b"!"
        path = folder / filename
        path.write_bytes(data)
        return path


GLINER_DIR = "gliner@aaaaaaaaaaaa"
ENCODER_DIR = "encoder@bbbbbbbbbbbb"


def _installed(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_pull_fetches_every_pinned_file_at_its_revision_and_stores_it_as_published(tmp_path: Path) -> None:
    hub = FakeHub()
    folder = pull(tmp_path, download=hub, models=FAKE)
    assert folder == tmp_path / GLINER_DIR
    assert sorted(hub.calls) == sorted((r, n, ("a" if r == "org/gliner" else "b") * 40) for r, n in FILES)
    assert (folder / "gliner_config.json").read_bytes() == CONFIG
    assert _installed(tmp_path) == sorted(
        f"{GLINER_DIR if r == 'org/gliner' else ENCODER_DIR}/{n}" for r, n in FILES
    ), "nothing else: no staging folder, no Hub metadata"
    assert check(tmp_path, models=FAKE) == folder


def test_pull_is_idempotent(tmp_path: Path) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    again = FakeHub()
    pull(tmp_path, download=again, models=FAKE)
    assert again.calls == []


@pytest.mark.parametrize("bad", ["gliner_config.json", "model.safetensors", "spm.model"])
def test_a_file_with_the_wrong_hash_means_nothing_is_installed(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ModelError, match=rf"{bad}.*Nothing was installed"):
        pull(tmp_path, download=FakeHub(tamper=bad), models=FAKE)
    assert _installed(tmp_path) == []


def test_a_failed_download_is_a_model_error_and_installs_nothing(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match=r"spm\.model failed \(ConnectionError.*Nothing was installed"):
        pull(tmp_path, download=FakeHub(fail="spm.model"), models=FAKE)
    assert _installed(tmp_path) == []


def test_a_folder_that_cant_be_written_is_a_model_error(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("not a folder")
    with pytest.raises(ModelError, match="can't write the model"):
        pull(blocked / "models", download=FakeHub(), models=FAKE)


def test_a_corrupted_file_is_fetched_again(tmp_path: Path) -> None:
    folder = pull(tmp_path, download=FakeHub(), models=FAKE)
    (folder / "model.safetensors").write_bytes(b"x" * len(FILES[("org/gliner", "model.safetensors")]))
    hub = FakeHub()
    pull(tmp_path, download=hub, models=FAKE)
    assert hub.calls == [("org/gliner", "model.safetensors", "a" * 40)]


def test_check_says_how_to_pull_when_files_are_missing(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="nona models pull"):
        check(tmp_path, models=FAKE)


@pytest.mark.parametrize("where", [GLINER_DIR, ENCODER_DIR])
def test_pickle_weights_in_either_folder_are_refused(tmp_path: Path, where: str) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    (tmp_path / where / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(ModelError, match="safetensors only"):
        check(tmp_path, models=FAKE)


def test_load_needs_the_ner_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    monkeypatch.setitem(sys.modules, "gliner", None)  # makes `import gliner` fail
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ModelError, match=r"uv sync --extra ner.*download\.pytorch\.org/whl/cpu"):
        load_ner_model(tmp_path, models=FAKE)


class _FakeGliner:
    """Stands in for `gliner.GLiNER`: records what `from_pretrained` sees, or raises."""

    seen: ClassVar[dict[str, object]] = {}
    error: ClassVar[Exception | None] = None

    def __init__(self) -> None:
        self.config = types.SimpleNamespace(max_len=384)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> _FakeGliner:
        if cls.error is not None:
            raise cls.error
        folder = Path(path)
        cls.seen = {
            "config": json.loads((folder / "gliner_config.json").read_text()),
            "weights": (folder / "model.safetensors").read_bytes(),
            "kwargs": kwargs,
            "offline": (os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE")),
        }
        return cls()

    def eval(self) -> None:
        pass


@pytest.fixture
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGliner]:
    monkeypatch.setitem(sys.modules, "gliner", types.SimpleNamespace(GLiNER=_FakeGliner))
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.delenv(var, raising=False)  # monkeypatch restores what load sets
    monkeypatch.setattr(_FakeGliner, "error", None)
    return _FakeGliner


def test_load_points_gliner_at_the_pinned_encoder_offline_without_writing(
    tmp_path: Path, fake_gliner: type[_FakeGliner]
) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    for p in tmp_path.rglob("*"):
        p.chmod(0o555 if p.is_dir() else 0o444)  # a read-only models folder (a container image)
    try:
        model = load_ner_model(tmp_path, models=FAKE)
    finally:
        for p in tmp_path.rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)
    seen = fake_gliner.seen
    assert seen["config"] == {"model_name": str((tmp_path / ENCODER_DIR).resolve()), "max_len": 384}
    assert seen["weights"] == FILES[("org/gliner", "model.safetensors")]
    assert seen["kwargs"] == {"local_files_only": True, "load_tokenizer": True}
    assert seen["offline"] == ("1", "1")
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert model.identity == f"org/gliner@{'a' * 40}"
    assert model.max_words == 384


def test_a_load_failure_is_a_model_error(tmp_path: Path, fake_gliner: type[_FakeGliner]) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    fake_gliner.error = ValueError("unexpected key in state dict")
    with pytest.raises(ModelError, match=r"loading the NER model .* failed .*ValueError"):
        load_ner_model(tmp_path, models=FAKE)


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
