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


@pytest.fixture
def isolated_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo everything `_go_offline` changes, so no later test inherits an offline process."""
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(var, "unset")  # registers the original state (absent or not) for undo
        monkeypatch.delenv(var)
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        monkeypatch.setattr(constants, "HF_HUB_OFFLINE", constants.HF_HUB_OFFLINE)
    monkeypatch.setattr(models, "_went_offline", False)


@pytest.mark.usefixtures("isolated_offline")
def test_load_needs_the_ner_extra_and_stays_online_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    monkeypatch.setitem(sys.modules, "gliner", None)  # makes `import gliner` fail
    with pytest.raises(
        ModelError, match=r"uv sync --extra ner.*--index-url https://download\.pytorch\.org/whl/cpu"
    ):
        load_ner_model(tmp_path, models=FAKE)
    assert "HF_HUB_OFFLINE" not in os.environ, "a failed load must not switch the process offline"
    assert models._went_offline is False
    assert "--extra-index-url" not in models._INSTALL_HINT, "pip would pick torch from either index"


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
        config = json.loads((folder / "gliner_config.json").read_text())
        encoder = Path(config["model_name"])
        cls.seen = {
            "config": config,
            "gliner_files": sorted(p.name for p in folder.iterdir()),
            "encoder_files": {p.name: p.read_bytes() for p in encoder.iterdir()},
            "weights": (folder / "model.safetensors").read_bytes(),
            "kwargs": kwargs,
            "offline": (os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE")),
        }
        return cls()

    def eval(self) -> None:
        pass


@pytest.fixture
def fake_gliner(monkeypatch: pytest.MonkeyPatch, isolated_offline: None) -> type[_FakeGliner]:
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGliner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
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
    config = seen["config"]
    assert isinstance(config, dict)
    assert config["max_len"] == 384
    assert not str(config["model_name"]).startswith(str(tmp_path)), "a staged copy, not the shared folder"
    assert seen["gliner_files"] == ["gliner_config.json", "model.safetensors"]
    assert seen["encoder_files"] == {n: d for (r, n), d in FILES.items() if r == "org/encoder"}
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


def test_a_small_file_edited_after_pull_is_refused_on_load(tmp_path: Path) -> None:
    folder = pull(tmp_path, download=FakeHub(), models=FAKE)
    config = folder / "gliner_config.json"
    config.write_bytes(CONFIG.replace(b"384", b"999"))  # same size, different content
    with pytest.raises(ModelError, match=r"gliner_config\.json .*no longer match their pinned hash"):
        check(tmp_path, models=FAKE)


def test_pull_reports_pickles_instead_of_saying_ok(tmp_path: Path) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    (tmp_path / ENCODER_DIR / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(ModelError, match="safetensors only"):
        pull(tmp_path, download=FakeHub(), models=FAKE)


def test_pull_removes_staging_left_by_a_killed_pull_but_not_a_running_one(tmp_path: Path) -> None:
    old = tmp_path / ".pull-killed"
    (old / "x").mkdir(parents=True)
    (old / "x" / "model.safetensors").write_bytes(b"partial")
    for p in (old / "x" / "model.safetensors", old / "x", old):
        os.utime(p, (0, 0))
    running = tmp_path / ".pull-running"  # its folders are old, but it is still writing the weights
    (running / "x").mkdir(parents=True)
    (running / "x" / "model.safetensors").write_bytes(b"growing")
    for p in (running / "x", running):
        os.utime(p, (0, 0))
    pull(tmp_path, download=FakeHub(), models=FAKE)
    assert not old.exists()
    assert running.exists()


def test_a_downloader_that_returns_no_file_is_named_in_the_error(tmp_path: Path) -> None:
    def lost(repo: str, filename: str, revision: str, folder: Path) -> Path:
        return folder / "elsewhere" / filename

    with pytest.raises(ModelError, match=r"downloading org/gliner@aaaaaaaaaaaa/.* gave no file"):
        pull(tmp_path, download=lost, models=FAKE)
    assert _installed(tmp_path) == []


def test_pull_after_a_load_says_to_use_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(models, "_went_offline", True)
    with pytest.raises(ModelError, match="pull from a separate process"):
        pull(tmp_path, download=FakeHub(fail="spm.model"), models=FAKE)


def test_load_copies_the_weights_where_symlinks_are_not_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gliner: type[_FakeGliner]
) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("symbolic links need privileges")  # Windows without Developer Mode

    def no_hard_links(*args: object, **kwargs: object) -> None:
        raise AssertionError("a hard link shares the original's metadata: never used")

    monkeypatch.setattr(Path, "symlink_to", refuse)
    monkeypatch.setattr(os, "link", no_hard_links)
    load_ner_model(tmp_path, models=FAKE)
    assert fake_gliner.seen["weights"] == FILES[("org/gliner", "model.safetensors")]


def test_a_file_dropped_next_to_the_pinned_encoder_is_never_seen_by_the_loader(
    tmp_path: Path, fake_gliner: type[_FakeGliner]
) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    (tmp_path / ENCODER_DIR / "tokenizer.json").write_text("{}")  # transformers would prefer it to spm.model
    (tmp_path / GLINER_DIR / "tokenizer_config.json").write_text("{}")
    load_ner_model(tmp_path, models=FAKE)
    assert "tokenizer.json" not in fake_gliner.seen["encoder_files"]  # type: ignore[operator]
    assert fake_gliner.seen["gliner_files"] == ["gliner_config.json", "model.safetensors"]


def test_an_unreadable_file_is_a_model_error(tmp_path: Path) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    config = tmp_path / ENCODER_DIR / "config.json"
    config.chmod(0o000)
    try:
        if os.access(config, os.R_OK):
            pytest.skip("running as a user who can read anything")
        with pytest.raises(ModelError, match="can't read the NER model"):
            check(tmp_path, models=FAKE)
    finally:
        config.chmod(0o644)


@pytest.mark.usefixtures("isolated_offline")
def test_a_broken_gliner_import_leaves_the_process_online(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pull(tmp_path, download=FakeHub(), models=FAKE)
    monkeypatch.setitem(sys.modules, "gliner", types.ModuleType("gliner"))  # installed, but no GLiNER in it
    with pytest.raises(ModelError, match="importing it failed"):
        load_ner_model(tmp_path, models=FAKE)
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert models._went_offline is False


def test_cli_pull_into_another_folder_says_how_to_use_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(models, "pull", lambda root: root / "gliner")
    monkeypatch.delenv("NONA_MODELS_DIR", raising=False)
    spaced = tmp_path / "my models"
    spaced.mkdir()
    monkeypatch.chdir(tmp_path)
    assert main(["models", "pull", "--dir", "my models"]) == 0  # relative: printed absolute, and quoted
    assert f"set NONA_MODELS_DIR='{spaced.resolve()}'" in capsys.readouterr().out
    monkeypatch.setenv("NONA_MODELS_DIR", str(tmp_path))
    assert main(["models", "pull", "--dir", str(tmp_path)]) == 0
    assert "NONA_MODELS_DIR" not in capsys.readouterr().out
