"""The NER model on disk: `nona models pull` downloads it, `load_ner_model()` loads it (ADR-0008, NFR-3).

Weights come **only** through `pull`, at pinned revisions, checked file by file against pinned SHA-256
hashes. Loading never touches the network. Everything lives in one folder (`NONA_MODELS_DIR`, default
`~/.cache/nona/models`):

    gliner_multi-v2.1@443d26d654e0/    gliner_config.json, model.safetensors
    mdeberta-v3-base@a0484667b223/     config.json, spm.model, tokenizer_config.json

GLiNER finds its encoder's config and tokenizer through `model_name` in `gliner_config.json`, by Hub name.
`pull` points it at the pinned encoder folder instead, so nothing resolves through a cache's `main`.

Only safetensors weights are fetched or accepted: a pickle (`pytorch_model.bin`) can run code when loaded,
and Nona sits in the path of every prompt (ADR-0008).

This module does file and network I/O, so it lives outside `nona.core`, which only sees the `NerModel`
shape (`nona.core.recognizers.ner`).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nona.core.recognizers.ner import NerPrediction


class ModelError(Exception):
    """The model can't be pulled or loaded. The message says what to do."""


@dataclass(frozen=True, slots=True)
class PinnedFile:
    name: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class PinnedRepo:
    repo: str
    revision: str
    files: tuple[PinnedFile, ...]

    @property
    def folder(self) -> str:
        return f"{self.repo.rsplit('/', 1)[-1]}@{self.revision[:12]}"


GLINER = PinnedRepo(
    "urchade/gliner_multi-v2.1",
    "443d26d654e0324125a96bebd8e796c14ff2efe6",
    (
        PinnedFile(
            "gliner_config.json", "e25f61d91620df84aae8076811ee592e926e94d341b82e7bc1be359718f83017", 477
        ),
        PinnedFile(
            "model.safetensors",
            "2100142f31627531497850659dcb3821c99d5e71c08a8e01a98e4b11ef32a199",
            1155830112,
        ),
    ),
)
ENCODER = PinnedRepo(
    "microsoft/mdeberta-v3-base",
    "a0484667b22365f84929a935b5e50a51f71f159d",
    (
        PinnedFile("config.json", "bcffcd343dc5efa5ef2d5a58d2b405eed108f01cc45b48d0a907b333ec41801f", 579),
        PinnedFile("spm.model", "13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3", 4305025),
        PinnedFile(
            "tokenizer_config.json", "3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d", 52
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class ModelSet:
    """GLiNER and the encoder it loads its config and tokenizer from. Tests pass small fakes."""

    gliner: PinnedRepo = GLINER
    encoder: PinnedRepo = ENCODER

    def local_name(self, repo: PinnedRepo, f: PinnedFile) -> str:
        """Where a pinned file is kept: GLiNER's config under `_ORIGINAL_CONFIG` (see `_link_encoder`)."""
        return _ORIGINAL_CONFIG if repo is self.gliner and f.name == "gliner_config.json" else f.name


PINNED = ModelSet()
# `pull` rewrites `model_name` in GLiNER's config; the pinned original is kept next to it, under this name.
_ORIGINAL_CONFIG = "gliner_config.pinned.json"

# (repo, filename, revision, folder) -> path of the downloaded file
Downloader = Callable[[str, str, str, Path], Path]


def models_dir() -> Path:
    env = os.environ.get("NONA_MODELS_DIR")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "nona" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _hub_download(repo: str, filename: str, revision: str, folder: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports] - `ner` extra
    except ImportError as exc:
        raise ModelError(_INSTALL_HINT) from exc
    return Path(hf_hub_download(repo, filename, revision=revision, local_dir=folder))


_INSTALL_HINT = "the NER model needs the `ner` extra: pip install 'nona-proxy[ner]'"


def pull(
    root: Path | None = None, *, download: Downloader = _hub_download, models: ModelSet = PINNED
) -> Path:
    """Download every pinned file that is missing or wrong, check its hash, and link GLiNER to its encoder.

    Returns the GLiNER folder. Idempotent: files already in place with the right hash aren't fetched again.
    """
    root = models_dir() if root is None else root
    root.mkdir(parents=True, exist_ok=True)
    for pinned in (models.gliner, models.encoder):
        folder = root / pinned.folder
        folder.mkdir(exist_ok=True)
        for f in pinned.files:
            target = folder / models.local_name(pinned, f)
            if target.is_file() and target.stat().st_size == f.size and _sha256(target) == f.sha256:
                continue
            with tempfile.TemporaryDirectory(dir=folder, prefix=".pull-") as tmp:
                got = download(pinned.repo, f.name, pinned.revision, Path(tmp))
                size, digest = got.stat().st_size, _sha256(got)
                if size != f.size or digest != f.sha256:
                    raise ModelError(
                        f"{pinned.repo}@{pinned.revision[:12]}/{f.name}: got {size} bytes, "
                        f"sha256 {digest[:12]}…; expected {f.size} bytes, sha256 {f.sha256[:12]}…. "
                        "Nothing was installed."
                    )
                shutil.move(got, target)
    _link_encoder(root, models)
    return root / models.gliner.folder


def _link_encoder(root: Path, models: ModelSet) -> None:
    """Write GLiNER's config with `model_name` set to the pinned encoder folder (absolute, so the working
    directory doesn't matter). Rewritten on load too, in case the folder was moved."""
    folder = root / models.gliner.folder
    config = json.loads((folder / _ORIGINAL_CONFIG).read_text(encoding="utf-8"))
    config["model_name"] = str((root / models.encoder.folder).resolve())
    target = folder / "gliner_config.json"
    text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if not target.is_file() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")


def check(root: Path | None = None, *, models: ModelSet = PINNED) -> Path:
    """The GLiNER folder, if every pinned file is in place with its pinned size and no pickle sits next to
    the weights. Hashes are checked by `pull`, not on every load (1.1 GB)."""
    root = models_dir() if root is None else root
    missing = []
    for pinned in (models.gliner, models.encoder):
        for f in pinned.files:
            name = models.local_name(pinned, f)
            path = root / pinned.folder / name
            if not path.is_file() or path.stat().st_size != f.size:
                missing.append(f"{pinned.folder}/{name}")
    if missing:
        raise ModelError(
            f"the NER model isn't pulled into {root} ({', '.join(missing)}): run `nona models pull`"
        )
    folder = root / models.gliner.folder
    pickles = sorted(p.name for p in folder.iterdir() if p.suffix in {".bin", ".pt", ".pth", ".pkl", ".ckpt"})
    if pickles:
        raise ModelError(f"refusing to load next to pickle weights {pickles} in {folder}: safetensors only")
    return folder


class GlinerModel:
    """Adapts a loaded GLiNER model to `NerModel`. One call at a time: torch modules aren't documented as
    safe to share across threads, and the proxy runs detection in a worker thread (S5)."""

    def __init__(self, model: Any, identity: str, max_words: int) -> None:
        self._model = model
        self._identity = identity
        self._max_words = max_words
        self._lock = threading.Lock()

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def max_words(self) -> int:
        return self._max_words

    def predict(self, text: str, labels: Sequence[str], threshold: float) -> list[NerPrediction]:
        with self._lock:
            found = self._model.predict_entities(text, list(labels), threshold=threshold, flat_ner=True)
        return [
            NerPrediction(int(e["start"]), int(e["end"]), str(e["label"]), float(e["score"])) for e in found
        ]


def load_ner_model(root: Path | None = None, *, models: ModelSet = PINNED) -> GlinerModel:
    """Load the pulled model, offline. Raises `ModelError` with what to do if it's missing."""
    root = models_dir() if root is None else root
    folder = check(root, models=models)
    _link_encoder(root, models)
    try:
        from gliner import GLiNER  # pyright: ignore[reportMissingImports] - `ner` extra
    except ImportError as exc:
        raise ModelError(_INSTALL_HINT) from exc
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    model = GLiNER.from_pretrained(str(folder), local_files_only=True, load_tokenizer=True)
    model.eval()
    max_words = int(getattr(model.config, "max_len", 384))
    return GlinerModel(model, f"{models.gliner.repo}@{models.gliner.revision}", max_words)
