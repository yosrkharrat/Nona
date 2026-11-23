"""The NER model on disk: `nona models pull` downloads it, `load_ner_model()` loads it (ADR-0008, NFR-3).

Weights come **only** through `pull`, at pinned revisions, checked file by file against pinned SHA-256
hashes. Loading never touches the network. Everything lives in one folder (`NONA_MODELS_DIR`, default
`~/.cache/nona/models`):

    gliner_multi-v2.1@443d26d654e0/    gliner_config.json, model.safetensors
    mdeberta-v3-base@a0484667b223/     config.json, spm.model, tokenizer_config.json

The files are stored exactly as published. GLiNER finds its encoder's config and tokenizer through
`model_name` in `gliner_config.json`, by Hub name; loading points it at the pinned encoder folder instead
(`load_ner_model`), so nothing resolves through a cache's `main`, and the models folder is never written to
after `pull`.

Only safetensors weights are fetched or accepted: a pickle (`pytorch_model.bin`) can run code when loaded,
and Nona sits in the path of every prompt (ADR-0008).

This module does file and network I/O, so it lives outside `nona.core`, which only sees the `NerModel`
shape (`nona.core.recognizers.ner`).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
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

    @property
    def repos(self) -> tuple[PinnedRepo, PinnedRepo]:
        return (self.gliner, self.encoder)


PINNED = ModelSet()
_PICKLES = {".bin", ".pt", ".pth", ".pkl", ".ckpt"}
_INSTALL_HINT = (
    "the NER model needs the `ner` extra: `uv sync --extra ner` in a checkout, or "
    "`pip install 'nona-proxy[ner]' --extra-index-url https://download.pytorch.org/whl/cpu` "
    "(CPU-only torch; without the index, Linux gets several GB of CUDA libraries)"
)

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


def _is_pinned(path: Path, f: PinnedFile) -> bool:
    return path.is_file() and path.stat().st_size == f.size and _sha256(path) == f.sha256


def _hub_download(repo: str, filename: str, revision: str, folder: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports] - `ner` extra
    except ImportError as exc:
        raise ModelError(_INSTALL_HINT) from exc
    return Path(hf_hub_download(repo, filename, revision=revision, local_dir=folder))


def pull(
    root: Path | None = None, *, download: Downloader = _hub_download, models: ModelSet = PINNED
) -> Path:
    """Download every pinned file that is missing or wrong, and install them once all their hashes match.

    Files are fetched into a staging folder and moved into place only when every file of both repos has
    its pinned size and hash, so a failed pull leaves what was installed before untouched. Idempotent: files
    already in place aren't fetched again. Returns the GLiNER folder.
    """
    root = models_dir() if root is None else root
    try:
        root.mkdir(parents=True, exist_ok=True)
        todo = [(p, f) for p in models.repos for f in p.files if not _is_pinned(root / p.folder / f.name, f)]
        if todo:
            with tempfile.TemporaryDirectory(dir=root, prefix=".pull-") as tmp:
                staged = [(p, f, _fetch(download, p, f, Path(tmp))) for p, f in todo]
                for p, f, path in staged:
                    (root / p.folder).mkdir(exist_ok=True)
                    os.replace(path, root / p.folder / f.name)
    except OSError as exc:
        raise ModelError(f"can't write the model into {root}: {exc}") from exc
    return root / models.gliner.folder


def _fetch(download: Downloader, pinned: PinnedRepo, f: PinnedFile, staging: Path) -> Path:
    where = f"{pinned.repo}@{pinned.revision[:12]}/{f.name}"
    folder = (
        staging / f"{pinned.folder}-{f.name.replace('/', '_')}"
    )  # one per file: the Hub client adds its own
    folder.mkdir()  # metadata next to what it downloads
    try:
        got = download(pinned.repo, f.name, pinned.revision, folder)
    except ModelError:
        raise
    except Exception as exc:  # network, missing revision or file, rate limits: any download failure
        raise ModelError(
            f"downloading {where} failed ({type(exc).__name__}: {exc}). Nothing was installed."
        ) from exc
    size, digest = got.stat().st_size, _sha256(got)
    if size != f.size or digest != f.sha256:
        raise ModelError(
            f"{where}: got {size} bytes, sha256 {digest[:12]}…; expected {f.size} bytes, "
            f"sha256 {f.sha256[:12]}…. Nothing was installed."
        )
    return got


def check(root: Path | None = None, *, models: ModelSet = PINNED) -> Path:
    """The GLiNER folder, if every pinned file is in place with its pinned size and neither folder holds
    pickle weights. Hashes are checked by `pull`, not on every load (1.1 GB). Never writes."""
    root = models_dir() if root is None else root
    missing = [
        f"{pinned.folder}/{f.name}"
        for pinned in models.repos
        for f in pinned.files
        if not (root / pinned.folder / f.name).is_file()
        or (root / pinned.folder / f.name).stat().st_size != f.size
    ]
    if missing:
        raise ModelError(
            f"the NER model isn't pulled into {root} ({', '.join(missing)}): run `nona models pull`"
        )
    for pinned in models.repos:
        folder = root / pinned.folder
        pickles = sorted(p.name for p in folder.iterdir() if p.suffix in _PICKLES)
        if pickles:
            raise ModelError(
                f"refusing to load next to pickle weights {pickles} in {folder}: safetensors only"
            )
    return root / models.gliner.folder


def _go_offline() -> None:
    """Make the Hub and transformers refuse the network for the rest of the process (NFR-3). The variables
    are read when those libraries are imported, so set them first, and patch the Hub's constant if it is
    already imported (`pull` imports it)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        setattr(constants, "HF_HUB_OFFLINE", True)  # noqa: B010 - a module attribute pyright can't see


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
    """Load the pulled model, offline, without writing to the models folder.

    GLiNER finds its encoder through `model_name` in its config, a Hub name. The config is rewritten in a
    temporary folder, next to a link to the weights, with `model_name` set to the pinned encoder folder: so
    nothing resolves by name, and a read-only or shared models folder works.
    """
    root = models_dir() if root is None else root
    folder = check(root, models=models)
    _go_offline()
    try:
        from gliner import GLiNER  # pyright: ignore[reportMissingImports] - `ner` extra
    except ImportError as exc:
        raise ModelError(_INSTALL_HINT) from exc
    try:
        config = json.loads((folder / "gliner_config.json").read_text(encoding="utf-8"))
        config["model_name"] = str((root / models.encoder.folder).resolve())
        with tempfile.TemporaryDirectory(prefix="nona-gliner-") as tmp:
            staged = Path(tmp)
            (staged / "gliner_config.json").write_text(json.dumps(config), encoding="utf-8")
            for f in models.gliner.files:
                if f.name != "gliner_config.json":
                    (staged / f.name).symlink_to((folder / f.name).resolve())
            model = GLiNER.from_pretrained(str(staged), local_files_only=True, load_tokenizer=True)
    except Exception as exc:  # a broken file, an incompatible library: fail closed with the cause attached
        raise ModelError(f"loading the NER model from {folder} failed ({type(exc).__name__}: {exc})") from exc
    model.eval()
    max_words = int(getattr(model.config, "max_len", 384))
    return GlinerModel(model, f"{models.gliner.repo}@{models.gliner.revision}", max_words)
