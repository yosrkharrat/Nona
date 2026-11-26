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
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
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
    "the NER model needs the `ner` extra: `uv sync --extra ner` in a checkout, or with pip, two steps: "
    "first `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU-only torch from its own "
    "index), then `pip install 'nona-proxy[ner]'`"
)
_STALE_STAGING_S = 24 * 3600  # a `.pull-*` folder this old was left by a pull that was killed
_HASHED_ON_LOAD_MAX = 16 << 20  # files up to this size (configs, tokenizer) are hashed on every load
_went_offline = False  # whether `load_ner_model` switched this process offline (see `_import_gliner_offline`)

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
    its pinned size and hash, so a failed download or hash check installs nothing. A disk error while
    moving them into place can leave some files updated; each of those was verified, and pulling again
    finishes the job. Idempotent: files already in place aren't fetched again. Ends with `check`, so a
    folder that loading would refuse (pickle weights) is reported now. Returns the GLiNER folder.
    """
    root = models_dir() if root is None else root
    try:
        root.mkdir(parents=True, exist_ok=True)
        _remove_stale_staging(root)
        todo = [(p, f) for p in models.repos for f in p.files if not _is_pinned(root / p.folder / f.name, f)]
        if todo:
            with tempfile.TemporaryDirectory(dir=root, prefix=".pull-") as tmp:
                staged = [(p, f, _fetch(download, p, f, Path(tmp))) for p, f in todo]
                for p, f, path in staged:
                    (root / p.folder).mkdir(exist_ok=True)
                    os.replace(path, root / p.folder / f.name)
    except OSError as exc:
        raise ModelError(f"can't write the model into {root}: {exc}") from exc
    return check(root, models=models)


def _remove_stale_staging(root: Path) -> None:
    """Delete `.pull-*` folders left by a pull that was killed (they can hold most of 1.2 GB). Only ones in
    which nothing changed for a day: a pull downloading in another process keeps writing to its folder.
    Another pull removing the same folder at the same time is fine."""
    cutoff = time.time() - _STALE_STAGING_S
    for old in root.glob(".pull-*"):
        try:
            if old.is_symlink() or not old.is_dir():
                continue
            newest = max((p.lstat().st_mtime for p in old.rglob("*")), default=old.lstat().st_mtime)
            if max(newest, old.lstat().st_mtime) < cutoff:
                shutil.rmtree(old, ignore_errors=True)  # best effort: remove all it can
        except OSError:
            continue


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
        hint = (
            " This process loaded the model and is offline for good (NFR-3): pull from a separate process."
            if _went_offline
            else ""
        )
        raise ModelError(
            f"downloading {where} failed ({type(exc).__name__}: {exc}). Nothing was installed.{hint}"
        ) from exc
    if not got.is_file():
        raise ModelError(f"downloading {where} gave no file at {got}. Nothing was installed.")
    size, digest = got.stat().st_size, _sha256(got)
    if size != f.size or digest != f.sha256:
        raise ModelError(
            f"{where}: got {size} bytes, sha256 {digest[:12]}…; expected {f.size} bytes, "
            f"sha256 {f.sha256[:12]}…. Nothing was installed."
        )
    return got


def _verified_small_files(root: Path, models: ModelSet) -> dict[tuple[str, str], bytes]:
    """The bytes of every small pinned file (configs, tokenizer), read once and checked against the pinned
    hash. Loading uses these bytes, not a second read, so a file can't change between check and use."""
    out: dict[tuple[str, str], bytes] = {}
    changed: list[str] = []
    for pinned in models.repos:
        for f in pinned.files:
            if f.size > _HASHED_ON_LOAD_MAX:
                continue
            with (root / pinned.folder / f.name).open("rb") as fh:
                data = fh.read(f.size + 1)  # never more than one byte past the pinned size
            if len(data) != f.size or hashlib.sha256(data).hexdigest() != f.sha256:
                changed.append(f"{pinned.folder}/{f.name}")
            out[(pinned.repo, f.name)] = data
    if changed:
        raise ModelError(
            f"{', '.join(changed)} in {root} no longer match their pinned hash: run `nona models pull`"
        )
    return out


def check(root: Path | None = None, *, models: ModelSet = PINNED) -> Path:
    """The GLiNER folder, if every pinned file is in place and neither folder holds pickle weights.

    Small files (configs, tokenizer) are checked against their pinned hash every time: a config edited in a
    shared folder could change what loads. The 1.1 GB weights are checked by size here and by hash in
    `pull`. Never writes. Any read error is a `ModelError`."""
    root = models_dir() if root is None else root
    try:
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
        _verified_small_files(root, models)
        for pinned in models.repos:
            folder = root / pinned.folder
            pickles = sorted(p.name for p in folder.iterdir() if p.suffix in _PICKLES)
            if pickles:
                raise ModelError(
                    f"refusing to load next to pickle weights {pickles} in {folder}: safetensors only"
                )
    except OSError as exc:
        raise ModelError(f"can't read the NER model in {root}: {exc}") from exc
    return root / models.gliner.folder


def _has_gliner() -> bool:
    """Whether `gliner` is installed, without importing it (that would import transformers online)."""
    if "gliner" in sys.modules:
        return sys.modules["gliner"] is not None  # None: imports of it are blocked
    try:
        return importlib.util.find_spec("gliner") is not None
    except (ImportError, ValueError):
        return False


_OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")


def _import_gliner_offline() -> Any:
    """Import `gliner.GLiNER` with the Hub and transformers offline, **for the rest of the process** (NFR-3).

    Those libraries read the setting when they are imported, so it can't be scoped to one call: the
    variables are set first, and the Hub's constant is patched if `pull` already imported it. If the import
    fails (the extra is missing, or torch is broken), everything is put back, so the process stays online
    and a `pull` can still run in it. Once it succeeds, a later `pull` in the same process fails, and its
    error says to pull from a separate process.
    """
    global _went_offline
    if not _has_gliner():
        raise ModelError(_INSTALL_HINT)
    saved_env = {var: os.environ.get(var) for var in _OFFLINE_VARS}
    constants = sys.modules.get("huggingface_hub.constants")
    saved_constant = getattr(constants, "HF_HUB_OFFLINE", None) if constants is not None else None
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if constants is not None:
        setattr(constants, "HF_HUB_OFFLINE", True)  # noqa: B010 - a module attribute pyright can't see
    try:
        from gliner import GLiNER  # pyright: ignore[reportMissingImports] - `ner` extra
    except Exception as exc:  # ImportError, or a broken torch raising anything while importing
        for var, value in saved_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        hub = sys.modules.get("huggingface_hub.constants")
        if hub is not None:
            setattr(hub, "HF_HUB_OFFLINE", bool(saved_constant))  # noqa: B010
        raise ModelError(f"{_INSTALL_HINT} (importing it failed: {type(exc).__name__}: {exc})") from exc
    _went_offline = True
    return GLiNER


def _link_or_copy(source: Path, target: Path) -> None:
    """A symlink, or a copy where symlinks need privileges (Windows without Developer Mode, which NFR-8
    doesn't list as supported: there every load copies the 1.1 GB weights into the temp folder). Never a
    hard link: it shares the file's metadata, and temp-folder cleanup could change the shared original."""
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copyfile(source, target)


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

    Both folders are rebuilt in a temporary folder that holds **only the pinned files**: the small ones
    written from the bytes just verified, the weights linked (or copied). GLiNER's `model_name`, a Hub name,
    is set to that encoder copy. So nothing resolves by name, a file dropped next to the pinned ones (a
    `tokenizer.json`, which transformers would prefer) is never seen, the small files can't change between
    the check and the load, and a read-only or shared models folder works. The weights are read through
    the link: checked by size here, by hash only in `pull` (hashing 1.1 GB on every start is too slow).

    **Memory (NFR-7).** Through a link, the weights are memory-mapped from the safetensors file, not read
    into memory: GLiNER builds the model on the `meta` device and puts the file's tensors in place
    (`low_cpu_mem_usage`). Only the pages inference reads are resident, and they are clean file pages the
    kernel can drop and read again: most of the 250k-row multilingual embedding table is never read. Loading
    peaked at 2.3-2.9 GB, and RSS was 1.88 GB after the `dev` split; now loading peaks at about 0.6 GB, and
    RSS is 1.27 GB after `dev` (ADR-0008, `bench/resources.py`). The file stays mapped for the life of the
    process: the link goes with the temp folder, the mapping doesn't. `pull` replaces files (`os.replace`),
    never writes into them, so a running process keeps the file it mapped. A copy (no symlinks) is read into
    memory as before, since the temp folder holding it is deleted.
    """
    root = models_dir() if root is None else root
    folder = check(root, models=models)
    GLiNER = _import_gliner_offline()  # noqa: N806 - a class
    try:
        small = _verified_small_files(root, models)
        with tempfile.TemporaryDirectory(prefix="nona-gliner-") as tmp:
            staged = {repo: Path(tmp) / repo.folder for repo in models.repos}
            for repo, target in staged.items():
                target.mkdir()
                for f in repo.files:
                    if (repo.repo, f.name) in small:
                        (target / f.name).write_bytes(small[(repo.repo, f.name)])
                    else:
                        _link_or_copy((root / repo.folder / f.name).resolve(), target / f.name)
            # Map the weights only if they are links into the models folder, which outlives the temp one.
            weights = list(staged[models.gliner].glob("*.safetensors"))
            mapped = bool(weights) and all(w.is_symlink() for w in weights)
            config_path = staged[models.gliner] / "gliner_config.json"
            config = json.loads(config_path.read_bytes())
            config["model_name"] = str(staged[models.encoder])
            config_path.write_text(json.dumps(config), encoding="utf-8")
            model = GLiNER.from_pretrained(
                str(staged[models.gliner]),
                local_files_only=True,
                load_tokenizer=True,
                low_cpu_mem_usage=mapped,
            )
    except ModelError:
        raise
    except Exception as exc:  # a broken file, an incompatible library: fail closed with the cause attached
        raise ModelError(f"loading the NER model from {folder} failed ({type(exc).__name__}: {exc})") from exc
    model.eval()
    max_words = int(getattr(model.config, "max_len", 384))
    return GlinerModel(model, f"{models.gliner.repo}@{models.gliner.revision}", max_words)
