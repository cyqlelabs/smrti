"""Where the decision model's weights come from.

The model is Laya — the same multilingual checkpoint Smrti has always
decided with — but it is no longer loaded through PyTorch. What ships is the
int8 ONNX build of it: 343 MB on disk against 1290 MB, ~560 MB resident
against 2.9 GB, and a runtime whose whole dependency list is onnxruntime,
tokenizers and numpy. On the small machines Smrti is meant to run beside,
that difference is the difference between a decision engine and a box that
has to be power-cycled.

Converting the checkpoint needs torch, so the conversion happens once and is
published as a release artifact. This module finds that artifact, and fetches
it when it is not already here.

Resolution order, first hit wins:

1. ``SMRTI_DECISIONS_MODEL`` — an explicit directory, for an air-gapped box
   or a model somebody built themselves.
2. ``~/.factor/decision-model`` — Factor runs the same artifact, and a
   machine running both should hold one copy of the weights, not two.
3. ``~/.smrti/decision-model`` — Smrti's own, downloaded on first use.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger("smrti.decisions.model")

# The published artifact. The tag moves only when the checkpoint or the
# conversion does, so an ordinary release never re-downloads 250 MB.
MODEL_TAG = "model-laya-multilingual-int8-v1"
MODEL_ASSET = "laya-multilingual-int8.tar.gz"
MODEL_SHA256 = "a724d4afc4d0a51e128b01b78cf64a5d285817c71c45918c2c4780658722d943"
MODEL_URL = f"https://github.com/cyqlelabs/factor/releases/download/{MODEL_TAG}/{MODEL_ASSET}"

# What the artifact is, once unpacked.
CONFIG_NAME = "edgejev.json"
DEFAULT_GRAPH = "model.onnx"

# Bounds what will be written to disk from that URL. The artifact is about
# 250 MB; an order of magnitude past it is not the model.
MAX_BYTES = 2 << 30

DOWNLOAD_TIMEOUT = 600.0


def _home(env: os._Environ[str] | dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("SMRTI_HOME") or Path.home() / ".smrti")


def _factor_model() -> Path:
    return Path(os.environ.get("FACTOR_HOME") or Path.home() / ".factor") / "decision-model"


def own_model_dir() -> Path:
    """Smrti's own copy, whether or not it is there yet."""
    return _home() / "decision-model"


def is_ready(directory: Path | str) -> bool:
    """Whether *directory* holds a usable model.

    Both halves are checked: an unpack interrupted halfway leaves the
    metadata without the graph, which loads as a confident nothing.
    """
    directory = Path(directory)
    config = directory / CONFIG_NAME
    try:
        meta = json.loads(config.read_text())
    except (OSError, ValueError):
        return False
    graph = directory / (meta.get("onnx_file") or DEFAULT_GRAPH)
    try:
        return graph.stat().st_size > 0
    except OSError:
        return False


def resolve(download: bool = True) -> Path:
    """The model directory to load, fetching Smrti's own copy if allowed.

    Raises ``FileNotFoundError`` when there is no model and none may be
    fetched — the caller turns that into ``DecisionUnavailable`` and takes
    the deterministic path, which is what every decision site does anyway.
    """
    explicit = os.environ.get("SMRTI_DECISIONS_MODEL", "").strip()
    if explicit:
        if is_ready(explicit):
            return Path(explicit)
        raise FileNotFoundError(
            f"SMRTI_DECISIONS_MODEL points at {explicit!r}, which holds no {CONFIG_NAME}"
        )

    shared = _factor_model()
    if is_ready(shared):
        logger.debug("using the decision model Factor already has at %s", shared)
        return shared

    own = own_model_dir()
    if is_ready(own):
        return own
    if not download:
        raise FileNotFoundError(f"no decision model at {own}")
    fetch(own)
    return own


def fetch(destination: Path, url: str = MODEL_URL, sha256: str = MODEL_SHA256) -> Path:
    """Download the artifact, check it, and unpack it into *destination*.

    Nothing is unpacked before the checksum matches, and the unpacked
    directory is renamed into place rather than filled in place: an
    interrupted fetch must not leave something that reads as installed.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("fetching the decision model (about 250 MB) from %s", url)

    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        archive = Path(scratch) / MODEL_ASSET
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response, archive.open("wb") as out:
            read = 0
            while chunk := response.read(1 << 20):
                read += len(chunk)
                if read > MAX_BYTES:
                    raise ValueError(f"{url} is larger than the model can be")
                digest.update(chunk)
                out.write(chunk)
        got = digest.hexdigest()
        if got != sha256:
            raise ValueError(
                f"the decision model did not match its checksum (got {got}); nothing was unpacked"
            )

        unpacked = Path(scratch) / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive) as tar:
            _extract(tar, unpacked)
        if not is_ready(unpacked):
            raise ValueError(f"the decision model unpacked without {CONFIG_NAME}")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(unpacked), str(destination))

    logger.info("the decision model is in place at %s", destination)
    return destination


def _extract(tar: tarfile.TarFile, directory: Path) -> None:
    """Unpack, refusing anything that is not a plain file or would land
    outside *directory*. The archive is one we publish; an extractor that
    trusts its input is the bug worth not having."""
    root = directory.resolve()
    for member in tar.getmembers():
        if member.isdir():
            continue
        if not member.isfile():
            raise ValueError(f"{MODEL_ASSET} carries {member.name!r}, which is not a file")
        target = (root / member.name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"{MODEL_ASSET} carries an entry outside the archive: {member.name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = tar.extractfile(member)
        if source is None:
            raise ValueError(f"{MODEL_ASSET} carries an unreadable entry: {member.name!r}")
        with source, target.open("wb") as out:
            shutil.copyfileobj(source, out, length=1 << 20)
