"""ONNX weights mapped from disk instead of copied into the heap.

ONNX Runtime parses a model whose tensors sit inline in the protobuf into the
heap twice — once as the protobuf, once as the session's own copy — and glibc
keeps the freed half. The GLiNER encoder ships that way: 1.1GB of weights cost
1.75GB resident before a single inference. A model whose tensors live in a
sidecar `.data` file is memory-mapped instead: the session reads the pages it
touches straight from the page cache, shares them with every other process
holding the file, and the kernel may drop them under pressure. Only the MatMul
weights the runtime pre-packs for its kernels are copied — 400MB for the
encoder against 1.1GB — and inference speed does not change.

`externalized` converts a model file once, into a cache keyed on the source
file, and returns the converted path. Both model loaders route their files
through it.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

# Tensors under this size stay inline: they are shapes and biases, and an
# external reference costs more to follow than the bytes it saves.
_INLINE_LIMIT = 1024


def cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "smrti" / "onnx"


def _key(src: Path) -> str:
    st = src.stat()
    return hashlib.sha256(f"{src}\0{st.st_size}\0{st.st_mtime_ns}".encode()).hexdigest()[:16]


def externalized(src: str | Path, *, with_siblings: bool = False) -> Path:
    """Return a copy of the model at *src* whose tensors live in a sidecar file.

    Converts once per source file and reuses the copy after that. A model
    already in that form is returned as it is. With *with_siblings* the other
    files in the source directory are copied next to the converted model, for
    a loader that takes a directory rather than a file.
    """
    # Names and siblings come from the path as given: in a Hugging Face cache
    # the file is a symlink into a blob store, and resolving it would rename
    # the copy after the blob and copy every blob as a sibling.
    src = Path(src)
    target_dir = cache_root() / _key(src.resolve())
    target = target_dir / src.name
    if target.exists():
        return target

    import onnx

    model = onnx.load(str(src), load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        return src
    onnx.load_external_data_for_model(model, str(src.parent))

    cache_root().mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{target_dir.name}.", dir=cache_root()))
    try:
        _write_external(model, src, tmp, with_siblings)
        del model
        _publish(tmp, target_dir, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        release_heap()
    return target


def _write_external(model, src: Path, into: Path, with_siblings: bool) -> None:
    import onnx

    onnx.save_model(
        model,
        str(into / src.name),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=src.name + ".data",
        size_threshold=_INLINE_LIMIT,
    )
    if with_siblings:
        for sibling in src.parent.iterdir():
            if sibling.is_file() and sibling.name != src.name:
                shutil.copy2(sibling, into / sibling.name)


def _publish(tmp: Path, target_dir: Path, target: Path) -> None:
    """Move the finished directory into place in one rename."""
    try:
        tmp.rename(target_dir)
    except OSError:
        if target.exists():
            # Another process converted the same file first; its copy wins.
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            # A leftover of an interrupted conversion: replace it.
            shutil.rmtree(target_dir)
            tmp.rename(target_dir)


def release_heap() -> None:
    """Hand the heap's free pages back to the kernel; glibc keeps them otherwise."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        # Not glibc: the allocator returns memory on its own.
        pass
