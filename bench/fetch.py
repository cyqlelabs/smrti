"""Download the benchmark datasets the harnesses need.

    make datasets

Neither dataset ships with the repo — LongMemEval-S is 265MB and HaluMem 32MB
— but `make bench` defaulted to paths under ``data/`` that nothing created, so
the documented command failed on a fresh clone with no way to fix it. This is
that way.

Stdlib only, on purpose: fetching a file is not worth a dependency, and the
benchmarks are already the part of the repo that is not in the wheel.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")

DATASETS = {
    "longmemeval_s.json": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s",
        "LongMemEval-S — 500 questions over ~115k-token histories",
    ),
    "HaluMem-Medium.jsonl": (
        "https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Medium.jsonl",
        "HaluMem-Medium — 20 personas, 3,467 questions",
    ),
}

# Anything smaller than this came back as an error page rather than a dataset.
_MIN_BYTES = 1_000_000


def _download(url: str, destination: str) -> None:
    """Stream *url* to *destination*, through a temporary name.

    Writing in place would leave a half-file behind on a dropped connection,
    and the next run would treat it as already downloaded.
    """
    partial = destination + ".part"
    with urllib.request.urlopen(url) as response, open(partial, "wb") as handle:
        total = int(response.headers.get("content-length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(
                    f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr
                )
    print(file=sys.stderr)
    if os.path.getsize(partial) < _MIN_BYTES:
        os.remove(partial)
        raise OSError("download returned too little data to be the dataset")
    os.replace(partial, destination)


def main(argv: list[str] | None = None) -> int:
    wanted = argv or list(DATASETS)
    os.makedirs(DATA_DIR, exist_ok=True)
    failed = False
    for name in wanted:
        if name not in DATASETS:
            print(f"unknown dataset: {name}", file=sys.stderr)
            failed = True
            continue
        url, description = DATASETS[name]
        destination = os.path.join(DATA_DIR, name)
        if os.path.exists(destination) and os.path.getsize(destination) >= _MIN_BYTES:
            print(f"{name}: already present", file=sys.stderr)
            continue
        print(f"{name}: {description}", file=sys.stderr)
        try:
            _download(url, destination)
        except (urllib.error.URLError, OSError) as exc:
            print(f"{name}: download failed — {exc}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
