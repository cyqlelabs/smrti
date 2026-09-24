"""Whether this machine should run the student rather than Laya.

Laya needs about a gigabyte free and, in practice, a CPU with AVX2: without
it onnxruntime's int8 kernels fall to SSE2 paths, and a 2011 dual core was
measured at 30–48 s per decision against deadlines of 4–5 s. The student
is a fraction of that at some cost in accuracy, so the choice is made on
what the machine can run, once, from what the kernel reports.
"""
from __future__ import annotations

import os
from pathlib import Path

# Under this much available memory Laya's graph would push the machine into
# swap; Factor's own supervisor refuses to start it below the same line.
LAYA_MIN_AVAILABLE_MB = 1024


def cpu_flags(path: str = "/proc/cpuinfo") -> set[str]:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def available_mb(path: str = "/proc/meminfo") -> int | None:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def prefers_student() -> tuple[bool, str]:
    """The verdict and the reason, for the log line that reports it. A
    machine the kernel says nothing about runs Laya, as before."""
    flags = cpu_flags()
    if flags and "avx2" not in flags:
        return True, "the CPU has no AVX2, which Laya's int8 kernels need"
    have = available_mb()
    if have is not None and have < LAYA_MIN_AVAILABLE_MB:
        return True, f"{have} MB available, under the {LAYA_MIN_AVAILABLE_MB} MB Laya needs"
    if os.environ.get("SMRTI_DECISIONS_ENGINE", "").strip().lower() == "student":
        return True, "SMRTI_DECISIONS_ENGINE=student"
    return False, ""
