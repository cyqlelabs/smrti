"""Whether this machine should run the student rather than Laya.

Laya needs a gigabyte to load and, in practice, a CPU with AVX2: without
it onnxruntime's int8 kernels fall to SSE2 paths, and a 2011 dual core was
measured at 30–48 s per decision against deadlines of 4–5 s. The student
is a fraction of that at some cost in accuracy, so the choice is made on
what the machine is, once, from what the kernel reports — the CPU's
flags and the memory it was built with, not how much happens to be free
at the moment of the first decision.
"""
from __future__ import annotations

from pathlib import Path

# A machine with less than this can hold Laya's graph beside an agent only
# by swapping; the box this exists for has 3.5 GB.
LAYA_MIN_TOTAL_MB = 4096


def cpu_flags(path: str = "/proc/cpuinfo") -> set[str]:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def total_mb(path: str = "/proc/meminfo") -> int | None:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("MemTotal:"):
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
    total = total_mb()
    if total is not None and total < LAYA_MIN_TOTAL_MB:
        return True, f"{total} MB of memory, under the {LAYA_MIN_TOTAL_MB} MB Laya needs beside an agent"
    return False, ""
