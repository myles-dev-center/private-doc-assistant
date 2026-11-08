"""What this machine can run, for a recommended number of parallel model requests.

A local model server answers parallel requests out of one pool of (V)RAM: each extra
slot needs its own context cache, and without a dedicated GPU the requests share the
CPU, so more parallelism mostly makes each answer slower. A cloud API has no such limit
here; there the limit is the provider's rate limit and your budget.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from functools import lru_cache

MAX_CONCURRENCY = 10


@lru_cache
def detect() -> dict:
    """{"cpu_cores", "ram_gb", "gpu", "vram_gb"} - None where it can't be found."""
    return {"cpu_cores": os.cpu_count(), "ram_gb": _ram_gb(), **_gpu()}


def recommend_local(hw: dict | None = None) -> tuple[int, str]:
    """(recommended parallel requests for a local ~9B model, why)."""
    hw = hw or detect()
    vram, ram = hw.get("vram_gb"), hw.get("ram_gb")
    if vram:
        n = 4 if vram >= 24 else 3 if vram >= 16 else 2 if vram >= 10 else 1
        return n, f"{hw['gpu']} with {vram:.0f} GB: about {n} model request{'s' if n > 1 else ''} fit side by side"
    gpu = f"{hw['gpu']}: its own memory can't be measured, so" if hw.get("gpu") else "No dedicated GPU found, so"
    why = f"{gpu} the safe choice is 1 - parallel requests would share the processor and each one gets slower"
    if ram and ram < 24:
        why += f"; {ram:.0f} GB RAM leaves little room for more than one context"
    return 1, why


def _ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong)
                    for n in ("total", "avail", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_ext")
                ]

            m = MemoryStatus()
            m.dwLength = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.total / 2**30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (OSError, ValueError, AttributeError):
        return None


def _gpu() -> dict:
    """A dedicated GPU, if any. Integrated graphics share system RAM and don't count."""
    if shutil.which("nvidia-smi"):
        try:
            out = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                .stdout.strip()
                .splitlines()
            )
            if out:
                name, mib = [x.strip() for x in out[0].split(",")]
                return {"gpu": name, "vram_gb": round(float(mib) / 1024, 1)}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if platform.system() == "Windows":
        # Without nvidia-smi the GPU's own memory can't be told from shared system RAM
        # (Windows reports integrated GPUs with gigabytes of "adapter RAM"), so other GPUs
        # are only named, and the recommendation stays conservative; raise it in the UI.
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name -join ';'"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            names = [n.strip() for n in out.split(";") if n.strip()]
            real = [n for n in names if any(k in n.lower() for k in ("intel", "amd", "radeon", "nvidia"))]
            return {"gpu": (real or names or [None])[0], "vram_gb": None}
        except (OSError, subprocess.SubprocessError):
            pass
    return {"gpu": None, "vram_gb": None}
