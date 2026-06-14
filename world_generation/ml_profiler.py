"""
ml_profiler.py — Lightweight ML debug & optimization decorator
==============================================================
Tracks CPU, RAM, GPU (optional), wall-clock time, and throughput
with minimal overhead and zero mandatory dependencies beyond stdlib.

Optional extras (auto-detected):
    pip install psutil pynvml torch

Usage
-----
    from ml_profiler import profile

    @profile()
    def train_epoch(model, loader): ...

    @profile(tag="forward", log="json", warmup=2, repeat=5)
    def forward_pass(x): ...

    with profile.context("data_loading"):
        batch = next(iter(loader))
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import statistics
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ──────────────────────────────────────────────────────────────────────────────
# Optional dependency shims
# ──────────────────────────────────────────────────────────────────────────────

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    pynvml = None  # type: ignore
    _HAS_NVML = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

class LogFormat(str, Enum):
    PLAIN = "plain"
    JSON  = "json"
    RICH  = "rich"   # colour + box-drawing (falls back to PLAIN if not a TTY)


@dataclass
class GPUSnapshot:
    index: int
    name: str
    util_pct: float          # GPU core utilisation %
    mem_used_mb: float
    mem_total_mb: float
    temp_c: Optional[float]


@dataclass
class Snapshot:
    """Single point-in-time resource measurement."""
    wall_s: float            # epoch timestamp
    cpu_pct: float           # process CPU %  (psutil) or -1
    ram_used_mb: float       # process RSS    (psutil) or -1
    ram_avail_mb: float      # system free RAM or -1
    gpus: List[GPUSnapshot] = field(default_factory=list)
    torch_alloc_mb: float = -1.0   # torch CUDA allocated memory


@dataclass
class RunStats:
    """Aggregated statistics for one decorated call."""
    tag: str
    calls: int = 0
    elapsed_s: List[float]        = field(default_factory=list)
    cpu_delta_pct: List[float]    = field(default_factory=list)
    ram_delta_mb: List[float]     = field(default_factory=list)
    gpu_util_pct: List[float]     = field(default_factory=list)
    gpu_mem_delta_mb: List[float] = field(default_factory=list)
    throughput: List[float]       = field(default_factory=list)   # items/s
    errors: int = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(lst: List[float]) -> str:
        if not lst:
            return "n/a"
        if len(lst) == 1:
            return f"{lst[0]:.3f}"
        return (f"mean={statistics.mean(lst):.3f}  "
                f"std={statistics.stdev(lst):.3f}  "
                f"min={min(lst):.3f}  max={max(lst):.3f}")

    def summary(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "calls": self.calls,
            "errors": self.errors,
            "elapsed_s": self._fmt(self.elapsed_s),
            "cpu_delta_pct": self._fmt(self.cpu_delta_pct),
            "ram_delta_mb": self._fmt(self.ram_delta_mb),
            "gpu_util_pct": self._fmt(self.gpu_util_pct),
            "gpu_mem_delta_mb": self._fmt(self.gpu_mem_delta_mb),
            "throughput_items_s": self._fmt(self.throughput),
        }


# Global registry — all decorated functions report here
_REGISTRY: Dict[str, RunStats] = {}
_REGISTRY_LOCK = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# Low-overhead resource sampling
# ──────────────────────────────────────────────────────────────────────────────

_PROC: Optional[Any] = None  # psutil.Process singleton


def _get_proc():
    global _PROC
    if _PROC is None and _HAS_PSUTIL:
        _PROC = psutil.Process(os.getpid())
    return _PROC


def _snap() -> Snapshot:
    """Capture current resource state. Called before & after the function."""
    wall = time.perf_counter()
    cpu = -1.0
    ram_used = -1.0
    ram_avail = -1.0
    gpus: List[GPUSnapshot] = []
    torch_alloc = -1.0

    proc = _get_proc()
    if proc is not None:
        try:
            with proc.oneshot():          # single /proc read — cheap
                cpu = proc.cpu_percent()  # % since last call (non-blocking)
                mem = proc.memory_info()
                ram_used = mem.rss / 1e6
        except Exception:
            pass
        try:
            ram_avail = psutil.virtual_memory().available / 1e6
        except Exception:
            pass

    if _HAS_NVML:
        try:
            n = pynvml.nvmlDeviceGetCount()
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = None
                gpus.append(GPUSnapshot(
                    index=i, name=name,
                    util_pct=float(util.gpu),
                    mem_used_mb=mem_info.used / 1e6,
                    mem_total_mb=mem_info.total / 1e6,
                    temp_c=temp,
                ))
        except Exception:
            pass

    if _HAS_TORCH and torch.cuda.is_available():
        try:
            torch_alloc = torch.cuda.memory_allocated() / 1e6
        except Exception:
            pass

    return Snapshot(
        wall_s=wall, cpu_pct=cpu,
        ram_used_mb=ram_used, ram_avail_mb=ram_avail,
        gpus=gpus, torch_alloc_mb=torch_alloc,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Background sampler (optional, for long-running calls)
# ──────────────────────────────────────────────────────────────────────────────

class _BackgroundSampler:
    """Polls resource usage on a daemon thread at `interval_s` seconds."""

    def __init__(self, interval_s: float = 0.5):
        self.interval = interval_s
        self.samples: List[Snapshot] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> List[Snapshot]:
        self._stop.set()
        self._thread.join(timeout=self.interval * 3)
        return self.samples

    def _run(self):
        while not self._stop.wait(self.interval):
            self.samples.append(_snap())


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

_ANSI = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
    "green":  "\033[32m",
    "cyan":   "\033[36m",
    "yellow": "\033[33m",
    "red":    "\033[31m",
    "blue":   "\033[34m",
    "magenta":"\033[35m",
}

def _c(key: str, text: str, use_color: bool) -> str:
    return f"{_ANSI[key]}{text}{_ANSI['reset']}" if use_color else text


def _format_report(
    tag: str,
    before: Snapshot,
    after: Snapshot,
    elapsed: float,
    bg_samples: List[Snapshot],
    n_items: Optional[int],
    fmt: LogFormat,
    call_index: int,
) -> str:
    color = fmt == LogFormat.RICH and sys.stdout.isatty()

    delta_ram = (after.ram_used_mb - before.ram_used_mb
                 if before.ram_used_mb >= 0 and after.ram_used_mb >= 0
                 else None)

    gpu_lines = []
    for g_before, g_after in zip(before.gpus, after.gpus):
        dm = g_after.mem_used_mb - g_before.mem_used_mb
        temp_str = (f"  temp={g_after.temp_c:.0f}°C"
                    if g_after.temp_c is not None else "")
        gpu_lines.append(
            f"  GPU[{g_before.index}] {g_before.name}: "
            f"util={g_after.util_pct:.1f}%  "
            f"mem={g_after.mem_used_mb:.1f}/{g_after.mem_total_mb:.1f} MB "
            f"(Δ{dm:+.1f} MB){temp_str}"
        )

    torch_delta = None
    if before.torch_alloc_mb >= 0 and after.torch_alloc_mb >= 0:
        torch_delta = after.torch_alloc_mb - before.torch_alloc_mb

    throughput_str = ""
    if n_items and elapsed > 0:
        thr = n_items / elapsed
        throughput_str = f"  throughput={thr:.1f} items/s"

    peak_ram_str = ""
    if bg_samples:
        peak = max(s.ram_used_mb for s in bg_samples if s.ram_used_mb >= 0)
        peak_ram_str = f"  peak_ram={peak:.1f} MB"

    if fmt == LogFormat.JSON:
        data: Dict[str, Any] = {
            "tag": tag,
            "call": call_index,
            "elapsed_s": round(elapsed, 6),
            "cpu_pct_after": after.cpu_pct,
        }
        if delta_ram is not None:
            data["ram_delta_mb"] = round(delta_ram, 2)
            data["ram_used_mb"] = round(after.ram_used_mb, 2)
        if gpu_lines:
            data["gpus"] = [
                {
                    "index": g.index,
                    "util_pct": g.util_pct,
                    "mem_used_mb": round(g.mem_used_mb, 2),
                    "mem_delta_mb": round(g.mem_used_mb - b.mem_used_mb, 2),
                }
                for g, b in zip(after.gpus, before.gpus)
            ]
        if torch_delta is not None:
            data["torch_alloc_delta_mb"] = round(torch_delta, 2)
        if n_items:
            data["throughput_items_s"] = round(n_items / elapsed, 2)
        return json.dumps(data, default=str)

    # PLAIN / RICH
    sep = _c("dim", "─" * 60, color)
    title = _c("bold", f"[profile] {tag}", color) + _c("dim", f" #{call_index}", color)

    parts = [sep, title]
    t_str = f"  time     = {_c('green', f'{elapsed*1000:.2f} ms', color)}"
    if throughput_str:
        t_str += _c("cyan", throughput_str, color)
    parts.append(t_str)

    if after.cpu_pct >= 0:
        bar_n = int(after.cpu_pct / 5)
        bar = "█" * bar_n + "░" * (20 - bar_n)
        parts.append(f"  cpu      = {_c('yellow', f'{after.cpu_pct:5.1f}%', color)} {bar}")

    if delta_ram is not None:
        sign = "+" if delta_ram >= 0 else ""
        parts.append(
            f"  ram      = {after.ram_used_mb:.1f} MB "
            f"(Δ{sign}{delta_ram:.1f} MB)"
            + (f"  avail={after.ram_avail_mb:.0f} MB" if after.ram_avail_mb >= 0 else "")
            + peak_ram_str
        )

    for gl in gpu_lines:
        parts.append(_c("magenta", gl, color))

    if torch_delta is not None:
        parts.append(f"  torch CUDA alloc Δ = {torch_delta:+.2f} MB")

    parts.append(sep)
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Core decorator
# ──────────────────────────────────────────────────────────────────────────────

class profile:
    """
    Decorator and context-manager for ML profiling.

    Parameters
    ----------
    tag : str
        Human-readable name shown in the report. Defaults to function name.
    log : {"plain", "json", "rich"}
        Output format.  "rich" adds ANSI colours when stdout is a TTY.
    logger : logging.Logger | None
        If given, output is sent via logger.info() instead of print().
    enabled : bool
        Set False to make the decorator a transparent no-op (zero overhead).
    warmup : int
        Skip reporting for the first N calls (let JIT / caches settle).
    repeat : int | None
        If given, run the wrapped function `repeat` times and aggregate.
        Useful for micro-benchmarks.  Return value is from the last call.
    n_items : int | Callable | None
        Items processed per call (e.g. batch size). Used for throughput.
        Can be a callable that receives the function's return value.
    background_sample_s : float | None
        If set, a daemon thread polls resources every N seconds during the
        call. Useful for long-running training steps.  Adds ~1 thread.
    raise_on_error : bool
        Re-raise exceptions after logging them (default True).
    """

    # ── class-level registry accessor ────────────────────────────────────────

    @classmethod
    def get_stats(cls, tag: Optional[str] = None) -> Union[RunStats, Dict[str, RunStats]]:
        with _REGISTRY_LOCK:
            if tag:
                return _REGISTRY.get(tag)
            return dict(_REGISTRY)

    @classmethod
    def print_summary(cls, fmt: LogFormat = LogFormat.PLAIN):
        with _REGISTRY_LOCK:
            items = list(_REGISTRY.values())
        if not items:
            print("[profile] No data collected yet.")
            return
        for rs in items:
            s = rs.summary()
            if fmt == LogFormat.JSON:
                print(json.dumps(s, indent=2))
            else:
                print("\n".join(f"  {k}: {v}" for k, v in s.items()))
                print()

    @classmethod
    def reset(cls):
        with _REGISTRY_LOCK:
            _REGISTRY.clear()

    # ── context manager ───────────────────────────────────────────────────────

    @classmethod
    @contextlib.contextmanager
    def context(cls, tag: str, **kwargs):
        p = cls(tag=tag, **kwargs)
        before = _snap()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            after = _snap()
            p._record_and_log(tag, before, after, elapsed, [], None, 1)

    # ── constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        tag: Optional[str] = None,
        *,
        log: Union[str, LogFormat] = LogFormat.RICH,
        logger: Optional[logging.Logger] = None,
        enabled: bool = True,
        warmup: int = 0,
        repeat: Optional[int] = None,
        n_items: Union[int, Callable, None] = None,
        background_sample_s: Optional[float] = None,
        raise_on_error: bool = True,
    ):
        self._tag = tag
        self._fmt = LogFormat(log)
        self._logger = logger
        self._enabled = enabled
        self._warmup = warmup
        self._repeat = repeat
        self._n_items = n_items
        self._bg_interval = background_sample_s
        self._raise = raise_on_error
        self._call_count: Dict[str, int] = {}

    # ── __call__ — makes the class usable as @profile() ──────────────────────

    def __call__(self, fn: Callable) -> Callable:
        if not self._enabled:
            return fn

        tag = self._tag or fn.__qualname__

        with _REGISTRY_LOCK:
            if tag not in _REGISTRY:
                _REGISTRY[tag] = RunStats(tag=tag)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            self._call_count.setdefault(tag, 0)
            self._call_count[tag] += 1
            call_idx = self._call_count[tag]
            silent = call_idx <= self._warmup

            repeat = self._repeat or 1
            result = None
            total_elapsed = 0.0
            bg_samples: List[Snapshot] = []

            before = _snap()
            t_start = time.perf_counter()

            sampler = None
            if self._bg_interval:
                sampler = _BackgroundSampler(self._bg_interval).start()

            try:
                for _ in range(repeat):
                    result = fn(*args, **kwargs)
                total_elapsed = time.perf_counter() - t_start
            except Exception as exc:
                total_elapsed = time.perf_counter() - t_start
                with _REGISTRY_LOCK:
                    _REGISTRY[tag].errors += 1
                if not silent:
                    self._emit(f"[profile] ERROR in '{tag}': {exc}\n"
                               + traceback.format_exc())
                if self._raise:
                    raise
                return None
            finally:
                if sampler:
                    bg_samples = sampler.stop()

            after = _snap()

            # resolve n_items
            n_items: Optional[int] = None
            if callable(self._n_items):
                try:
                    n_items = int(self._n_items(result))
                except Exception:
                    pass
            elif self._n_items is not None:
                n_items = int(self._n_items)

            if repeat > 1 and n_items:
                n_items *= repeat

            if not silent:
                self._record_and_log(
                    tag, before, after, total_elapsed,
                    bg_samples, n_items, call_idx,
                )
            else:
                # still record stats, just don't print
                self._update_registry(tag, before, after,
                                      total_elapsed, n_items)

            return result

        return wrapper

    # ── internals ─────────────────────────────────────────────────────────────

    def _record_and_log(
        self,
        tag: str,
        before: Snapshot,
        after: Snapshot,
        elapsed: float,
        bg_samples: List[Snapshot],
        n_items: Optional[int],
        call_idx: int,
    ):
        self._update_registry(tag, before, after, elapsed, n_items)
        report = _format_report(
            tag, before, after, elapsed,
            bg_samples, n_items, self._fmt, call_idx,
        )
        self._emit(report)

    def _update_registry(
        self,
        tag: str,
        before: Snapshot,
        after: Snapshot,
        elapsed: float,
        n_items: Optional[int],
    ):
        with _REGISTRY_LOCK:
            rs = _REGISTRY.setdefault(tag, RunStats(tag=tag))
            rs.calls += 1
            rs.elapsed_s.append(elapsed)
            if before.cpu_pct >= 0 and after.cpu_pct >= 0:
                rs.cpu_delta_pct.append(after.cpu_pct - before.cpu_pct)
            if before.ram_used_mb >= 0 and after.ram_used_mb >= 0:
                rs.ram_delta_mb.append(after.ram_used_mb - before.ram_used_mb)
            for g_b, g_a in zip(before.gpus, after.gpus):
                rs.gpu_util_pct.append(g_a.util_pct)
                rs.gpu_mem_delta_mb.append(g_a.mem_used_mb - g_b.mem_used_mb)
            if n_items and elapsed > 0:
                rs.throughput.append(n_items / elapsed)

    def _emit(self, text: str):
        if self._logger:
            self._logger.info(text)
        else:
            print(text, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience aliases
# ──────────────────────────────────────────────────────────────────────────────

def profile_fn(fn=None, **kwargs):
    """One-liner alias: @profile_fn  or  @profile_fn(tag='x')."""
    if fn is not None:
        return profile()(fn)
    return profile(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test / demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math

    # ── 1. Basic usage ────────────────────────────────────────────────────────
    @profile(tag="matrix_ops", log="rich", warmup=1, repeat=3)
    def heavy_compute(n: int):
        return sum(math.sqrt(i) for i in range(n))

    heavy_compute(500_000)   # warmup (silent)
    heavy_compute(500_000)   # reported (mean of 3 repeats)

    # ── 2. JSON output ────────────────────────────────────────────────────────
    @profile(tag="data_prep", log="json", n_items=128)
    def prepare_batch():
        time.sleep(0.02)
        return list(range(128))

    prepare_batch()

    # ── 3. Context manager ────────────────────────────────────────────────────
    with profile.context("inference_loop", log="plain"):
        time.sleep(0.01)

    # ── 4. Aggregate summary ──────────────────────────────────────────────────
    print("\n\n=== AGGREGATE SUMMARY ===")
    profile.print_summary()
