# -*- coding: utf-8 -*-
"""Append-only lineage log for world-generation jobs.

Persists the actual run_id/split/seed/params/schema_version/timestamps for every
generation and pool_manager job, keyed by instance_id. This replaces the previous
fabricated "lineage" block in Map_Elite's manager-state snapshot, which recorded a
hardcoded placeholder instead of real per-request provenance.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()


class LineageRecorder:
    """Thread-safe JSONL writer, one file per instance_id."""

    def __init__(self, root: Optional[Path] = None):
        self._root = Path(root) if root else Path(__file__).resolve().parent / "instances"

    def _path(self, instance_id: str) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root / f"lineage_{instance_id}.jsonl"

    def record(self, instance_id: str, entry: Dict[str, Any]) -> None:
        line = json.dumps(entry, sort_keys=True, default=str)
        with _LOCK:
            with open(self._path(instance_id), "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


_DEFAULT_RECORDER = LineageRecorder()


def record_job(instance_id: str, job, status: str) -> None:
    """Record a completed/failed generation job's real provenance for instance_id."""
    request = job.request
    _DEFAULT_RECORDER.record(instance_id, {
        "instance_id": instance_id,
        "job_id": job.job_id,
        "run_id": request.params.get("run_id"),
        "queue_lane": request.queue_lane,
        "workflow": request.workflow,
        "split": request.split,
        "seed": request.seed,
        "params": request.params,
        "schema_version": request.schema_version,
        "idempotency_key": request.idempotency_key,
        "correlation_id": request.correlation_id,
        "priority": request.priority,
        "status": status,
        "created_at": request.created_at,
        "completed_at": job.completed_at,
    })
