# -*- coding: utf-8 -*-
"""In-process request queue management for the world-gen <-> agent pool_manager path.

Resolves the long-standing TODOs in Map_Grammars_Elite.pool_manager / API.pool_manager
about the pending-request counter being an ad-hoc dict with no real backpressure
semantics, no prioritization, and no capture of failed/stuck requests.

Scope note (tracked follow-up, not addressed here): this queue is process-local, in
memory only (persisted best-effort alongside manager state). A cross-process/durable
queue (e.g. backed by Redis or a DB) would be needed to share backpressure state across
multiple API worker processes; that is a separate infra change and out of scope for
this pass.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


class QueueBackpressureError(RuntimeError):
    """Raised when a queue_key is at capacity and the request cannot be admitted.

    Documented backpressure behavior: a request is rejected outright unless its
    priority is strictly higher than the lowest-priority request currently admitted,
    in which case that lower-priority request is preempted (moved to the dead-letter
    log with reason="preempted_by_higher_priority") to make room.
    """

    def __init__(self, queue_key: str, pending: int, max_pending: int):
        self.queue_key = queue_key
        self.pending = pending
        self.max_pending = max_pending
        super().__init__(
            f"Queue backpressure exceeded for {queue_key} ({pending}/{max_pending} pending)"
        )


@dataclass
class DeadLetter:
    request_id: str
    queue_key: str
    agent_name: str
    priority: int
    reason: str
    enqueued_at: float
    failed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "queue_key": self.queue_key,
            "agent_name": self.agent_name,
            "priority": self.priority,
            "reason": self.reason,
            "enqueued_at": self.enqueued_at,
            "failed_at": self.failed_at,
            "wait_seconds": round(max(0.0, self.failed_at - self.enqueued_at), 4),
        }


@dataclass
class GenerationRequestContract:
    """Versioned request contract used by both generation job submission and gameplay pool refills.

    The contract intentionally keeps the exact generation params plus lineage metadata in one
    serializable object, so a request can be replayed and its provenance can be checked later.
    """

    queue_lane: str = "agentic"
    workflow: str = "train"
    split: str = "train"
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    idempotency_key: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 0
    schema_version: int = 1
    created_at: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        self.queue_lane = str(self.queue_lane or "agentic")
        self.workflow = str(self.workflow or "train")
        self.split = str(self.split or "train")
        self.params = dict(self.params or {})
        if self.seed is not None:
            self.params.setdefault("seed", self.seed)
        self.idempotency_key = self.idempotency_key or self.request_id
        self.correlation_id = self.correlation_id or self.idempotency_key or self.request_id

    @property
    def reproducibility_key(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "queue_lane": self.queue_lane,
            "workflow": self.workflow,
            "split": self.split,
            "seed": self.seed,
            "params": self.params,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return f"{self.idempotency_key}|{digest}"


@dataclass
class GenerationQueueJob:
    job_id: str
    request: GenerationRequestContract
    lane: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "lane": self.lane,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "schema_version": self.request.schema_version,
            "idempotency_key": self.request.idempotency_key,
            "correlation_id": self.request.correlation_id,
            "reproducibility_key": self.request.reproducibility_key,
        }


class GenerationQueueRouter:
    """Separate queue lanes for agentic generation and human gameplay traffic.

    Also runs jobs asynchronously (submit -> track -> fetch) on a dedicated thread
    pool per lane, so the human lane's worker capacity is isolated from agentic
    train/test/validate traffic instead of sharing one pool.
    """

    def __init__(self, max_pending_per_lane: int = 8, workers_per_lane: Optional[Dict[str, int]] = None):
        self._lock = threading.RLock()
        self._max_pending_per_lane = {"agentic": int(max_pending_per_lane), "human": int(max_pending_per_lane)}
        self._workers_per_lane = dict(workers_per_lane or {"agentic": 2, "human": 4})
        self._executors: Dict[str, ThreadPoolExecutor] = {}
        self._pending_counts: Dict[str, int] = defaultdict(int)
        self._jobs_by_idempotency: Dict[str, GenerationQueueJob] = {}
        self._jobs_by_id: Dict[str, GenerationQueueJob] = {}
        self._jobs_by_lane: Dict[str, Dict[str, GenerationQueueJob]] = defaultdict(dict)
        self._dead_letters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def _replay_conflict(self, existing: GenerationQueueJob, request: GenerationRequestContract) -> None:
        if existing.request.reproducibility_key == request.reproducibility_key:
            return
        raise ValueError(
            "idempotency_key conflict: same idempotency_key was already used for a different generation specification "
            f"({existing.request.queue_lane}/{existing.request.workflow}/{existing.request.split} vs "
            f"{request.queue_lane}/{request.workflow}/{request.split})"
        )

    def submit(self, request: GenerationRequestContract) -> GenerationQueueJob:
        if not isinstance(request, GenerationRequestContract):
            request = GenerationRequestContract(**dict(request))

        lane = request.queue_lane
        idem = request.idempotency_key
        with self._lock:
            existing = self._jobs_by_idempotency.get(idem)
            if existing is not None:
                self._replay_conflict(existing, request)
                return existing

            max_pending = self._max_pending_per_lane.get(lane, self._max_pending_per_lane.get("agentic", 8))
            if self._pending_counts.get(lane, 0) >= max_pending:
                raise QueueBackpressureError(lane, self._pending_counts.get(lane, 0), max_pending)

            job = GenerationQueueJob(job_id=request.request_id, request=request, lane=lane)
            self._jobs_by_idempotency[idem] = job
            self._jobs_by_id[job.job_id] = job
            self._jobs_by_lane[lane][job.job_id] = job
            self._pending_counts[lane] = self._pending_counts.get(lane, 0) + 1
            return job

    def get_job(self, job_id: str) -> Optional[GenerationQueueJob]:
        """Look up a job by job_id regardless of lane/idempotency_key, for track/fetch."""
        with self._lock:
            return self._jobs_by_id.get(job_id)

    def _executor_for_lane(self, lane: str) -> ThreadPoolExecutor:
        with self._lock:
            executor = self._executors.get(lane)
            if executor is None:
                workers = self._workers_per_lane.get(lane, self._workers_per_lane.get("agentic", 2))
                executor = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix=f"genq-{lane}")
                self._executors[lane] = executor
            return executor

    def run_async(self, request, fn, on_done=None) -> GenerationQueueJob:
        """Submit + run fn() on the lane's worker pool; returns immediately with a job handle.

        A duplicate submission (same idempotency_key) returns the existing job without
        scheduling fn again, whatever its current status is. NOTE: this is a best-effort,
        in-process-only guard — two racing submissions for a brand-new idempotency_key can
        both observe status == 'queued' before either flips to 'running'; closing that
        race needs a durable/cross-process queue, out of scope here (see module docstring).
        """
        job = self.submit(request)
        if job.status != "queued":
            return job

        def _run():
            try:
                result = fn()
                with self._lock:
                    job.result = result
                    job.status = "completed"
                    job.completed_at = time.time()
                    self._jobs_by_lane.get(job.lane, {}).pop(job.job_id, None)
                    self._pending_counts[job.lane] = max(0, self._pending_counts.get(job.lane, 0) - 1)
            except Exception as exc:
                with self._lock:
                    job.error = str(exc)
                    job.status = "failed"
                    job.completed_at = time.time()
                    self._jobs_by_lane.get(job.lane, {}).pop(job.job_id, None)
                    self._pending_counts[job.lane] = max(0, self._pending_counts.get(job.lane, 0) - 1)
                    self._dead_letters[job.lane].append({
                        "job_id": job.job_id,
                        "reason": job.error,
                        "idempotency_key": job.request.idempotency_key,
                        "correlation_id": job.request.correlation_id,
                        "failed_at": job.completed_at,
                    })
            if on_done is not None:
                try:
                    on_done(job)
                except Exception:
                    pass

        with self._lock:
            job.status = "running"
        self._executor_for_lane(job.lane).submit(_run)
        return job

    def mark_done(self, job_id: str, lane: Optional[str] = None) -> None:
        with self._lock:
            for current_lane, jobs in self._jobs_by_lane.items():
                if job_id in jobs:
                    lane = current_lane
                    break
            if lane is None:
                return
            job = self._jobs_by_lane.get(lane, {}).pop(job_id, None)
            if job is None:
                return
            self._pending_counts[lane] = max(0, self._pending_counts.get(lane, 0) - 1)
            job.status = "completed"
            job.completed_at = time.time()

    def mark_failed(self, job_id: str, reason: str, lane: Optional[str] = None) -> None:
        with self._lock:
            for current_lane, jobs in self._jobs_by_lane.items():
                if job_id in jobs:
                    lane = current_lane
                    break
            if lane is None:
                return
            job = self._jobs_by_lane.get(lane, {}).pop(job_id, None)
            if job is None:
                return
            self._pending_counts[lane] = max(0, self._pending_counts.get(lane, 0) - 1)
            job.status = "failed"
            job.completed_at = time.time()
            self._dead_letters[lane].append({
                "job_id": job_id,
                "reason": reason,
                "idempotency_key": job.request.idempotency_key,
                "correlation_id": job.request.correlation_id,
                "failed_at": job.completed_at,
            })

    def snapshot(self, lane: str) -> Dict[str, Any]:
        lane_name = str(lane or "agentic")
        with self._lock:
            return {
                "pending": self._pending_counts.get(lane_name, 0),
                "max_pending": self._max_pending_per_lane.get(lane_name, 8),
                "dead_letter_count": len(self._dead_letters.get(lane_name, [])),
            }

    def dead_letters(self, lane: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if lane is None:
                merged: List[Dict[str, Any]] = []
                for entries in self._dead_letters.values():
                    merged.extend(entries)
                return merged
            return list(self._dead_letters.get(str(lane), []))


class DifficultyPoolManager:
    """Thread-safe live pool for difficulty-specific map supply.

    The current difficulty is consumed from the active pool while background pre-warming keeps
    adjacent difficulties ready for transition without forcing gameplay to wait.
    """

    def __init__(self, capacity: int = 8):
        self._lock = threading.RLock()
        self._capacity = max(1, int(capacity))
        self._pools: Dict[str, Deque[Any]] = defaultdict(deque)
        self._current_difficulty: Optional[str] = None
        self._transition_ready: Optional[str] = None

    def seed(self, difficulty: str, values: List[Any]) -> None:
        difficulty_name = str(difficulty)
        with self._lock:
            queue = self._pools[difficulty_name]
            queue.clear()
            for value in values[: self._capacity]:
                queue.append(value)

    def refill(self, difficulty: str, values: List[Any]) -> None:
        """Append produced items up to capacity without discarding what's already queued."""
        difficulty_name = str(difficulty)
        with self._lock:
            queue = self._pools[difficulty_name]
            for value in values:
                if len(queue) >= self._capacity:
                    break
                queue.append(value)

    def pending(self, difficulty: str) -> int:
        with self._lock:
            return len(self._pools.get(str(difficulty), ()))

    def consume(self, difficulty: str) -> Any:
        """Pop one item for an explicit difficulty, independent of current_difficulty."""
        difficulty_name = str(difficulty)
        with self._lock:
            queue = self._pools.get(difficulty_name)
            if queue is None or not queue:
                raise RuntimeError(f"No ready items for difficulty '{difficulty_name}'")
            return queue.popleft()

    def set_current_difficulty(self, difficulty: str) -> None:
        with self._lock:
            self._current_difficulty = str(difficulty)
            self._transition_ready = None

    def prepare_transition(self, difficulty: str) -> None:
        with self._lock:
            self._transition_ready = str(difficulty)

    def preview_difficulty(self, difficulty: str) -> List[Any]:
        with self._lock:
            return list(self._pools.get(str(difficulty), deque()))

    def commit_transition(self, difficulty: str) -> None:
        with self._lock:
            self._current_difficulty = str(difficulty)
            self._transition_ready = None

    @property
    def current_difficulty(self) -> Optional[str]:
        with self._lock:
            return self._current_difficulty

    def consume_current(self) -> Any:
        with self._lock:
            if self._current_difficulty is None:
                raise RuntimeError("No current difficulty is active")
            queue = self._pools.get(self._current_difficulty)
            if queue is None or not queue:
                raise RuntimeError(f"No ready items for current difficulty '{self._current_difficulty}'")
            return queue.popleft()


class RequestQueueManager:
    """Thread-safe, per-queue_key admission control with priority and dead-lettering.

    Not a worker/execution queue: callers still execute the request synchronously in
    the calling thread. This class only tracks how many requests are in flight per
    queue_key (backpressure), lets higher-priority requests preempt lower-priority
    ones when at capacity, and records anything that fails or gets preempted instead
    of silently dropping it.
    """

    def __init__(self, max_dead_letters_per_key: int = 200):
        self._lock = threading.Lock()
        self._pending: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._max_pending: Dict[str, int] = {}
        self._dead_letters: Dict[str, List[DeadLetter]] = {}
        self._max_dead_letters_per_key = max_dead_letters_per_key

    @staticmethod
    def new_request_id() -> str:
        return uuid.uuid4().hex[:12]

    def admit(
        self,
        queue_key: str,
        agent_name: str,
        priority: int = 0,
        max_pending: int = 8,
        request_id: Optional[str] = None,
    ) -> str:
        """Admit a request into queue_key, or raise QueueBackpressureError."""
        request_id = request_id or self.new_request_id()
        max_pending = max(1, int(max_pending or 1))
        with self._lock:
            pending = self._pending.setdefault(queue_key, {})
            self._max_pending[queue_key] = max_pending
            if len(pending) >= max_pending:
                lowest_id, lowest = min(pending.items(), key=lambda kv: kv[1]["priority"])
                if priority > lowest["priority"]:
                    pending.pop(lowest_id)
                    self._record_dead_letter(
                        queue_key, lowest_id, lowest["agent_name"], lowest["priority"],
                        lowest["enqueued_at"], reason="preempted_by_higher_priority",
                    )
                else:
                    raise QueueBackpressureError(queue_key, len(pending), max_pending)
            pending[request_id] = {
                "priority": int(priority),
                "agent_name": agent_name,
                "enqueued_at": time.time(),
            }
        return request_id

    def release(self, queue_key: str, request_id: str) -> None:
        """Mark a request as successfully completed."""
        with self._lock:
            pending = self._pending.get(queue_key)
            if pending is not None:
                pending.pop(request_id, None)

    def fail(self, queue_key: str, request_id: str, reason: str) -> None:
        """Mark a request as failed/stuck; capture it in the dead-letter log."""
        with self._lock:
            pending = self._pending.get(queue_key, {})
            entry = pending.pop(request_id, None)
            if entry is None:
                entry = {"priority": 0, "agent_name": "unknown", "enqueued_at": time.time()}
            self._record_dead_letter(
                queue_key, request_id, entry["agent_name"], entry["priority"],
                entry["enqueued_at"], reason=reason,
            )

    def _record_dead_letter(self, queue_key, request_id, agent_name, priority, enqueued_at, reason):
        letters = self._dead_letters.setdefault(queue_key, [])
        letters.append(DeadLetter(
            request_id=request_id, queue_key=queue_key, agent_name=agent_name,
            priority=priority, reason=reason, enqueued_at=enqueued_at,
        ))
        if len(letters) > self._max_dead_letters_per_key:
            del letters[: len(letters) - self._max_dead_letters_per_key]

    def snapshot(self, queue_key: str) -> Dict[str, Any]:
        with self._lock:
            pending = self._pending.get(queue_key, {})
            return {
                "pending": len(pending),
                "max_pending": self._max_pending.get(queue_key, 8),
                "dead_letter_count": len(self._dead_letters.get(queue_key, [])),
            }

    def dead_letters(self, queue_key: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if queue_key is not None:
                return [dl.to_dict() for dl in self._dead_letters.get(queue_key, [])]
            return [dl.to_dict() for letters in self._dead_letters.values() for dl in letters]

    def to_state(self) -> Dict[str, Any]:
        """Serializable snapshot for best-effort persistence alongside manager state."""
        with self._lock:
            return {
                "max_pending": dict(self._max_pending),
                "dead_letters": {
                    key: [dl.to_dict() for dl in letters]
                    for key, letters in self._dead_letters.items()
                },
            }

    def load_state(self, data: Dict[str, Any]) -> None:
        """Restore dead-letter history (pending counts are intentionally not restored:
        in-flight requests from a previous process cannot still be pending)."""
        if not isinstance(data, dict):
            return
        with self._lock:
            for key, value in (data.get("max_pending") or {}).items():
                try:
                    self._max_pending[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
            for key, letters in (data.get("dead_letters") or {}).items():
                restored = []
                for item in letters:
                    if not isinstance(item, dict):
                        continue
                    restored.append(DeadLetter(
                        request_id=str(item.get("request_id", "")),
                        queue_key=str(key),
                        agent_name=str(item.get("agent_name", "")),
                        priority=int(item.get("priority", 0)),
                        reason=str(item.get("reason", "")),
                        enqueued_at=float(item.get("enqueued_at", 0.0)),
                        failed_at=float(item.get("failed_at", 0.0)),
                    ))
                if restored:
                    self._dead_letters[str(key)] = restored
