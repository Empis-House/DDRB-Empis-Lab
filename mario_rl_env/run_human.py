"""
run_human.py – Play the Mario level as a human via keyboard.

Controls
--------
  Arrow RIGHT / D  →  walk right
  Arrow LEFT  / A  →  walk left
    Arrow DOWN / S   →  enter rare red/blue pipe (when standing on it)
  Z / Space        →  jump
  X / Shift        →  run (hold with direction)
  R                →  restart episode
  Q / Esc          →  quit

Usage
-----
  python run_human.py path/to/map.txt
    python run_human.py --infinite --api-url http://localhost:8000 --instance-id 1 --agent-name player1
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import sys
import os
import json
import random
import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import pandas as pd

# ensure local imports work when running directly
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .constants import FPS
except ImportError:
    from constants import FPS

try:
    from mario_rl_env.artifact_registry import ArtifactRegistry, DEFAULT_REGISTRY_PATH, ExecutionConfig
except ImportError:
    from artifact_registry import ArtifactRegistry, DEFAULT_REGISTRY_PATH, ExecutionConfig

_REGISTRY: Optional[ArtifactRegistry] = None


def _get_registry() -> ArtifactRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ArtifactRegistry(DEFAULT_REGISTRY_PATH)
    return _REGISTRY


class WorldGenerationApiClient:
    """Thin client for the world generation HTTP API used by infinite mode."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def wait_until_ready(self, retries: int = 10, delay: float = 1.0) -> None:
        root_url = f"{self.base_url}/"
        for attempt in range(1, retries + 1):
            try:
                request = Request(root_url, headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=5) as response:
                    if response.status == 200:
                        return
            except Exception:
                if attempt == retries:
                    raise RuntimeError(f"API did not become ready at {root_url}")
                time.sleep(delay)

    def post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request) as response:
                return json.load(response)
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API request failed: {exc.code} {exc.reason} {message}")
        except URLError as exc:
            raise RuntimeError(f"API connection failed: {exc.reason}")

    def get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        request = Request(url, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request) as response:
                return json.load(response)
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API request failed: {exc.code} {exc.reason} {message}")
        except URLError as exc:
            raise RuntimeError(f"API connection failed: {exc.reason}")

    def wait_for_job_result(
        self,
        job_id: str,
        poll_interval: float = 0.5,
        timeout: float = 600.0,
    ) -> dict:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            status = self.get(f"/jobs/{job_id}")
            job_status = str(status.get("status", "unknown"))
            if job_status in {"completed", "failed"}:
                result_response = self.get(f"/jobs/{job_id}/result")
                if result_response.get("status") == "failed":
                    raise RuntimeError(result_response.get("error") or f"Generation job {job_id} failed")
                result = result_response.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(f"Generation job {job_id} completed without a result")
                return result
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for generation job {job_id}")
            time.sleep(max(0.0, float(poll_interval)))

    def create_instance(
        self,
        function: str = "Performance_KL_Progression",
        alphas: Optional[list] = None,
        variety_domains: Optional[list] = None,
        grid_points: Optional[list] = None,
        instance_id: Optional[str] = None,
    ) -> str:
        if alphas is None:
            alphas = [0.0, 0.1]
        if variety_domains is None:
            variety_domains = ["overworld", "underground", "athletic"]
        if grid_points is None:
            grid_points = [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]
        if instance_id is None:
            instance_id = f"human-{uuid.uuid4().hex[:12]}"

        response = self.post(
            "/instances",
            {
                "function": function,
                "alphas": alphas,
                "variety_domains": variety_domains,
                "grid_points": grid_points,
                "instance_id": instance_id,
            },
        )
        return str(response.get("instance_id", instance_id))

    def generate_mapping(self, instance_id: str, request_payload: dict) -> None:
        payload = {
            "queue_lane": "human",
            "workflow": "live_pool",
            "split": "live",
            "run_id": request_payload.get("run_id"),
            "example_code": request_payload.get("example_code", ['1-1','1-2','1-3','2-1','3-1','3-3','4-1','4-2','5-1','5-3','6-1','6-2','6-3','7-1','8-1']),
            "Level_Len": request_payload.get("Level_Len",60),
            "module": request_payload.get("module", request_payload.get("Level_Len", 40)),
            "batch_size": request_payload.get("batch_size", 1),
            "epochs": request_payload.get("epochs", 30),
            "use_grammar_degree_grid": request_payload.get("use_grammar_degree_grid", True),
            "grammar_min_degree": request_payload.get("grammar_min_degree", 2),
            "grammar_end_out_degree": request_payload.get("grammar_end_out_degree", 2),
            "grammar_bulk_repeats": request_payload.get("grammar_bulk_repeats", 1),
            "grammar_max_attempts": request_payload.get("grammar_max_attempts", 256),
        }
        job = self.post(f"/instances/{instance_id}/generate_mapping/jobs", payload)
        job_id = job.get("job_id")
        if not job_id:
            raise RuntimeError("API did not return a job_id for generate_mapping")
        self.wait_for_job_result(str(job_id))

    def fetch_levels(
        self,
        instance_id: str,
        agent_name: str,
        request_payload: dict,
        batch_size: int = 1,
        priority: int = 0,
        difficulty: Optional[str] = None,
        refill_threshold: int = 4,
    ) -> Dict[str, Any]:
        """Pull up to `batch_size` levels in a single round-trip (reduces the number
        of HTTP calls needed to keep the prefetch queue warm vs. one call per level)."""
        try:
            from world_generation.level_formats import rows_to_tilemap
        except ImportError:
            import pandas as pd

            def rows_to_tilemap(rows):
                return pd.DataFrame(rows)

        payload = {
            "agent_name": agent_name,
            "request": request_payload,
            "batch_size": max(1, int(batch_size or 1)),
            "priority": priority,
            "queue_lane": "human",
            "workflow": "live_pool",
            "split": "live",
            "refill_threshold": max(1, int(refill_threshold or 1)),
        }
        if difficulty is not None:
            payload["difficulty"] = str(difficulty)

        job = self.post(f"/instances/{instance_id}/pool_manager/jobs", payload)
        job_id = job.get("job_id")
        if not job_id:
            raise RuntimeError("API did not return a job_id for pool_manager")
        response = self.wait_for_job_result(str(job_id))
        tilemaps = response.get("tilemaps") or []
        if not tilemaps and response.get("tilemap") is not None:
            tilemaps = [response["tilemap"]]
        return {
            "tilemaps": [rows_to_tilemap(rows) for rows in tilemaps],
            "generation": int(response.get("generation", 0)),
        }


class LevelStreamBuffer:
    """Client-side queue of generated levels, refilled from pool_manager."""

    _MAX_DEAD_LETTERS = 50
    _RETRY_BACKOFF_SECONDS = 1.0

    def __init__(
        self,
        api_client: WorldGenerationApiClient,
        instance_id: str,
        agent_name: str,
        request_payload: dict,
        prefetch_levels: int,
        priority: int = 0,
        difficulty: Optional[str] = None,
    ):
        self.api_client = api_client
        self.instance_id = instance_id
        self.agent_name = agent_name
        self.request_payload = request_payload
        self.prefetch_levels = max(1, int(prefetch_levels))
        self.priority = priority
        self.difficulty = str(difficulty) if difficulty is not None else None
        # Fetch several levels per HTTP round-trip instead of one-per-request: cuts
        # the number of pool_manager calls needed to keep the queue warm roughly
        # by a factor of fetch_chunk_size.
        self.fetch_chunk_size = max(1, min(4, self.prefetch_levels))
        self._queue = deque()
        self._pending: deque[Future] = deque()
        self._pending_count = 0
        self._generation: Optional[int] = None
        self._executor = ThreadPoolExecutor(max_workers=max(1, min(4, self.prefetch_levels)))
        self._dead_letters: deque = deque(maxlen=self._MAX_DEAD_LETTERS)
        self._next_retry_at = 0.0

    def _accept_result(self, result: Dict[str, Any]) -> None:
        generation = int(result.get("generation", 0))
        tilemaps = result.get("tilemaps", [])
        if self._generation is None or generation > self._generation:
            self._generation = generation
            self._queue.clear()
        elif generation < self._generation:
            return
        for tilemap in tilemaps:
            self._queue.append({"tilemap": tilemap, "generation": generation})

    def _submit_fetch(self) -> None:
        if time.monotonic() < self._next_retry_at:
            return
        chunk = self.fetch_chunk_size
        self._pending.append(
            self._executor.submit(
                self.api_client.fetch_levels,
                self.instance_id,
                self.agent_name,
                self.request_payload,
                chunk,
                self.priority,
                difficulty=self.difficulty,
                refill_threshold=max(4, self.prefetch_levels),
            )
        )
        self._pending_count += chunk

    def _record_dead_letter(self, exc: Exception) -> None:
        # Capture failed fetches instead of silently dropping them, and back off
        # briefly so a stuck/unhealthy API isn't hammered with immediate retries.
        self._dead_letters.append({
            "agent_name": self.agent_name,
            "reason": str(exc),
            "at": time.time(),
        })
        self._next_retry_at = time.monotonic() + self._RETRY_BACKOFF_SECONDS

    def _drain_ready(self) -> None:
        ready_count = 0
        for future in self._pending:
            if future.done():
                ready_count += 1
            else:
                break

        for _ in range(ready_count):
            future = self._pending.popleft()
            self._pending_count = max(0, self._pending_count - self.fetch_chunk_size)
            try:
                self._accept_result(future.result())
            except Exception as exc:
                self._record_dead_letter(exc)

    def _wait_for_next(self):
        if not self._pending:
            self._submit_fetch()
        if not self._pending:
            return  # still in backoff window; nothing in flight to wait on
        future = self._pending.popleft()
        self._pending_count = max(0, self._pending_count - self.fetch_chunk_size)
        try:
            self._accept_result(future.result())
        except Exception as exc:
            self._record_dead_letter(exc)

    def refill(self, min_size: int = 1) -> None:
        self._drain_ready()
        target = max(int(min_size), self.prefetch_levels)
        while len(self._queue) + self._pending_count < target:
            before = len(self._pending)
            self._submit_fetch()
            if len(self._pending) == before:
                break  # in backoff window; avoid spinning

    def pop_next(self) -> "pd.DataFrame":
        self._drain_ready()
        if not self._queue:
            self.refill(min_size=1)
            self._drain_ready()
        if not self._queue:
            self._wait_for_next()
        if not self._queue:
            raise RuntimeError(
                f"No levels available for {self.agent_name} after fetch failures; "
                f"see dead_letters() ({len(self._dead_letters)} recorded)"
            )
        return self._queue.popleft()["tilemap"]

    def dead_letters(self) -> List[Dict[str, Any]]:
        return list(self._dead_letters)

    def size(self) -> int:
        self._drain_ready()
        return len(self._queue) + self._pending_count

    def latest_generation(self) -> Optional[int]:
        self._drain_ready()
        return self._generation

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class InfiniteLane:
    alpha: float
    instance_id: str
    stream: LevelStreamBuffer
    agent_name_suffix: str
    game_state: Optional[Any] = None
    retry_preview_state: Optional[Any] = None
    chunks_appended: int = 0
    tiles_appended: int = 0


@dataclass
class HoldHudResult:
    action: str
    text: str = ""


def _normalize_alpha_schedule(alpha_schedule: Optional[List[float]]) -> List[float]:
    """Return sorted unique alpha values with a safe default."""
    if not alpha_schedule:
        return [0.0, 0.05, 0.1]
    dedup = sorted({float(a) for a in alpha_schedule})
    return dedup if dedup else [0.0, 0.05, 0.1]


def _full_interval_alpha_schedule(step: float = 0.05) -> List[float]:
    """Build [0, 1] alpha schedule with constant spacing used by pipe routing."""
    step = max(0.01, min(1.0, float(step)))
    points = [0.0]
    x = 0.0
    while x + step < 1.0:
        x += step
        points.append(round(x, 6))
    if points[-1] != 1.0:
        points.append(1.0)
    return points


def _alpha_window(alpha: float, window_size: float = 0.1) -> List[float]:
    """Return a clamped [low, high] alpha window on [0, 1] with target width."""
    width = max(0.0, float(window_size))
    center = min(1.0, max(0.0, float(alpha)))
    half = width / 2.0

    low = center - half
    high = center + half

    if low < 0.0:
        high = min(1.0, high - low)
        low = 0.0
    if high > 1.0:
        low = max(0.0, low - (high - 1.0))
        high = 1.0

    return [float(low), float(high)]


def _build_stream_request_payload(
    base_request_payload: dict,
    alpha: float,
    alpha_window_size: float,
) -> dict:
    payload = dict(base_request_payload or {})
    payload["target_alpha"] = float(alpha)
    payload["alpha_window_size"] = max(0.0, float(alpha_window_size))
    payload["alpha_window"] = _alpha_window(alpha, alpha_window_size)
    return payload


def _build_alpha_streams(
    api_client: WorldGenerationApiClient,
    agent_name: str,
    base_request_payload: dict,
    alpha_schedule: List[float],
    prefetch_levels: int,
    alpha_window_size: float = 0.1,
) -> List[InfiniteLane]:
    """Create one Map-Elites instance+stream per alpha for directional difficulty routing."""
    streams: List[InfiniteLane] = []
    for idx, alpha in enumerate(alpha_schedule):
        stream_request_payload = _build_stream_request_payload(
            base_request_payload,
            alpha=alpha,
            alpha_window_size=alpha_window_size,
        )
        instance_id = api_client.create_instance(alphas=_alpha_window(alpha, alpha_window_size))
        stream = LevelStreamBuffer(
            api_client=api_client,
            instance_id=instance_id,
            agent_name=f"{agent_name}|a={alpha}",
            request_payload=stream_request_payload,
            prefetch_levels=prefetch_levels,
            difficulty=f"alpha:{alpha}",
        )
        # Initialize only the starting lane; other alpha lanes generate on first use.
        if idx == 0:
            stream.refill(min_size=prefetch_levels)
        streams.append(
            InfiniteLane(
                alpha=float(alpha),
                instance_id=instance_id,
                stream=stream,
                agent_name_suffix=f"|a={alpha}",
            )
        )
    return streams


def _build_streams_for_provided_instance(
    api_client: WorldGenerationApiClient,
    instance_id: str,
    agent_name: str,
    request_payload: dict,
    alpha_schedule: List[float],
    prefetch_levels: int,
    alpha_window_size: float,
) -> List[InfiniteLane]:
    """
    Build one client stream per alpha using the same existing instance.

    This preserves pipe-based alpha movement even when the caller pins
    generation to a pre-created instance via --instance-id.
    """
    streams: List[InfiniteLane] = []
    for idx, alpha in enumerate(alpha_schedule):
        stream_request_payload = _build_stream_request_payload(
            request_payload,
            alpha=alpha,
            alpha_window_size=alpha_window_size,
        )
        stream = LevelStreamBuffer(
            api_client=api_client,
            instance_id=instance_id,
            agent_name=f"{agent_name}|provided|i={instance_id}|a={alpha}",
            request_payload=stream_request_payload,
            prefetch_levels=prefetch_levels,
            difficulty=f"alpha:{alpha}",
        )
        # Keep startup responsive: only warm the initial stream eagerly.
        if idx == 0:
            stream.refill(min_size=prefetch_levels)
        streams.append(
            InfiniteLane(
                alpha=float(alpha),
                instance_id=instance_id,
                stream=stream,
                agent_name_suffix=f"|provided|i={instance_id}|a={alpha}",
            )
        )
    return streams


def _worldgen_hud_line(
    infinite_mode: bool,
    alpha_streams: Optional[List[InfiniteLane]],
    current_alpha_idx: int,
    alpha_window_size: float,
    levels_completed: int,
    tiles_appended: int,
    level_len: Optional[int],
    pipe_alpha_step: float,
) -> str:
    """Build concise HUD diagnostics for infinite world-generation mode."""
    if not infinite_mode:
        return "WORLD: static map"
    if not alpha_streams:
        return "WORLD: infinite (initializing streams...)"

    idx = max(0, min(len(alpha_streams) - 1, int(current_alpha_idx)))
    current = alpha_streams[idx]
    current_alpha = float(current.alpha)
    low, high = _alpha_window(current_alpha, alpha_window_size)

    stream = current.stream
    queued = stream.size() if stream is not None else 0
    generation = stream.latest_generation() if stream is not None else None
    instance_id = str(current.instance_id)
    instance_short = instance_id[:8] if instance_id else "-"

    len_token = f"LEN:{int(level_len)}" if level_len is not None else "LEN:-"
    generation_token = f"G:{int(generation)}" if generation is not None else "G:-"
    return (
        f"WG:INF A:{current_alpha:0.2f} W:[{low:0.2f},{high:0.2f}] "
        f"S:{idx + 1}/{len(alpha_streams)} dA:{pipe_alpha_step:0.2f} Q:{queued:02d} {generation_token} I:{instance_short} "
        f"APP:{levels_completed} +T:{tiles_appended} {len_token}"
    )


def _default_time_trial_ranking_path() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base, "history", "human_time_trial_ranking.json")


def _format_clock(seconds: float) -> str:
    clamped = max(0.0, float(seconds))
    minutes = int(clamped // 60)
    secs = clamped - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_time_trial_ranking(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"updated_at": _iso_now(), "entries": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {"updated_at": _iso_now(), "entries": []}

    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    if not isinstance(entries, list):
        entries = []
    return {
        "updated_at": str(raw.get("updated_at", _iso_now())) if isinstance(raw, dict) else _iso_now(),
        "entries": entries,
    }


def _sorted_ranking_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _score_key(entry: Dict[str, Any]) -> tuple[float, float, float, str]:
        survived = float(entry.get("survived_seconds", 0.0) or 0.0)
        bonus = float(entry.get("bonus_time_seconds", 0.0) or 0.0)
        remaining = float(entry.get("remaining_seconds", 0.0) or 0.0)
        player = str(entry.get("player", ""))
        return (survived, bonus, remaining, player)

    return sorted(entries, key=_score_key, reverse=True)


def _rank_position(entries: List[Dict[str, Any]], player_name: str) -> int:
    key = str(player_name or "").strip()
    ordered = _sorted_ranking_entries(entries)
    if not key:
        return len(ordered) + 1
    for idx, row in enumerate(ordered, start=1):
        if str(row.get("player", "")).strip() == key:
            return idx
    return len(ordered) + 1


def _format_rank_change_message(previous_rank: Optional[int], new_rank: int, total: int) -> str:
    if total <= 0:
        return "Rank: #1/1"
    if previous_rank is None or previous_rank <= 0 or previous_rank == new_rank:
        return f"Rank: #{new_rank}/{total}"
    if new_rank < previous_rank:
        return f"Rank improved to #{new_rank}/{total}"
    return f"Rank dropped to #{new_rank}/{total}"


def _save_time_trial_ranking(path: str, entries: List[Dict[str, Any]]) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "updated_at": _iso_now(),
        "entries": _sorted_ranking_entries(entries),
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    os.replace(tmp_path, path)


def _build_rank_preview(
    ranking_entries: List[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> tuple[int, int, List[Dict[str, Any]]]:
    tagged = list(ranking_entries)
    marker = dict(candidate)
    marker["_candidate"] = True
    tagged.append(marker)
    ordered = _sorted_ranking_entries(tagged)
    rank = 1
    for idx, row in enumerate(ordered, start=1):
        if row.get("_candidate"):
            rank = idx
            break
    clean = [{k: v for k, v in row.items() if k != "_candidate"} for row in ordered]
    return rank, len(clean), clean


def _draw_between_tries_ui(
    env: Any,
    title: str,
    summary_lines: List[str],
    ranking_lines: List[str],
    countdown_seconds: float,
    fps: int,
    hint_text: str = "ENTER/SPACE: continue   Q/ESC: quit",
    ranking_title: str = "Ranking",
    editable_label: str = "",
    editable_text: str = "",
    primary_label: str = "PLAY",
    secondary_label: str = "RESTART",
) -> None:
    try:
        import pygame
    except ImportError:
        return

    renderer = getattr(env, "_renderer", None)
    if renderer is not None:
        try:
            renderer.draw(env._gs)
        except Exception:
            pass
    if not pygame.get_init():
        return
    screen = getattr(renderer, "_screen", None) if renderer is not None else None
    if screen is None:
        return

    width, height = screen.get_size()
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((10, 14, 26, 160))
    screen.blit(overlay, (0, 0))

    panel = pygame.Rect(
        max(16, width // 10),
        max(16, height // 10),
        min(width - 32, int(width * 0.74)),
        min(height - 32, int(height * 0.74)),
    )
    pygame.draw.rect(screen, (18, 24, 36), panel, border_radius=18)
    pygame.draw.rect(screen, (120, 170, 255), panel, width=2, border_radius=18)
    pygame.draw.rect(screen, (31, 44, 72), panel.inflate(-10, -10), border_radius=14)

    title_size = max(22, min(34, panel.h // 10))
    main_size = max(16, min(22, panel.h // 16))
    small_size = max(13, min(18, panel.h // 20))

    font_title = pygame.font.SysFont("consolas", title_size, bold=True)
    font_main = pygame.font.SysFont("consolas", main_size, bold=True)
    font_small = pygame.font.SysFont("consolas", small_size)

    y = panel.y + 18
    title_label = font_title.render(title, True, (244, 247, 255))
    screen.blit(title_label, (panel.x + 18, y))
    y += title_label.get_height() + 12

    timer_text = f"TIME LEFT: {_format_clock(countdown_seconds)}"
    timer_color = (255, 118, 118) if countdown_seconds <= 15.0 else (255, 224, 120)
    timer_label = font_main.render(timer_text, True, timer_color)
    screen.blit(timer_label, (panel.x + 18, y))
    y += timer_label.get_height() + 10

    if editable_label:
        editor_label = font_small.render(editable_label, True, (164, 214, 255))
        screen.blit(editor_label, (panel.x + 18, y))
        y += editor_label.get_height() + 6

        input_rect = pygame.Rect(panel.x + 18, y, max(220, panel.w - 36), font_main.get_height() + 16)
        pygame.draw.rect(screen, (10, 16, 28), input_rect, border_radius=10)
        pygame.draw.rect(screen, (154, 199, 255), input_rect, width=2, border_radius=10)

        shown_text = editable_text or " "
        input_text = font_main.render(shown_text, True, (246, 248, 255))
        screen.blit(input_text, (input_rect.x + 10, input_rect.y + 8))

        caret_x = min(input_rect.right - 12, input_rect.x + 10 + input_text.get_width() + 2)
        pygame.draw.line(screen, (255, 235, 120), (caret_x, input_rect.y + 6), (caret_x, input_rect.bottom - 6), 2)
        y = input_rect.bottom + 12

    footer_y = panel.bottom - 52
    line_gap = max(4, small_size // 3)
    section_gap = max(8, small_size // 2)

    available_summary_h = max(0, footer_y - 64 - y)
    max_summary_lines = max(1, min(8, available_summary_h // max(1, font_small.get_height() + line_gap)))
    for line in summary_lines[:max_summary_lines]:
        screen.blit(font_small.render(line, True, (236, 240, 255)), (panel.x + 18, y))
        y += font_small.get_height() + line_gap

    if ranking_lines and y < footer_y - (font_main.get_height() + font_small.get_height() + section_gap):
        y += section_gap
        ranking_label = font_main.render(ranking_title, True, (170, 220, 255))
        screen.blit(ranking_label, (panel.x + 18, y))
        y += ranking_label.get_height() + 6

        available_rank_h = max(0, footer_y - y - 12)
        max_ranking_lines = max(1, min(5, available_rank_h // max(1, font_small.get_height() + line_gap)))
        for line in ranking_lines[:max_ranking_lines]:
            screen.blit(font_small.render(line, True, (214, 228, 255)), (panel.x + 18, y))
            y += font_small.get_height() + line_gap

    primary_rect = pygame.Rect(panel.right - 192, footer_y, 156, 30)
    secondary_rect = pygame.Rect(panel.x + 18, footer_y, 146, 30)
    pygame.draw.rect(screen, (84, 214, 132), primary_rect, border_radius=10)
    pygame.draw.rect(screen, (91, 129, 239), secondary_rect, border_radius=10)
    pygame.draw.rect(screen, (230, 239, 255), primary_rect, width=1, border_radius=10)
    pygame.draw.rect(screen, (230, 239, 255), secondary_rect, width=1, border_radius=10)

    primary_label_surf = font_main.render(primary_label, True, (8, 18, 24))
    secondary_label_surf = font_main.render(secondary_label, True, (240, 244, 255))
    screen.blit(primary_label_surf, (primary_rect.centerx - primary_label_surf.get_width() // 2, primary_rect.centery - primary_label_surf.get_height() // 2))
    screen.blit(secondary_label_surf, (secondary_rect.centerx - secondary_label_surf.get_width() // 2, secondary_rect.centery - secondary_label_surf.get_height() // 2))

    hint_label = font_small.render(hint_text, True, (245, 247, 255))
    screen.blit(hint_label, (panel.x + 18, panel.bottom - 24))

    pygame.display.flip()
    if renderer is not None and getattr(renderer, "_clock", None):
        renderer._clock.tick(fps)


def _intermission_wait(
    env: Any,
    title: str,
    summary_lines: List[str],
    ranking_lines: List[str],
    timer_deadline: float,
    fps: int,
    hint_text: str = "ENTER/SPACE: continue   Q/ESC: quit",
    ranking_title: str = "Ranking",
    editable_label: str = "",
    initial_text: str = "",
    max_text_length: int = 24,
    primary_label: str = "PLAY",
    secondary_label: str = "RESTART",
) -> HoldHudResult:
    try:
        import pygame
    except ImportError:
        return HoldHudResult(action="continue", text=initial_text)
    if not pygame.get_init():
        return HoldHudResult(action="continue", text=initial_text)

    current_text = str(initial_text or "")[:max(1, int(max_text_length))]

    while True:
        now = time.monotonic()
        remaining = max(0.0, timer_deadline - now)
        if remaining <= 0.0:
            return HoldHudResult(action="timeout", text=current_text)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return HoldHudResult(action="quit", text=current_text)
            if event.type == pygame.KEYDOWN:
                typed = str(getattr(event, "unicode", "") or "")

                if editable_label:
                    if event.key == pygame.K_RETURN:
                        return HoldHudResult(action="continue", text=current_text)
                    if event.key == pygame.K_BACKSPACE:
                        current_text = current_text[:-1]
                        continue
                    if typed and typed.isprintable() and len(current_text) < max(1, int(max_text_length)):
                        current_text += typed
                        continue
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        continue
                    if event.key == pygame.K_r:
                        continue
                    continue

                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return HoldHudResult(action="quit", text=current_text)
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return HoldHudResult(action="continue", text=current_text)
                if event.key == pygame.K_r:
                    return HoldHudResult(action="restart", text=current_text)

        _draw_between_tries_ui(
            env=env,
            title=title,
            summary_lines=summary_lines,
            ranking_lines=ranking_lines,
            countdown_seconds=remaining,
            fps=fps,
            hint_text=hint_text,
            ranking_title=ranking_title,
            editable_label=editable_label,
            editable_text=current_text,
            primary_label=primary_label,
            secondary_label=secondary_label,
        )


def _rename_lane_agent_names(alpha_streams: Optional[List[InfiniteLane]], player_name: str) -> None:
    if not alpha_streams:
        return
    base_name = str(player_name or "human_player").strip() or "human_player"
    for lane in alpha_streams:
        lane.stream.agent_name = f"{base_name}{lane.agent_name_suffix}"


def _pull_next_chunk(
    stream: LevelStreamBuffer,
    rare_pipe_replace_rate: float,
    rare_pipe_blue_share: float,
    rare_pipe_rng: random.Random,
    refill_target: int,
    rare_pipe_state: Optional[Dict[str, Any]] = None,
    rare_pipe_guarantee_window: int = 6,
) -> "pd.DataFrame":
    """Pop next chunk and apply rare-pipe conversion; keeps queue warm."""
    if stream.size() <= 1:
        stream.refill(min_size=max(2, refill_target))
    next_chunk = stream.pop_next()
    chunk = _inject_rare_pipes(
        next_chunk,
        replace_rate=rare_pipe_replace_rate,
        blue_share=rare_pipe_blue_share,
        rng=rare_pipe_rng,
    )
    if rare_pipe_state is not None:
        chunk = _enforce_rare_pipe_guarantee(
            chunk,
            state=rare_pipe_state,
            guarantee_window=rare_pipe_guarantee_window,
            rng=rare_pipe_rng,
        )
    return chunk


def _build_game_state_for_chunk(level_chunk: "pd.DataFrame") -> Any:
    try:
        from .game_state import GameState
        from .tilemap import parse_tilemap
    except ImportError:
        from game_state import GameState
        from tilemap import parse_tilemap

    return GameState(tilemap_source=parse_tilemap(level_chunk).clone())


def _prime_lane_state(
    lane: InfiniteLane,
    rare_pipe_replace_rate: float,
    rare_pipe_blue_share: float,
    rare_pipe_rng: random.Random,
    rare_pipe_state: Optional[Dict[str, Any]],
    rare_pipe_guarantee_window: int,
) -> None:
    if lane.game_state is not None:
        return
    lane.game_state = _build_game_state_for_chunk(
        _pull_next_chunk(
            lane.stream,
            rare_pipe_replace_rate=rare_pipe_replace_rate,
            rare_pipe_blue_share=rare_pipe_blue_share,
            rare_pipe_rng=rare_pipe_rng,
            refill_target=max(2, lane.stream.prefetch_levels),
            rare_pipe_state=rare_pipe_state,
            rare_pipe_guarantee_window=rare_pipe_guarantee_window,
        )
    )
    lane.retry_preview_state = copy.deepcopy(lane.game_state)


def _save_env_to_lane(env: Any, lane: InfiniteLane) -> None:
    lane.game_state = copy.deepcopy(env._gs)


def _load_lane_into_env(env: Any, lane: InfiniteLane) -> int:
    if lane.game_state is None:
        raise RuntimeError(f"Lane alpha={lane.alpha:.2f} is not initialized")
    env._gs = copy.deepcopy(lane.game_state)
    env._tilemap_source = env._gs.tilemap_source.clone()
    lane.retry_preview_state = None
    return int(env._gs.tilemap.width)


def _prime_preview_lanes(
    alpha_streams: Optional[List[InfiniteLane]],
    current_idx: int,
    rare_pipe_replace_rate: float,
    rare_pipe_blue_share: float,
    rare_pipe_rng: random.Random,
    rare_pipe_state: Optional[Dict[str, Any]],
    rare_pipe_guarantee_window: int,
) -> None:
    if not alpha_streams:
        return
    watch = [current_idx, max(0, current_idx - 1), min(len(alpha_streams) - 1, current_idx + 1)]
    seen = set()
    for idx in watch:
        if idx in seen:
            continue
        seen.add(idx)
        _prime_lane_state(
            alpha_streams[idx],
            rare_pipe_replace_rate=rare_pipe_replace_rate,
            rare_pipe_blue_share=rare_pipe_blue_share,
            rare_pipe_rng=rare_pipe_rng,
            rare_pipe_state=rare_pipe_state,
            rare_pipe_guarantee_window=rare_pipe_guarantee_window,
        )


def _load_recycled_retry_preview(
    env: Any,
    alpha_streams: Optional[List[InfiniteLane]],
    current_alpha_idx: int,
) -> tuple[bool, int]:
    if not alpha_streams:
        return False, current_alpha_idx

    ordered_indexes = [
        current_alpha_idx,
        min(len(alpha_streams) - 1, current_alpha_idx + 1),
        max(0, current_alpha_idx - 1),
    ]
    ordered_indexes.extend(idx for idx in range(len(alpha_streams)) if idx not in ordered_indexes)

    for idx in ordered_indexes:
        lane = alpha_streams[idx]
        if lane.retry_preview_state is None:
            continue
        lane.game_state = copy.deepcopy(lane.retry_preview_state)
        lane.retry_preview_state = None
        _load_lane_into_env(env, lane)
        return True, idx

    return False, current_alpha_idx


def _schedule_alpha_refills(
    alpha_streams: List[InfiniteLane],
    current_idx: int,
    prefetch_levels: int,
) -> None:
    """Proactively top-up current and neighboring difficulty streams before depletion."""
    if not alpha_streams:
        return
    watch = {current_idx, max(0, current_idx - 1), min(len(alpha_streams) - 1, current_idx + 1)}
    for idx in watch:
        stream = alpha_streams[idx].stream
        if stream.size() < max(2, prefetch_levels // 2):
            stream.refill(min_size=prefetch_levels)


def _append_level_to_env(env: Any, level_chunk: "pd.DataFrame") -> int:
    """Append a tile chunk to the active environment map and spawn its goombas."""
    import numpy as np

    try:
        from .tilemap import parse_tilemap
        from .entities import Goomba
        from .constants import TILE_SIZE, GOOMBA_W
    except ImportError:
        from tilemap import parse_tilemap
        from entities import Goomba
        from constants import TILE_SIZE, GOOMBA_W

    gs = env._gs
    chunk_tm = parse_tilemap(level_chunk)
    current_tm = gs.tilemap
    source_tm = gs.tilemap_source
    col_offset = current_tm.width

    target_h = max(current_tm.height, chunk_tm.height)

    def _pad_rows(tiles: np.ndarray, height: int) -> np.ndarray:
        if tiles.shape[0] >= height:
            return tiles
        pad = np.zeros((height - tiles.shape[0], tiles.shape[1]), dtype=tiles.dtype)
        return np.vstack([tiles, pad])

    left_tiles = _pad_rows(current_tm.tiles, target_h)
    right_tiles = _pad_rows(chunk_tm.tiles, target_h)
    combined_tiles = np.hstack([left_tiles, right_tiles]).astype(np.int8, copy=False)
    combined_spawns = list(current_tm.enemy_spawns) + [
        (col + col_offset, row) for (col, row) in chunk_tm.enemy_spawns
    ]

    for tm in (current_tm, source_tm):
        tm.tiles = combined_tiles.copy() if tm is source_tm else combined_tiles
        tm.enemy_spawns = list(combined_spawns)
        tm.width = combined_tiles.shape[1]
        tm.height = combined_tiles.shape[0]

    for (col, row) in chunk_tm.enemy_spawns:
        x = (col + col_offset) * TILE_SIZE + (TILE_SIZE - GOOMBA_W) // 2
        y = row * TILE_SIZE
        gs.goombas.append(Goomba(x=float(x), y=float(y), vx=0.0))

    return chunk_tm.width


def _inject_rare_pipes(
    level_chunk: "pd.DataFrame",
    replace_rate: float,
    blue_share: float,
    rng: random.Random,
) -> "pd.DataFrame":
    """
    Replace a subset of regular upward pipes with rare red/blue pipe symbols.

    Input pipe symbols:
      top:  < >
      body: [ ]

    Output symbols:
      red:  r R / u U
      blue: v V / w W
    """
    if replace_rate <= 0:
        return level_chunk

    try:
        import pandas as pd
    except ImportError:
        return level_chunk

    if not isinstance(level_chunk, pd.DataFrame):
        chunk = pd.DataFrame(level_chunk)
    else:
        chunk = level_chunk.copy(deep=True)

    h, w = chunk.shape
    if h < 2 or w < 2:
        return chunk

    def _cell(r: int, c: int) -> str:
        return str(chunk.iat[r, c]).strip()

    candidates = []
    for row in range(h - 1):
        for col in range(w - 1):
            if _cell(row, col) != "<" or _cell(row, col + 1) != ">":
                continue
            # Only consider complete upward-pipe tops with body beneath.
            if _cell(row + 1, col) != "[" or _cell(row + 1, col + 1) != "]":
                continue
            candidates.append((row, col))

    for row, col in candidates:
        if rng.random() > replace_rate:
            continue

        if rng.random() < blue_share:
            top_l, top_r = "v", "V"
            body_l, body_r = "w", "W"
        else:
            top_l, top_r = "r", "R"
            body_l, body_r = "u", "U"

        chunk.iat[row, col] = top_l
        chunk.iat[row, col + 1] = top_r

        body_row = row + 1
        while body_row < h and _cell(body_row, col) == "[" and _cell(body_row, col + 1) == "]":
            chunk.iat[body_row, col] = body_l
            chunk.iat[body_row, col + 1] = body_r
            body_row += 1

    return chunk


def _pipe_candidates(
    chunk: "pd.DataFrame",
    top_left: str,
    top_right: str,
    body_left: str,
    body_right: str,
) -> List[tuple[int, int]]:
    """Return (row, col) candidates for pipe tops with matching body below."""
    h, w = chunk.shape

    def _cell(r: int, c: int) -> str:
        return str(chunk.iat[r, c]).strip()

    out: List[tuple[int, int]] = []
    for row in range(h - 1):
        for col in range(w - 1):
            if _cell(row, col) != top_left or _cell(row, col + 1) != top_right:
                continue
            if _cell(row + 1, col) != body_left or _cell(row + 1, col + 1) != body_right:
                continue
            out.append((row, col))
    return out


def _spawn_synthetic_rare_pipe(
    chunk: "pd.DataFrame",
    color: str,
    rng: random.Random,
) -> bool:
    """
    Create a minimal 2-tile-tall rare pipe on top of existing solid support.

    Returns True if a synthetic pipe was placed.
    """
    h, w = chunk.shape
    if h < 3 or w < 2:
        return False

    support_tiles = {
        "X", "S", "B", "b", "Q", "?",
        "<", ">", "[", "]",
        "r", "R", "u", "U",
        "v", "V", "w", "W",
    }
    replaceable = {"-", "o", "E"}

    def _cell(r: int, c: int) -> str:
        return str(chunk.iat[r, c]).strip()

    candidates: List[tuple[int, int]] = []
    # support_row is the row under the pipe body.
    for support_row in range(2, h):
        top_row = support_row - 2
        body_row = support_row - 1
        for col in range(w - 1):
            if _cell(support_row, col) not in support_tiles or _cell(support_row, col + 1) not in support_tiles:
                continue
            if _cell(top_row, col) not in replaceable or _cell(top_row, col + 1) not in replaceable:
                continue
            if _cell(body_row, col) not in replaceable or _cell(body_row, col + 1) not in replaceable:
                continue
            candidates.append((top_row, col))

    if not candidates:
        return False

    row, col = candidates[rng.randrange(len(candidates))]
    if color == "red":
        top_l, top_r, body_l, body_r = "r", "R", "u", "U"
    else:
        top_l, top_r, body_l, body_r = "v", "V", "w", "W"

    chunk.iat[row, col] = top_l
    chunk.iat[row, col + 1] = top_r
    chunk.iat[row + 1, col] = body_l
    chunk.iat[row + 1, col + 1] = body_r
    return True


def _recolor_pipe_at(
    chunk: "pd.DataFrame",
    row: int,
    col: int,
    src_body_left: str,
    src_body_right: str,
    dst_top_left: str,
    dst_top_right: str,
    dst_body_left: str,
    dst_body_right: str,
) -> None:
    """Recolor one full vertical pipe (top + contiguous body)."""
    h, _ = chunk.shape

    def _cell(r: int, c: int) -> str:
        return str(chunk.iat[r, c]).strip()

    chunk.iat[row, col] = dst_top_left
    chunk.iat[row, col + 1] = dst_top_right

    body_row = row + 1
    while body_row < h and _cell(body_row, col) == src_body_left and _cell(body_row, col + 1) == src_body_right:
        chunk.iat[body_row, col] = dst_body_left
        chunk.iat[body_row, col + 1] = dst_body_right
        body_row += 1


def _rare_pipe_presence(chunk: "pd.DataFrame") -> Dict[str, bool]:
    """Check whether chunk currently contains red/blue rare pipe tops."""
    values = set(str(v).strip() for v in chunk.values.ravel().tolist())
    return {
        "red": ("r" in values and "R" in values),
        "blue": ("v" in values and "V" in values),
    }


def _enforce_rare_pipe_guarantee(
    chunk: "pd.DataFrame",
    state: Dict[str, Any],
    guarantee_window: int,
    rng: random.Random,
) -> "pd.DataFrame":
    """
    Ensure both rare pipe colors appear eventually.

    Conditions:
    1) Bootstrap: until each color is seen at least once, force missing colors.
    2) Max-gap: if a color has been absent for `guarantee_window` chunks, force it.
    """
    present = _rare_pipe_presence(chunk)
    seen = state.setdefault("seen", {"red": False, "blue": False})
    miss = state.setdefault("miss", {"red": 0, "blue": 0})

    forced_colors: List[str] = []
    for color in ("red", "blue"):
        if not seen.get(color, False):
            forced_colors.append(color)
        elif miss.get(color, 0) >= max(1, guarantee_window):
            forced_colors.append(color)

    for color in forced_colors:
        if present.get(color, False):
            continue

        regular = _pipe_candidates(chunk, "<", ">", "[", "]")
        if regular:
            row, col = regular[rng.randrange(len(regular))]
            if color == "red":
                _recolor_pipe_at(chunk, row, col, "[", "]", "r", "R", "u", "U")
            else:
                _recolor_pipe_at(chunk, row, col, "[", "]", "v", "V", "w", "W")
            present = _rare_pipe_presence(chunk)
            continue

        # Fallback: recolor opposite rare pipe if available.
        if color == "red":
            opp = _pipe_candidates(chunk, "v", "V", "w", "W")
            if opp:
                row, col = opp[rng.randrange(len(opp))]
                _recolor_pipe_at(chunk, row, col, "w", "W", "r", "R", "u", "U")
                present = _rare_pipe_presence(chunk)
        else:
            opp = _pipe_candidates(chunk, "r", "R", "u", "U")
            if opp:
                row, col = opp[rng.randrange(len(opp))]
                _recolor_pipe_at(chunk, row, col, "u", "U", "v", "V", "w", "W")
                present = _rare_pipe_presence(chunk)

        # Last-resort fallback: synthesize a rare pipe even when no natural
        # pipes are present in this chunk.
        if not present.get(color, False):
            placed = _spawn_synthetic_rare_pipe(chunk, color=color, rng=rng)
            if placed:
                present = _rare_pipe_presence(chunk)

    for color in ("red", "blue"):
        if present.get(color, False):
            seen[color] = True
            miss[color] = 0
        else:
            miss[color] = int(miss.get(color, 0)) + 1

    return chunk


def _replace_level_in_env(env: Any, level_chunk: "pd.DataFrame") -> int:
    """Replace active world with a new level while preserving Mario counters."""
    try:
        from .tilemap import parse_tilemap
    except ImportError:
        from tilemap import parse_tilemap

    gs = env._gs
    next_tm = parse_tilemap(level_chunk)
    mario_stats = {
        "coins": gs.mario.coins,
        "score": gs.mario.score,
        "kills": gs.mario.kills,
        "state": gs.mario.state,
    }

    gs.tilemap_source = next_tm.clone()
    gs.tilemap = gs.tilemap_source.clone()
    gs.mario = gs._spawn_mario()
    gs.mario.coins = mario_stats["coins"]
    gs.mario.score = mario_stats["score"]
    gs.mario.kills = mario_stats["kills"]
    gs.mario.state = mario_stats["state"]
    gs.goombas = gs._spawn_goombas()

    # Keep episode continuity but reset local progression against new map origin.
    gs.done = False
    gs._max_x = gs.mario.x
    gs._prev_x = gs.mario.x
    return gs.tilemap.width


def _transport_via_pipe(
    env: Any,
    alpha_streams: Optional[List[InfiniteLane]],
    current_alpha_idx: int,
    pipe_color: str,
    rare_pipe_replace_rate: float,
    rare_pipe_blue_share: float,
    rare_pipe_rng: random.Random,
    rare_pipe_state: Dict[str, Any],
    rare_pipe_guarantee_window: int,
) -> tuple[bool, int]:
    """Enter a rare red/blue pipe and transport the player to another level."""
    if not alpha_streams:
        print(f"  → {pipe_color.title()} pipe has no stream target in standard mode")
        return False, current_alpha_idx

    if pipe_color == "blue":
        target_idx = min(len(alpha_streams) - 1, current_alpha_idx + 1)
    else:
        target_idx = max(0, current_alpha_idx - 1)

    current_lane = alpha_streams[current_alpha_idx]
    target_lane = alpha_streams[target_idx]
    _save_env_to_lane(env, current_lane)
    _prime_lane_state(
        target_lane,
        rare_pipe_replace_rate=rare_pipe_replace_rate,
        rare_pipe_blue_share=rare_pipe_blue_share,
        rare_pipe_rng=rare_pipe_rng,
        rare_pipe_state=rare_pipe_state,
        rare_pipe_guarantee_window=rare_pipe_guarantee_window,
    )
    prev_alpha = current_lane.alpha
    target_alpha = target_lane.alpha
    new_w = _load_lane_into_env(env, target_lane)
    print(
        f"  → Entered {pipe_color} pipe: alpha {prev_alpha:.2f} → {target_alpha:.2f} "
        f"(width={new_w})"
    )
    return True, target_idx


def run(
    map_path: Optional[Union[str, "pd.DataFrame"]] = None,
    fps: int = FPS,
    infinite_mode: bool = False,
    api_base_url: Optional[str] = None,
    instance_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_hash: Optional[str] = None,
    case_id: Optional[str] = None,
    pool_request_payload: Optional[dict] = None,
    alpha_schedule: Optional[List[float]] = None,
    alpha_full_interval: bool = True,
    pipe_alpha_step: float = 0.05,
    alpha_window_size: float = 0.1,
    prefetch_levels: int = 3,
    append_trigger_tiles: int = 16,
    rare_pipe_replace_rate: float = 0.18,
    rare_pipe_blue_share: float = 0.05,
    rare_pipe_seed: Optional[int] = None,
    rare_pipe_guarantee_window: int = 6,
) -> None:
    try:
        import pygame
    except ImportError:
        print("pygame is required.  Install with: pip install pygame")
        sys.exit(1)

    from env import MarioEnv
    from metrics import MetricsTracker
    from constants import TILE_SIZE

    resolved_agent_name = str(agent_name or agent_hash or "human_player").strip() or "human_player"
    run_id = f"human-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    registry = _get_registry()
    registry.record_run_start(
        agent_name=resolved_agent_name,
        mode="human_play",
        run_id=run_id,
        case_id=case_id,
        metadata={"infinite_mode": infinite_mode, "objective": "freeplay"},
    )

    api_client: Optional[WorldGenerationApiClient] = None
    alpha_streams: Optional[List[InfiniteLane]] = None
    current_alpha_idx: int = 0
    rare_pipe_rng = random.Random(rare_pipe_seed)
    rare_pipe_state: Dict[str, Any] = {
        "seen": {"red": False, "blue": False},
        "miss": {"red": 0, "blue": 0},
    }

    # Validate infinite mode requirements
    if infinite_mode:
        normalized_alphas = (
            _full_interval_alpha_schedule(pipe_alpha_step)
            if alpha_full_interval
            else _normalize_alpha_schedule(alpha_schedule)
        )
        if not api_base_url:
            api_base_url = "http://127.0.0.1:8000"
        if not pool_request_payload:
            pool_request_payload = {
                "example_code": ['1-1','1-2','1-3','2-1','3-1','3-3','4-1','4-2','5-1','5-3','6-1','6-2','6-3','7-1','8-1'],
                "Level_Len": 40,
                "module": 40,
                "batch_size": 1,
                "epochs": 10,
                "use_grammar_degree_grid": True,
                "grammar_min_degree": 2,
                "grammar_end_out_degree": 2,
                "grammar_bulk_repeats": 1,
                "grammar_max_attempts": 256,
            }
        pool_request_payload.setdefault("run_id", run_id)
        api_client = WorldGenerationApiClient(api_base_url)

        print("Waiting for world generation API...")
        try:
            api_client.wait_until_ready()
            print("✓ API ready")
        except RuntimeError as e:
            print(f"✗ Failed to reach API: {e}")
            sys.exit(1)

        # Build one stream per alpha to route red/blue pipes by difficulty direction.
        try:
            if instance_id:
                print("Using provided instance-id with alpha-indexed client streams")
                alpha_streams = _build_streams_for_provided_instance(
                    api_client=api_client,
                    instance_id=instance_id,
                    agent_name=resolved_agent_name,
                    request_payload=pool_request_payload,
                    alpha_schedule=normalized_alphas,
                    prefetch_levels=prefetch_levels,
                    alpha_window_size=alpha_window_size,
                )
            else:
                print("Creating alpha-mapped instances...")
                alpha_streams = _build_alpha_streams(
                    api_client=api_client,
                    agent_name=resolved_agent_name,
                    base_request_payload=pool_request_payload,
                    alpha_schedule=normalized_alphas,
                    prefetch_levels=prefetch_levels,
                    alpha_window_size=alpha_window_size,
                )
            current_alpha_idx = 0
            _prime_lane_state(
                alpha_streams[current_alpha_idx],
                rare_pipe_replace_rate=rare_pipe_replace_rate,
                rare_pipe_blue_share=rare_pipe_blue_share,
                rare_pipe_rng=rare_pipe_rng,
                rare_pipe_state=rare_pipe_state,
                rare_pipe_guarantee_window=rare_pipe_guarantee_window,
            )
            current_map = alpha_streams[current_alpha_idx].game_state.tilemap_source.clone()
            print(
                f"✓ Alpha streams ready ({len(alpha_streams)} streams), "
                f"starting alpha={alpha_streams[current_alpha_idx].alpha:.2f}"
            )
        except RuntimeError as e:
            print(f"✗ Failed to fetch level: {e}")
            sys.exit(1)
    else:
        if map_path is None:
            raise ValueError("map_path is required when not in infinite mode")
        current_map = map_path

    env = MarioEnv(map_path=current_map, obs_mode="rgb",
                   render_mode="human", fps=fps)
    if infinite_mode and alpha_streams:
        _load_lane_into_env(env, alpha_streams[current_alpha_idx])
    tracker = MetricsTracker(keep_steps=True)

    try:
        obs, _ = env.reset()
        if infinite_mode and alpha_streams:
            _schedule_alpha_refills(
                alpha_streams=alpha_streams,
                current_idx=current_alpha_idx,
                prefetch_levels=prefetch_levels,
            )
        try:
            env.set_hud_extra_line(
                _worldgen_hud_line(
                    infinite_mode=infinite_mode,
                    alpha_streams=alpha_streams,
                    current_alpha_idx=current_alpha_idx,
                    alpha_window_size=alpha_window_size,
                    levels_completed=0,
                    tiles_appended=0,
                    level_len=(pool_request_payload or {}).get("Level_Len") if pool_request_payload else None,
                    pipe_alpha_step=pipe_alpha_step,
                )
            )
        except Exception:
            pass
        tracker.new_episode()
        episode = 0
        total_steps = 0
        levels_completed = 0
        tiles_appended = 0
        pipe_cooldown = 0

        print("=" * 50)
        mode_info = " (Infinite Runner)" if infinite_mode else ""
        print(f"  Mario RL – Human Mode{mode_info}")
        print("  Controls: ←/→/A/D  move   ↓/S enter pipe   Z/Space  jump")
        print("            X/Shift  run    R  restart   Q  quit")
        print("=" * 50)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        obs, _ = env.reset()
                        tracker.new_episode()
                        episode += 1
                        print(f"\n[Episode {episode}] Restarted")

            keys = pygame.key.get_pressed()
            actions = {
                'right': keys[pygame.K_RIGHT] or keys[pygame.K_d],
                'left':  keys[pygame.K_LEFT]  or keys[pygame.K_a],
                'down':  keys[pygame.K_DOWN]  or keys[pygame.K_s],
                'jump':  keys[pygame.K_z]     or keys[pygame.K_SPACE],
                'run':   keys[pygame.K_x]     or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT],
            }

            action_idx = _actions_to_idx(actions)

            obs, reward, done, trunc, info = env.step_with_actions(actions)
            tracker.record(action_idx, reward, info)
            total_steps += 1

            try:
                env.set_hud_extra_line(
                    _worldgen_hud_line(
                        infinite_mode=infinite_mode,
                        alpha_streams=alpha_streams,
                        current_alpha_idx=current_alpha_idx,
                        alpha_window_size=alpha_window_size,
                        levels_completed=levels_completed,
                        tiles_appended=tiles_appended,
                        level_len=(pool_request_payload or {}).get("Level_Len") if pool_request_payload else None,
                        pipe_alpha_step=pipe_alpha_step,
                    )
                )
            except Exception:
                pass

            if pipe_cooldown > 0:
                pipe_cooldown -= 1

            if pipe_cooldown == 0 and info.get("pipe_entry") in ("red", "blue"):
                try:
                    transported, current_alpha_idx = _transport_via_pipe(
                        env,
                        alpha_streams,
                        current_alpha_idx,
                        info["pipe_entry"],
                        rare_pipe_replace_rate=rare_pipe_replace_rate,
                        rare_pipe_blue_share=rare_pipe_blue_share,
                        rare_pipe_rng=rare_pipe_rng,
                        rare_pipe_state=rare_pipe_state,
                        rare_pipe_guarantee_window=rare_pipe_guarantee_window,
                    )
                    if transported:
                        pipe_cooldown = max(18, fps // 3)
                        obs = env.render() if env.render_mode == "rgb_array" else obs
                except RuntimeError as e:
                    print(f"  ⚠ Pipe transport failed: {e}")

            if infinite_mode:
                if alpha_streams:
                    _schedule_alpha_refills(
                        alpha_streams=alpha_streams,
                        current_idx=current_alpha_idx,
                        prefetch_levels=prefetch_levels,
                    )
                mario_col = int(env._gs.mario.cx // TILE_SIZE)
                remaining_tiles = env._gs.tilemap.width - mario_col
                if remaining_tiles <= max(4, append_trigger_tiles):
                    try:
                        if not alpha_streams:
                            raise RuntimeError("alpha streams not initialized")
                        current_lane = alpha_streams[current_alpha_idx]
                        next_chunk = _pull_next_chunk(
                            current_lane.stream,
                            rare_pipe_replace_rate=rare_pipe_replace_rate,
                            rare_pipe_blue_share=rare_pipe_blue_share,
                            rare_pipe_rng=rare_pipe_rng,
                            refill_target=prefetch_levels,
                            rare_pipe_state=rare_pipe_state,
                            rare_pipe_guarantee_window=rare_pipe_guarantee_window,
                        )
                        added = _append_level_to_env(env, next_chunk)
                        current_lane.chunks_appended += 1
                        current_lane.tiles_appended += added
                        _save_env_to_lane(env, current_lane)
                        tiles_appended += added
                        levels_completed += 1
                        print(
                            f"  → Appended chunk #{levels_completed} "
                            f"(+{added} tiles, alpha={current_lane.alpha:.2f}, "
                            f"total_width={env._gs.tilemap.width})"
                        )
                    except RuntimeError as e:
                        print(f"  ⚠ Failed to append next level chunk: {e}")

            if done or trunc:
                ep_rec = tracker.end_episode(info)
                status = "WON 🎉" if ep_rec.won else ("TRUNCATED" if trunc else "DIED 💀")
                print(f"\n[Episode {episode}] {status}  "
                      f"reward={ep_rec.total_reward:.1f}  "
                      f"steps={ep_rec.steps}  "
                      f"coins={ep_rec.coins}  "
                      f"max_x={ep_rec.max_x:.0f}")

                if infinite_mode:
                    # Keep the continuous world; only reset on death/truncation.
                    obs, _ = env.reset()
                    pipe_cooldown = 0
                else:
                    obs, _ = env.reset()
                    pipe_cooldown = 0

                tracker.new_episode()
                episode += 1
    finally:
        env.close()
        if alpha_streams:
            for alpha_stream in alpha_streams:
                alpha_stream.stream.close()

    print(f"\nTotal episodes: {episode}   Total steps: {total_steps}")
    if infinite_mode:
        print(f"Appended chunks: {levels_completed}   Added tiles: {tiles_appended}")
    tracker.print_summary()
    save = input("Save replay? [y/N] ").strip().lower()
    if save == "y":
        tracker.save_csv("metrics_human.csv")
        tracker.save_replay("replay_human_last.json")
        registry.register(
            artifact_kind="metrics_csv",
            artifact_path="metrics_human.csv",
            case_id=case_id,
            run_id=run_id,
            agent_name=resolved_agent_name,
            mode="human_play",
        )
        registry.register(
            artifact_kind="replay",
            artifact_path="replay_human_last.json",
            case_id=case_id,
            run_id=run_id,
            agent_name=resolved_agent_name,
            mode="human_play",
        )

    registry.record_run_complete(
        agent_name=resolved_agent_name,
        mode="human_play",
        run_id=run_id,
        case_id=case_id,
        metrics=tracker.summary(),
    )


def run_survival_time_trial(
    map_path: Optional[Union[str, "pd.DataFrame"]] = None,
    fps: int = FPS,
    infinite_mode: bool = True,
    api_base_url: Optional[str] = None,
    instance_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_hash: Optional[str] = None,
    case_id: Optional[str] = None,
    pool_request_payload: Optional[dict] = None,
    alpha_schedule: Optional[List[float]] = None,
    alpha_full_interval: bool = True,
    pipe_alpha_step: float = 0.1,
    alpha_window_size: float = 0.1,
    prefetch_levels: int = 3,
    append_trigger_tiles: int = 16,
    rare_pipe_replace_rate: float = 0.18,
    rare_pipe_blue_share: float = 0.05,
    rare_pipe_seed: Optional[int] = None,
    rare_pipe_guarantee_window: int = 6,
    initial_time_seconds: float = 90.0,
    lives_total: int = 5,
    bonus_time_coin_seconds: float = 5.0,
    bonus_time_kill_seconds: float = 8.0,
    bonus_time_blue_pipe_seconds: float = 20,
    ranking_path: Optional[str] = None,
) -> None:
    try:
        import pygame
    except ImportError:
        print("pygame is required.  Install with: pip install pygame")
        sys.exit(1)

    from env import MarioEnv
    from metrics import MetricsTracker
    from constants import TILE_SIZE

    ranking_file = ranking_path or _default_time_trial_ranking_path()
    ranking_data = _load_time_trial_ranking(ranking_file)
    ranking_entries = list(ranking_data.get("entries", []))

    initial_time_seconds = max(1.0, float(initial_time_seconds))
    lives_total = max(1, int(lives_total))
    bonus_time_coin_seconds = max(0.0, float(bonus_time_coin_seconds))
    bonus_time_kill_seconds = max(0.0, float(bonus_time_kill_seconds))
    bonus_time_blue_pipe_seconds = max(0.0, float(bonus_time_blue_pipe_seconds))
    current_player_name = str(agent_name or agent_hash or "human_player").strip() or "human_player"

    run_id = f"human-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    registry = _get_registry()
    registry.record_run_start(
        agent_name=current_player_name,
        mode="human_play",
        run_id=run_id,
        case_id=case_id,
        metadata={"infinite_mode": infinite_mode, "objective": "survive"},
    )

    api_client: Optional[WorldGenerationApiClient] = None
    alpha_streams: Optional[List[InfiniteLane]] = None
    current_alpha_idx: int = 0
    rare_pipe_rng = random.Random(rare_pipe_seed)
    rare_pipe_state: Dict[str, Any] = {
        "seen": {"red": False, "blue": False},
        "miss": {"red": 0, "blue": 0},
    }

    if infinite_mode:
        normalized_alphas = (
            _full_interval_alpha_schedule(pipe_alpha_step)
            if alpha_full_interval
            else _normalize_alpha_schedule(alpha_schedule)
        )
        if not api_base_url:
            api_base_url = "http://127.0.0.1:8000"
        if not pool_request_payload:
            pool_request_payload = {
                "example_code": ['1-1','1-2','1-3','2-1','3-1','3-3','4-1','4-2','5-1','5-3','6-1','6-2','6-3','7-1','8-1'],
                "Level_Len": 40,
                "module": 40,
                "batch_size": 1,
                "epochs": 10,
                "use_grammar_degree_grid": True,
                "grammar_min_degree": 2,
                "grammar_end_out_degree": 2,
                "grammar_bulk_repeats": 1,
                "grammar_max_attempts": 256,
            }
            pool_request_payload.setdefault("run_id", run_id)
        api_client = WorldGenerationApiClient(api_base_url)
        print("Waiting for world generation API...")
        try:
            api_client.wait_until_ready()
            print("API ready")
        except RuntimeError as e:
            print(f"Failed to reach API: {e}")
            sys.exit(1)

        try:
            if instance_id:
                alpha_streams = _build_streams_for_provided_instance(
                    api_client=api_client,
                    instance_id=instance_id,
                    agent_name=current_player_name,
                    request_payload=pool_request_payload,
                    alpha_schedule=normalized_alphas,
                    prefetch_levels=prefetch_levels,
                )
            else:
                alpha_streams = _build_alpha_streams(
                    api_client=api_client,
                    agent_name=current_player_name,
                    base_request_payload=pool_request_payload,
                    alpha_schedule=normalized_alphas,
                    prefetch_levels=prefetch_levels,
                    alpha_window_size=alpha_window_size,
                )
            current_alpha_idx = 0
            _prime_lane_state(
                alpha_streams[current_alpha_idx],
                rare_pipe_replace_rate=rare_pipe_replace_rate,
                rare_pipe_blue_share=rare_pipe_blue_share,
                rare_pipe_rng=rare_pipe_rng,
                rare_pipe_state=rare_pipe_state,
                rare_pipe_guarantee_window=rare_pipe_guarantee_window,
            )
            current_map = alpha_streams[current_alpha_idx].game_state.tilemap_source.clone()
        except RuntimeError as e:
            print(f"Failed to fetch initial level: {e}")
            sys.exit(1)
    else:
        if map_path is None:
            raise ValueError("map_path is required when not in infinite mode")
        current_map = map_path

    env = MarioEnv(map_path=current_map, obs_mode="rgb", render_mode="human", fps=fps)
    if infinite_mode and alpha_streams:
        _load_lane_into_env(env, alpha_streams[current_alpha_idx])
    tracker = MetricsTracker(keep_steps=True)

    timer_deadline = time.monotonic() + initial_time_seconds
    lives_used = 0
    bonus_time_total = 0.0
    coins_total = 0
    kills_total = 0
    rare_pipe_uses_total = 0
    reward_total = 0.0
    tries: List[Dict[str, Any]] = []
    run_end_reason = "completed"
    restart_requested = False

    life_started_at = time.monotonic()
    life_start_remaining = initial_time_seconds
    life_coins = 0
    life_kills = 0
    life_rare_pipe_uses = 0
    life_bonus = 0.0
    life_reward = 0.0
    life_finalized = False

    last_coins = 0
    last_kills = 0

    def append_life(reason: str, status: str) -> None:
        nonlocal life_finalized
        if life_finalized:
            return
        life_finalized = True
        remaining = max(0.0, timer_deadline - time.monotonic())
        survived = max(0.0, life_start_remaining - remaining)
        tries.append(
            {
                "try": len(tries) + 1,
                "status": status,
                "reason": reason,
                "survived_seconds": float(survived),
                "coins": int(life_coins),
                "kills": int(life_kills),
                "rare_pipe_uses": int(life_rare_pipe_uses),
                "bonus_time_seconds": float(life_bonus),
                "reward": float(life_reward),
                "remaining_seconds_after_try": float(remaining),
            }
        )

    def make_entry() -> Dict[str, Any]:
        remaining = max(0.0, timer_deadline - time.monotonic())
        survived = max(0.0, initial_time_seconds + bonus_time_total - remaining)
        return {
            "player": current_player_name,
            "timestamp": _iso_now(),
            "objective": "survive",
            "survived_seconds": float(survived),
            "initial_time_seconds": float(initial_time_seconds),
            "bonus_time_seconds": float(bonus_time_total),
            "remaining_seconds": float(remaining),
            "lives_total": int(lives_total),
            "lives_used": int(lives_used),
            "lives_left": int(max(0, lives_total - lives_used)),
            "coins": int(coins_total),
            "kills": int(kills_total),
            "rare_pipe_uses": int(rare_pipe_uses_total),
            "total_reward": float(reward_total),
            "tries": list(tries),
        }

    try:
        obs, _ = env.reset()
        tracker.new_episode()
        if infinite_mode and alpha_streams:
            _schedule_alpha_refills(
                alpha_streams=alpha_streams,
                current_idx=current_alpha_idx,
                prefetch_levels=prefetch_levels,
            )

        initial_alpha = 0.0
        if alpha_streams:
            try:
                initial_alpha = float(alpha_streams[current_alpha_idx].alpha)
            except Exception:
                initial_alpha = 0.0
        try:
            env.set_overlay_box_lines(
                [
                    f"TIMER  {_format_clock(initial_time_seconds)}",
                    f"ALPHAx10  {initial_alpha * 10.0:0.1f}",
                    f"LIVES  {lives_total}/{lives_total}",
                ]
            )
        except Exception:
            pass

        print("=" * 50)
        mode_info = " (Infinite Runner)" if infinite_mode else ""
        print(f"  Mario RL - Time Trial Survival{mode_info}")
        print("  Controls: <-/->/A/D move, down/S pipe, Z/Space jump, X/Shift run")
        print("  Objective: survive as long as possible before the clock reaches zero")
        print("=" * 50)

        ordered_entry_rows = _sorted_ranking_entries(ranking_entries)
        ranked_lines = [
            f"#{idx} {row.get('player', 'unknown')} {float(row.get('survived_seconds', 0.0) or 0.0):.1f}s"
            for idx, row in enumerate(ordered_entry_rows[:5], start=1)
        ] or ["No runs logged yet"]
        start_hold = _intermission_wait(
            env=env,
            title="Ready to Run",
            summary_lines=[
                f"Objective: survive with {lives_total} lives and a shared timer",
                f"Blue pipes add +{bonus_time_blue_pipe_seconds:.0f}s when available",
                "Unplayed nearby preview worlds are recycled on retries when possible",
            ],
            ranking_lines=ranked_lines,
            timer_deadline=time.monotonic() + 3600.0,
            fps=fps,
            hint_text="ENTER/SPACE: start   R: restart run   Q/ESC: quit",
            editable_label="PLAYER NAME",
            initial_text=current_player_name,
            primary_label="PLAY",
            secondary_label="RESTART",
        )
        current_player_name = str(start_hold.text or current_player_name).strip() or "human_player"
        _rename_lane_agent_names(alpha_streams, current_player_name)
        if start_hold.action == "restart":
            restart_requested = True
            run_end_reason = "restart"
            running = False
        elif start_hold.action == "quit":
            run_end_reason = "quit"
            running = False
        else:
            running = True

        episode = 1
        total_steps = 0
        levels_completed = 0
        tiles_appended = 0
        pipe_cooldown = 0

        while running:
            now = time.monotonic()
            remaining = max(0.0, timer_deadline - now)
            if remaining <= 0.0:
                run_end_reason = "timeout"
                append_life(reason="timeout", status="TIMEOUT")
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run_end_reason = "quit"
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    run_end_reason = "quit"
                    running = False

            if not running:
                break

            keys = pygame.key.get_pressed()
            actions = {
                "right": keys[pygame.K_RIGHT] or keys[pygame.K_d],
                "left": keys[pygame.K_LEFT] or keys[pygame.K_a],
                "down": keys[pygame.K_DOWN] or keys[pygame.K_s],
                "jump": keys[pygame.K_z] or keys[pygame.K_SPACE],
                "run": keys[pygame.K_x] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT],
            }

            action_idx = _actions_to_idx(actions)
            obs, reward, done, trunc, info = env.step_with_actions(actions)
            tracker.record(action_idx, reward, info)
            reward_total += float(reward)
            life_reward += float(reward)
            total_steps += 1

            current_coins = int(info.get("coins", last_coins))
            current_kills = int(info.get("kills", last_kills))
            gained_coins = max(0, current_coins - last_coins)
            gained_kills = max(0, current_kills - last_kills)

            if gained_coins > 0:
                add = gained_coins * bonus_time_coin_seconds
                timer_deadline += add
                bonus_time_total += add
                life_bonus += add
                coins_total += gained_coins
                life_coins += gained_coins
            if gained_kills > 0:
                add = gained_kills * bonus_time_kill_seconds
                timer_deadline += add
                bonus_time_total += add
                life_bonus += add
                kills_total += gained_kills
                life_kills += gained_kills

            last_coins = current_coins
            last_kills = current_kills

            if pipe_cooldown > 0:
                pipe_cooldown -= 1

            if pipe_cooldown == 0 and info.get("pipe_entry") in ("red", "blue"):
                try:
                    pipe_color = str(info.get("pipe_entry", ""))
                    transported, current_alpha_idx = _transport_via_pipe(
                        env,
                        alpha_streams,
                        current_alpha_idx,
                        pipe_color,
                        rare_pipe_replace_rate=rare_pipe_replace_rate,
                        rare_pipe_blue_share=rare_pipe_blue_share,
                        rare_pipe_rng=rare_pipe_rng,
                        rare_pipe_state=rare_pipe_state,
                        rare_pipe_guarantee_window=rare_pipe_guarantee_window,
                    )
                    if transported:
                        pipe_cooldown = max(18, fps // 3)
                        _prime_preview_lanes(
                            alpha_streams=alpha_streams,
                            current_idx=current_alpha_idx,
                            rare_pipe_replace_rate=rare_pipe_replace_rate,
                            rare_pipe_blue_share=rare_pipe_blue_share,
                            rare_pipe_rng=rare_pipe_rng,
                            rare_pipe_state=rare_pipe_state,
                            rare_pipe_guarantee_window=rare_pipe_guarantee_window,
                        )
                        if pipe_color == "blue":
                            timer_deadline += bonus_time_blue_pipe_seconds
                            bonus_time_total += bonus_time_blue_pipe_seconds
                            life_bonus += bonus_time_blue_pipe_seconds
                        rare_pipe_uses_total += 1
                        life_rare_pipe_uses += 1
                        obs = env.render() if env.render_mode == "rgb_array" else obs
                except RuntimeError as e:
                    print(f"  Pipe transport failed: {e}")

            if infinite_mode:
                if alpha_streams:
                    _schedule_alpha_refills(
                        alpha_streams=alpha_streams,
                        current_idx=current_alpha_idx,
                        prefetch_levels=prefetch_levels,
                    )
                mario_col = int(env._gs.mario.cx // TILE_SIZE)
                remaining_tiles = env._gs.tilemap.width - mario_col
                if remaining_tiles <= max(4, append_trigger_tiles):
                    try:
                        if not alpha_streams:
                            raise RuntimeError("alpha streams not initialized")
                        current_lane = alpha_streams[current_alpha_idx]
                        next_chunk = _pull_next_chunk(
                            current_lane.stream,
                            rare_pipe_replace_rate=rare_pipe_replace_rate,
                            rare_pipe_blue_share=rare_pipe_blue_share,
                            rare_pipe_rng=rare_pipe_rng,
                            refill_target=prefetch_levels,
                            rare_pipe_state=rare_pipe_state,
                            rare_pipe_guarantee_window=rare_pipe_guarantee_window,
                        )
                        added = _append_level_to_env(env, next_chunk)
                        current_lane.chunks_appended += 1
                        current_lane.tiles_appended += added
                        _save_env_to_lane(env, current_lane)
                        tiles_appended += added
                        levels_completed += 1
                    except RuntimeError as e:
                        print(f"  Failed to append next chunk: {e}")

            remaining = max(0.0, timer_deadline - time.monotonic())
            env.set_hud_extra_line("")

            current_alpha = 0.0
            if alpha_streams:
                try:
                    current_alpha = float(alpha_streams[current_alpha_idx].alpha)
                except Exception:
                    current_alpha = 0.0
            overlay_lines = [
                f"TIMER  {_format_clock(remaining)}",
                f"ALPHAx10  {current_alpha * 10.0:0.1f}",
                f"LIVES  {max(0, lives_total - lives_used)}/{lives_total}",
            ]
            try:
                env.set_overlay_box_lines(overlay_lines)
            except Exception:
                pass

            if done or trunc:
                ep_rec = tracker.end_episode(info)
                status = "WON" if ep_rec.won else ("TRUNCATED" if trunc else "DIED")
                append_life(reason=status.lower(), status=status)
                lives_used += 1
                lives_left = max(0, lives_total - lives_used)

                remaining = max(0.0, timer_deadline - time.monotonic())
                print(
                    f"[Life {lives_used}] {status} | time={_format_clock(remaining)} "
                    f"| reward={life_reward:.1f} | bonus=+{bonus_time_total:.1f}s | lives_left={lives_left}"
                )

                if remaining <= 0.0:
                    run_end_reason = "timeout"
                    break
                if lives_left <= 0:
                    run_end_reason = "lives_exhausted"
                    break

                hold_result = _intermission_wait(
                    env=env,
                    title="Retry Ready",
                    summary_lines=[
                        f"Next life: {lives_used + 1}/{lives_total}",
                        f"Time left: {_format_clock(remaining)}",
                        f"Last life reward: {life_reward:.1f}",
                    ],
                    ranking_lines=[
                        f"#{idx} {row.get('player', 'unknown')} {float(row.get('survived_seconds', 0.0) or 0.0):.1f}s"
                        for idx, row in enumerate(_sorted_ranking_entries(ranking_entries)[:5], start=1)
                    ] or ["No runs logged yet"],
                    timer_deadline=time.monotonic() + 3600.0,
                    fps=fps,
                    hint_text="ENTER/SPACE: next try   R: restart run   Q/ESC: quit",
                    primary_label="NEXT TRY",
                    secondary_label="RESTART",
                )
                if hold_result.action == "restart":
                    restart_requested = True
                    run_end_reason = "restart"
                    break
                if hold_result.action == "quit":
                    run_end_reason = "quit"
                    break

                restarted_from_preview = False
                if infinite_mode:
                    restarted_from_preview, current_alpha_idx = _load_recycled_retry_preview(
                        env=env,
                        alpha_streams=alpha_streams,
                        current_alpha_idx=current_alpha_idx,
                    )
                    if restarted_from_preview:
                        obs = env._make_obs()
                        _prime_preview_lanes(
                            alpha_streams=alpha_streams,
                            current_idx=current_alpha_idx,
                            rare_pipe_replace_rate=rare_pipe_replace_rate,
                            rare_pipe_blue_share=rare_pipe_blue_share,
                            rare_pipe_rng=rare_pipe_rng,
                            rare_pipe_state=rare_pipe_state,
                            rare_pipe_guarantee_window=rare_pipe_guarantee_window,
                        )
                if not restarted_from_preview:
                    obs, _ = env.reset()
                tracker.new_episode()
                episode += 1
                pipe_cooldown = 0

                life_started_at = time.monotonic()
                life_start_remaining = max(0.0, timer_deadline - life_started_at)
                life_coins = 0
                life_kills = 0
                life_rare_pipe_uses = 0
                life_bonus = 0.0
                life_reward = 0.0
                life_finalized = False
                last_coins = 0
                last_kills = 0

        current_episode = getattr(tracker, "_current", None)
        if current_episode is not None and int(getattr(current_episode, "steps", 0)) > 0:
            tracker.end_episode({})

        if not life_finalized and (run_end_reason in ("quit", "timeout") or total_steps > 0):
            append_life(reason=run_end_reason, status=run_end_reason.upper())

    finally:
        if run_end_reason == "lives_exhausted":
            try:
                candidate = make_entry()
                rank, total, ordered = _build_rank_preview(ranking_entries, candidate)
                summary_lines = [
                    f"Try complete: used {lives_used}/{lives_total} lives",
                    f"Survived: {candidate['survived_seconds']:.1f}s",
                    f"Bonus time: +{bonus_time_total:.1f}s",
                    f"Coins/Kills/Pipes: {coins_total}/{kills_total}/{rare_pipe_uses_total}",
                    f"Projected rank: #{rank}/{total}",
                ]
                ranking_lines = []
                for idx, row in enumerate(ordered[:5], start=1):
                    ranking_lines.append(
                        f"#{idx} {row.get('player', 'unknown')} survive={float(row.get('survived_seconds', 0.0)):.1f}s"
                    )
                _intermission_wait(
                    env=env,
                    title="Try Complete",
                    summary_lines=summary_lines,
                    ranking_lines=ranking_lines,
                    timer_deadline=time.monotonic() + 3600.0,
                    fps=fps,
                    hint_text="ENTER/SPACE: finish   R: restart run   Q/ESC: quit",
                    primary_label="CONTINUE",
                    secondary_label="RESTART",
                )
            except Exception:
                pass

        env.close()
        if alpha_streams:
            for alpha_stream in alpha_streams:
                alpha_stream.stream.close()

    if restart_requested:
        registry.record_run_complete(
            agent_name=current_player_name,
            mode="human_play",
            run_id=run_id,
            case_id=case_id,
            metrics={"outcome": "restarted"},
        )
        return run_survival_time_trial(
            map_path=map_path,
            fps=fps,
            infinite_mode=infinite_mode,
            api_base_url=api_base_url,
            instance_id=instance_id,
            agent_name=current_player_name,
            case_id=case_id,
            pool_request_payload=pool_request_payload,
            alpha_schedule=alpha_schedule,
            alpha_full_interval=alpha_full_interval,
            pipe_alpha_step=pipe_alpha_step,
            alpha_window_size=alpha_window_size,
            prefetch_levels=prefetch_levels,
            append_trigger_tiles=append_trigger_tiles,
            rare_pipe_replace_rate=rare_pipe_replace_rate,
            rare_pipe_blue_share=rare_pipe_blue_share,
            rare_pipe_seed=rare_pipe_seed,
            rare_pipe_guarantee_window=rare_pipe_guarantee_window,
            initial_time_seconds=initial_time_seconds,
            lives_total=lives_total,
            bonus_time_coin_seconds=bonus_time_coin_seconds,
            bonus_time_kill_seconds=bonus_time_kill_seconds,
            bonus_time_blue_pipe_seconds=bonus_time_blue_pipe_seconds,
            ranking_path=ranking_path,
        )

    final_entry = make_entry()
    final_entry["tries"] = list(tries)
    previous_rank = _rank_position(ranking_entries, current_player_name)
    ranking_entries.append(final_entry)
    ordered = _sorted_ranking_entries(ranking_entries)
    rank = 1
    for idx, row in enumerate(ordered, start=1):
        if row is final_entry:
            rank = idx
            break

    _save_time_trial_ranking(ranking_file, ordered)
    tracker.save_csv("metrics_human.csv")
    tracker.save_replay("replay_human_last.json")
    registry.register(
        artifact_kind="metrics_csv",
        artifact_path="metrics_human.csv",
        case_id=case_id,
        run_id=run_id,
        agent_name=current_player_name,
        mode="human_play",
    )
    registry.register(
        artifact_kind="replay",
        artifact_path="replay_human_last.json",
        case_id=case_id,
        run_id=run_id,
        agent_name=current_player_name,
        mode="human_play",
    )
    registry.register(
        artifact_kind="ranking",
        artifact_path=ranking_file,
        case_id=case_id,
        run_id=run_id,
        agent_name=current_player_name,
        mode="human_play",
        metrics={"rank": rank, "total": len(ordered)},
    )
    registry.record_run_complete(
        agent_name=current_player_name,
        mode="human_play",
        run_id=run_id,
        case_id=case_id,
        metrics={
            "survived_seconds": final_entry["survived_seconds"],
            "bonus_time_seconds": final_entry["bonus_time_seconds"],
            "remaining_seconds": final_entry["remaining_seconds"],
            "rank": rank,
            "total": len(ordered),
        },
    )

    rank_change_message = _format_rank_change_message(previous_rank, rank, len(ordered))
    final_summary_lines = [
        f"Run ended: {run_end_reason}",
        f"Survived: {final_entry['survived_seconds']:.1f}s",
        f"Bonus time: +{final_entry['bonus_time_seconds']:.1f}s",
        f"Remaining: {final_entry['remaining_seconds']:.1f}s",
        f"Current rank: #{rank}/{len(ordered)}",
    ]
    if previous_rank not in (None, 0) and previous_rank != rank:
        final_summary_lines.append(rank_change_message)

    ranking_preview = [
        f"#{idx} {row.get('player', 'unknown')} {float(row.get('survived_seconds', 0.0) or 0.0):.1f}s"
        for idx, row in enumerate(ordered[:5], start=1)
    ]
    final_hold = _intermission_wait(
        env=env,
        title="Game Over",
        summary_lines=final_summary_lines,
        ranking_lines=ranking_preview,
        timer_deadline=time.monotonic() + 3600.0,
        fps=fps,
        hint_text="ENTER/SPACE: play again   R: restart run   Q/ESC: quit",
        ranking_title="Ranking",
        primary_label="PLAY AGAIN",
        secondary_label="RESTART",
    )

    if final_hold.action in ("continue", "timeout"):
        return run_survival_time_trial(
            map_path=map_path,
            fps=fps,
            infinite_mode=infinite_mode,
            api_base_url=api_base_url,
            instance_id=instance_id,
            agent_name=current_player_name,
            case_id=case_id,
            pool_request_payload=pool_request_payload,
            alpha_schedule=alpha_schedule,
            alpha_full_interval=alpha_full_interval,
            pipe_alpha_step=pipe_alpha_step,
            alpha_window_size=alpha_window_size,
            prefetch_levels=prefetch_levels,
            append_trigger_tiles=append_trigger_tiles,
            rare_pipe_replace_rate=rare_pipe_replace_rate,
            rare_pipe_blue_share=rare_pipe_blue_share,
            rare_pipe_seed=rare_pipe_seed,
            rare_pipe_guarantee_window=rare_pipe_guarantee_window,
            initial_time_seconds=initial_time_seconds,
            lives_total=lives_total,
            bonus_time_coin_seconds=bonus_time_coin_seconds,
            bonus_time_kill_seconds=bonus_time_kill_seconds,
            bonus_time_blue_pipe_seconds=bonus_time_blue_pipe_seconds,
            ranking_path=ranking_path,
        )
    if final_hold.action == "restart":
        return run_survival_time_trial(
            map_path=map_path,
            fps=fps,
            infinite_mode=infinite_mode,
            api_base_url=api_base_url,
            instance_id=instance_id,
            agent_name=current_player_name,
            case_id=case_id,
            pool_request_payload=pool_request_payload,
            alpha_schedule=alpha_schedule,
            alpha_full_interval=alpha_full_interval,
            pipe_alpha_step=pipe_alpha_step,
            alpha_window_size=alpha_window_size,
            prefetch_levels=prefetch_levels,
            append_trigger_tiles=append_trigger_tiles,
            rare_pipe_replace_rate=rare_pipe_replace_rate,
            rare_pipe_blue_share=rare_pipe_blue_share,
            rare_pipe_seed=rare_pipe_seed,
            rare_pipe_guarantee_window=rare_pipe_guarantee_window,
            initial_time_seconds=initial_time_seconds,
            lives_total=lives_total,
            bonus_time_coin_seconds=bonus_time_coin_seconds,
            bonus_time_kill_seconds=bonus_time_kill_seconds,
            bonus_time_blue_pipe_seconds=bonus_time_blue_pipe_seconds,
            ranking_path=ranking_path,
        )

    print(f"Run ended: {run_end_reason}")
    print(
        f"Time Trial result: survived={final_entry['survived_seconds']:.1f}s "
        f"bonus=+{final_entry['bonus_time_seconds']:.1f}s "
        f"remaining={final_entry['remaining_seconds']:.1f}s "
        f"rank=#{rank}/{len(ordered)}"
    )
    print(f"Ranking file updated: {ranking_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-playable Mario RL")
    parser.add_argument("map", nargs="?", help="Path to the .txt tile map (required for standard mode)")
    parser.add_argument("--fps", type=int, default=FPS, help="FPS (default: %(default)s)")
    parser.add_argument(
        "--objective",
        default="survive",
        choices=["survive", "freeplay"],
        help="Gameplay objective. 'survive' runs countdown time-trial mode (default: %(default)s)",
    )
    parser.add_argument(
        "--time-seconds",
        type=float,
        default=60.0,
        help="Initial time budget in seconds for survival mode (default: %(default)s)",
    )
    parser.add_argument(
        "--lives",
        type=int,
        default=5,
        help="Lives available in survival mode (clock keeps running) (default: %(default)s)",
    )
    parser.add_argument(
        "--bonus-coin-seconds",
        type=float,
        default=3.0,
        help="Time bonus per coin in survival mode (default: %(default)s)",
    )
    parser.add_argument(
        "--bonus-kill-seconds",
        type=float,
        default=5.0,
        help="Time bonus per enemy kill in survival mode (default: %(default)s)",
    )
    parser.add_argument(
        "--bonus-rare-pipe-seconds",
        type=float,
        default=20,
        help="Time bonus when entering a blue pipe in survival mode (default: %(default)s)",
    )
    parser.add_argument(
        "--ranking-path",
        default=_default_time_trial_ranking_path(),
        help="Path to survival ranking JSON file (default: history/human_time_trial_ranking.json)",
    )
    parser.add_argument("--infinite", action="store_true",
                        help="Enable infinite runner mode (creates new instance if --instance-id not provided)")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000",
                        help="World generation API base URL (default: %(default)s)")
    parser.add_argument("--instance-id", help="Map elite instance ID (optional; new instance created if not provided)")
    parser.add_argument("--agent-name", default="human_player",
                        help="Agent name identifier for pool manager (default: %(default)s)")
    parser.add_argument("--agent-hash", default=None,
                        help="Deprecated alias for --agent-name")
    parser.add_argument(
        "--case-id",
        default=None,
        help="Optional identifier grouping this run with others for comparison/reporting",
    )
    parser.add_argument(
        "--alphas",
        default="0.0, 0.1",
        help="Comma-separated alpha schedule (used only when --alpha-full-interval is disabled)",
    )
    parser.add_argument(
        "--alpha-full-interval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use full [0,1] alpha schedule with constant pipe step (default: %(default)s)",
    )
    parser.add_argument(
        "--pipe-alpha-step",
        type=float,
        default=0.1,
        help="Constant alpha step applied per rare pipe transition across [0,1] (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha-window-size",
        type=float,
        default=0.2,
        help="Dynamic alpha window length used for each instance around schedule alpha (default: %(default)s)",
    )
    parser.add_argument("--level-len", type=int, default=40,
                        help="Level length for pool manager requests (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for pool manager requests (default: %(default)s)")
    parser.add_argument(
        "--grammar-degree-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use grammar-graph degree-constrained generation for map-elites chunks (default: %(default)s)",
    )
    parser.add_argument(
        "--grammar-min-degree",
        type=int,
        default=2,
        help="Minimum in/out degree for start (inner) grammar token candidates (default: %(default)s)",
    )
    parser.add_argument(
        "--grammar-end-out-degree",
        type=int,
        default=2,
        help="Minimum out-degree required for level ending grammar tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--grammar-bulk-repeats",
        type=int,
        default=1,
        help="How many full passes over eligible start tokens to seed archive diversity (default: %(default)s)",
    )
    parser.add_argument(
        "--grammar-max-attempts",
        type=int,
        default=64,
        help="Max constrained generation retries per candidate sample (default: %(default)s)",
    )
    parser.add_argument("--prefetch-levels", type=int, default=3,
                        help="How many chunks to prefetch and keep buffered (default: %(default)s)")
    parser.add_argument("--append-trigger-tiles", type=int, default=16,
                        help="Append next chunk when <= N tiles remain ahead (default: %(default)s)")
    parser.add_argument(
        "--rare-pipe-rate",
        type=float,
        default=0.18,
        help="Chance to convert each generated normal pipe into red/blue rare pipe (default: %(default)s)",
    )
    parser.add_argument(
        "--rare-pipe-blue-share",
        type=float,
        default=0.5,
        help="Fraction of converted pipes that become blue (rest become red) (default: %(default)s)",
    )
    parser.add_argument(
        "--rare-pipe-seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic rare-pipe placement",
    )
    parser.add_argument(
        "--rare-pipe-guarantee-window",
        type=int,
        default=6,
        help="Force missing red/blue rare pipe if absent for this many pulled chunks (default: %(default)s)",
    )
    args = parser.parse_args()

    is_survive_mode = str(args.objective).strip().lower() == "survive"

    if args.infinite:
        alphas = None
        if not args.alpha_full_interval:
            alphas = [float(value) for value in str(args.alphas).split(",") if value.strip()]
        pool_request_payload = {
            "example_code": ['1-1','1-2','1-3','2-1','3-1','3-3','4-1','4-2','5-1','5-3','6-1','6-2','6-3','7-1','8-1'],
            "Level_Len": args.level_len,
            "module": args.level_len,
            "batch_size": args.batch_size,
            "epochs": 10,
            "use_grammar_degree_grid": bool(args.grammar_degree_grid),
            "grammar_min_degree": max(1, int(args.grammar_min_degree)),
            "grammar_end_out_degree": max(1, int(args.grammar_end_out_degree)),
            "grammar_bulk_repeats": max(1, int(args.grammar_bulk_repeats)),
            "grammar_max_attempts": max(16, int(args.grammar_max_attempts)),

        }
        if is_survive_mode:
            run_survival_time_trial(
                infinite_mode=True,
                api_base_url=args.api_url,
                instance_id=args.instance_id,
                agent_name=(args.agent_name or args.agent_hash),
                case_id=args.case_id,
                pool_request_payload=pool_request_payload,
                alpha_schedule=alphas,
                alpha_full_interval=bool(args.alpha_full_interval),
                pipe_alpha_step=max(0.01, min(1.0, args.pipe_alpha_step)),
                alpha_window_size=max(0.0, min(1.0, args.alpha_window_size)),
                prefetch_levels=args.prefetch_levels,
                append_trigger_tiles=args.append_trigger_tiles,
                rare_pipe_replace_rate=max(0.0, min(1.0, args.rare_pipe_rate)),
                rare_pipe_blue_share=max(0.0, min(1.0, args.rare_pipe_blue_share)),
                rare_pipe_seed=args.rare_pipe_seed,
                rare_pipe_guarantee_window=max(1, args.rare_pipe_guarantee_window),
                initial_time_seconds=args.time_seconds,
                lives_total=args.lives,
                bonus_time_coin_seconds=args.bonus_coin_seconds,
                bonus_time_kill_seconds=args.bonus_kill_seconds,
                bonus_time_blue_pipe_seconds=args.bonus_rare_pipe_seconds,
                ranking_path=args.ranking_path,
                fps=args.fps,
            )
        else:
            run(
                infinite_mode=True,
                api_base_url=args.api_url,
                instance_id=args.instance_id,
                agent_name=(args.agent_name or args.agent_hash),
                case_id=args.case_id,
                pool_request_payload=pool_request_payload,
                alpha_schedule=alphas,
                alpha_full_interval=bool(args.alpha_full_interval),
                pipe_alpha_step=max(0.01, min(1.0, args.pipe_alpha_step)),
                alpha_window_size=max(0.0, min(1.0, args.alpha_window_size)),
                prefetch_levels=args.prefetch_levels,
                append_trigger_tiles=args.append_trigger_tiles,
                rare_pipe_replace_rate=max(0.0, min(1.0, args.rare_pipe_rate)),
                rare_pipe_blue_share=max(0.0, min(1.0, args.rare_pipe_blue_share)),
                rare_pipe_seed=args.rare_pipe_seed,
                rare_pipe_guarantee_window=max(1, args.rare_pipe_guarantee_window),
                fps=args.fps,
            )
    else:
        if not args.map:
            print("Error: map argument is required for standard mode")
            sys.exit(1)
        if is_survive_mode:
            run_survival_time_trial(
                map_path=args.map,
                infinite_mode=False,
                api_base_url=args.api_url,
                instance_id=args.instance_id,
                agent_name=(args.agent_name or args.agent_hash),
                case_id=args.case_id,
                alpha_schedule=None,
                initial_time_seconds=args.time_seconds,
                lives_total=args.lives,
                bonus_time_coin_seconds=args.bonus_coin_seconds,
                bonus_time_kill_seconds=args.bonus_kill_seconds,
                bonus_time_blue_pipe_seconds=args.bonus_rare_pipe_seconds,
                ranking_path=args.ranking_path,
                fps=args.fps,
            )
        else:
            run(map_path=args.map, case_id=args.case_id, fps=args.fps)


def _actions_to_idx(actions: dict) -> int:
    """Map a dict of pressed keys to a discrete action index."""
    r = actions.get('right')
    l = actions.get('left')
    j = actions.get('jump')
    run = actions.get('run')
    if r and run and j:  return 7
    if r and run:        return 6
    if r and j:          return 4
    if l and j:          return 5
    if r:                return 1
    if l:                return 2
    if j:                return 3
    return 0


if __name__ == "__main__":
    main()
