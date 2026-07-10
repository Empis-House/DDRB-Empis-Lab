"""
run_human.py – Play the Mario level as a human via keyboard.

Controls
--------
  Arrow RIGHT / D  →  walk right
  Arrow LEFT  / A  →  walk left
  Z / Space        →  jump
  X / Shift        →  run (hold with direction)
  R                →  restart episode
  Q / Esc          →  quit

Usage
-----
  python run_human.py path/to/map.txt
  python run_human.py --infinite --api-url http://localhost:8000 --instance-id 1 --agent-hash player1
"""

from __future__ import annotations

import argparse
import sys
import os
import json
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Optional, Union
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

    def create_instance(
        self,
        function: str = "Performance",
        alphas: Optional[list] = None,
        variety_domains: Optional[list] = None,
        grid_points: Optional[list] = None,
    ) -> str:
        if alphas is None:
            alphas = [0.0, 0.1, 1.0]
        if variety_domains is None:
            variety_domains = ["overworld", "underground", "athletic"]
        if grid_points is None:
            grid_points = [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]

        response = self.post(
            "/instances",
            {
                "function": function,
                "alphas": alphas,
                "variety_domains": variety_domains,
                "grid_points": grid_points,
            },
        )
        return response["instance_id"]

    def generate_mapping(self, instance_id: str, request_payload: dict) -> None:
        payload = {
            "example_code": request_payload.get("example_code", ["1-1"]),
            "Level_Len": request_payload.get("Level_Len", 40),
            "module": request_payload.get("module", request_payload.get("Level_Len", 40)),
            "batch_size": request_payload.get("batch_size", 1),
            "epochs": request_payload.get("epochs", 10),
        }
        self.post(f"/instances/{instance_id}/generate_mapping", payload)

    def fetch_next_level(
        self,
        instance_id: str,
        agent_hash: str,
        request_payload: dict,
    ) -> "pd.DataFrame":
        try:
            from world_generation.level_formats import rows_to_tilemap
        except ImportError:
            import pandas as pd

            def rows_to_tilemap(rows):
                return pd.DataFrame(rows)

        response = self.post(
            f"/instances/{instance_id}/pool_manager",
            {
                "agent_hash": agent_hash,
                "request": request_payload,
            },
        )
        return rows_to_tilemap(response["tilemap"])


class LevelStreamBuffer:
    """Client-side queue of generated levels, refilled from pool_manager."""

    def __init__(
        self,
        api_client: WorldGenerationApiClient,
        instance_id: str,
        agent_hash: str,
        request_payload: dict,
        prefetch_levels: int,
    ):
        self.api_client = api_client
        self.instance_id = instance_id
        self.agent_hash = agent_hash
        self.request_payload = request_payload
        self.prefetch_levels = max(1, int(prefetch_levels))
        self._queue = deque()

    def refill(self, min_size: int = 1) -> None:
        target = max(int(min_size), self.prefetch_levels)
        while len(self._queue) < target:
            self._queue.append(
                self.api_client.fetch_next_level(
                    self.instance_id,
                    self.agent_hash,
                    self.request_payload,
                )
            )

    def pop_next(self) -> "pd.DataFrame":
        if not self._queue:
            self.refill(min_size=1)
        return self._queue.popleft()

    def size(self) -> int:
        return len(self._queue)


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


def run(
    map_path: Optional[Union[str, "pd.DataFrame"]] = None,
    fps: int = FPS,
    infinite_mode: bool = False,
    api_base_url: Optional[str] = None,
    instance_id: Optional[str] = None,
    agent_hash: Optional[str] = None,
    pool_request_payload: Optional[dict] = None,
    prefetch_levels: int = 3,
    append_trigger_tiles: int = 16,
) -> None:
    try:
        import pygame
    except ImportError:
        print("pygame is required.  Install with: pip install pygame")
        sys.exit(1)

    from env import MarioEnv
    from metrics import MetricsTracker
    from constants import TILE_SIZE

    api_client: Optional[WorldGenerationApiClient] = None
    level_stream: Optional[LevelStreamBuffer] = None

    # Validate infinite mode requirements
    if infinite_mode:
        if not api_base_url:
            api_base_url = "http://127.0.0.1:8000"
        if not agent_hash:
            agent_hash = "human_player"
        if not pool_request_payload:
            pool_request_payload = {
                "example_code": ["1-1","1-2","4-2","5-1"],
                "Level_Len": 40,
                "module": 40,
                "batch_size": 1,
                "epochs": 10,
            }
        api_client = WorldGenerationApiClient(api_base_url)

        print("Waiting for world generation API...")
        try:
            api_client.wait_until_ready()
            print("✓ API ready")
        except RuntimeError as e:
            print(f"✗ Failed to reach API: {e}")
            sys.exit(1)

        # Create instance if not provided
        if not instance_id:
            print("Creating new Map Elite instance...")
            try:
                instance_id = api_client.create_instance()
                print(f"✓ Instance created: {instance_id}")

                # Generate mapping for the instance
                print("Generating level mappings (this may take a moment)...")
                api_client.generate_mapping(instance_id, pool_request_payload)
                print(f"✓ Mappings generated")
            except RuntimeError as e:
                print(f"✗ Failed to create/initialize instance: {e}")
                sys.exit(1)

        # Fetch initial level and prefetch the stream queue.
        try:
            level_stream = LevelStreamBuffer(
                api_client=api_client,
                instance_id=instance_id,
                agent_hash=agent_hash,
                request_payload=pool_request_payload,
                prefetch_levels=prefetch_levels,
            )
            print("Fetching initial stream batch...")
            level_stream.refill(min_size=prefetch_levels)
            current_map = level_stream.pop_next()
            print(f"✓ Level loaded from instance {instance_id}")
        except RuntimeError as e:
            print(f"✗ Failed to fetch level: {e}")
            sys.exit(1)
    else:
        if map_path is None:
            raise ValueError("map_path is required when not in infinite mode")
        current_map = map_path

    env = MarioEnv(map_path=current_map, obs_mode="rgb",
                   render_mode="human", fps=fps)
    tracker = MetricsTracker(keep_steps=True)

    obs, _ = env.reset()
    tracker.new_episode()
    episode = 0
    total_steps = 0
    levels_completed = 0
    tiles_appended = 0

    print("=" * 50)
    mode_info = " (Infinite Runner)" if infinite_mode else ""
    print(f"  Mario RL – Human Mode{mode_info}")
    print("  Controls: ←/→/A/D  move   Z/Space  jump")
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
            'jump':  keys[pygame.K_z]     or keys[pygame.K_SPACE],
            'run':   keys[pygame.K_x]     or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT],
        }

        action_idx = _actions_to_idx(actions)

        obs, reward, done, trunc, info = env.step(action_idx)
        tracker.record(action_idx, reward, info)
        total_steps += 1

        if infinite_mode:
            mario_col = int(env._gs.mario.cx // TILE_SIZE)
            remaining_tiles = env._gs.tilemap.width - mario_col
            if remaining_tiles <= max(4, append_trigger_tiles):
                try:
                    if level_stream.size() <= 1:
                        print("  → Refilling level stream buffer...")
                        level_stream.refill(min_size=prefetch_levels)
                    next_chunk = level_stream.pop_next()
                    added = _append_level_to_env(env, next_chunk)
                    tiles_appended += added
                    levels_completed += 1
                    print(
                        f"  → Appended chunk #{levels_completed} "
                        f"(+{added} tiles, total_width={env._gs.tilemap.width})"
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
            else:
                obs, _ = env.reset()

            tracker.new_episode()
            episode += 1

    env.close()
    print(f"\nTotal episodes: {episode}   Total steps: {total_steps}")
    if infinite_mode:
        print(f"Appended chunks: {levels_completed}   Added tiles: {tiles_appended}")
    tracker.print_summary()
    save = input("Save replay? [y/N] ").strip().lower()
    if save == "y":
        tracker.save_csv("metrics_human.csv")
        tracker.save_replay("replay_human_last.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-playable Mario RL")
    parser.add_argument("map", nargs="?", help="Path to the .txt tile map (required for standard mode)")
    parser.add_argument("--fps", type=int, default=FPS, help="FPS (default: %(default)s)")
    parser.add_argument("--infinite", action="store_true",
                        help="Enable infinite runner mode (creates new instance if --instance-id not provided)")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000",
                        help="World generation API base URL (default: %(default)s)")
    parser.add_argument("--instance-id", help="Map elite instance ID (optional; new instance created if not provided)")
    parser.add_argument("--agent-hash", default="human_player",
                        help="Agent hash identifier for pool manager (default: %(default)s)")
    parser.add_argument("--level-len", type=int, default=40,
                        help="Level length for pool manager requests (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for pool manager requests (default: %(default)s)")
    parser.add_argument("--prefetch-levels", type=int, default=3,
                        help="How many chunks to prefetch and keep buffered (default: %(default)s)")
    parser.add_argument("--append-trigger-tiles", type=int, default=16,
                        help="Append next chunk when <= N tiles remain ahead (default: %(default)s)")
    args = parser.parse_args()

    if args.infinite:
        pool_request_payload = {
            "example_code": ["1-1"],
            "Level_Len": args.level_len,
            "module": args.level_len,
            "batch_size": args.batch_size,
            "epochs": 10,
        }
        run(
            infinite_mode=True,
            api_base_url=args.api_url,
            instance_id=args.instance_id,
            agent_hash=args.agent_hash,
            pool_request_payload=pool_request_payload,
            prefetch_levels=args.prefetch_levels,
            append_trigger_tiles=args.append_trigger_tiles,
            fps=args.fps,
        )
    else:
        if not args.map:
            print("Error: map argument is required for standard mode")
            sys.exit(1)
        run(map_path=args.map, fps=args.fps)


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
