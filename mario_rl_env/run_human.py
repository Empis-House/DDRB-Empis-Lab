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
from typing import Union, Optional, Callable, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ensure local imports work when running directly
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .constants import FPS
except ImportError:
    from constants import FPS


def _api_post(url: str, payload: dict) -> dict:
    """Make a POST request to the world generation API."""
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


def _fetch_level_from_pool_manager(
    api_base_url: str,
    instance_id: str,
    agent_hash: str,
    request_payload: dict
) -> "pandas.DataFrame":
    """Fetch a new level from the pool manager API."""
    import pandas as pd
    url = f"{api_base_url}/instances/{instance_id}/pool_manager"
    response = _api_post(url, {
        "agent_hash": agent_hash,
        "request": request_payload,
    })
    return pd.DataFrame(response["tilemap"])


def _create_map_elite_instance(
    api_base_url: str,
    function: str = "Performance",
    alphas: list = None,
    variety_domains: list = None,
    grid_points: list = None,
) -> str:
    """Create a new Map Elite instance and return its ID."""
    if alphas is None:
        alphas = [0.0, 0.1, 1.0]
    if variety_domains is None:
        variety_domains = ["calculate_slope", "colliders_score"]
    if grid_points is None:
        grid_points = [[0, 0.5], [0, 1]]
    
    url = f"{api_base_url}/instances"
    response = _api_post(url, {
        "function": function,
        "alphas": alphas,
        "variety_domains": variety_domains,
        "grid_points": grid_points,
    })
    return response["instance_id"]


def _generate_mapping(
    api_base_url: str,
    instance_id: str,
    example_code: list = None,
    level_len: int = 40,
    batch_size: int = 5,
    epochs: int = 10,
) -> None:
    """Initialize mapping generation for a Map Elite instance."""
    if example_code is None:
        example_code = ["1-1"]
    
    url = f"{api_base_url}/instances/{instance_id}/generate_mapping"
    _api_post(url, {
        "example_code": example_code,
        "Level_Len": level_len,
        "module": level_len,
        "batch_size": batch_size,
        "epochs": epochs,
    })


def run(
    map_path: Optional[Union[str, "pandas.DataFrame"]] = None,
    fps: int = FPS,
    infinite_mode: bool = False,
    api_base_url: Optional[str] = None,
    instance_id: Optional[str] = None,
    agent_hash: Optional[str] = None,
    pool_request_payload: Optional[dict] = None,
) -> None:
    try:
        import pygame
    except ImportError:
        print("pygame is required.  Install with: pip install pygame")
        sys.exit(1)

    from env import MarioEnv
    from metrics import MetricsTracker

    # Validate infinite mode requirements
    if infinite_mode:
        if not api_base_url:
            api_base_url = "http://127.0.0.1:8000"
        if not agent_hash:
            agent_hash = "human_player"
        if not pool_request_payload:
            pool_request_payload = {
                "example_code": ["1-1"],
                "Level_Len": 40,
                "module": 40,
                "batch_size": 1,
                "epochs": 10,
            }
        
        # Create instance if not provided
        if not instance_id:
            print("Creating new Map Elite instance...")
            try:
                instance_id = _create_map_elite_instance(api_base_url)
                print(f"✓ Instance created: {instance_id}")
                
                # Generate mapping for the instance
                print("Generating level mappings (this may take a moment)...")
                _generate_mapping(
                    api_base_url,
                    instance_id,
                    example_code=pool_request_payload.get("example_code", ["3-2"]),
                    level_len=pool_request_payload.get("Level_Len", 40),
                    batch_size=pool_request_payload.get("batch_size", 5),
                    epochs=pool_request_payload.get("epochs", 10),
                )
                print(f"✓ Mappings generated")
            except RuntimeError as e:
                print(f"✗ Failed to create/initialize instance: {e}")
                sys.exit(1)
        
        # Fetch initial level
        try:
            print("Fetching initial level...")
            current_map = _fetch_level_from_pool_manager(
                api_base_url, instance_id, agent_hash, pool_request_payload
            )
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

        if done or trunc:
            ep_rec = tracker.end_episode(info)
            status = "WON 🎉" if ep_rec.won else ("TRUNCATED" if trunc else "DIED 💀")
            print(f"\n[Episode {episode}] {status}  "
                  f"reward={ep_rec.total_reward:.1f}  "
                  f"steps={ep_rec.steps}  "
                  f"coins={ep_rec.coins}  "
                  f"max_x={ep_rec.max_x:.0f}")

            if infinite_mode and ep_rec.won:
                # Fetch next level from pool manager
                try:
                    print(f"  → Fetching next level from pool manager...")
                    current_map = _fetch_level_from_pool_manager(
                        api_base_url, instance_id, agent_hash, pool_request_payload
                    )
                    levels_completed += 1
                    print(f"  → Level {levels_completed} loaded")
                    env.close()
                    env = MarioEnv(map_path=current_map, obs_mode="rgb",
                                 render_mode="human", fps=fps)
                except RuntimeError as e:
                    print(f"  ⚠ Failed to fetch level: {e}")
                    print(f"  → Restarting current level")

            obs, _ = env.reset()
            tracker.new_episode()
            episode += 1

    env.close()
    print(f"\nTotal episodes: {episode}   Total steps: {total_steps}")
    if infinite_mode:
        print(f"Levels completed: {levels_completed}")
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
