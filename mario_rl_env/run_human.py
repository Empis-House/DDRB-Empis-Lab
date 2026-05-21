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
"""

from __future__ import annotations

import argparse
import sys
import os
from typing import Union

# ensure local imports work when running directly
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .constants import FPS
except ImportError:
    from constants import FPS


def run(map_path: Union[str, "pandas.DataFrame"], fps: int = FPS) -> None:
    try:
        import pygame
    except ImportError:
        print("pygame is required.  Install with: pip install pygame")
        sys.exit(1)

    from env import MarioEnv
    from metrics import MetricsTracker

    env = MarioEnv(map_path=map_path, obs_mode="rgb",
                   render_mode="human", fps=fps)
    tracker = MetricsTracker(keep_steps=True)

    obs, _ = env.reset()
    tracker.new_episode()
    episode = 0
    total_steps = 0

    print("=" * 50)
    print("  Mario RL – Human Mode")
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
            obs, _ = env.reset()
            tracker.new_episode()
            episode += 1

    env.close()
    print(f"\nTotal episodes: {episode}   Total steps: {total_steps}")
    tracker.print_summary()
    save = input("Save replay? [y/N] ").strip().lower()
    if save == "y":
        tracker.save_csv("metrics_human.csv")
        tracker.save_replay("replay_human_last.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-playable Mario RL")
    parser.add_argument("map", help="Path to the .txt tile map")
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

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
