"""
run_agent.py – Run an AI agent in the Mario RL environment.

Built-in agents
---------------
  random      – Uniform random action each frame
  right       – Always press RIGHT (baseline heuristic)
  right_jump  – RIGHT + occasional jump (simple heuristic)

Usage
-----
  python run_agent.py map.txt --agent random --episodes 50
  python run_agent.py map.txt --agent right  --episodes 100 --render
  python run_agent.py map.txt --agent right_jump --episodes 200 --obs features

The environment exposes a standard gymnasium API so you can plug in
any RL library (Stable-Baselines3, RLlib, CleanRL, etc.) instead.
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from typing import Callable, Union

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd


# ── Built-in agents ───────────────────────────────────────────────────────────

class RandomAgent:
    """Uniform random action from the action space."""
    def __init__(self, n_actions: int):
        self.n = n_actions

    def act(self, obs: np.ndarray) -> int:
        return np.random.randint(self.n)


class RightAgent:
    """Always walk right — sanity-check baseline."""
    def act(self, obs: np.ndarray) -> int:
        return 1  # RIGHT


class RightJumpAgent:
    """
    Walk right, jump over obstacles and enemies.

    When obs_mode == 'features' (recommended), the agent reads the
    normalised feature vector to detect nearby enemies and walls and
    jumps proactively.  Otherwise it uses periodic timed jumps.
    """

    def __init__(self):
        self._t = 0
        self._jump_cooldown = 0
        self._jump_every = 40   # periodic jump interval (fallback)

    def act(self, obs: np.ndarray) -> int:
        self._t += 1
        if self._jump_cooldown > 0:
            self._jump_cooldown -= 1

        should_jump = False

        if obs.ndim == 1 and len(obs) > 8:
            # ── feature-mode heuristics ───────────────────────────────────
            on_ground = obs[4] > 0.5

            # indices 8 .. 8+H*W-1 are the tile window;
            # goomba features start at 8 + OBS_TILES_H*OBS_TILES_W
            try:
                from .constants import OBS_TILES_H, OBS_TILES_W
            except ImportError:
                from constants import OBS_TILES_H, OBS_TILES_W
            tile_start = 8
            tile_end   = tile_start + OBS_TILES_H * OBS_TILES_W
            goomba_start = tile_end

            # check nearest enemy (first goomba block)
            if goomba_start + 3 < len(obs):
                enemy_exists = obs[goomba_start] > 0.5
                enemy_dx     = obs[goomba_start + 1]  # normalised by SCREEN_W
                if enemy_exists and -0.05 < enemy_dx < 0.25:
                    should_jump = True   # enemy is close and ahead

            # periodic jump to clear pits / platforms
            if on_ground and self._t % self._jump_every < 4:
                should_jump = True

            if should_jump and self._jump_cooldown == 0 and on_ground:
                self._jump_cooldown = 28
                return 7   # RUN_RIGHT_JUMP

            return 6   # RUN_RIGHT

        # ── fallback: tile / rgb obs ──────────────────────────────────────
        if self._jump_cooldown == 0 and self._t % self._jump_every < 4:
            self._jump_cooldown = 28
            return 4   # RIGHT_JUMP
        return 1   # RIGHT


# ── Runner ────────────────────────────────────────────────────────────────────

def run(
    map_path: Union[str, pd.DataFrame, Callable[[], Union[str, pd.DataFrame]]],
    agent_name: str = "random",
    batch_size: int = 1,
    n_episodes: int = 10,
    obs_mode: str = "features",
    render: bool = False,
    save_csv: str = "",
    save_replay: str = "",
    verbose: bool = True,
) -> dict:
    """
    Run *n_episodes* of the specified agent and return summary stats.

    If `map_path` is a callable, it will be invoked every `batch_size`
    episodes to obtain a fresh generated level from a pool manager.

    Returns
    -------
    dict from MetricsTracker.summary()
    """
    from env import MarioEnv, N_ACTIONS, ACTION_MAP
    from metrics import MetricsTracker

    if batch_size <= 0:
        batch_size = 1

    level_provider = map_path if callable(map_path) else None
    initial_map = map_path() if callable(map_path) else map_path

    render_mode = "human" if render else "rgb_array" if obs_mode == "rgb" else None
    env = MarioEnv(
        map_path=initial_map,
        obs_mode=obs_mode,
        render_mode=render_mode,
        fps=60 if render else 10000,
    )

    # Agent selection
    if agent_name == "random":
        agent = RandomAgent(N_ACTIONS)
    elif agent_name == "right":
        agent = RightAgent()
    elif agent_name == "right_jump":
        agent = RightJumpAgent()
    else:
        raise ValueError(f"Unknown agent '{agent_name}'. "
                         f"Choose: random, right, right_jump")

    tracker = MetricsTracker(keep_steps=True)
    t_start = time.time()

    for ep in range(n_episodes):
        if ep > 0 and level_provider is not None and ep % batch_size == 0:
            obs, _ = env.reset(map_path=level_provider())
        else:
            obs, _ = env.reset()

        tracker.new_episode()
        done = trunc = False

        while not (done or trunc):
            action = agent.act(obs)
            obs, reward, done, trunc, info = env.step(action)
            tracker.record(action, reward, info)

            if render:
                try:
                    import pygame
                    for evt in pygame.event.get():
                        if evt.type == pygame.QUIT:
                            done = True

                except ImportError:
                    pass

        ep_rec = tracker.end_episode(info)

        if verbose:
            status = "WIN" if ep_rec.won else "DEAD"
            print(f"  EP {ep+1:04d}/{n_episodes}  [{status}]  "
                  f"reward={ep_rec.total_reward:+8.2f}  "
                  f"steps={ep_rec.steps:5d}  "
                  f"max_x={ep_rec.max_x:6.0f}  "
                  f"coins={ep_rec.coins}  kills={ep_rec.kills}")

    elapsed = time.time() - t_start
    env.close()

    if save_csv:
        tracker.save_csv(save_csv)
    if save_replay:
        tracker.save_replay(save_replay)

    summary = tracker.summary()
    if verbose:
        print()
        tracker.print_summary()
        fps_sim = sum(e.steps for e in tracker._episodes) / max(elapsed, 1e-9)
        print(f"  Simulated at {fps_sim:,.0f} steps/sec  "
              f"({elapsed:.1f}s total)")

    return summary


# ── Action-distribution analysis ─────────────────────────────────────────────

def analyse_replay(replay_path: str) -> None:
    """Print statistics from a saved JSON replay file."""
    import json
    with open(replay_path) as f:
        data = json.load(f)

    rewards = np.array(data['rewards'])
    xs = np.array(data['mario_x'])
    actions = np.array(data['actions'])

    from env import MarioEnv
    action_names = MarioEnv.action_meanings()

    print(f"\nReplay EP {data['episode_id']} — {data['steps']} steps")
    print(f"  Total reward : {data['total_reward']:.2f}")
    print(f"  Won          : {data['won']}")
    print(f"  Max x        : {data['max_x']:.1f}")
    print(f"  Reward stats : mean={rewards.mean():.4f}  "
          f"std={rewards.std():.4f}  "
          f"min={rewards.min():.3f}  "
          f"max={rewards.max():.3f}")

    print("\n  Action distribution:")
    for a in range(len(action_names)):
        frac = np.mean(actions == a)
        bar = "█" * int(frac * 40)
        print(f"    {a}  {action_names[a]:<18} {bar} {frac*100:.1f}%")

    print("\n  Progress (x position):")
    quarters = np.array_split(xs, 4)
    for i, q in enumerate(quarters):
        print(f"    Q{i+1}: {q.mean():.0f} px  max={q.max():.0f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a built-in agent against the Mario RL environment.")
    parser.add_argument("map",
                        help="Path to .txt tile map")
    parser.add_argument("--agent",    default="right_jump",
                        choices=["random", "right", "right_jump"],
                        help="Agent to use (default: right_jump)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to run (default: 20)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Number of episodes per generated level batch (default: 1)")
    parser.add_argument("--obs",      default="features",
                        choices=["rgb", "tiles", "features"],
                        help="Observation mode (default: features)")
    parser.add_argument("--render",   action="store_true",
                        help="Open Pygame window (slows training)")
    parser.add_argument("--save-csv",    default="",
                        metavar="FILE",
                        help="Save episode CSV to FILE")
    parser.add_argument("--save-replay", default="",
                        metavar="FILE",
                        help="Save last episode replay JSON to FILE")
    parser.add_argument("--analyse",  default="",
                        metavar="REPLAY_FILE",
                        help="Analyse a saved replay JSON instead of running")
    args = parser.parse_args()

    if args.analyse:
        analyse_replay(args.analyse)
        return

    print(f"\nMario RL – Agent: {args.agent}  "
          f"Episodes: {args.episodes}  Obs: {args.obs}\n")

    run(
        map_path=args.map,
        agent_name=args.agent,
        batch_size=args.batch,
        n_episodes=args.episodes,
        obs_mode=args.obs,
        render=args.render,
        save_csv=args.save_csv,
        save_replay=args.save_replay,
    )


if __name__ == "__main__":
    main()
