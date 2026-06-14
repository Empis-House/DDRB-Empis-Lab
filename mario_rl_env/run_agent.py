"""
run_agent.py – Run an AI agent in the Mario RL environment.

Built-in agents
---------------
  random          – Uniform random action each frame
  right           – Always press RIGHT (baseline heuristic)
  right_jump      – RIGHT + occasional jump (simple heuristic)
  smart_heuristic – Fast heuristic that avoids gaps, obstacles, and nearby enemies

Usage
-----
  python run_agent.py map.txt --agent random --episodes 50
  python run_agent.py map.txt --agent right  --episodes 100 --render
  python run_agent.py map.txt --agent right_jump --episodes 200 --obs features
  python run_agent.py map.txt --agent smart_heuristic --episodes 50

The environment exposes a standard gymnasium API so you can plug in
any RL library (Stable-Baselines3, RLlib, CleanRL, etc.) instead.
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from typing import Callable, Optional, Union

import gymnasium as gym
from mario_rl_env.ml_agents import AGENT_MANAGER, RLAgent, run_training_loop

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
            # goomba features start after the tile window or after the
            # optional tagged landing/collider layers.
            try:
                from .constants import OBS_TILES_H, OBS_TILES_W
            except ImportError:
                from constants import OBS_TILES_H, OBS_TILES_W
            tile_start = 8
            tile_size = OBS_TILES_H * OBS_TILES_W
            tile_end = tile_start + tile_size
            landing_start = tile_end
            collider_start = landing_start + tile_size
            if len(obs) >= collider_start + tile_size:
                goomba_start = collider_start + tile_size
            else:
                goomba_start = tile_end

            enemy_exists = False
            enemy_dx = 0.0
            if goomba_start + 3 < len(obs):
                enemy_exists = obs[goomba_start] > 0.5
                enemy_dx = obs[goomba_start + 1]  # normalised by SCREEN_W

            if enemy_exists and -0.05 < enemy_dx < 0.25:
                should_jump = True   # enemy is close and ahead

            # tile-based terrain heuristics when the structural layers are present
            if len(obs) >= collider_start + tile_size:
                landing_mask = obs[landing_start:collider_start]
                collider_mask = obs[collider_start:collider_start + tile_size]
                center = (OBS_TILES_H // 2) * OBS_TILES_W + (OBS_TILES_W // 2)
                for offset in (1, 2, 3):
                    idx = center + offset
                    if idx >= tile_size:
                        break
                    if collider_mask[idx] > 0.5:
                        should_jump = True
                        break
                    if on_ground and landing_mask[idx] < 0.5:
                        should_jump = True
                        break

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


class SmartHeuristicAgent:
    """Lightweight heuristic player agent that avoids gaps and obstacles."""

    def __init__(self):
        self._t = 0
        self._jump_cooldown = 0

    def act(self, obs: np.ndarray) -> int:
        self._t += 1
        if self._jump_cooldown > 0:
            self._jump_cooldown -= 1

        should_jump = False

        if obs.ndim == 1 and len(obs) > 8:
            on_ground = obs[4] > 0.5
            coins = obs[5] if len(obs) > 5 else 0.0
            coin_pressure = coins < 0.06
            try:
                from .constants import OBS_TILES_H, OBS_TILES_W
            except ImportError:
                from constants import OBS_TILES_H, OBS_TILES_W

            tile_start = 8
            tile_size = OBS_TILES_H * OBS_TILES_W
            tile_end = tile_start + tile_size
            landing_start = tile_end
            collider_start = landing_start + tile_size
            use_structural = len(obs) >= collider_start + tile_size
            feature_window = obs[tile_start:tile_end]
            center = (OBS_TILES_H // 2) * OBS_TILES_W + (OBS_TILES_W // 2)

            def get_tile(idx: int) -> float:
                if 0 <= idx < tile_size:
                    return feature_window[idx]
                return 0.0

            def is_solid(idx: int) -> bool:
                return get_tile(idx) > 0.1

            def has_gap(offset: int) -> bool:
                ahead = center + offset
                below = ahead + OBS_TILES_W
                return get_tile(ahead) < 0.1 and get_tile(below) < 0.1

            # enemy detection from feature vector
            enemy_exists = False
            enemy_dx = 0.0
            if use_structural:
                goomba_start = collider_start + tile_size
            else:
                goomba_start = tile_end
            if goomba_start + 3 < len(obs):
                enemy_exists = obs[goomba_start] > 0.5
                enemy_dx = obs[goomba_start + 1]

            if enemy_exists and on_ground and -0.10 < enemy_dx < 0.35:
                should_jump = True

            if use_structural:
                landing_mask = obs[landing_start:collider_start]
                collider_mask = obs[collider_start:collider_start + tile_size]
                for offset in (1, 2, 3):
                    idx = center + offset
                    if idx >= tile_size:
                        break
                    if collider_mask[idx] > 0.5:
                        should_jump = True
                        break
                    if on_ground and landing_mask[idx] < 0.5:
                        should_jump = True
                        break
            else:
                # no structural layers: infer gaps and obstacles from tile grid
                for offset in (1, 2, 3):
                    if is_solid(center + offset):
                        should_jump = True
                        break
                if on_ground and not should_jump:
                    if has_gap(1) or has_gap(2):
                        should_jump = True

            # occasional jump for platform clearance
            if on_ground and self._t % 30 < 4:
                should_jump = True

            # coin-driven jump pressure has equal priority to obstacle avoidance
            if on_ground and coin_pressure and self._t % 20 < 5:
                should_jump = True

            if should_jump and self._jump_cooldown == 0 and on_ground:
                self._jump_cooldown = 24
                return 7   # RUN_RIGHT_JUMP

            return 6   # RUN_RIGHT

        if self._jump_cooldown == 0 and self._t % 30 < 4:
            self._jump_cooldown = 24
            return 4   # RIGHT_JUMP
        return 1   # RIGHT


# ── Runner ────────────────────────────────────────────────────────────────────

def run(
    map_path: Optional[Union[str, pd.DataFrame, Callable[[], Union[str, pd.DataFrame, gym.Env]]]] = None,
    env: Optional[Union[gym.Env, Callable[[], gym.Env]]] = None,
    agent_name: str = "random",
    batch_size: int = 1,
    n_episodes: int = 10,
    obs_mode: str = "features",
    render: bool = False,
    render_last: bool = False,
    save_csv: str = "",
    save_replay: str = "",
    history_path: str = "",
    checkpoint_path: str = "",
    verbose: bool = True,
    debug: bool = False,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01,
) -> dict:
    """
    Run *n_episodes* of the specified agent and return summary stats.

    If `env` is provided, the runner will use the supplied environment directly.
    If `env` is a callable, a fresh environment is created every `batch_size`
    episodes for batched training. If `map_path` is a callable, it is used to
    generate fresh Mario maps when no `env` is provided.

    Returns
    -------
    dict from MetricsTracker.summary() or training history for ML agents
    """
    from env import MarioEnv, N_ACTIONS, ACTION_MAP
    from metrics import MetricsTracker

    if batch_size <= 0:
        batch_size = 1

    if env is None and map_path is None:
        raise ValueError("Either map_path or env must be provided.")

    render_mode = "human" if render or render_last else "rgb_array" if obs_mode == "rgb" else None

    env_provider = None
    if env is not None:
        if callable(env):
            env_provider = env
            env = env_provider()
        else:
            env_provider = None
    elif callable(map_path):
        env_provider = lambda: MarioEnv(
            map_path=map_path(),
            obs_mode=obs_mode,
            render_mode=render_mode,
            fps=60 if render else 10000,
        )
        env = env_provider()
    else:
        env = MarioEnv(
            map_path=map_path,
            obs_mode=obs_mode,
            render_mode=render_mode,
            fps=60 if render else 10000,
        )

    # Agent selection
    training_agent = False
    if agent_name == "random":
        agent = RandomAgent(N_ACTIONS)
    elif agent_name == "right":
        agent = RightAgent()
    elif agent_name == "right_jump":
        agent = RightJumpAgent()
    elif agent_name == "smart_heuristic":
        agent = SmartHeuristicAgent()
    elif agent_name in AGENT_MANAGER.list_agents():
        agent = AGENT_MANAGER.create(
            agent_name,
            N_ACTIONS,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
        )
        training_agent = True
    else:
        raise ValueError(
            f"Unknown agent '{agent_name}'. "
            f"Choose: random, right, right_jump, or one of {AGENT_MANAGER.list_agents()}"
        )

    if training_agent:
        if checkpoint_path and os.path.exists(checkpoint_path):
            if verbose:
                print(f"[Agent] Resuming from checkpoint: {checkpoint_path}")
            agent = RLAgent.load(checkpoint_path)
            # Validate loaded agent action size against environment
            try:
                loaded_size = getattr(agent, "action_size", None)
                if loaded_size is not None and loaded_size != N_ACTIONS:
                    print(f"[Agent] Warning: loaded agent action_size={loaded_size} != env N_ACTIONS={N_ACTIONS}")
                    if hasattr(agent, "resize_action_space"):
                        agent.resize_action_space(N_ACTIONS)
                        print(f"[Agent] Resized loaded agent action space to {N_ACTIONS}")
                    else:
                        # best-effort: overwrite attribute and hope for the best
                        agent.action_size = N_ACTIONS
                        print(f"[Agent] Overwrote agent.action_size -> {N_ACTIONS}")
            except Exception as exc:
                print(f"[Agent] Could not validate/adjust loaded agent: {exc}")

        training_environment = env_provider if env_provider is not None else env
        if verbose:
            print(f"[Agent] Training agent: {agent_name}")
            print(f"[Agent] Episodes: {n_episodes}, Batch size: {batch_size}")
            if history_path:
                print(f"[Agent] History path: {history_path}")
            if checkpoint_path:
                print(f"[Agent] Checkpoint path: {checkpoint_path}")
        training_history = run_training_loop(
            training_environment,
            agent,
            n_episodes=n_episodes,
            max_steps=None,
            render=render,
            render_last=render_last,
            verbose=verbose,
            batch_size=batch_size,
            history_path=history_path,
            checkpoint_path=checkpoint_path,
            debug=debug,
        )
        if hasattr(env, "close"):
            env.close()
        return {"history": training_history}

    tracker = MetricsTracker(keep_steps=True)
    t_start = time.time()

    for ep in range(n_episodes):
        if ep > 0 and env_provider is not None and ep % batch_size == 0:
            if hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass
            env = env_provider()
            obs, _ = env.reset()
        else:
            obs, _ = env.reset()

        tracker.new_episode()
        done = trunc = False

        while not (done or trunc):
            action = agent.act(obs)
            obs, reward, done, trunc, info = env.step(action)
            tracker.record(action, reward, info)

            if render and (not render_last or ep == n_episodes):
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
    agent_choices = ["random", "right", "right_jump", "smart_heuristic"] + [
        name for name in AGENT_MANAGER.list_agents()
        if name not in {"random"}
    ]
    parser = argparse.ArgumentParser(
        description="Run a built-in agent against the Mario RL environment.")
    parser.add_argument("map",
                        help="Path to .txt tile map")
    parser.add_argument("--agent",    default="right_jump",
                        choices=agent_choices,
                        help="Agent to use (default: right_jump)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to run (default: 20)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Number of episodes per generated level batch (default: 1)")
    parser.add_argument("--obs",      default="features",
                        choices=["rgb", "tiles", "features", "features_layers"],
                        help="Observation mode (default: features)")
    parser.add_argument("--render",   action="store_true",
                        help="Open Pygame window for all episodes (slows training)")
    parser.add_argument("--render-last", action="store_true",
                        help="Open Pygame window only for the final episode")
    parser.add_argument("--save-csv",    default="",
                        metavar="FILE",
                        help="Save episode CSV to FILE")
    parser.add_argument("--save-replay", default="",
                        metavar="FILE",
                        help="Save last episode replay JSON to FILE")
    parser.add_argument("--history-file", default="",
                        metavar="FILE",
                        help="Save training history to FILE (JSON or CSV)")
    parser.add_argument("--save-agent", default="",
                        metavar="FILE",
                        help="Save trained agent checkpoint to FILE. If FILE exists, resume training from it.")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Tabular Q learning rate alpha (default: 0.2)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor gamma for tabular Q (default: 0.99)")
    parser.add_argument("--epsilon", type=float, default=0.4,
                        help="Initial epsilon for epsilon-greedy exploration (default: 0.4)")
    parser.add_argument("--epsilon-decay", type=float, default=0.95,
                        help="Epsilon decay factor per episode (default: 0.95)")
    parser.add_argument("--min-epsilon", type=float, default=0.05,
                        help="Minimum epsilon after decay (default: 0.05)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging for training (prints sample obs, encoded state, Q deltas)")
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
        render_last=args.render_last,
        save_csv=args.save_csv,
        save_replay=args.save_replay,
        history_path=args.history_file,
        checkpoint_path=args.save_agent,
        debug=args.debug,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
    )


if __name__ == "__main__":
    main()
