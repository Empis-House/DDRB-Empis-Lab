"""
metrics.py – Episode statistics tracking, CSV logging, and replay recording.

ML analysis helpers
-------------------
MetricsTracker   – attach to any MarioEnv and call record() each step.
                   Call summary() to get aggregate stats.
                   Call save_csv() to dump all episodes to a CSV file.
                   Call save_replay() to dump a full episode replay to JSON.

Usage example
-------------
    tracker = MetricsTracker()
    obs, _ = env.reset()
    tracker.new_episode()

    while True:
        action = agent.act(obs)
        obs, reward, done, trunc, info = env.step(action)
        tracker.record(action, reward, info)
        if done or trunc:
            tracker.end_episode(info)
            break

    tracker.save_csv("metrics.csv")
    tracker.save_replay("replay_ep0.json")
    print(tracker.summary())
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ── Per-episode data ──────────────────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    episode_id: int
    total_reward: float = 0.0
    steps: int = 0
    coins: int = 0
    score: int = 0
    kills: int = 0
    max_x: float = 0.0
    won: bool = False
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    # per-step data (stored if keep_steps=True)
    step_actions:  List[int]   = field(default_factory=list)
    step_rewards:  List[float] = field(default_factory=list)
    step_mario_x:  List[float] = field(default_factory=list)
    step_mario_y:  List[float] = field(default_factory=list)
    step_on_ground: List[bool] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return (self.end_time or time.time()) - self.start_time


class MetricsTracker:
    """
    Lightweight metrics collector.

    Parameters
    ----------
    keep_steps : bool
        If True, store per-step data for replays (uses more memory).
    max_episodes : int | None
        Rolling buffer size.  Older episodes are discarded when exceeded.
    """

    def __init__(self, keep_steps: bool = True, max_episodes: int = 1000):
        self.keep_steps = keep_steps
        self.max_episodes = max_episodes
        self._episodes: List[EpisodeRecord] = []
        self._current: Optional[EpisodeRecord] = None
        self._ep_count = 0

    # ── Recording API ─────────────────────────────────────────────────────────

    def new_episode(self) -> None:
        """Call at the start of each episode (after env.reset())."""
        self._current = EpisodeRecord(episode_id=self._ep_count)
        self._ep_count += 1

    def record(self, action: int, reward: float, info: dict) -> None:
        """Call after every env.step()."""
        if self._current is None:
            return
        ep = self._current
        ep.total_reward += reward
        ep.steps += 1
        if self.keep_steps:
            ep.step_actions.append(int(action))
            ep.step_rewards.append(float(reward))
            ep.step_mario_x.append(float(info.get('mario_x', 0)))
            ep.step_mario_y.append(float(info.get('mario_y', 0)))
            ep.step_on_ground.append(bool(info.get('on_ground', False)))

    def end_episode(self, info: dict) -> EpisodeRecord:
        """Call when done or truncated.  Returns the finalised record."""
        if self._current is None:
            raise RuntimeError("Call new_episode() before end_episode().")
        ep = self._current
        ep.end_time = time.time()
        ep_info = info.get('episode', {})
        ep.total_reward = ep_info.get('total_reward', ep.total_reward)
        ep.steps        = ep_info.get('steps',        ep.steps)
        ep.coins        = ep_info.get('coins',        ep.coins)
        ep.score        = ep_info.get('score',        ep.score)
        ep.kills        = ep_info.get('kills',        ep.kills)
        ep.max_x        = ep_info.get('max_x',        ep.max_x)
        ep.won          = ep_info.get('won',          ep.won)

        self._episodes.append(ep)
        if len(self._episodes) > self.max_episodes:
            self._episodes.pop(0)
        self._current = None
        return ep

    # ── Analysis ──────────────────────────────────────────────────────────────

    def summary(self, last_n: int = 100) -> dict:
        """Return aggregate statistics over the last *last_n* episodes."""
        if not self._episodes:
            return {"episodes_recorded": 0}

        recent = self._episodes[-last_n:]
        rewards  = [e.total_reward for e in recent]
        max_xs   = [e.max_x        for e in recent]
        steps    = [e.steps        for e in recent]
        wins     = [int(e.won)     for e in recent]
        kills    = [e.kills        for e in recent]
        coins    = [e.coins        for e in recent]

        return {
            "episodes_recorded":    len(self._episodes),
            "last_n":               len(recent),
            "mean_reward":          float(np.mean(rewards)),
            "std_reward":           float(np.std(rewards)),
            "max_reward":           float(np.max(rewards)),
            "min_reward":           float(np.min(rewards)),
            "mean_max_x":           float(np.mean(max_xs)),
            "max_max_x":            float(np.max(max_xs)),
            "mean_steps":           float(np.mean(steps)),
            "win_rate":             float(np.mean(wins)),
            "mean_kills_per_ep":    float(np.mean(kills)),
            "mean_coins_per_ep":    float(np.mean(coins)),
        }

    def action_distribution(self, last_n: int = 10) -> Dict[int, float]:
        """Fraction of time each action was taken across last *last_n* episodes."""
        if not self.keep_steps:
            return {}
        recent = self._episodes[-last_n:]
        all_acts: List[int] = []
        for ep in recent:
            all_acts.extend(ep.step_actions)
        if not all_acts:
            return {}
        total = len(all_acts)
        counts: Dict[int, int] = {}
        for a in all_acts:
            counts[a] = counts.get(a, 0) + 1
        return {a: c / total for a, c in sorted(counts.items())}

    def reward_curve(self, window: int = 20) -> List[float]:
        """Smoothed reward curve (moving average over *window* episodes)."""
        rewards = [e.total_reward for e in self._episodes]
        if len(rewards) < window:
            return rewards
        return [
            float(np.mean(rewards[i:i + window]))
            for i in range(len(rewards) - window + 1)
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_csv(self, path: str) -> None:
        """Write per-episode summary rows to a CSV file."""
        rows = [
            {
                "episode_id":    e.episode_id,
                "total_reward":  e.total_reward,
                "steps":         e.steps,
                "coins":         e.coins,
                "score":         e.score,
                "kills":         e.kills,
                "max_x":         e.max_x,
                "won":           int(e.won),
                "duration_sec":  round(e.duration_sec, 3),
            }
            for e in self._episodes
        ]
        if not rows:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[MetricsTracker] Saved {len(rows)} episode(s) → {path}")

    def save_replay(self, path: str, episode_idx: int = -1) -> None:
        """
        Save a full episode replay to JSON.
        Fields: actions, rewards, mario_x, mario_y, on_ground lists.
        Useful for offline analysis, visualisation, or imitation learning.
        """
        if not self._episodes:
            print("[MetricsTracker] No episodes to save.")
            return
        ep = self._episodes[episode_idx]
        data = {
            "episode_id":   ep.episode_id,
            "total_reward": ep.total_reward,
            "steps":        ep.steps,
            "won":          ep.won,
            "max_x":        ep.max_x,
            "actions":      ep.step_actions,
            "rewards":      ep.step_rewards,
            "mario_x":      ep.step_mario_x,
            "mario_y":      ep.step_mario_y,
            "on_ground":    ep.step_on_ground,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            json.dump(data, fh, indent=2)
        print(f"[MetricsTracker] Replay saved → {path}")

    def load_replay(self, path: str) -> dict:
        """Load a replay JSON file for offline inspection."""
        with open(path) as fh:
            return json.load(fh)

    def print_summary(self, last_n: int = 100) -> None:
        s = self.summary(last_n)
        print("=" * 52)
        print(f"  Episode summary  (last {s['last_n']} / {s['episodes_recorded']} total)")
        print("=" * 52)
        print(f"  Mean reward : {s['mean_reward']:+.2f}  ±{s['std_reward']:.2f}")
        print(f"  Max  reward : {s['max_reward']:+.2f}")
        print(f"  Mean max_x  : {s['mean_max_x']:.1f}  px")
        print(f"  Win rate    : {s['win_rate']*100:.1f} %")
        print(f"  Mean steps  : {s['mean_steps']:.1f}")
        print(f"  Mean kills  : {s['mean_kills_per_ep']:.2f}")
        print(f"  Mean coins  : {s['mean_coins_per_ep']:.2f}")
        print("=" * 52)
