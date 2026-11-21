"""Shared experimental protocol for custom and NES-backed Mario environments."""

from __future__ import annotations

import hashlib
import json
import re
from importlib import import_module
from typing import Any, Optional

import gymnasium as gym

from .constants import MAX_STEPS_PER_EPISODE, HORIZONTAL_STUCK_STEPS


PROTOCOL_VERSION = "mario-rl-benchmark-v1"
ACTION_MEANINGS = (
    "NOOP",
    "RIGHT",
    "LEFT",
    "JUMP",
    "RIGHT_JUMP",
    "LEFT_JUMP",
    "RUN_RIGHT",
    "RUN_RIGHT_JUMP",
)

# JoypadSpace button names corresponding positionally to the custom simulator's
# ACTION_MAP. SMB uses A for jump and B for run.
GYM_JOYPAD_ACTIONS = (
    ("NOOP",),
    ("right",),
    ("left",),
    ("A",),
    ("right", "A"),
    ("left", "A"),
    ("right", "B"),
    ("right", "A", "B"),
)

PROTOCOL_METADATA = {
    "name": PROTOCOL_VERSION,
    "action_meanings": list(ACTION_MEANINGS),
    "frames_per_action": 1,
    "max_episode_steps": MAX_STEPS_PER_EPISODE,
    "horizontal_stuck_steps": HORIZONTAL_STUCK_STEPS,
    "progress_unit": "horizontal tile widths advanced (the first step is the zero point)",
    "progress_reward_per_new_tile": 1.0,
    "coin_reward": 2.0,
    "death_penalty": -15.0,
    "win_bonus": 100.0,
    "step_penalty": -0.01,
}


class BenchmarkProtocolWrapper(gym.Wrapper):
    """Normalize actions' experiment metrics while retaining backend physics.

    The wrapper uses horizontal tile widths as the common progress unit:
    custom-engine tiles are 64 px wide; NES tiles are 16 px wide. It does not
    change either environment's movement, collision, enemy, or level rules.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        backend: str,
        max_episode_steps: int = MAX_STEPS_PER_EPISODE,
        horizontal_stuck_steps: int = HORIZONTAL_STUCK_STEPS,
        seed: Optional[int] = None,
        auto_render: bool = False,
        level_id: Optional[str] = None,
    ) -> None:
        super().__init__(env)
        if backend not in {"custom", "nes"}:
            raise ValueError(f"Unsupported benchmark backend: {backend}")
        if not hasattr(env.action_space, "n") or env.action_space.n != len(ACTION_MEANINGS):
            raise ValueError(
                f"Benchmark environments must expose {len(ACTION_MEANINGS)} discrete actions"
            )
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if horizontal_stuck_steps <= 0:
            raise ValueError("horizontal_stuck_steps must be positive")

        self.backend = backend
        self.max_episode_steps = int(max_episode_steps)
        self.horizontal_stuck_steps = int(horizontal_stuck_steps)
        self._seed = seed
        self._auto_render = auto_render
        self.level_id = level_id
        self._episode_index = 0
        self._steps = 0
        self._start_x: Optional[float] = None
        self._max_progress = 0.0
        self._coins = 0
        self._total_reward = 0.0
        self._furthest_x: Optional[float] = None
        self._horizontal_stuck_steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None and self._seed is not None:
            seed = self._seed + self._episode_index
        self._episode_index += 1
        observation, info = self.env.reset(seed=seed, options=options)
        self._steps = 0
        self._start_x = None
        self._max_progress = 0.0
        self._coins = 0
        self._total_reward = 0.0
        self._furthest_x = None
        self._horizontal_stuck_steps = 0
        info = dict(info)
        info["benchmark"] = {
            "protocol": PROTOCOL_VERSION,
            "backend": self.backend,
            "actions": list(ACTION_MEANINGS),
            "level_id": self.level_id,
        }
        return observation, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside the shared benchmark action space")

        observation, native_reward, terminated, truncated, raw_info = self.env.step(action)
        info = dict(raw_info)
        x_position = self._read_x_position(info)
        if self._start_x is None:
            self._start_x = x_position
        if self._furthest_x is None:
            self._furthest_x = x_position
        if x_position > self._furthest_x:
            self._furthest_x = x_position
            self._horizontal_stuck_steps = 0
        else:
            self._horizontal_stuck_steps += 1
        stuck_horizontally = (
            self._horizontal_stuck_steps >= self.horizontal_stuck_steps
        )
        stuck_horizontally = bool(
            stuck_horizontally
            or info.get("stuck_horizontally", False)
        )
        progress = max(0.0, (x_position - self._start_x) / self._pixels_per_tile)
        previous_max = self._max_progress
        self._max_progress = max(previous_max, progress)

        coin_count = info.get("coins")
        coin_delta = 0
        if coin_count is not None:
            coin_delta = max(0, int(coin_count) - self._coins)
            self._coins = max(self._coins, int(coin_count))

        episode_info = info.get("episode", {})
        won = bool(
            info.get("won")
            or info.get("flag_get")
            or info.get("clear")
            or (isinstance(episode_info, dict) and episode_info.get("won"))
        )
        died = bool(info.get("died") or info.get("death"))
        if terminated and not won and not died:
            died = True
        terminated = bool(terminated or won or died)
        self._steps += 1
        truncated = bool(
            (
                truncated
                or stuck_horizontally
                or self._steps >= self.max_episode_steps
            ) and not terminated
        )

        reward = max(0.0, self._max_progress - previous_max)
        reward += 2.0 * coin_delta
        reward -= 0.01
        if won:
            reward += 100.0
        elif died:
            reward -= 15.0
        self._total_reward += reward

        info.update({
            "backend": self.backend,
            "level_id": self.level_id,
            "native_reward": float(native_reward),
            "benchmark_reward": float(reward),
            "progress_tiles": float(progress),
            "progress_max_tiles": float(self._max_progress),
            "mario_x": float(progress),
            "coins": self._coins,
            "won": won,
            "died": died,
            "step": self._steps,
            "stuck_horizontally": stuck_horizontally,
        })
        if terminated or truncated:
            normalized_episode = dict(episode_info) if isinstance(episode_info, dict) else {}
            normalized_episode.update({
                "total_reward": self._total_reward,
                "steps": self._steps,
                "coins": self._coins,
                "max_x": float(self._max_progress),
                "max_progress_tiles": float(self._max_progress),
                "won": won,
                "died": died,
                "termination_reason": (
                    "win" if won
                    else "death" if died
                    else "horizontal_stuck" if stuck_horizontally
                    else "truncated"
                ),
                "stuck_horizontally": stuck_horizontally,
                "level_id": self.level_id,
            })
            info["episode"] = normalized_episode
        if self.backend == "nes" and self._auto_render:
            self.env.render()
        return observation, float(reward), terminated, truncated, info

    def set_auto_render(self, enabled: bool) -> None:
        if self.backend == "nes":
            self._auto_render = bool(enabled)
        setter = getattr(self.env, "set_auto_render", None)
        if setter is not None:
            setter(enabled)

    @property
    def _pixels_per_tile(self) -> int:
        return 64 if self.backend == "custom" else 16

    def _read_x_position(self, info: dict) -> float:
        key = "mario_x" if self.backend == "custom" else "x_pos"
        if key not in info:
            raise KeyError(
                f"{self.backend} environment info is missing required progress field {key!r}"
            )
        return float(info[key])


def make_custom_env(
    map_path: Any,
    *,
    render_mode: Optional[str] = None,
    fps: int = 60,
    obs_mode: str = "rgb",
    seed: Optional[int] = None,
) -> BenchmarkProtocolWrapper:
    from .env import MarioEnv
    from .tilemap import parse_tilemap

    tilemap = parse_tilemap(map_path)
    identity = {
        "shape": list(tilemap.tiles.shape),
        "enemy_spawns": tilemap.enemy_spawns,
    }
    digest = hashlib.sha256(tilemap.tiles.tobytes())
    digest.update(json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return BenchmarkProtocolWrapper(
        MarioEnv(map_path=tilemap, obs_mode=obs_mode, render_mode=render_mode, fps=fps),
        backend="custom",
        seed=seed,
        level_id=f"custom:{digest.hexdigest()}",
    )


def make_classic_env(
    stage: str,
    *,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> BenchmarkProtocolWrapper:
    if not re.fullmatch(r"\d+-\d+", stage):
        raise ValueError(f"Stage must have the form WORLD-STAGE, got {stage!r}")

    try:
        import_module("gym_super_mario_bros")  # Registers the package's Gymnasium IDs.
        joypad_space = import_module("nes_py.wrappers")
    except ImportError as exc:
        raise RuntimeError(
            "Classic-level benchmarking requires the optional gym-super-mario-bros "
            "and nes-py dependencies. Install requirements-benchmark.txt with Python 3.13+."
        ) from exc

    env = gym.make(f"SuperMarioBros-{stage}-v0", render_mode=render_mode)
    env = joypad_space.JoypadSpace(env, [list(action) for action in GYM_JOYPAD_ACTIONS])
    return BenchmarkProtocolWrapper(
        env,
        backend="nes",
        seed=seed,
        auto_render=render_mode == "human",
        level_id=f"SuperMarioBros-{stage}-v0",
    )
