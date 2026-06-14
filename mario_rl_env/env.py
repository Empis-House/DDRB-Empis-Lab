"""
env.py – Gymnasium-compatible Super Mario Bros. environment.

────────────────────────────────────────────────────────────────
Observation modes (set via ``obs_mode`` constructor arg):
  'rgb'      – (H, W, 3) uint8 raw pixel frame (default)
  'tiles'    – (OBS_TILES_H, OBS_TILES_W) int8 tile-type window around Mario
  'features' – 1-D float32 vector of hand-crafted features (fast for training)

Action space (Discrete 8):
  0  NOOP
  1  RIGHT
  2  LEFT
  3  JUMP
  4  RIGHT + JUMP
  5  LEFT  + JUMP
  6  RIGHT + RUN
  7  RIGHT + RUN + JUMP

Reward (per step):
  + progress_right * REWARD_PER_TILE_RIGHT   (only for new furthest x)
  - regression_left * |REWARD_PER_TILE_LEFT|
  + REWARD_KILL_ENEMY  on stomp
  + REWARD_COIN        on question block hit
  + REWARD_DEATH       on death / falling in pit  (negative constant)
  + REWARD_WIN         on reaching end of level
  + REWARD_TIME_STEP   each step (small negative, encourages speed)

Info dict (always):
  step, mario_x, mario_y, mario_vx, mario_vy, on_ground,
  coins, score, kills, mario_state, total_reward, n_goombas

Info dict (episode end only, key 'episode'):
  total_reward, steps, coins, score, kills, max_x, won
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from .constants import (
        SCREEN_H, SCREEN_W, TILE_SIZE, FPS,
        OBS_TILES_W, OBS_TILES_H,
        MAX_STEPS_PER_EPISODE,
    )
    from .tilemap import TileMap, TileType, SOLID_TILES, parse_tilemap
    from .game_state import GameState
    from .renderer import Renderer
except ImportError:
    from constants import (
        SCREEN_H, SCREEN_W, TILE_SIZE, FPS,
        OBS_TILES_W, OBS_TILES_H,
        MAX_STEPS_PER_EPISODE,
    )
    from tilemap import TileMap, TileType, SOLID_TILES, parse_tilemap
    from game_state import GameState
    from renderer import Renderer


# ── Action definitions ────────────────────────────────────────────────────────

ACTION_MAP: List[dict] = [
    {},                                            # 0 NOOP
    {'right': True},                               # 1 RIGHT
    {'left': True},                                # 2 LEFT
    {'jump': True},                                # 3 JUMP
    {'right': True, 'jump': True},                 # 4 RIGHT+JUMP
    {'left': True,  'jump': True},                 # 5 LEFT+JUMP
    {'right': True, 'run': True},                  # 6 RUN RIGHT
    {'right': True, 'run': True, 'jump': True},    # 7 RUN+JUMP RIGHT
]

N_ACTIONS = len(ACTION_MAP)

# feature vector length (must match _make_feature_obs)
N_FEATURES = 2 + 2 + 1 + 1 + 1 + 1 + OBS_TILES_H * OBS_TILES_W + 4 * 5
N_FEATURES_LAYERS = 2 + 2 + 1 + 1 + 1 + 1 + OBS_TILES_H * OBS_TILES_W * 3 + 4 * 5
# mario(x,y) + vel(vx,vy) + on_ground + coins + kills + mario_state
# + local tile window
# + landing mask layer
# + collider mask layer
# + up to 5 nearest goomba (dx, dy, vx, exists)


class MarioEnv(gym.Env):
    """
    Gymnasium environment wrapping the Mario game engine.

    Parameters
    ----------
    map_path : str
        Path to the comma-separated .txt tile map.
    obs_mode : str
        One of 'rgb', 'tiles', 'features'.
    render_mode : str | None
        'human'   – open a Pygame window
        'rgb_array' – headless; use env.render() to get numpy arrays
        None      – fully headless, no rendering
    fps : int
        Frames per second cap (only applied in 'human' mode).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        map_path: str,
        obs_mode: str = "rgb",
        render_mode: Optional[str] = None,
        fps: int = FPS,
    ):
        super().__init__()
        assert obs_mode in ("rgb", "tiles", "features", "features_layers"), \
            f"obs_mode must be 'rgb', 'tiles', 'features', or 'features_layers'. Got: {obs_mode}"

        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.fps = fps

        # Load tilemap once; GameState clones it per episode
        self._load_tilemap(map_path)

        # Renderer (lazy init)
        self._renderer: Optional[Renderer] = None
        if render_mode in ("human", "rgb_array"):
            self._renderer = Renderer(headless=(render_mode == "rgb_array"))

        # ── Spaces ────────────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(N_ACTIONS)

        if obs_mode == "rgb":
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(SCREEN_H, SCREEN_W, 3),
                dtype=np.uint8,
            )
        elif obs_mode == "tiles":
            self.observation_space = spaces.Box(
                low=0, high=len(TileType),
                shape=(OBS_TILES_H, OBS_TILES_W),
                dtype=np.int8,
            )
        elif obs_mode == "features_layers":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(N_FEATURES_LAYERS,),
                dtype=np.float32,
            )
        else:  # features
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(N_FEATURES,),
                dtype=np.float32,
            )

    def _load_tilemap(self, map_path: Any) -> None:
        self._tilemap_source = parse_tilemap(map_path)
        self._gs = GameState(tilemap_source=self._tilemap_source)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        map_path: Optional[Any] = None,
    ) -> Tuple[np.ndarray, dict]:
        if map_path is not None:
            self._load_tilemap(map_path)
        super().reset(seed=seed)
        self._gs.reset()
        obs = self._make_obs()
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        acts = ACTION_MAP[int(action)]
        result = self._gs.step(acts)
        obs = self._make_obs()

        if self.render_mode == "human" and self._renderer:
            self._renderer.draw(self._gs)
            self._renderer.flip(self.fps)

        return obs, result.reward, result.done, result.truncated, result.info

    def render(self) -> Optional[np.ndarray]:
        """
        Returns an RGB array when render_mode == 'rgb_array',
        or None for 'human' / headless modes.
        """
        if self._renderer is None:
            return None
        if not self._renderer._initialised:
            self._renderer.init()
        self._renderer.draw(self._gs)
        if self.render_mode == "human":
            self._renderer.flip(self.fps)
            return None
        return self._renderer.get_rgb_array()

    def close(self) -> None:
        if self._renderer:
            self._renderer.close()

    # ── Observation builders ─────────────────────────────────────────────────

    def _make_obs(self) -> np.ndarray:
        if self.obs_mode == "rgb":
            return self._make_rgb_obs()
        elif self.obs_mode == "tiles":
            return self._make_tile_obs()
        elif self.obs_mode == "features_layers":
            return self._make_feature_obs(include_structural_layers=True)
        else:
            return self._make_feature_obs()

    def _make_rgb_obs(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = Renderer(headless=True)
        if not self._renderer._initialised:
            self._renderer.init()
        self._renderer.draw(self._gs)
        return self._renderer.get_rgb_array()

    def _make_tile_obs(self) -> np.ndarray:
        """
        Build a OBS_TILES_H × OBS_TILES_W int8 grid centred on Mario.
        AIR (0) is used for tiles outside the map boundary.
        Enemy positions are encoded as a special value (len(TileType)).
        """
        gs = self._gs
        mario = gs.mario
        tm = gs.tilemap

        # Mario's tile
        mario_col = int(mario.cx // TILE_SIZE)
        mario_row = int(mario.cy // TILE_SIZE)

        half_w = OBS_TILES_W // 2
        half_h = OBS_TILES_H // 2

        grid = np.zeros((OBS_TILES_H, OBS_TILES_W), dtype=np.int8)
        for ri, row in enumerate(range(mario_row - half_h,
                                        mario_row - half_h + OBS_TILES_H)):
            for ci, col in enumerate(range(mario_col - half_w,
                                            mario_col - half_w + OBS_TILES_W)):
                grid[ri, ci] = int(tm.get(col, row))

        # overlay goombas
        GOOMBA_TILE_VAL = int8_clamp(len(TileType))
        for g in gs.goombas:
            if not g.is_alive():
                continue
            gc = int(g.cx / TILE_SIZE)
            gr = int(g.cy / TILE_SIZE)
            ri = gr - (mario_row - half_h)
            ci = gc - (mario_col - half_w)
            if 0 <= ri < OBS_TILES_H and 0 <= ci < OBS_TILES_W:
                grid[ri, ci] = GOOMBA_TILE_VAL

        return grid

    def _make_feature_obs(self, include_structural_layers: bool = False) -> np.ndarray:
        """
        Build a compact float32 feature vector.

        Layout:
          [0]  mario_x / map_pixel_width        (normalised 0-1)
          [1]  mario_y / map_pixel_height        (normalised 0-1)
          [2]  mario_vx / MARIO_RUN_SPEED        (clamped -1..1)
          [3]  mario_vy / MAX_FALL_SPEED         (clamped -1..1)
          [4]  on_ground                         (0 or 1)
          [5]  coins / 100                       (normalised)
          [6]  kills / 50
          [7]  mario_state_encoded               (0=SMALL,1=BIG,2=DEAD)
          [8 .. 8+H*W-1]  flattened tile window  (int8 → float, /N_TILES)
          [.. .. ..] optional structural layers for world description:
             landing mask and collider mask around Mario
          last 20:  up to 5 nearest goombas
                    per goomba: [exists, dx/screen_w, dy/screen_h, goomba_vx_dir]
        """
        try:
            from .constants import (MARIO_RUN_SPEED, MAX_FALL_SPEED,
                                    SCREEN_W, SCREEN_H)
            from .entities import MarioState
        except ImportError:
            from constants import (MARIO_RUN_SPEED, MAX_FALL_SPEED,
                                   SCREEN_W, SCREEN_H)
            from entities import MarioState

        gs = self._gs
        mario = gs.mario
        tm = gs.tilemap

        pw = tm.pixel_width or 1
        ph = tm.pixel_height or 1
        n_tiles = max(1, len(TileType))

        feats: List[float] = [
            mario.x / pw,
            mario.y / ph,
            np.clip(mario.vx / MARIO_RUN_SPEED, -1, 1),
            np.clip(mario.vy / MAX_FALL_SPEED, -1, 1),
            float(mario.on_ground),
            mario.coins / 100.0,
            mario.kills / 50.0,
            {'SMALL': 0.0, 'BIG': 1.0, 'DEAD': 2.0}.get(mario.state.name, 0.0),
        ]

        # tile window
        tile_grid = self._make_tile_obs().flatten().astype(np.float32) / n_tiles
        feats.extend(tile_grid.tolist())

        if include_structural_layers:
            landing_mask = self._make_landing_mask().flatten().astype(np.float32)
            collider_mask = self._make_collider_mask().flatten().astype(np.float32)
            feats.extend(landing_mask.tolist())
            feats.extend(collider_mask.tolist())

        # nearest 5 goombas
        alive = [g for g in gs.goombas if g.is_alive()]
        alive.sort(key=lambda g: abs(g.cx - mario.cx))
        for i in range(5):
            if i < len(alive):
                g = alive[i]
                feats += [
                    1.0,
                    (g.cx - mario.cx) / SCREEN_W,
                    (g.cy - mario.cy) / SCREEN_H,
                    1.0 if g.vx > 0 else -1.0,
                ]
            else:
                feats += [0.0, 0.0, 0.0, 0.0]

        return np.array(feats, dtype=np.float32)

    def _make_landing_mask(self) -> np.ndarray:
        """Return a dense landing surface mask in Mario's local tile window."""
        gs = self._gs
        mario = gs.mario
        tm = gs.tilemap

        mario_col = int(mario.cx // TILE_SIZE)
        mario_row = int(mario.cy // TILE_SIZE)
        half_w = OBS_TILES_W // 2
        half_h = OBS_TILES_H // 2

        mask = np.zeros((OBS_TILES_H, OBS_TILES_W), dtype=np.float32)
        for ri, row in enumerate(range(mario_row - half_h,
                                        mario_row - half_h + OBS_TILES_H)):
            for ci, col in enumerate(range(mario_col - half_w,
                                            mario_col - half_w + OBS_TILES_W)):
                if tm.get(col, row) == TileType.AIR and tm.get(col, row + 1) in SOLID_TILES:
                    mask[ri, ci] = 1.0
        return mask

    def _make_collider_mask(self) -> np.ndarray:
        """Return a binary collider mask for solid terrain in Mario's local tile window."""
        gs = self._gs
        mario = gs.mario
        tm = gs.tilemap

        mario_col = int(mario.cx // TILE_SIZE)
        mario_row = int(mario.cy // TILE_SIZE)
        half_w = OBS_TILES_W // 2
        half_h = OBS_TILES_H // 2

        mask = np.zeros((OBS_TILES_H, OBS_TILES_W), dtype=np.float32)
        for ri, row in enumerate(range(mario_row - half_h,
                                        mario_row - half_h + OBS_TILES_H)):
            for ci, col in enumerate(range(mario_col - half_w,
                                            mario_col - half_w + OBS_TILES_W)):
                if tm.get(col, row) in SOLID_TILES:
                    mask[ri, ci] = 1.0
        return mask

    # ── Utility ───────────────────────────────────────────────────────────────

    @property
    def game_state(self) -> GameState:
        """Direct access to the underlying game state (read/write)."""
        return self._gs

    @staticmethod
    def action_meanings() -> List[str]:
        return [
            "NOOP",
            "RIGHT",
            "LEFT",
            "JUMP",
            "RIGHT_JUMP",
            "LEFT_JUMP",
            "RUN_RIGHT",
            "RUN_RIGHT_JUMP",
        ]


def int8_clamp(v: int) -> int:
    return max(-128, min(127, v))
