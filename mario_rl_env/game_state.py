"""
game_state.py – Complete mutable world state for one episode.

GameState holds Mario, all active Goombas, the TileMap copy,
step counter, etc.  The ``step()`` method advances the simulation
one frame given a dict of pressed actions.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from .constants import (
        TILE_SIZE, GOOMBA_SPEED,
        SCREEN_W,
        MARIO_WALK_SPEED, BOUNCE_VY,
        REWARD_PER_TILE_RIGHT, REWARD_PER_TILE_LEFT,
        REWARD_KILL_ENEMY, REWARD_COIN, REWARD_DEATH,
        REWARD_WIN, REWARD_TIME_STEP,
        MAX_STEPS_PER_EPISODE,
    )
    from .tilemap import TileMap, parse_tilemap
    from .entities import Mario, Goomba, MarioState, GoombaState
    from .physics import move_mario, move_goomba, check_mario_goomba
except ImportError:
    from constants import (
        TILE_SIZE, GOOMBA_SPEED,
        SCREEN_W,
        MARIO_WALK_SPEED, BOUNCE_VY,
        REWARD_PER_TILE_RIGHT, REWARD_PER_TILE_LEFT,
        REWARD_KILL_ENEMY, REWARD_COIN, REWARD_DEATH,
        REWARD_WIN, REWARD_TIME_STEP,
        MAX_STEPS_PER_EPISODE,
    )
    from tilemap import TileMap, parse_tilemap
    from entities import Mario, Goomba, MarioState, GoombaState
    from physics import move_mario, move_goomba, check_mario_goomba


@dataclass
class StepResult:
    reward: float
    done: bool
    truncated: bool
    info: dict


@dataclass
class GameState:
    """
    All mutable state for a single episode.

    Parameters
    ----------
    tilemap_source : TileMap
        The *original* (unmodified) tilemap loaded from file.
        A fresh copy is cloned per episode so blocks can be consumed.
    """
    tilemap_source: TileMap        # immutable reference (never modified)
    tilemap: TileMap = field(init=False)  # per-episode clone
    mario: Mario = field(init=False)
    goombas: List[Goomba] = field(init=False, default_factory=list)

    step_count: int = field(init=False, default=0)
    total_reward: float = field(init=False, default=0.0)
    done: bool = field(init=False, default=False)

    # Track furthest x for progress reward
    _max_x: float = field(init=False, default=0.0)
    _prev_x: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.reset()

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Initialise (or reinitialise) the episode."""
        self.tilemap = self.tilemap_source.clone()
        self.mario = self._spawn_mario()
        self.goombas = self._spawn_goombas()
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self._max_x = self.mario.x
        self._prev_x = self.mario.x

    def step(self, actions: dict) -> StepResult:
        """
        Advance simulation by one frame.

        Parameters
        ----------
        actions : dict
            Keys: 'left', 'right', 'jump', 'run'  (values: bool)

        Returns
        -------
        StepResult
        """
        if self.done:
            return StepResult(0.0, True, False,
                              {'step': self.step_count,
                               'message': 'episode already done'})

        reward = REWARD_TIME_STEP
        info: dict = {}

        # ── Physics ──────────────────────────────────────────────────────
        events = move_mario(self.mario, self.tilemap, actions)

        # Activate goombas when they enter the visible viewport so they
        # remain idle until the player can see them. This prevents enemies
        # from moving before the camera reaches them.
        try:
            half_w = SCREEN_W // 2
            cam_x = max(0, min(int(self.mario.cx) - half_w,
                               self.tilemap.pixel_width - SCREEN_W))
        except Exception:
            # conservative fallback if tilemap pixel width missing
            cam_x = max(0, int(self.mario.cx) - SCREEN_W // 2)

        for g in self.goombas:
            # if goomba hasn't been activated yet, start moving when any
            # part of its bounding box becomes visible in the viewport
            if g.is_alive() and abs(g.vx) < 1e-6:
                if (g.x + g.w) >= cam_x and g.x <= cam_x + SCREEN_W:
                    # default direction: move left
                    g.vx = -GOOMBA_SPEED

        for g in self.goombas:
            move_goomba(g, self.tilemap)

        # ── Enemy interactions ────────────────────────────────────────────
        for g in self.goombas:
            result = check_mario_goomba(self.mario, g)
            if result == 'stomp':
                g.state = GoombaState.SQUISHED
                g.squish_timer = 15
                g.vx = 0.0
                self.mario.vy = BOUNCE_VY
                self.mario.kills += 1
                reward += REWARD_KILL_ENEMY
                self.mario.score += 100
                info['killed_goomba'] = True
            elif result == 'hit':
                self._damage_mario()
                reward += REWARD_DEATH
                info['took_damage'] = True

        # ── Remove dead goombas ───────────────────────────────────────────
        self.goombas = [g for g in self.goombas if g.state != GoombaState.DEAD]

        # ── Block events ─────────────────────────────────────────────────
        if events['coin']:
            self.mario.coins += 1
            self.mario.score += 200
            reward += REWARD_COIN
            info['coin'] = True

        # ── Pit / death ───────────────────────────────────────────────────
        if events['fell_pit']:
            self.mario.state = MarioState.DEAD
            reward += REWARD_DEATH
            info['fell_pit'] = True

        # ── Progress reward ───────────────────────────────────────────────
        dx = self.mario.x - self._prev_x
        if dx > 0:
            progress = (self.mario.x - self._max_x)
            if progress > 0:
                reward += progress / TILE_SIZE * REWARD_PER_TILE_RIGHT
                self._max_x = self.mario.x
        elif dx < 0:
            reward += abs(dx) / TILE_SIZE * REWARD_PER_TILE_LEFT
        self._prev_x = self.mario.x

        # ── Win condition ─────────────────────────────────────────────────
        if self.mario.x >= (self.tilemap.width - 4) * TILE_SIZE:
            reward += REWARD_WIN
            info['won'] = True
            self.done = True

        # ── Death check ───────────────────────────────────────────────────
        if self.mario.state == MarioState.DEAD:
            self.done = True
            info['died'] = True

        # ── Truncation ────────────────────────────────────────────────────
        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS_PER_EPISODE and not self.done

        self.total_reward += reward

        info.update({
            'step':         self.step_count,
            'mario_x':      self.mario.x,
            'mario_y':      self.mario.y,
            'mario_vx':     self.mario.vx,
            'mario_vy':     self.mario.vy,
            'on_ground':    self.mario.on_ground,
            'coins':        self.mario.coins,
            'score':        self.mario.score,
            'kills':        self.mario.kills,
            'mario_state':  self.mario.state.name,
            'total_reward': self.total_reward,
            'n_goombas':    len(self.goombas),
        })

        if self.done or truncated:
            info['episode'] = {
                'total_reward': self.total_reward,
                'steps':        self.step_count,
                'coins':        self.mario.coins,
                'score':        self.mario.score,
                'kills':        self.mario.kills,
                'max_x':        self._max_x,
                'won':          info.get('won', False),
            }

        return StepResult(
            reward=reward,
            done=self.done,
            truncated=truncated,
            info=info,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _spawn_mario(self) -> Mario:
        """
        Place Mario just above ground at the first column that is both
        open (air above ground) and clear of enemy spawns by ≥3 tiles.
        """
        tm = self.tilemap_source
        from constants import MARIO_W, MARIO_H

        enemy_cols = {c for c, r in tm.enemy_spawns}
        clearance = 3

        start_col = 1
        start_row = 1
        for col in range(1, min(8, tm.width)):

            solid_rows = [r for r in range(tm.height) if tm.is_solid(col, r)]
            landings = [x for x in solid_rows if not x-1 in solid_rows]
            if not landings:
                start_col = 1
                start_row = 1
                break

            lowest_land = max(landings)
            if lowest_land >= 2:
                start_col = col
                start_row = lowest_land - 1
                break

        x = start_col * TILE_SIZE + (TILE_SIZE - MARIO_W) // 2
        y = start_row * TILE_SIZE
        return Mario(x=float(x), y=float(y))

    def _spawn_goombas(self) -> List[Goomba]:
        """Instantiate one Goomba per enemy spawn in the tilemap."""
        goombas: List[Goomba] = []
        from constants import GOOMBA_W, GOOMBA_H
        # avoid spawning enemies right next to Mario's spawn column
        clearance_cols = 3
        try:
            mario_col = int(self.mario.cx // TILE_SIZE)
        except Exception:
            mario_col = None

        for (col, row) in self.tilemap_source.enemy_spawns:
            if mario_col is not None and abs(col - mario_col) <= clearance_cols:
                # skip enemy spawn near mario start
                continue
            x = col * TILE_SIZE + (TILE_SIZE - GOOMBA_W) // 2
            y = row * TILE_SIZE
            # spawn idle; activate when entering the viewport
            g = Goomba(x=float(x), y=float(y), vx=0.0)
            goombas.append(g)
        return goombas

    def _damage_mario(self) -> None:
        """Apply one hit to Mario."""
        if self.mario.state == MarioState.BIG:
            self.mario.state = MarioState.SMALL
            self.mario.invincible_frames = 90   # 1.5 s invincibility
        else:
            self.mario.state = MarioState.DEAD
            self.mario.vy = -8.0   # death bounce
