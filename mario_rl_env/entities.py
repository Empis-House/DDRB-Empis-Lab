"""
entities.py – Dataclass definitions for Mario and Goomba.

All positions are in pixel-space (float).
(x, y) is the TOP-LEFT corner of the entity's AABB bounding box.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple


class MarioState(Enum):
    SMALL      = auto()   # one tile tall
    BIG        = auto()   # two tiles tall
    DEAD       = auto()   # death animation, then episode ends


class GoombaState(Enum):
    ALIVE  = auto()
    SQUISHED = auto()   # stomped; plays squish for N frames then removed
    DEAD   = auto()


@dataclass
class Mario:
    x: float           # left edge (px)
    y: float           # top  edge (px)
    vx: float = 0.0    # velocity x  (px/frame)
    vy: float = 0.0    # velocity y  (px/frame)

    on_ground: bool = False
    facing_right: bool = True
    running: bool = False
    jump_buffer: int = 0         # frames of buffered jump input
    coyote_time: int = 0         # frames of coyote time for jumps

    state: MarioState = MarioState.SMALL
    invincible_frames: int = 0   # brief invincibility after taking damage
    stomp_grace_frames: int = 0  # brief grace period after a stomp

    # statistics this episode
    coins: int = 0
    score: int = 0
    kills: int = 0

    @property
    def w(self) -> int:
        try:
            from .constants import MARIO_W
        except ImportError:
            from constants import MARIO_W
        return MARIO_W

    @property
    def h(self) -> int:
        try:
            from .constants import MARIO_H, MARIO_H_BIG
        except ImportError:
            from constants import MARIO_H, MARIO_H_BIG
        return MARIO_H_BIG if self.state == MarioState.BIG else MARIO_H

    @property
    def cx(self) -> float:
        """Centre x."""
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        """Centre y."""
        return self.y + self.h / 2

    def rect(self) -> Tuple[float, float, float, float]:
        """(left, top, right, bottom) in pixel-space."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def is_alive(self) -> bool:
        return self.state != MarioState.DEAD


@dataclass
class Goomba:
    x: float
    y: float
    vx: float = 0.0     # set to -GOOMBA_SPEED on creation
    vy: float = 0.0

    state: GoombaState = GoombaState.ALIVE
    squish_timer: int = 0    # frames remaining in squish animation
    on_ground: bool = False

    @property
    def w(self) -> int:
        try:
            from .constants import GOOMBA_W
        except ImportError:
            from constants import GOOMBA_W
        return GOOMBA_W

    @property
    def h(self) -> int:
        try:
            from .constants import GOOMBA_H
        except ImportError:
            from constants import GOOMBA_H
        return GOOMBA_H

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def rect(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def is_alive(self) -> bool:
        return self.state == GoombaState.ALIVE
