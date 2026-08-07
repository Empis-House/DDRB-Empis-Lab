"""
physics.py – AABB collision detection and entity movement.

Per-axis separation avoids corner-sliding artefacts:
  1. Apply gravity to vy.
  2. Move entity along X; resolve X collisions.
  3. Move entity along Y; resolve Y collisions.

All coordinates are in pixel-space (float).
Tile coordinates are (col, row) integers.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional

try:
    from .constants import (
        TILE_SIZE,
        GRAVITY, MAX_FALL_SPEED,
        MARIO_WALK_SPEED, MARIO_RUN_SPEED, MARIO_JUMP_VY,
        GOOMBA_SPEED,
    )
    from .tilemap import TileMap, TileType
    from .entities import Mario, Goomba, MarioState, GoombaState
except ImportError:
    from constants import (
        TILE_SIZE,
        GRAVITY, MAX_FALL_SPEED,
        MARIO_WALK_SPEED, MARIO_RUN_SPEED, MARIO_JUMP_VY,
        GOOMBA_SPEED,
    )
    from tilemap import TileMap, TileType
    from entities import Mario, Goomba, MarioState, GoombaState


# ── Rect helpers ──────────────────────────────────────────────────────────────

def _overlaps(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> bool:
    """AABB overlap test."""
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _tiles_for_rect(x: float, y: float, w: int, h: int):
    """Yield (col, row) tile indices under the given AABB."""
    # Use a tiny epsilon instead of subtracting a full pixel so contact at
    # tile boundaries remains stable with sub-pixel velocities.
    eps = 1e-6
    col_lo = int(math.floor(x / TILE_SIZE))
    col_hi = int(math.floor((x + w - eps) / TILE_SIZE))
    row_lo = int(math.floor(y / TILE_SIZE))
    row_hi = int(math.floor((y + h - eps) / TILE_SIZE))
    for row in range(row_lo, row_hi + 1):
        for col in range(col_lo, col_hi + 1):
            yield col, row


# ── Mario movement ─────────────────────────────────────────────────────────────

def move_mario(mario: Mario, tilemap: TileMap, actions: dict) -> dict:
    """
    Apply one physics frame to *mario* given a dict of active actions:
        actions = {
            'left':  bool,
            'right': bool,
            'jump':  bool,
            'run':   bool,
        }

    Returns an *events* dict:
        {
            'coin':       bool,   # hit a question block
            'brick_broken': bool,
            'fell_pit':   bool,   # fell below tilemap
            'pipe_entry': Optional[str],  # 'red' | 'blue' | None
        }
    """
    events = {'coin': False, 'brick_broken': False, 'fell_pit': False, 'pipe_entry': None}

    if mario.state == MarioState.DEAD:
        # dead Mario: just fall through the world
        mario.vy = min(mario.vy + GRAVITY, MAX_FALL_SPEED)
        mario.y += mario.vy
        return events

    speed = MARIO_RUN_SPEED if actions.get('run') else MARIO_WALK_SPEED

    # ── horizontal input ──────────────────────────────────────────────────
    if actions.get('right'):
        mario.vx = speed
        mario.facing_right = True
    elif actions.get('left'):
        mario.vx = -speed
        mario.facing_right = False
    else:
        mario.vx = 0.0

    mario.running = bool(actions.get('run'))

    # ── jump ──────────────────────────────────────────────────────────────
    if actions.get('jump') and mario.on_ground:
        mario.vy = MARIO_JUMP_VY
        mario.on_ground = False

    # ── gravity ───────────────────────────────────────────────────────────
    mario.vy = min(mario.vy + GRAVITY, MAX_FALL_SPEED)

    # ── invincibility timer ───────────────────────────────────────────────
    if mario.invincible_frames > 0:
        mario.invincible_frames -= 1
    if mario.stomp_grace_frames > 0:
        mario.stomp_grace_frames -= 1

    # ── X movement ────────────────────────────────────────────────────────
    mario.x += mario.vx
    # clamp to left world boundary
    if mario.x < 0:
        mario.x = 0.0
        mario.vx = 0.0

    _resolve_x(mario, tilemap)

    # ── Y movement ────────────────────────────────────────────────────────
    mario.on_ground = False
    mario.y += mario.vy
    coin, brick = _resolve_y(mario, tilemap)
    events['coin'] = coin or _collect_overlapping_coins(mario, tilemap)
    events['brick_broken'] = brick
    events['pipe_entry'] = _detect_pipe_entry(mario, tilemap, actions)

    # ── pit detection ─────────────────────────────────────────────────────
    if mario.y > tilemap.pixel_height-32:
        events['fell_pit'] = True

    return events


def _collect_overlapping_coins(mario: Mario, tilemap: TileMap) -> bool:
    """Collect any passable coin tiles overlapped by Mario this frame."""
    collected = False
    for col, row in _tiles_for_rect(mario.x, mario.y, mario.w, mario.h):
        collected = tilemap.collect_coin(col, row) or collected
    return collected


def _detect_pipe_entry(mario: Mario, tilemap: TileMap, actions: dict) -> Optional[str]:
    """Return pipe color when crouching on an entry pipe top, else None."""
    # Accept tiny grounding jitter while pressing down near a pipe top.
    if not actions.get('down'):
        return None
    if not mario.on_ground and abs(mario.vy) > 1.0:
        return None

    foot_row = int((mario.y + mario.h) // TILE_SIZE)
    left_col = int(mario.x // TILE_SIZE)
    right_col = int((mario.x + mario.w - 1) // TILE_SIZE)

    candidates = {
        tilemap.get(left_col, foot_row),
        tilemap.get(right_col, foot_row),
        tilemap.get(int(mario.cx // TILE_SIZE), foot_row),
    }
    if TileType.PIPE_RED_TOP_LEFT in candidates or TileType.PIPE_RED_TOP_RIGHT in candidates:
        return 'red'
    if TileType.PIPE_BLUE_TOP_LEFT in candidates or TileType.PIPE_BLUE_TOP_RIGHT in candidates:
        return 'blue'
    return None


def _resolve_x(mario: Mario, tilemap: TileMap) -> None:
    """Push mario out of solid tiles along X axis."""
    for col, row in _tiles_for_rect(mario.x, mario.y, mario.w, mario.h):
        if tilemap.is_solid(col, row):
            tile_x1 = col * TILE_SIZE
            tile_x2 = tile_x1 + TILE_SIZE
            # overlap? push out
            if mario.x + mario.w > tile_x1 and mario.x < tile_x2:
                if mario.vx > 0:
                    mario.x = tile_x1 - mario.w
                elif mario.vx < 0:
                    mario.x = tile_x2
                mario.vx = 0.0


def _resolve_y(mario: Mario, tilemap: TileMap) -> Tuple[bool, bool]:
    """Push mario out of solid tiles along Y axis.
    Returns (coin_awarded, brick_broken)."""
    coin = False
    brick = False
    for col, row in _tiles_for_rect(mario.x, mario.y, mario.w, mario.h):
        if tilemap.is_solid(col, row):
            tile_y1 = row * TILE_SIZE
            tile_y2 = tile_y1 + TILE_SIZE
            if mario.y + mario.h > tile_y1 and mario.y < tile_y2:
                if mario.vy > 0:
                    # landing on top
                    mario.y = tile_y1 - mario.h
                    mario.vy = 0.0
                    mario.on_ground = True
                elif mario.vy < 0:
                    # hitting ceiling
                    mario.y = tile_y2
                    mario.vy = 0.0
                    # hit block from below
                    coin = coin or tilemap.hit_block(col, row)
                    if mario.state == MarioState.BIG:
                        brk = tilemap.break_brick(col, row)
                        brick = brick or brk
    return coin, brick


# ── Goomba movement ───────────────────────────────────────────────────────────

def move_goomba(goomba: Goomba, tilemap: TileMap) -> None:
    """Apply one frame of movement to *goomba*."""
    if goomba.state == GoombaState.DEAD:
        return
    if goomba.state == GoombaState.SQUISHED:
        goomba.squish_timer -= 1
        if goomba.squish_timer <= 0:
            goomba.state = GoombaState.DEAD
        return

    # gravity
    goomba.vy = min(goomba.vy + GRAVITY, MAX_FALL_SPEED)

    # horizontal
    goomba.x += goomba.vx
    _resolve_goomba_x(goomba, tilemap)

    # vertical
    goomba.on_ground = False
    goomba.y += goomba.vy

    # TileMap treats rows below the map as solid to keep Mario bounded.
    # For enemies, falling below the map should be a pit death instead.
    if goomba.y + goomba.h > tilemap.pixel_height:
        goomba.state = GoombaState.DEAD
        return

    _resolve_goomba_y(goomba, tilemap)

    # fall off world = permanent death
    if goomba.y > tilemap.pixel_height + 64:
        goomba.state = GoombaState.DEAD


def _resolve_goomba_x(goomba: Goomba, tilemap: TileMap) -> None:
    for col, row in _tiles_for_rect(goomba.x, goomba.y, goomba.w, goomba.h):
        if tilemap.is_solid(col, row):
            tile_x1 = col * TILE_SIZE
            tile_x2 = tile_x1 + TILE_SIZE
            if goomba.x + goomba.w > tile_x1 and goomba.x < tile_x2:
                if goomba.vx > 0:
                    goomba.x = tile_x1 - goomba.w
                    goomba.vx = -GOOMBA_SPEED
                elif goomba.vx < 0:
                    goomba.x = tile_x2
                    goomba.vx = GOOMBA_SPEED

    # reverse at world left boundary
    if goomba.x < 0:
        goomba.x = 0.0
        goomba.vx = GOOMBA_SPEED


def _resolve_goomba_y(goomba: Goomba, tilemap: TileMap) -> None:
    for col, row in _tiles_for_rect(goomba.x, goomba.y, goomba.w, goomba.h):
        if tilemap.is_solid(col, row):
            tile_y1 = row * TILE_SIZE
            tile_y2 = tile_y1 + TILE_SIZE
            if goomba.y + goomba.h > tile_y1 and goomba.y < tile_y2:
                if goomba.vy >= 0:
                    goomba.y = tile_y1 - goomba.h
                    goomba.vy = 0.0
                    goomba.on_ground = True
                elif goomba.vy < 0:
                    goomba.y = tile_y2
                    goomba.vy = 0.0


# ── Collision helpers ─────────────────────────────────────────────────────────

def check_mario_goomba(mario: Mario, goomba: Goomba) -> Optional[str]:
    """
    Returns:
        'stomp'  – Mario jumped on top of Goomba
        'hit'    – Side/bottom collision (Mario takes damage)
        None     – No collision
    """
    if not goomba.is_alive():
        return None

    ml, mt, mr, mb = mario.rect()
    gl, gt, gr, gb = goomba.rect()

    if not _overlaps(ml, mt, mr, mb, gl, gt, gr, gb):
        return None

    # stomp: Mario falling, his bottom was above goomba's vertical centre last frame
    if mario.vy > 0 and mb - mario.vy <= gt + goomba.h * 0.4:
        return 'stomp'

    if mario.invincible_frames > 0 or mario.stomp_grace_frames > 0:
        return None

    return 'hit'
