"""
renderer.py – Pygame-based renderer for the Mario RL environment.

Uses only coloured rectangles — no external sprite assets required.
Supports a scrolling viewport that follows Mario.

Usage
-----
  renderer = Renderer(tilemap)
  renderer.init()            # call once to create the window
  surface = renderer.draw(game_state)
  renderer.flip()            # blit to screen and advance display
  renderer.close()
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

from constants import (
    TILE_SIZE, SCREEN_W, SCREEN_H, SCREEN_TILES_W, SCREEN_TILES_H,
    COL_SKY, COL_GROUND, COL_BRICK, COL_QUESTION, COL_QUESTION_USED,
    COL_PIPE_DARK, COL_PIPE_LITE, COL_MARIO, COL_MARIO_BIG,
    COL_GOOMBA, COL_GOOMBA_DEAD, COL_HUD_BG, COL_WHITE, COL_YELLOW, COL_RED,
)
from tilemap import TileMap, TileType
from entities import Mario, Goomba, MarioState, GoombaState
from game_state import GameState


# ── Tile colour lookup ────────────────────────────────────────────────────────

TILE_COLORS = {
    TileType.GROUND:         COL_GROUND,
    TileType.BRICK:          COL_BRICK,
    TileType.QUESTION:       COL_QUESTION,
    TileType.QUESTION_USED:  COL_QUESTION_USED,
    TileType.PIPE_TOP_LEFT:  COL_PIPE_DARK,
    TileType.PIPE_TOP_RIGHT: COL_PIPE_DARK,
    TileType.PIPE_LEFT:      COL_PIPE_LITE,
    TileType.PIPE_RIGHT:     COL_PIPE_LITE,
}


class Renderer:
    """
    Pygame renderer.  Can run in *headless* mode (``headless=True``),
    which skips creating a window and returns surfaces as numpy arrays
    via ``get_rgb_array()``.
    """

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._surface: Optional["pygame.Surface"] = None
        self._screen:  Optional["pygame.Surface"] = None
        self._font:    Optional["pygame.font.Font"] = None
        self._clock    = None
        self._initialised = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def init(self) -> None:
        if self._initialised:
            return
        if not _PYGAME_AVAILABLE:
            raise RuntimeError("pygame is required for rendering. "
                               "Install with: pip install pygame")
        if self.headless:
            import os
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        pygame.init()
        if not self.headless:
            self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Mario RL")
        self._surface = pygame.Surface((SCREEN_W, SCREEN_H))
        self._font = pygame.font.SysFont("monospace", 14, bold=True)
        self._clock = pygame.time.Clock()
        self._initialised = True

    def close(self) -> None:
        if self._initialised and _PYGAME_AVAILABLE:
            pygame.quit()
        self._initialised = False

    # ── Main draw entry ───────────────────────────────────────────────────────

    def draw(self, gs: GameState) -> None:
        """Render the current game state onto the internal surface."""
        if not self._initialised:
            self.init()

        surf = self._surface
        surf.fill(COL_SKY)

        # viewport offset: keep Mario centred horizontally
        cam_x = self._camera_x(gs)

        self._draw_tiles(surf, gs.tilemap, cam_x)
        self._draw_goombas(surf, gs.goombas, cam_x)
        self._draw_mario(surf, gs.mario, cam_x, gs.step_count)
        self._draw_hud(surf, gs)

    def flip(self, fps: int = 60) -> None:
        """Blit surface → screen and tick clock."""
        if not self._initialised:
            return
        if not self.headless and self._screen is not None:
            self._screen.blit(self._surface, (0, 0))
            pygame.display.flip()
        if self._clock:
            self._clock.tick(fps)

    def get_rgb_array(self) -> np.ndarray:
        """Return the rendered frame as (H, W, 3) uint8 numpy array."""
        if not self._initialised:
            self.init()
        return pygame.surfarray.array3d(self._surface).transpose(1, 0, 2)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _camera_x(self, gs: GameState) -> int:
        half_w = SCREEN_W // 2
        cx = int(gs.mario.cx) - half_w
        max_cam = gs.tilemap.pixel_width - SCREEN_W
        return max(0, min(cx, max_cam))

    # ── Tile drawing ──────────────────────────────────────────────────────────

    def _draw_tiles(self, surf: "pygame.Surface",
                    tm: TileMap, cam_x: int) -> None:
        col_lo = cam_x // TILE_SIZE
        col_hi = min(col_lo + SCREEN_TILES_W + 1, tm.width)
        row_hi = tm.height

        for row in range(row_hi):
            for col in range(col_lo, col_hi):
                t = TileType(int(tm.tiles[row, col]))
                if t == TileType.AIR:
                    continue
                color = TILE_COLORS.get(t, COL_GROUND)
                rx = col * TILE_SIZE - cam_x
                ry = row * TILE_SIZE
                rect = pygame.Rect(rx, ry, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(surf, color, rect)
                # subtle border for visual depth
                pygame.draw.rect(surf, _darken(color, 0.7), rect, 1)

                # question block '?' text
                if t == TileType.QUESTION:
                    _blit_char(surf, '?', rx, ry, COL_HUD_BG, TILE_SIZE)

                # pipe highlight stripe
                if t in (TileType.PIPE_TOP_LEFT, TileType.PIPE_LEFT):
                    stripe = pygame.Rect(rx + 4, ry + 4,
                                        TILE_SIZE // 5, TILE_SIZE - 8)
                    pygame.draw.rect(surf, (120, 220, 100), stripe)

    # ── Mario drawing ─────────────────────────────────────────────────────────

    def _draw_mario(self, surf: "pygame.Surface",
                    mario: Mario, cam_x: int, step: int) -> None:
        if mario.state == MarioState.DEAD:
            # blinking red silhouette
            if (step // 4) % 2 == 0:
                return
        if mario.invincible_frames > 0 and (step // 3) % 2 == 0:
            return  # blink when invincible

        rx = int(mario.x) - cam_x
        ry = int(mario.y)
        w, h = mario.w, mario.h

        body_color = COL_MARIO_BIG if mario.state.name == 'BIG' else COL_MARIO
        hat_h = max(6, h // 4)

        # body
        pygame.draw.rect(surf, body_color,
                         pygame.Rect(rx, ry + hat_h, w, h - hat_h))
        # hat
        hat_w = int(w * 0.8)
        hat_x = rx + (w - hat_w) // 2 if mario.facing_right else rx
        pygame.draw.rect(surf, (180, 20, 0),
                         pygame.Rect(hat_x, ry, hat_w, hat_h))
        # eyes (simple dot)
        eye_x = rx + (w * 3 // 4 if mario.facing_right else w // 4)
        eye_y = ry + hat_h + 2
        pygame.draw.circle(surf, (0, 0, 0), (eye_x, eye_y), 2)

    # ── Goomba drawing ────────────────────────────────────────────────────────

    def _draw_goombas(self, surf: "pygame.Surface",
                      goombas: list, cam_x: int) -> None:
        for g in goombas:
            if g.state == GoombaState.DEAD:
                continue
            rx = int(g.x) - cam_x
            ry = int(g.y)
            if g.state == GoombaState.SQUISHED:
                # flat squished goomba
                sq_h = max(4, g.h // 3)
                pygame.draw.rect(surf, COL_GOOMBA_DEAD,
                                 pygame.Rect(rx, ry + g.h - sq_h, g.w, sq_h))
            else:
                # body
                pygame.draw.rect(surf, COL_GOOMBA,
                                 pygame.Rect(rx, ry, g.w, g.h))
                # angry eyebrow lines
                ex = rx + g.w // 4
                pygame.draw.line(surf, (0, 0, 0),
                                 (ex, ry + 5), (ex + g.w // 4, ry + 9), 2)
                ex2 = rx + g.w * 3 // 4
                pygame.draw.line(surf, (0, 0, 0),
                                 (ex2, ry + 9), (ex2 - g.w // 4, ry + 5), 2)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, surf: "pygame.Surface", gs: GameState) -> None:
        hud_h = 28
        hud = pygame.Surface((SCREEN_W, hud_h))
        hud.fill(COL_HUD_BG)
        m = gs.mario

        text = (f"  SCORE:{m.score:06d}  COINS:{m.coins:02d}  "
                f"KILLS:{m.kills:02d}  STEP:{gs.step_count:05d}  "
                f"X:{int(m.x):04d}  STATE:{m.state.name}")
        label = self._font.render(text, True, COL_WHITE)
        hud.blit(label, (4, 6))
        surf.blit(hud, (0, SCREEN_H - hud_h))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _darken(color: Tuple, factor: float) -> Tuple:
    return tuple(int(c * factor) for c in color)


def _blit_char(surf: "pygame.Surface", char: str,
               tx: int, ty: int, color, size: int) -> None:
    try:
        font = pygame.font.SysFont("monospace", size - 4, bold=True)
        label = font.render(char, True, color)
        surf.blit(label, (tx + (size - label.get_width()) // 2,
                          ty + (size - label.get_height()) // 2))
    except Exception:
        pass
