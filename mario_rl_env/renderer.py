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
import os
import math
from typing import Optional, Tuple

import numpy as np

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    from .constants import (
        TILE_SIZE, SCREEN_W, SCREEN_H, SCREEN_TILES_W, SCREEN_TILES_H,
        COL_SKY, COL_GROUND, COL_BRICK, COL_QUESTION, COL_QUESTION_USED,
        COL_PIPE_DARK, COL_PIPE_LITE, COL_MARIO, COL_MARIO_BIG,
        COL_GOOMBA, COL_GOOMBA_DEAD, COL_HUD_BG, COL_WHITE, COL_YELLOW, COL_RED,
    )
    from .tilemap import TileMap, TileType
    from .entities import Mario, Goomba, MarioState, GoombaState
    from .game_state import GameState
except ImportError:
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
        self._font_small: Optional["pygame.font.Font"] = None
        self._clock    = None
        self._window_scale: float = 1.7
        self._hud_margin: int = 8
        self._hud_height: int = 56
        self._game_view_size: Tuple[int, int] = (SCREEN_W, SCREEN_H)
        self._last_gs: Optional[GameState] = None
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
        self._window_scale = _read_window_scale()
        if not self.headless:
            window_w = max(1, int(SCREEN_W * self._window_scale))
            game_h = max(1, int(SCREEN_H * self._window_scale))
            window_h = game_h + self._hud_margin + self._hud_height + self._hud_margin
            self._screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
            pygame.display.set_caption("Mario RL")
        self._surface = pygame.Surface((SCREEN_W, SCREEN_H))
        self._font = pygame.font.SysFont("consolas", 15, bold=True)
        self._font_small = pygame.font.SysFont("consolas", 13)
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
        self._last_gs = gs

    def flip(self, fps: int = 60) -> None:
        """Blit surface → screen and tick clock."""
        if not self._initialised:
            return
        if not self.headless and self._screen is not None:
            screen_w, screen_h = self._screen.get_size()
            game_h = max(1, screen_h - (self._hud_margin + self._hud_height + self._hud_margin))
            game_target = (screen_w, game_h)
            self._game_view_size = game_target

            if game_target == (SCREEN_W, SCREEN_H):
                self._screen.blit(self._surface, (0, 0))
            else:
                scaled = pygame.transform.scale(self._surface, game_target)
                self._screen.blit(scaled, (0, 0))

            self._draw_hud_panel(self._screen)
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

    def _draw_hud_panel(self, screen: "pygame.Surface") -> None:
        gs = self._last_gs
        if gs is None:
            return

        screen_w, _ = screen.get_size()
        game_h = self._game_view_size[1]
        panel_x = self._hud_margin
        panel_y = game_h + self._hud_margin
        panel_w = max(1, screen_w - 2 * self._hud_margin)
        panel_h = self._hud_height

        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(screen, COL_HUD_BG, panel_rect, border_radius=6)
        pygame.draw.rect(screen, _darken(COL_WHITE, 0.4), panel_rect, width=1, border_radius=6)

        m = gs.mario
        level_goal_x = max(1, (gs.tilemap.width - 4) * TILE_SIZE)
        progress_pct = max(0.0, min(100.0, (m.x / level_goal_x) * 100.0))

        line_1 = (
            f"SCORE:{m.score:06d}  COINS:{m.coins:02d}  KILLS:{m.kills:02d}  "
            f"STEP:{gs.step_count:05d}  STATE:{m.state.name}"
        )
        line_2 = (
            f"X:{int(m.x):04d}  Y:{int(m.y):03d}  VX:{m.vx:+5.2f}  VY:{m.vy:+6.2f}  "
            f"G:{int(m.on_ground)}  GOOMBAS:{len(gs.goombas):02d}  "
            f"R:{gs.total_reward:+8.2f}  PROG:{progress_pct:5.1f}%"
        )

        label_1 = self._font.render(line_1, True, COL_WHITE)
        label_2 = self._font_small.render(line_2, True, COL_YELLOW)
        screen.blit(label_1, (panel_x + 10, panel_y + 7))
        screen.blit(label_2, (panel_x + 10, panel_y + 30))


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


def _read_window_scale() -> float:
    """Read optional renderer window scale from env var MARIO_WINDOW_SCALE."""
    raw = os.environ.get("MARIO_WINDOW_SCALE", "1.7")
    try:
        value = float(raw)
    except ValueError:
        return 1.7
    return max(1.0, min(value, 4.0))
