"""
renderer.py – Pygame-based renderer for the Mario RL environment.

Uses coloured rectangles by default.  If PNG sprites are placed in
  mario_rl_env/sprites/
they are loaded automatically and the coloured-rectangle fallback is
kept for any sprite that is missing.

Run  tools/download_smb_sprites.py  to populate the sprites folder.

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
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    from .constants import (
        TILE_SIZE, SCREEN_W, SCREEN_H, SCREEN_TILES_W, SCREEN_TILES_H,
        COL_SKY, COL_GROUND, COL_BRICK, COL_QUESTION, COL_QUESTION_USED, COL_COIN,
        COL_PIPE_DARK, COL_PIPE_LITE, COL_PIPE_RED_DARK, COL_PIPE_RED_LITE,
        COL_PIPE_BLUE_DARK, COL_PIPE_BLUE_LITE, COL_MARIO, COL_MARIO_BIG,
        COL_GOOMBA, COL_GOOMBA_DEAD, COL_HUD_BG, COL_WHITE, COL_YELLOW, COL_RED,
    )
    from .tilemap import TileMap, TileType
    from .entities import Mario, Goomba, MarioState, GoombaState
    from .game_state import GameState
except ImportError:
    from constants import (
        TILE_SIZE, SCREEN_W, SCREEN_H, SCREEN_TILES_W, SCREEN_TILES_H,
        COL_SKY, COL_GROUND, COL_BRICK, COL_QUESTION, COL_QUESTION_USED, COL_COIN,
        COL_PIPE_DARK, COL_PIPE_LITE, COL_PIPE_RED_DARK, COL_PIPE_RED_LITE,
        COL_PIPE_BLUE_DARK, COL_PIPE_BLUE_LITE, COL_MARIO, COL_MARIO_BIG,
        COL_GOOMBA, COL_GOOMBA_DEAD, COL_HUD_BG, COL_WHITE, COL_YELLOW, COL_RED,
    )
    from tilemap import TileMap, TileType
    from entities import Mario, Goomba, MarioState, GoombaState
    from game_state import GameState


# ── Tile colour lookup ────────────────────────────────────────────────────────

TILE_COLORS = {
    TileType.COIN:           COL_COIN,
    TileType.GROUND:         COL_GROUND,
    TileType.BRICK:          COL_BRICK,
    TileType.QUESTION:       COL_QUESTION,
    TileType.QUESTION_USED:  COL_QUESTION_USED,
    TileType.PIPE_TOP_LEFT:  COL_PIPE_DARK,
    TileType.PIPE_TOP_RIGHT: COL_PIPE_DARK,
    TileType.PIPE_LEFT:      COL_PIPE_LITE,
    TileType.PIPE_RIGHT:     COL_PIPE_LITE,
    TileType.PIPE_RED_TOP_LEFT:   COL_PIPE_RED_DARK,
    TileType.PIPE_RED_TOP_RIGHT:  COL_PIPE_RED_DARK,
    TileType.PIPE_RED_LEFT:       COL_PIPE_RED_LITE,
    TileType.PIPE_RED_RIGHT:      COL_PIPE_RED_LITE,
    TileType.PIPE_BLUE_TOP_LEFT:  COL_PIPE_BLUE_DARK,
    TileType.PIPE_BLUE_TOP_RIGHT: COL_PIPE_BLUE_DARK,
    TileType.PIPE_BLUE_LEFT:      COL_PIPE_BLUE_LITE,
    TileType.PIPE_BLUE_RIGHT:     COL_PIPE_BLUE_LITE,
}


# ── Sprite loader ─────────────────────────────────────────────────────────────

class _SpriteLoader:
    """
    Loads individual PNG sprites from the ``sprites/`` folder next to this
    file.  Returns scaled ``pygame.Surface`` objects, or ``None`` when a file
    is absent (renderer falls back to rectangles).

    Expected filenames (all PNG, transparent background):
      mario_small_stand.png  mario_small_walk1.png  mario_small_walk2.png
      mario_small_jump.png   mario_big_stand.png    mario_big_walk1.png
      mario_big_walk2.png    mario_big_jump.png     mario_dead.png
      goomba_walk1.png       goomba_walk2.png       goomba_squish.png
      tile_ground.png        tile_brick.png         tile_question.png
      tile_question_used.png tile_coin.png
      tile_pipe_tl.png       tile_pipe_tr.png       tile_pipe_l.png
      tile_pipe_r.png        tile_pipe_red_tl.png   tile_pipe_red_tr.png
      tile_pipe_red_l.png    tile_pipe_red_r.png    tile_pipe_blue_tl.png
      tile_pipe_blue_tr.png  tile_pipe_blue_l.png   tile_pipe_blue_r.png

    """

    _SPRITES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites")

    def __init__(self) -> None:
        # cache: (name, w, h) → Surface | None
        self._cache: Dict[Tuple[str, int, int], Optional["pygame.Surface"]] = {}

    def get(self, name: str, w: int, h: int) -> Optional["pygame.Surface"]:
        """Return a surface scaled to (w, h), or None if the sprite is missing."""
        key = (name, w, h)
        if key not in self._cache:
            self._cache[key] = self._load(name, w, h)
        return self._cache[key]

    def _load(self, name: str, w: int, h: int) -> Optional["pygame.Surface"]:
        path = os.path.join(self._SPRITES_DIR, f"{name}.png")
        if not os.path.isfile(path):
            return None
        try:
            raw = pygame.image.load(path).convert_alpha()
            if raw.get_size() == (w, h):
                return raw
            return pygame.transform.scale(raw, (w, h))
        except Exception:
            return None

    @property
    def available(self) -> bool:
        """True if the sprites directory exists and contains at least one PNG."""
        if not os.path.isdir(self._SPRITES_DIR):
            return False
        return any(f.endswith(".png") for f in os.listdir(self._SPRITES_DIR))


# Tile type → sprite name mapping
_TILE_SPRITE: Dict["TileType", str] = {}   # filled after TileType is imported


def _init_tile_sprite_map() -> None:
    global _TILE_SPRITE
    _TILE_SPRITE = {
        TileType.GROUND:             "tile_ground",
        TileType.BRICK:              "tile_brick",
        TileType.QUESTION:           "tile_question",
        TileType.QUESTION_USED:      "tile_question_used",
        TileType.COIN:               "tile_coin",
        TileType.PIPE_TOP_LEFT:      "tile_pipe_tl",
        TileType.PIPE_TOP_RIGHT:     "tile_pipe_tr",
        TileType.PIPE_LEFT:          "tile_pipe_l",
        TileType.PIPE_RIGHT:         "tile_pipe_r",
        TileType.PIPE_RED_TOP_LEFT:  "tile_pipe_red_tl",
        TileType.PIPE_RED_TOP_RIGHT: "tile_pipe_red_tr",
        TileType.PIPE_RED_LEFT:      "tile_pipe_red_l",
        TileType.PIPE_RED_RIGHT:     "tile_pipe_red_r",
        TileType.PIPE_BLUE_TOP_LEFT: "tile_pipe_blue_tl",
        TileType.PIPE_BLUE_TOP_RIGHT:"tile_pipe_blue_tr",
        TileType.PIPE_BLUE_LEFT:     "tile_pipe_blue_l",
        TileType.PIPE_BLUE_RIGHT:    "tile_pipe_blue_r",
    }


# ── Renderer ──────────────────────────────────────────────────────────────────

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
        self._font_overlay: Optional["pygame.font.Font"] = None
        self._clock    = None
        self._window_scale: float = 1.7
        self._hud_margin: int = 8
        self._hud_height: int = 56
        self._game_view_size: Tuple[int, int] = (SCREEN_W, SCREEN_H)
        self._last_gs: Optional[GameState] = None
        self._hud_extra_line: str = ""
        self._overlay_box_lines: List[str] = []
        self._initialised = False
        self._sprites = _SpriteLoader()

    def set_hud_extra_line(self, text: str) -> None:
        """Set optional HUD line for caller-provided runtime diagnostics."""
        self._hud_extra_line = str(text or "")

    def set_overlay_box_lines(self, lines: List[str]) -> None:
        """Set top-right in-game overlay text lines."""
        self._overlay_box_lines = [str(line) for line in (lines or []) if str(line).strip()]

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
        self._font_overlay = pygame.font.SysFont("consolas", 18, bold=True)
        self._clock = pygame.time.Clock()
        _init_tile_sprite_map()
        self._initialised = True

    def close(self) -> None:
        if self._initialised and _PYGAME_AVAILABLE and pygame.get_init():
            pygame.quit()
        self._initialised = False
        self._screen = None
        self._surface = None
        self._font = None
        self._font_small = None
        self._font_overlay = None
        self._clock = None

    # ── Main draw entry ───────────────────────────────────────────────────────

    def draw(self, gs: GameState) -> None:
        """Render the current game state onto the internal surface."""
        if not self._initialised:
            self.init()

        surf = self._surface
        surf.fill(COL_SKY)
        #self._draw_backdrop(surf, gs)

        # viewport offset: keep Mario centred horizontally
        cam_x = self._camera_x(gs)

        self._draw_tiles(surf, gs.tilemap, cam_x)
        self._draw_goombas(surf, gs.goombas, cam_x)
        self._draw_mario(surf, gs.mario, cam_x, gs.step_count)
        self._last_gs = gs

    def flip(self, fps: int = 60) -> None:
        """Blit surface → screen and tick clock."""
        if not self._initialised or not _PYGAME_AVAILABLE or not pygame.get_init():
            return
        if not self.headless and self._screen is not None:
            screen_w, screen_h = self._screen.get_size()
            game_h = max(1, screen_h - (self._hud_margin + self._hud_height + self._hud_margin))
            game_area = pygame.Rect(0, 0, screen_w, game_h)

            # Use fit-downscale when window is smaller than native resolution.
            # Otherwise, use integer upscaling to keep pixel-art edges crisp.
            if game_area.width < SCREEN_W or game_area.height < SCREEN_H:
                fit_scale = min(game_area.width / SCREEN_W, game_area.height / SCREEN_H)
                target_w = max(1, int(SCREEN_W * fit_scale))
                target_h = max(1, int(SCREEN_H * fit_scale))
            else:
                scale_x = max(1, game_area.width // SCREEN_W)
                scale_y = max(1, game_area.height // SCREEN_H)
                pixel_scale = max(1, min(scale_x, scale_y))
                target_w = SCREEN_W * pixel_scale
                target_h = SCREEN_H * pixel_scale
            self._game_view_size = (target_w, target_h)

            self._screen.fill((0, 0, 0), game_area)
            if (target_w, target_h) == (SCREEN_W, SCREEN_H):
                scaled = self._surface
            else:
                scaled = pygame.transform.scale(self._surface, (target_w, target_h))

            dest_x = (game_area.width - target_w) // 2
            dest_y = (game_area.height - target_h) // 2
            self._screen.blit(scaled, (dest_x, dest_y))

            self._draw_overlay_box(self._screen)
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

    # ── Backdrop ─────────────────────────────────────────────────────────────

    def _draw_backdrop(self, surf: "pygame.Surface", gs: GameState) -> None:
        horizon = int(SCREEN_H * 0.65)
        hill_dark = (67, 137, 68)
        hill_light = (135, 206, 127)

        for y in range(0, horizon + 1):
            t = max(0.0, min(1.0, y / max(1, horizon)))
            r = 75
            g = 195
            b = 255
            pygame.draw.line(surf, (r, g, b), (0, y), (SCREEN_W, y))

        hill_y = horizon + 28
        for x in range(0, SCREEN_W + 1, 48):
            center = x + (gs.mario.cx * 0.08) % 48
            pygame.draw.ellipse(surf, hill_light, (center - 60, hill_y, 140, 80))
            pygame.draw.ellipse(surf, hill_dark, (center - 22, hill_y + 16, 120, 70))

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
                rx = col * TILE_SIZE - cam_x
                ry = row * TILE_SIZE
                rect = pygame.Rect(rx, ry, TILE_SIZE, TILE_SIZE)

                # ── sprite path ──
                sprite_name = _TILE_SPRITE.get(t)
                if sprite_name:
                    spr = self._sprites.get(sprite_name, TILE_SIZE, TILE_SIZE)
                    if spr is not None:
                        surf.blit(spr, (rx, ry))
                        continue

                # ── rectangle fallback ──
                color = TILE_COLORS.get(t, COL_GROUND)
                pygame.draw.rect(surf, color, rect)
                pygame.draw.rect(surf, _darken(color, 0.7), rect, 1)

                if t == TileType.QUESTION:
                    _blit_char(surf, '?', rx, ry, COL_HUD_BG, TILE_SIZE)

                if t == TileType.COIN:
                    coin_r = max(4, TILE_SIZE // 4)
                    pygame.draw.circle(
                        surf,
                        (255, 245, 90),
                        (rx + TILE_SIZE // 2, ry + TILE_SIZE // 2),
                        coin_r,
                    )

                if t in (
                    TileType.PIPE_TOP_LEFT,
                    TileType.PIPE_LEFT,
                    TileType.PIPE_RED_TOP_LEFT,
                    TileType.PIPE_RED_LEFT,
                    TileType.PIPE_BLUE_TOP_LEFT,
                    TileType.PIPE_BLUE_LEFT,
                ):
                    stripe = pygame.Rect(rx + 4, ry + 4,
                                        TILE_SIZE // 5, TILE_SIZE - 8)
                    stripe_color = (120, 220, 100)
                    if t in (TileType.PIPE_RED_TOP_LEFT, TileType.PIPE_RED_LEFT):
                        stripe_color = (255, 170, 170)
                    elif t in (TileType.PIPE_BLUE_TOP_LEFT, TileType.PIPE_BLUE_LEFT):
                        stripe_color = (180, 210, 255)
                    pygame.draw.rect(surf, stripe_color, stripe)

    # ── Mario drawing ─────────────────────────────────────────────────────────

    def _draw_mario(self, surf: "pygame.Surface",
                    mario: Mario, cam_x: int, step: int) -> None:
        if mario.state == MarioState.DEAD:
            if (step // 4) % 2 == 0:
                return
        if mario.invincible_frames > 0 and (step // 3) % 2 == 0:
            return

        rx = int(mario.x) - cam_x
        ry = int(mario.y)
        w, h = mario.w, mario.h

        # ── pick sprite name ──
        if mario.state == MarioState.DEAD:
            sprite_name = "mario_dead"
        else:
            prefix = "mario_big" if mario.state == MarioState.BIG else "mario_small"
            if not mario.on_ground:
                sprite_name = f"{prefix}_jump"
            elif abs(mario.vx) > 0.3:
                frame = (step // 6) % 2 + 1
                sprite_name = f"{prefix}_walk{frame}"
            else:
                sprite_name = f"{prefix}_stand"

        spr = self._sprites.get(sprite_name, w, h)
        if spr is not None:
            if not mario.facing_right:
                spr = pygame.transform.flip(spr, True, False)
            surf.blit(spr, (rx, ry))
            return

        # ── rectangle fallback ──
        body_color = COL_MARIO_BIG if mario.state == MarioState.BIG else COL_MARIO
        hat_h = max(6, h // 4)
        pygame.draw.rect(surf, body_color,
                         pygame.Rect(rx, ry + hat_h, w, h - hat_h))
        hat_w = int(w * 0.8)
        hat_x = rx + (w - hat_w) // 2 if mario.facing_right else rx
        pygame.draw.rect(surf, (180, 20, 0),
                         pygame.Rect(hat_x, ry, hat_w, hat_h))
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
                spr = self._sprites.get("goomba_squish", g.w, max(4, g.h // 3))
                if spr is not None:
                    surf.blit(spr, (rx, ry + g.h - max(4, g.h // 3)))
                    continue
                sq_h = max(4, g.h // 3)
                pygame.draw.rect(surf, COL_GOOMBA_DEAD,
                                 pygame.Rect(rx, ry + g.h - sq_h, g.w, sq_h))
            else:
                frame = (getattr(g, "_step", 0) // 10) % 2 + 1
                spr = self._sprites.get(f"goomba_walk{frame}", g.w, g.h)
                if spr is not None:
                    surf.blit(spr, (rx, ry))
                    continue
                # fallback
                pygame.draw.rect(surf, COL_GOOMBA,
                                 pygame.Rect(rx, ry, g.w, g.h))
                ex = rx + g.w // 4
                pygame.draw.line(surf, (0, 0, 0),
                                 (ex, ry + 5), (ex + g.w // 4, ry + 9), 2)
                ex2 = rx + g.w * 3 // 4
                pygame.draw.line(surf, (0, 0, 0),
                                 (ex2, ry + 9), (ex2 - g.w // 4, ry + 5), 2)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_overlay_box(self, screen: "pygame.Surface") -> None:
        if not self._overlay_box_lines:
            return

        font = self._font_overlay or self._font
        if font is None:
            return

        screen_w, _ = screen.get_size()
        game_h = self._game_view_size[1]
        pad = 12

        labels = [font.render(line, True, COL_WHITE) for line in self._overlay_box_lines[:6]]
        if not labels:
            return

        text_h = sum(lbl.get_height() for lbl in labels)
        line_gap = 6
        box_h = text_h + line_gap * (len(labels) - 1) + 2 * pad
        box_w = max(lbl.get_width() for lbl in labels) + 2 * pad

        x = max(8, screen_w - box_w - 12)
        y = 12

        # Keep box inside game viewport (above bottom HUD panel).
        if y + box_h > game_h - 8:
            y = max(8, game_h - box_h - 8)

        rect = pygame.Rect(x, y, box_w, box_h)
        pygame.draw.rect(screen, (18, 20, 34), rect, border_radius=8)
        pygame.draw.rect(screen, (230, 235, 255), rect, width=2, border_radius=8)

        cursor_y = y + pad
        for label in labels:
            screen.blit(label, (x + pad, cursor_y))
            cursor_y += label.get_height() + line_gap

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
        pygame.draw.rect(screen, (8, 16, 28), panel_rect, border_radius=8)
        pygame.draw.rect(screen, (130, 205, 255), panel_rect, width=2, border_radius=8)

        m = gs.mario
        level_goal_x = max(1, (gs.tilemap.width - 4) * TILE_SIZE)
        progress_pct = max(0.0, min(100.0, (m.x / level_goal_x) * 100.0))

        header = self._font.render("MARIO", True, (255, 225, 120))
        score_label = self._font_small.render(f"SCORE {m.score:06d}", True, COL_WHITE)
        coin_label = self._font_small.render(f"COINS {m.coins:02d}", True, (255, 236, 150))
        world_label = self._font_small.render(f"WORLD {max(1, int(m.x // TILE_SIZE) + 1):02d}", True, COL_WHITE)
        status_label = self._font_small.render(f"STATE {m.state.name}", True, (180, 235, 255))
        step_label = self._font_small.render(f"STEP {gs.step_count:05d}", True, (220, 236, 255))

        screen.blit(header, (panel_x + 12, panel_y + 6))
        screen.blit(score_label, (panel_x + 120, panel_y + 8))
        screen.blit(coin_label, (panel_x + 260, panel_y + 8))
        screen.blit(world_label, (panel_x + 370, panel_y + 8))
        screen.blit(step_label, (panel_x + 480, panel_y + 8))
        screen.blit(status_label, (panel_x + 620, panel_y + 8))

        progress_bar_x = panel_x + 12
        progress_bar_y = panel_y + 30
        progress_bar_w = max(120, panel_w - 180)
        progress_bar = pygame.Rect(progress_bar_x, progress_bar_y, progress_bar_w, 8)
        pygame.draw.rect(screen, (30, 46, 72), progress_bar, border_radius=4)
        fill_w = int(progress_bar_w * (progress_pct / 100.0))
        pygame.draw.rect(screen, (120, 220, 120), pygame.Rect(progress_bar_x, progress_bar_y, fill_w, 8), border_radius=4)

        line_2 = (
            f"X:{int(m.x):04d}  Y:{int(m.y):03d}  VX:{m.vx:+5.2f}  VY:{m.vy:+6.2f}  "
            f"G:{int(m.on_ground)}  GOOMBAS:{len(gs.goombas):02d}  "
            f"R:{gs.total_reward:+8.2f}  PROG:{progress_pct:5.1f}%"
        )
        label_2 = self._font_small.render(line_2, True, COL_YELLOW)
        screen.blit(label_2, (panel_x + 12, panel_y + 42))

        if self._hud_extra_line:
            extra = self._font_small.render(self._hud_extra_line, True, COL_WHITE)
            screen.blit(extra, (panel_x + 12, panel_y + 58))


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
    raw = os.environ.get("MARIO_WINDOW_SCALE", "0.5")
    try:
        value = float(raw)
    except ValueError:
        return 0.5
    return max(1.0, min(value, 4.0))
