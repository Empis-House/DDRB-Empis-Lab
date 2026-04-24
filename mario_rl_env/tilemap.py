"""
tilemap.py – Parse a comma-separated .txt tile map into a numpy array.

Tile legend (characters in the .txt file):
  -   AIR            (passable, empty)
  X   GROUND         (solid)
  S   BRICK          (solid, breakable by big Mario)
  Q   QUESTION BLOCK (solid, yields coin on first hit)
  ?   QUESTION BLOCK (same)
  E   ENEMY SPAWN    (Goomba spawn position; tile is AIR)
  <   PIPE_TOP_LEFT  (solid)
  >   PIPE_TOP_RIGHT (solid)
  [   PIPE_LEFT      (solid)
  ]   PIPE_RIGHT     (solid)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Set
import numpy as np


class TileType(IntEnum):
    AIR            = 0
    GROUND         = 1
    BRICK          = 2
    QUESTION       = 3
    QUESTION_USED  = 4
    PIPE_TOP_LEFT  = 5
    PIPE_TOP_RIGHT = 6
    PIPE_LEFT      = 7
    PIPE_RIGHT     = 8


# Character → TileType mapping
CHAR_TO_TILE: dict[str, TileType] = {
    '-': TileType.AIR,
    'X': TileType.GROUND,
    'S': TileType.BRICK,
    'Q': TileType.QUESTION,
    '?': TileType.QUESTION,
    'E': TileType.AIR,          # enemy spawn; handled separately
    '<': TileType.PIPE_TOP_LEFT,
    '>': TileType.PIPE_TOP_RIGHT,
    '[': TileType.PIPE_LEFT,
    ']': TileType.PIPE_RIGHT,
}

SOLID_TILES: Set[TileType] = {
    TileType.GROUND,
    TileType.BRICK,
    TileType.QUESTION,
    TileType.QUESTION_USED,
    TileType.PIPE_TOP_LEFT,
    TileType.PIPE_TOP_RIGHT,
    TileType.PIPE_LEFT,
    TileType.PIPE_RIGHT,
}

# Tiles that yield a coin when hit from below
HITTABLE_TILES: Set[TileType] = {TileType.QUESTION}


@dataclass
class TileMap:
    """Immutable (mostly) description of the level geometry."""
    tiles: np.ndarray                         # shape (rows, cols), dtype int8
    enemy_spawns: List[Tuple[int, int]]       # [(col, row), ...]
    width: int                                # cols
    height: int                               # rows
    # mutable state for question blocks already activated
    used_blocks: Set[Tuple[int, int]] = field(default_factory=set)

    @property
    def pixel_width(self) -> int:
        from constants import TILE_SIZE
        return self.width * TILE_SIZE

    @property
    def pixel_height(self) -> int:
        from constants import TILE_SIZE
        return self.height * TILE_SIZE

    def clone(self) -> "TileMap":
        """Return a copy suitable for a new episode."""
        return TileMap(
            tiles=self.tiles.copy(),
            enemy_spawns=list(self.enemy_spawns),
            width=self.width,
            height=self.height,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def get(self, col: int, row: int) -> TileType:
        """Return tile type, treating out-of-bounds sensibly."""
        if row >= self.height:
            return TileType.GROUND   # solid floor below level
        if col < 0 or col >= self.width or row < 0:
            return TileType.AIR
        return TileType(int(self.tiles[row, col]))

    def is_solid(self, col: int, row: int) -> bool:
        return self.get(col, row) in SOLID_TILES

    def hit_block(self, col: int, row: int) -> bool:
        """
        Hit a block from below.  Returns True if a coin was given.
        Converts QUESTION → QUESTION_USED.
        """
        t = self.get(col, row)
        if t == TileType.QUESTION:
            self.tiles[row, col] = TileType.QUESTION_USED
            self.used_blocks.add((col, row))
            return True
        return False

    def break_brick(self, col: int, row: int) -> bool:
        """
        Big Mario breaks a BRICK from below.  Returns True if broken.
        """
        t = self.get(col, row)
        if t == TileType.BRICK:
            self.tiles[row, col] = TileType.AIR
            return True
        return False


def parse_tilemap(filepath: str) -> TileMap:
    """
    Load a comma-separated .txt tile map from *filepath*.

    Each row is one line; each cell is separated by a comma.
    Rows need not all be the same length – shorter rows are padded with AIR.
    """
    with open(filepath, "r") as fh:
        raw_lines = fh.readlines()

    rows: List[List[int]] = []
    enemy_spawns: List[Tuple[int, int]] = []

    for row_idx, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        cells = line.split(",")
        row_tiles: List[int] = []
        for col_idx, cell in enumerate(cells):
            cell = cell.strip()
            if cell == "E":
                enemy_spawns.append((col_idx, row_idx))
                row_tiles.append(int(TileType.AIR))
            else:
                row_tiles.append(int(CHAR_TO_TILE.get(cell, TileType.AIR)))
        rows.append(row_tiles)

    if not rows:
        raise ValueError(f"Empty tilemap: {filepath}")

    max_cols = max(len(r) for r in rows)
    for r in rows:
        r.extend([int(TileType.AIR)] * (max_cols - len(r)))

    tiles = np.array(rows, dtype=np.int8)
    return TileMap(
        tiles=tiles,
        enemy_spawns=enemy_spawns,
        width=max_cols,
        height=len(rows),
    )
