"""
tilemap.py – Parse a comma-separated .txt tile map into a numpy array.

Tile legend (characters in the .txt file):
  -   AIR            (passable, empty)
    o   COIN           (collectable, passable)
  X   GROUND         (solid)
    S/B/b BRICK        (solid, breakable by big Mario)
  Q   QUESTION BLOCK (solid, yields coin on first hit)
  ?   QUESTION BLOCK (same)
  E   ENEMY SPAWN    (Goomba spawn position; tile is AIR)
  <   PIPE_TOP_LEFT  (solid)
  >   PIPE_TOP_RIGHT (solid)
  [   PIPE_LEFT      (solid)
  ]   PIPE_RIGHT     (solid)
    r/R/u/U RED PIPE   (solid)
    v/V/w/W BLUE PIPE  (solid)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Set, Union
import numpy as np
import pandas as pd


class TileType(IntEnum):
    AIR            = 0
    COIN           = 1
    GROUND         = 2
    BRICK          = 3
    QUESTION       = 4
    QUESTION_USED  = 5
    PIPE_TOP_LEFT  = 6
    PIPE_TOP_RIGHT = 7
    PIPE_LEFT      = 8
    PIPE_RIGHT     = 9
    PIPE_RED_TOP_LEFT   = 10
    PIPE_RED_TOP_RIGHT  = 11
    PIPE_RED_LEFT       = 12
    PIPE_RED_RIGHT      = 13
    PIPE_BLUE_TOP_LEFT  = 14
    PIPE_BLUE_TOP_RIGHT = 15
    PIPE_BLUE_LEFT      = 16
    PIPE_BLUE_RIGHT     = 17


# Character → TileType mapping
CHAR_TO_TILE: dict[str, TileType] = {
    '-': TileType.AIR,
    'o': TileType.COIN,
    'X': TileType.GROUND,
    'S': TileType.BRICK,
    'B': TileType.BRICK,
    'b': TileType.BRICK,
    'Q': TileType.QUESTION,
    '?': TileType.QUESTION,
    'E': TileType.AIR,          # enemy spawn; handled separately
    '<': TileType.PIPE_TOP_LEFT,
    '>': TileType.PIPE_TOP_RIGHT,
    '[': TileType.PIPE_LEFT,
    ']': TileType.PIPE_RIGHT,
    'r': TileType.PIPE_RED_TOP_LEFT,
    'R': TileType.PIPE_RED_TOP_RIGHT,
    'u': TileType.PIPE_RED_LEFT,
    'U': TileType.PIPE_RED_RIGHT,
    'v': TileType.PIPE_BLUE_TOP_LEFT,
    'V': TileType.PIPE_BLUE_TOP_RIGHT,
    'w': TileType.PIPE_BLUE_LEFT,
    'W': TileType.PIPE_BLUE_RIGHT,
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
    TileType.PIPE_RED_TOP_LEFT,
    TileType.PIPE_RED_TOP_RIGHT,
    TileType.PIPE_RED_LEFT,
    TileType.PIPE_RED_RIGHT,
    TileType.PIPE_BLUE_TOP_LEFT,
    TileType.PIPE_BLUE_TOP_RIGHT,
    TileType.PIPE_BLUE_LEFT,
    TileType.PIPE_BLUE_RIGHT,
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
        try:
            from .constants import TILE_SIZE
        except ImportError:
            from constants import TILE_SIZE
        return self.width * TILE_SIZE

    @property
    def pixel_height(self) -> int:
        try:
            from .constants import TILE_SIZE
        except ImportError:
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

    def collect_coin(self, col: int, row: int) -> bool:
        """Collect a free coin tile (o). Returns True when a coin was collected."""
        t = self.get(col, row)
        if t == TileType.COIN:
            self.tiles[row, col] = TileType.AIR
            return True
        return False


def parse_tilemap(source: Union[str, pd.DataFrame, TileMap]) -> TileMap:
    """
    Load a tile map from one of:
      - filepath (str): comma-separated .txt file path
      - pandas.DataFrame: grid of tile characters (rows × cols)
      - TileMap: returned as-is (clone)

    Backward-compatible: when passed a string path this behaves like before.
    """
    # If already a TileMap, return it directly
    if isinstance(source, TileMap):
        return source

    rows: List[List[int]] = []
    enemy_spawns: List[Tuple[int, int]] = []

    # If a DataFrame, iterate its rows
    if isinstance(source, pd.DataFrame):
        df: pd.DataFrame = source
        for row_idx in range(df.shape[0]):
            cells = [str(x).strip() for x in df.iloc[row_idx].tolist()]
            if not any(cells):
                continue
            row_tiles: List[int] = []
            for col_idx, cell in enumerate(cells):
                if cell == "E":
                    enemy_spawns.append((col_idx, row_idx))
                    row_tiles.append(int(TileType.AIR))
                else:
                    row_tiles.append(int(CHAR_TO_TILE.get(cell, TileType.AIR)))
            rows.append(row_tiles)

    else:
        # Assume a file path-like string
        filepath = str(source)
        with open(filepath, "r") as fh:
            raw_lines = fh.readlines()
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
        raise ValueError(f"Empty tilemap: {source}")

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
