from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


BAND_HEIGHT = 14


def _normalize_token(token: Any) -> str:
    """Normalize tokens across int/float/string representations.

    Examples:
    - 0 -> "0"
    - "0.0" -> "0"
    - " '12' " -> "12"
    """
    text = str(token).strip().strip("\"'")
    if text.endswith(".0"):
        integer_candidate = text[:-2]
        if integer_candidate.lstrip("-").isdigit():
            return integer_candidate
    return text


def _parse_level_tokens(level: Any) -> List[str]:
    """Return a normalized flat token list from common level encodings."""
    if isinstance(level, str):
        raw = level.strip()
        if raw == "":
            return []

        # Support payloads such as "[0,1,2]" or "0,1,2" in addition to legacy char streams.
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1].strip()

        if "," in raw:
            return [_normalize_token(token) for token in raw.split(",") if token.strip()]

        if " " in raw:
            return [_normalize_token(token) for token in raw.split() if token.strip()]

        return [_normalize_token(token) for token in list(raw)]

    if isinstance(level, np.ndarray):
        return [_normalize_token(token) for token in level.flatten().tolist()]

    if isinstance(level, pd.Series):
        return [_normalize_token(token) for token in level.tolist()]

    if isinstance(level, Sequence):
        return [_normalize_token(token) for token in level]

    raise TypeError(f"Unsupported level format: {type(level).__name__}")


def build_token_structure_lookup(structures_df: pd.DataFrame) -> Dict[str, str]:
    """Build a token -> structure-string lookup from the structures dataframe."""
    if "token" not in structures_df.columns or "Structures" not in structures_df.columns:
        raise ValueError("structures_df must include 'token' and 'Structures' columns")

    lookup: Dict[str, str] = {}
    for _, row in structures_df[["token", "Structures"]].dropna().iterrows():
        token = _normalize_token(row["token"])
        structure = str(row["Structures"])
        lookup[token] = structure
    return lookup


def level_string_to_tilemap(level: Any, token_to_structure: Mapping[str, str]) -> pd.DataFrame:
    """Translate tokenized level data into the canonical tilemap dataframe."""
    level_tokens = _parse_level_tokens(level)
    if not level_tokens:
        raise ValueError("empty str level is no plotable")

    rows = []
    for key in level_tokens:
        if key not in token_to_structure:
            raise IndexError(f"Not tokenized Structure: {key}")

        structure = token_to_structure[key]
        row = list(structure)
        if len(row) != BAND_HEIGHT:
            raise ValueError(
                f"Structure for token '{key}' has {len(row)} rows, expected {BAND_HEIGHT}"
            )
        rows.append(row)

    return pd.DataFrame(rows, columns=[f"{i}" for i in range(BAND_HEIGHT)]).transpose()


def tilemap_to_rows(tilemap: Any) -> List[List[Any]]:
    """Normalize known tilemap objects to list[list]."""
    if isinstance(tilemap, pd.DataFrame):
        return tilemap.values.tolist()

    if isinstance(tilemap, np.ndarray):
        return tilemap.tolist()

    if isinstance(tilemap, list):
        return tilemap

    raise TypeError(f"Unsupported tilemap format: {type(tilemap).__name__}")


def rows_to_tilemap(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Convert a list-like tile grid into a pandas dataframe."""
    return pd.DataFrame(rows)


def parse_level_line(line: str) -> List[str]:
    """Parse one text row into tile tokens.

    Supports both plain-char rows and CSV-like rows.
    """
    raw = line.strip()
    if not raw:
        return []

    if "," in raw:
        return [token.strip() for token in raw.split(",") if token.strip()]

    return list(raw)
