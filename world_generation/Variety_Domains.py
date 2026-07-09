# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:00:20 2023
Check List:
    *More Functions
    *Mapping Functions frome Grids
@author: Juan J. Rueda M.
"""
""""""
# import sys
# sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB\DDRB-Empis-Lab\Super_Mario_Brothers_Maps")

import pandas as pd
import time
import numpy as np
import itertools
from scipy.stats import linregress
from pathlib import Path
from typing import Dict, List

try:
    from .evaluation_utils import metric_values, normalized_mean, normalize_level_tokens
except ImportError:  # pragma: no cover - supports direct script imports
    from evaluation_utils import metric_values, normalized_mean, normalize_level_tokens


_DOMAIN_ROOT = Path(__file__).resolve().parent.parent / "Super_Mario_Brothers_Maps" / "Domains"
_DOMAIN_NAMES = ("overworld", "underground", "athletic")
_DOMAIN_WINDOW = (2, 2)
_DOMAIN_STRIDE = (1, 1)
_DOMAIN_SMOOTHING = 1e-6

_TOKEN_STRUCTURE_LOOKUP_CACHE: Dict[int, Dict[int, str]] = {}
_DOMAIN_PATTERN_CACHE: Dict[str, object] = {}


def _token_to_structure_lookup(df) -> Dict[int, str]:
    cache_key = id(df)
    if cache_key not in _TOKEN_STRUCTURE_LOOKUP_CACHE:
        tokens = df["token"].astype(int).tolist()
        structures = df["Structures"].astype(str).tolist()
        _TOKEN_STRUCTURE_LOOKUP_CACHE[cache_key] = dict(zip(tokens, structures))
    return _TOKEN_STRUCTURE_LOOKUP_CACHE[cache_key]


def _level_to_grid(Level, df) -> List[List[str]]:
    tokens = normalize_level_tokens(Level)
    if not tokens:
        return []

    lookup = _token_to_structure_lookup(df)
    columns = [list(lookup[token]) for token in tokens if token in lookup]
    if not columns:
        return []

    height = min(len(col) for col in columns)
    width = len(columns)
    return [[columns[x][y] for x in range(width)] for y in range(height)]


def _domain_reference_counts(domain_name: str):
    if domain_name in _DOMAIN_PATTERN_CACHE:
        return _DOMAIN_PATTERN_CACHE[domain_name]

    try:
        from .tile_pattern_kl import load_level_set, tile_pattern_distribution
    except ImportError:  # pragma: no cover - supports direct script imports
        from tile_pattern_kl import load_level_set, tile_pattern_distribution

    domain_path = _DOMAIN_ROOT / domain_name
    levels = load_level_set(domain_path, recursive=True)
    counts = tile_pattern_distribution(levels, window=_DOMAIN_WINDOW, stride=_DOMAIN_STRIDE)
    _DOMAIN_PATTERN_CACHE[domain_name] = counts
    return counts


def _domain_similarity_score(Level, df, domain_name: str) -> float:
    if domain_name not in _DOMAIN_NAMES:
        raise ValueError(f"Unknown domain '{domain_name}'")

    grid = _level_to_grid(Level, df)
    if not grid:
        return 0.0

    try:
        from .tile_pattern_kl import (
            _probability_vectors,
            crossed_divergence,
            tile_pattern_distribution,
        )
    except ImportError:  # pragma: no cover - supports direct script imports
        from tile_pattern_kl import _probability_vectors, crossed_divergence, tile_pattern_distribution

    level_counts = tile_pattern_distribution([grid], window=_DOMAIN_WINDOW, stride=_DOMAIN_STRIDE)
    reference_counts = _domain_reference_counts(domain_name)
    p_probs, q_probs = _probability_vectors(reference_counts, level_counts, smoothing=_DOMAIN_SMOOTHING)
    CoD = crossed_divergence(p_probs, q_probs)

    # Similarity score in [0,1], where 1 means close to domain distribution.
    return 1.0 / (1.0 + float(CoD))


def overworld_domain_score(Level, df):
    return _domain_similarity_score(Level=Level, df=df, domain_name="overworld")


def underground_domain_score(Level, df):
    return _domain_similarity_score(Level=Level, df=df, domain_name="underground")


def athletic_domain_score(Level, df):
    return _domain_similarity_score(Level=Level, df=df, domain_name="athletic")

variety_domains_json = {"overworld": overworld_domain_score,
                        "underground": underground_domain_score,
                        "athletic": athletic_domain_score}


def tile_pattern_kl_diversity_between_sets(
    set_a_path,
    set_b_path,
    window=(2, 2),
    stride=(1, 1),
    smoothing=1e-6,
    recursive=True,
):
    """Evaluate diversity between two level sets via tile-pattern KL metrics."""
    try:
        from .tile_pattern_kl import evaluate_level_folders
    except ImportError:  # pragma: no cover - supports direct script imports
        from tile_pattern_kl import evaluate_level_folders

    return evaluate_level_folders(
        set_a_path=Path(set_a_path),
        set_b_path=Path(set_b_path),
        window=tuple(window),
        stride=tuple(stride),
        smoothing=float(smoothing),
        recursive=bool(recursive),
    )


def tile_pattern_kl_diversity_between_domains(
    domain_a,
    domain_b,
    domain_root=Path("Super_Mario_Brothers_Maps") / "Domains",
    window=(2, 2),
    stride=(1, 1),
    smoothing=1e-6,
    recursive=True,
):
    """Evaluate diversity between named domain clusters."""
    try:
        from .tile_pattern_kl import evaluate_named_domains
    except ImportError:  # pragma: no cover - supports direct script imports
        from tile_pattern_kl import evaluate_named_domains

    return evaluate_named_domains(
        domain_a=str(domain_a),
        domain_b=str(domain_b),
        domain_root=Path(domain_root),
        window=tuple(window),
        stride=tuple(stride),
        smoothing=float(smoothing),
        recursive=bool(recursive),
    )