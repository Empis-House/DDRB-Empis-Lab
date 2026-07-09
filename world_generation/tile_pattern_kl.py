from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from .level_formats import parse_level_line
except ImportError:
    from level_formats import parse_level_line


Pattern = Tuple[str, ...]
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
_DEFAULT_DOMAIN_ROOT = _PROJECT_ROOT / "Super_Mario_Brothers_Maps" / "Domains"
_KNOWN_DOMAINS = ("overworld", "underground", "athletic")


def load_level_grid(path: Path) -> List[List[str]]:
    """Load a level text file as a 2D tile grid [rows][cols]."""
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_level_line(line)
            if parsed:
                rows.append(parsed)

    if not rows:
        raise ValueError(f"Level file has no parseable rows: {path}")

    min_width = min(len(row) for row in rows)
    if min_width <= 0:
        raise ValueError(f"Level file has empty rows: {path}")

    # Keep a rectangular grid by trimming to the smallest width.
    return [row[:min_width] for row in rows]


def load_level_set(path: Path, recursive: bool = True) -> List[List[List[str]]]:
    """Load all .txt levels under path. Path can be a file or directory."""
    if path.is_file():
        return [load_level_grid(path)]

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if not path.is_dir():
        raise ValueError(f"Expected file or directory path, got: {path}")

    pattern = "**/*.txt" if recursive else "*.txt"
    files = sorted(p for p in path.glob(pattern) if p.is_file())
    if not files:
        raise ValueError(f"No .txt levels found under: {path}")

    return [load_level_grid(level_path) for level_path in files]


def _iter_patterns(
    grid: Sequence[Sequence[str]],
    window_height: int,
    window_width: int,
    stride_y: int,
    stride_x: int,
) -> Iterable[Pattern]:
    height = len(grid)
    width = len(grid[0]) if height else 0

    if window_height > height or window_width > width:
        return

    for y in range(0, height - window_height + 1, stride_y):
        for x in range(0, width - window_width + 1, stride_x):
            tiles: List[str] = []
            for dy in range(window_height):
                row = grid[y + dy]
                tiles.extend(row[x:x + window_width])
            yield tuple(tiles)


def tile_pattern_distribution(
    level_set: Sequence[Sequence[Sequence[str]]],
    window: Tuple[int, int] = (2, 2),
    stride: Tuple[int, int] = (1, 1),
) -> Counter[Pattern]:
    """Build a tile-pattern histogram for a level set."""
    window_height, window_width = window
    stride_y, stride_x = stride

    if window_height <= 0 or window_width <= 0:
        raise ValueError("window dimensions must be positive")
    if stride_y <= 0 or stride_x <= 0:
        raise ValueError("stride dimensions must be positive")

    counts: Counter[Pattern] = Counter()
    for grid in level_set:
        for pat in _iter_patterns(grid, window_height, window_width, stride_y, stride_x):
            counts[pat] += 1

    if not counts:
        raise ValueError("No tile patterns extracted. Check level sizes/window.")

    return counts


def _probability_vectors(
    p_counts: Counter[Pattern],
    q_counts: Counter[Pattern],
    smoothing: float,
) -> Tuple[List[float], List[float]]:
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")

    all_patterns = set(p_counts) | set(q_counts)
    if not all_patterns:
        raise ValueError("No patterns available to compare")

    p_total = float(sum(p_counts.values()))
    q_total = float(sum(q_counts.values()))
    vocab_size = float(len(all_patterns))

    p_den = p_total + smoothing * vocab_size
    q_den = q_total + smoothing * vocab_size

    p_probs: List[float] = []
    q_probs: List[float] = []

    for pat in all_patterns:
        p_probs.append((p_counts.get(pat, 0) + smoothing) / p_den)
        q_probs.append((q_counts.get(pat, 0) + smoothing) / q_den)

    return p_probs, q_probs


def kl_divergence(p_probs: Sequence[float], q_probs: Sequence[float]) -> float:
    """Compute KL(P||Q) using natural log."""
    if len(p_probs) != len(q_probs):
        raise ValueError("Probability vectors must have equal length")

    value = 0.0
    for p, q in zip(p_probs, q_probs):
        if p <= 0.0:
            continue
        if q <= 0.0:
            raise ValueError("q contains zero probability while p is positive")
        value += p * math.log(p / q)
    return value


def crossed_divergence(p_probs: Sequence[float], q_probs: Sequence[float], w=0.5) -> float:
    """Symmetric and finite divergence between two distributions."""
    if len(p_probs) != len(q_probs):
        raise ValueError("Probability vectors must have equal length")

    m = [(p + q) * 0.5 for p, q in zip(p_probs, q_probs)]
    return w * kl_divergence(p_probs, m) + 1-w * kl_divergence(q_probs, m)


def compare_level_sets_tile_pattern_kl(
    set_a: Sequence[Sequence[Sequence[str]]],
    set_b: Sequence[Sequence[Sequence[str]]],
    window: Tuple[int, int] = (2, 2),
    stride: Tuple[int, int] = (1, 1),
    smoothing: float = 1e-6,
) -> Dict[str, float]:
    """Mimic tile-pattern KL analysis between two level sets.

    Returns forward/reverse KL and CoD to quantify distribution shift.
    """
    if not set_a:
        raise ValueError("set_a must contain at least one level")
    if not set_b:
        raise ValueError("set_b must contain at least one level")

    a_counts = tile_pattern_distribution(set_a, window=window, stride=stride)
    b_counts = tile_pattern_distribution(set_b, window=window, stride=stride)
    p, q = _probability_vectors(a_counts, b_counts, smoothing=smoothing)

    kld_a_to_b = kl_divergence(p, q)
    kld_b_to_a = kl_divergence(q, p)
    CoD = crossed_divergence(p, q)

    return {
        "patterns_set_a": sum(a_counts.values()),
        "patterns_set_b": sum(b_counts.values()),
        "unique_patterns_union": len(set(a_counts) | set(b_counts)),
        "kld_set_a_to_set_b": kld_a_to_b,
        "kld_set_b_to_set_a": kld_b_to_a,
        "kld_symmetric": 0.5 * (kld_a_to_b + kld_b_to_a),
        "CoD": CoD,
    }


def evaluate_level_folders(
    set_a_path: Path,
    set_b_path: Path,
    window: Tuple[int, int] = (2, 2),
    stride: Tuple[int, int] = (1, 1),
    smoothing: float = 1e-6,
    recursive: bool = True,
) -> Dict[str, float]:
    """Convenience helper that compares two folders (or files) of levels."""
    levels_a = load_level_set(set_a_path, recursive=recursive)
    levels_b = load_level_set(set_b_path, recursive=recursive)
    return compare_level_sets_tile_pattern_kl(
        levels_a,
        levels_b,
        window=window,
        stride=stride,
        smoothing=smoothing,
    )


def domain_paths(domain_root: Path = _DEFAULT_DOMAIN_ROOT) -> Dict[str, Path]:
    """Return standard domain folder paths keyed by domain name."""
    base = Path(domain_root)
    return {name: base / name for name in _KNOWN_DOMAINS}


def evaluate_named_domains(
    domain_a: str,
    domain_b: str,
    domain_root: Path = _DEFAULT_DOMAIN_ROOT,
    window: Tuple[int, int] = (2, 2),
    stride: Tuple[int, int] = (1, 1),
    smoothing: float = 1e-6,
    recursive: bool = True,
) -> Dict[str, float]:
    """Compare two known domains by name (overworld/underground/athletic)."""
    paths = domain_paths(domain_root=domain_root)
    if domain_a not in paths:
        valid = ", ".join(sorted(paths))
        raise ValueError(f"Unknown domain_a '{domain_a}'. Valid domains: {valid}")
    if domain_b not in paths:
        valid = ", ".join(sorted(paths))
        raise ValueError(f"Unknown domain_b '{domain_b}'. Valid domains: {valid}")

    result = evaluate_level_folders(
        set_a_path=paths[domain_a],
        set_b_path=paths[domain_b],
        window=window,
        stride=stride,
        smoothing=smoothing,
        recursive=recursive,
    )
    result["domain_a"] = domain_a
    result["domain_b"] = domain_b
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile-pattern KL divergence between two level sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--set-a", help="Folder or file for level set A")
    parser.add_argument("--set-b", help="Folder or file for level set B")
    parser.add_argument("--domain-a", choices=_KNOWN_DOMAINS, help="Named domain for set A")
    parser.add_argument("--domain-b", choices=_KNOWN_DOMAINS, help="Named domain for set B")
    parser.add_argument(
        "--domain-root",
        default=str(_DEFAULT_DOMAIN_ROOT),
        help="Base folder that contains domain subfolders",
    )
    parser.add_argument("--window", type=int, nargs=2, default=(2, 2), metavar=("H", "W"))
    parser.add_argument("--stride", type=int, nargs=2, default=(1, 1), metavar=("SY", "SX"))
    parser.add_argument("--smoothing", type=float, default=1e-6)
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subfolders")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    use_named_domains = bool(args.domain_a or args.domain_b)
    if use_named_domains:
        if not (args.domain_a and args.domain_b):
            raise ValueError("Both --domain-a and --domain-b are required when using named domains")
        result = evaluate_named_domains(
            domain_a=args.domain_a,
            domain_b=args.domain_b,
            domain_root=Path(args.domain_root),
            window=tuple(args.window),
            stride=tuple(args.stride),
            smoothing=args.smoothing,
            recursive=not args.no_recursive,
        )
    else:
        if not (args.set_a and args.set_b):
            raise ValueError("Either provide --set-a/--set-b or --domain-a/--domain-b")
        result = evaluate_level_folders(
            set_a_path=Path(args.set_a),
            set_b_path=Path(args.set_b),
            window=tuple(args.window),
            stride=tuple(args.stride),
            smoothing=args.smoothing,
            recursive=not args.no_recursive,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
