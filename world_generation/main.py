"""world_generation/main.py — local & container terminal test runner.

Exercises the full world-generation pipeline **without** requiring a running
HTTP server. Suitable for:

* Local development smoke-tests
* Docker container acceptance checks::

      docker run --rm world-gen-api python -m world_generation.main

* Quick CI / integration validation

Usage::

    # From the repo root
    python -m world_generation.main

    # With extra options
    python -m world_generation.main --examples 1-1 1-2 --epochs 5
    python -m world_generation.main --rebuild-df
    python -m world_generation.main --check-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Support both package import and direct `python world_generation/main.py`
try:
    from .Data_Preprocessing import load_structures_df, DEFAULT_EXAMPLE_CODES
    from .Performance_Evaluation import (
        Performance,
        performance,
        get_performance_model_status,
        train_performance_predictor,
        update_agent_history,
    )
    from .Map_Grammars_Elite import Map_Elite
    from .Variety_Domains import (
        athletic_domain_score,
        overworld_domain_score,
        underground_domain_score,
    )
except ImportError:
    _pkg = Path(__file__).resolve().parent.parent
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))
    from world_generation.Data_Preprocessing import load_structures_df, DEFAULT_EXAMPLE_CODES
    from world_generation.Performance_Evaluation import (
        Performance,
        performance,
        get_performance_model_status,
        train_performance_predictor,
        update_agent_history,
    )
    from world_generation.Map_Grammars_Elite import Map_Elite
    from world_generation.Variety_Domains import (
        athletic_domain_score,
        overworld_domain_score,
        underground_domain_score,
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_load_df(rebuild: bool = False, verbose: bool = True) -> Any:
    """Stage 1 – Load or build the structures DataFrame."""
    if verbose:
        print("[1/4] Loading structures DataFrame ...")
    df = load_structures_df(rebuild=rebuild)
    if verbose:
        print(f"      {len(df)} structure rows loaded.")
    return df


def stage_performance(df: Any, level_len: int = 40, verbose: bool = True) -> float:
    """Stage 2 – Evaluate a sample structural performance score."""
    if verbose:
        print("[2/4] Evaluating sample structural performance ...")
    sample_tokens = list(range(min(len(df), level_len)))
    score = performance(sample_tokens, df=df)
    if verbose:
        print(f"      Score: {score:.4f}")
    return score


def stage_map_elite(
    example_codes: List[str],
    level_len: int,
    alphas: Optional[List[float]] = None,
    epochs: int = 10,
    verbose: bool = True,
) -> Map_Elite:
    """Stage 3 – Create a Map_Elite instance."""
    if alphas is None:
        alphas = [0.0, 0.5, 1.0]
    if verbose:
        print("[3/4] Creating Map Elite instance ...")
    grid_points = [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]
    instance = Map_Elite(
        Performance,
        alphas=alphas,
        Variety_Domains=[overworld_domain_score, underground_domain_score, athletic_domain_score],
        Grid_points=grid_points,
    )
    if verbose:
        print(f"      Instance ready (alphas={alphas}).")
    return instance


def stage_generate(
    instance: Map_Elite,
    example_codes: List[str],
    level_len: int,
    epochs: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Stage 4 – Run map generation and return results."""
    if verbose:
        print(f"[4/4] Running map generation (epochs={epochs}) ...")
    instance.Generate_Mapping(
        example_code=example_codes,
        Level_Len=level_len,
        batch_size=3,
        epochs=epochs,
    )
    result = {
        "optimal_count": instance.get_optimal_count(),
        "generation": instance.get_current_generation(),
    }
    if verbose:
        print(f"      optimal_count={result['optimal_count']}  generation={result['generation']}")
    return result


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    example_codes: Optional[List[str]] = None,
    level_len: int = 40,
    alphas: Optional[List[float]] = None,
    epochs: int = 10,
    rebuild_df: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full local world-generation pipeline without HTTP.

    Returns a summary dict with performance score, optimal level count, and
    model status.
    """
    if example_codes is None:
        example_codes = ["1-1"]

    df = stage_load_df(rebuild=rebuild_df, verbose=verbose)
    struct_score = stage_performance(df, level_len=level_len, verbose=verbose)
    instance = stage_map_elite(example_codes, level_len, alphas=alphas, epochs=epochs, verbose=verbose)
    gen_result = stage_generate(instance, example_codes, level_len, epochs=epochs, verbose=verbose)

    result: Dict[str, Any] = {
        "structural_score": struct_score,
        "model_status": get_performance_model_status(),
        **gen_result,
    }

    if verbose:
        print("\n--- Pipeline Summary ---")
        print(json.dumps(result, indent=2))

    return result


# ---------------------------------------------------------------------------
# Check-only mode – verifies imports and DataFrame availability
# ---------------------------------------------------------------------------

def check_imports(verbose: bool = True) -> bool:
    """Verify that all world-generation modules import and the DataFrame is reachable."""
    ok = True
    checks = [
        ("Data_Preprocessing", load_structures_df),
        ("Performance_Evaluation.performance", performance),
        ("Performance_Evaluation.Performance", Performance),
        ("Map_Grammars_Elite.Map_Elite", Map_Elite),
        ("Variety_Domains.overworld_domain_score", overworld_domain_score),
        ("Variety_Domains.underground_domain_score", underground_domain_score),
        ("Variety_Domains.athletic_domain_score", athletic_domain_score),
    ]
    for label, obj in checks:
        status = "ok" if callable(obj) else "MISSING"
        if status != "ok":
            ok = False
        if verbose:
            print(f"  {status:8s}  {label}")

    structures_path = Path(__file__).resolve().parent / "Universal_Structures.txt"
    df_status = "ok" if structures_path.exists() else "absent (will be built on first use)"
    if verbose:
        print(f"  {df_status:8s}  Universal_Structures.txt")

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local world-generation pipeline test runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--examples", nargs="+", default=["1-1"], metavar="LEVEL",
        help="Level IDs to use (e.g. 1-1 1-2)",
    )
    parser.add_argument("--level-len", type=int, default=40, help="Level token length")
    parser.add_argument("--epochs", type=int, default=10, help="Map generation epochs")
    parser.add_argument(
        "--rebuild-df", action="store_true",
        help="Force rebuild of the structures DataFrame from level files",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only verify imports and structures file; do not run generation",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.check_only:
        print("=== Import & dependency check ===")
        ok = check_imports(verbose=True)
        sys.exit(0 if ok else 1)

    run_pipeline(
        example_codes=args.examples,
        level_len=args.level_len,
        epochs=args.epochs,
        rebuild_df=args.rebuild_df,
        verbose=True,
    )


if __name__ == "__main__":
    main()
