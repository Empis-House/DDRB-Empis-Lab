"""
run_search.py – Generate Mario levels with MAP-Elites using example-based seeding.

Overview
--------
  Generates levels by evolving chunks derived from example Mario levels.
  Levels are evaluated and placed in a MAP-Elites grid defined by
  variety domains (behavioural descriptors) and grid accumulation points.

Usage
-----
  # Minimal – required arguments only
  python run_search.py 1-1 1-2 3-1 --level-len 200 \ --alphas '[0.2, 0.5, 0.8]' \ --variety-domains '["landings", "slope"]' \ --grid-points '[[0,0.5,1], [0,0.5,1]]'

  # With chunking module and output file
  python run_search.py 1-1 2-1 --level-len 400 \
      --alphas '[0.1, 0.5, 0.9]' \
      --variety-domains '["landings", "slope", "colliders"]' \
      --grid-points '[[0,0.4,0.8,1], [0,0.3,0.6,0.9], [0,0.2,0.4,0.6]]' \
      --module 50 \
      --save-as output/elites.json

Arguments
---------
  example_codes       One or more Mario level codes to seed generation from
                      (e.g. 1-1 1-2 3-1). Passed as positional arguments.
  --level-len INT     Minimum length of each generated level (required).
  --alphas JSON       Performance curve thresholds as a JSON float array
                      (e.g. '[0.2, 0.5, 0.8]').
  --variety-domains JSON
                      Behavioural descriptor names as a JSON string array
                      (e.g. '["jumps", "enemies", "gaps"]').
  --grid-points JSON  MAP-Elites accumulation points as a JSON array of arrays,
                      one inner array per variety domain
                      (e.g. '[[0,5,10], [0,3,6]]').
  --module INT        Chunk length for sub-level generation (default: 0,
                      meaning no chunking).
  --save-as FILE      Path to save the resulting elite archive (default: none).

Notes
-----
  - The number of arrays in --grid-points must match the number of entries
    in --variety-domains.
  - JSON arguments must be quoted to prevent shell expansion; use single
    quotes on Unix/macOS and double quotes with inner escaping on Windows.
"""

from __future__ import annotations

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import time
import numpy as np
import itertools
from scipy.stats import linregress

def transform_to_cartesian(grid_points):
    cartesian_list = list(itertools.product(*grid_points))
    
    return cartesian_list
    

def run(example_codes, # List of Mario level code which generation is based on. ex: 1-1, 1-2, 3-1
        level_len, # min generated level length
        alphas, # Performance (Y) level-curves list
        variety_domains, # List of descriptive functions to map levels from.
        grid_points, # Lists of acumulation points to Map-elites
        module=0, # The chuncking module is the sub-generated levels lenght 
        save_as = None
        ):
    
    if level_len == 0:
        raise ValueError("Levels of 0 lenght are not generable")
    if module == 0:
        module = level_len

    if len(variety_domains) != len(grid_points):
        raise KeyError("The number of arrays in --grid-points must match the number of entriesin --variety-domains.")

    """if save_as != None:
        save_as = f'Super_Mario_Brothers_Maps\\generated_levels\\Search_Testing\\{save_as}'
    """
    from Performance_Evaluation import Performance
    from Map_Grammars_Elite import Map_Elite
    from Variety_Domains import variety_domains_json

    variety_functions = []
    for tag in variety_domains:
        variety_functions += [variety_domains_json[tag]]


    cartesian_list = transform_to_cartesian(grid_points)
    Map_Performance = Map_Elite(Performance,alphas=alphas, Variety_Domains = variety_functions, Grid_points =grid_points)
    Map_Performance.Generate_Mapping(example_code=example_codes, Level_Len=level_len,module=module,batch_size=5,epochs=10)
    Map_Performance.Plot_Coordinate(cartesian_list=cartesian_list, level_curves=[0.8], level_name=save_as) 

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MAP-Elites level generator against the Mario RL environment.")

    parser.add_argument("example_codes",
                        nargs="+",
                        help="Mario level codes to base generation on (e.g. 1-1 1-2 3-1)")
    parser.add_argument("--level-len",    type=int, required=True,
                        help="Minimum generated level length")
    parser.add_argument("--alphas",       type=json.loads, required=True,
                        metavar="JSON",
                        help="Performance (Y) level-curves list as a JSON array (e.g. '[0.1, 0.5, 0.9]')")
    parser.add_argument("--variety-domains", nargs="+", required=True,
                    metavar="DOMAIN",
                    help="Descriptive functions list (e.g. --variety-domains landings slope colliders)")
    parser.add_argument("--grid-points",  type=json.loads, required=True,
                        metavar="JSON",
                        help="MAP-Elites accumulation points as a JSON array (e.g. '[[0,1],[0.1,0.2]]')")
    parser.add_argument("--module",       type=int, default=0,
                        help="Sub-generated level chunk length (default: 0)")
    parser.add_argument("--save-as",      default=None,
                        metavar="FILE",
                        help="Save output to FILE (default: None)")

    args = parser.parse_args()

    run(
        example_codes=args.example_codes,
        level_len=args.level_len,
        alphas=args.alphas,
        variety_domains=args.variety_domains,
        grid_points=args.grid_points,
        module=args.module,
        save_as=args.save_as,
    )


if __name__ == "__main__":
    main()
