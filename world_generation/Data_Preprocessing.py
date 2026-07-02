# -*- coding: utf-8 -*-
"""Data preprocessing utilities for world generation.

Builds the ``Universal_Structures`` DataFrame consumed by
:mod:`world_generation.Performance_Evaluation` and
:class:`~world_generation.Map_Grammars_Elite.Map_Elite` from the raw level
text files shipped in ``Super_Mario_Brothers_Maps/Processed/``.

All paths are resolved relative to this module's location so the package
works regardless of the current working directory (local dev, Docker, etc.).

Check List:
    *Extract Example Levels strings
    *Hazards and Rewards Search
    *Hazards and Rewards _Score implementation in Performance_Evaluation
    *Reward decoupling
@author: Juan J. Rueda M.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Path constants – resolved once at import time so the module is CWD-agnostic
# ---------------------------------------------------------------------------
_MODULE_DIR: Path = Path(__file__).resolve().parent          # world_generation/
_PROJECT_ROOT: Path = _MODULE_DIR.parent                     # repo root
_LEVELS_DIR: Path = _PROJECT_ROOT / "Super_Mario_Brothers_Maps" / "Processed"
_STRUCTURES_DIR: Path = _PROJECT_ROOT / "Super_Mario_Brothers_Maps" / "structures"
_DEFAULT_OUTPUT: Path = _MODULE_DIR / "Universal_Structures.txt"

DEFAULT_EXAMPLE_CODES: List[str] = [
    "1-1", "1-2", "1-3", "2-1", "3-1", "3-3",
    "4-1", "4-2", "5-1", "5-3", "6-1", "6-2",
    "6-3", "7-1", "8-1",
]

############ Limitation: ASCII is a 7-bit character set containing 128 characters.
#### Solution: The grammars could be closed-managed, i.e. We can define a grammar with only 128 maximum structures,
#### but we can create as many as we want. Also if we guarantee that grammars match at least some structures, so that
#### form a full-connected network, which could transition between grammars with >128 structures considered.


# ---------------------------------------------------------------------------
# Row-level feature extractors (unchanged logic, parameterised tile sets)
# ---------------------------------------------------------------------------

#Function to join elements of a row into one string
def Structure(row):
    return "".join(row[6:])

#Function to check landings in a row and remove anything else
def Landing_Search(row, lands_ID={"X", "S", "<", ">", "[", "]", "B", "b", "Q", "?"}):  #Each mechanic stitching must have a Fi-2pop evaluation
    boolean_lands = ""
    for i in range(1, 14):  #this range is dependant on level size
        #if character is a landing, it stays
        if (row["{}".format(i)] in lands_ID) and not (row["{}".format(i - 1)] in lands_ID):  #
            boolean_lands += "1"
        #if it's anything else, it is removed to nothing
        else:
            boolean_lands += "0"

    return int(boolean_lands, 2)

#function to check "colliders" or tiles that you can't pass through
def Colliders_Search(row, collider_ID={"X", "S", "<", ">", "[", "]", "B", "b", "Q", "?"}):
    boolean_colliders = "".join(row.apply(lambda x: "1" if x in collider_ID else "0"))
    return int(boolean_colliders, 2)

def Rewards_Search(row, collider_ID={"o", "?", "Q"}):
    boolean_colliders = "".join(row.apply(lambda x: "1" if x in collider_ID else "0"))
    return int(boolean_colliders, 2)

def Hazard_Search(row, collider_ID={"E", "B"}):
    boolean_colliders = "".join(row.apply(lambda x: "1" if x in collider_ID else "0"))
    return int(boolean_colliders, 2)


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def build_structures_df(
    example_codes: List[str],
    levels_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the structures DataFrame from raw level text files.

    Args:
        example_codes: List of level IDs (e.g. ``["1-1", "1-2"]``).
        levels_dir: Directory containing ``mario-<id>.txt`` files.
            Defaults to ``Super_Mario_Brothers_Maps/Processed/`` in the repo root.

    Returns:
        DataFrame with columns: token, Structures, Landings, Colliders, Reward, Hazard.
    """
    if levels_dir is None:
        levels_dir = _LEVELS_DIR

    cols_base = ["Structures", "token", "Landings", "Colliders", "Reward", "Hazard"]
    cols_tile = ["{}".format(i) for i in range(13)]

    Complete_Level = pd.DataFrame(columns=cols_base + cols_tile)

    for example_code in example_codes:
        Temporal = pd.DataFrame(columns=["Structures", "token", "Landings", "Colliders"] + cols_tile)
        level_path = levels_dir / f"mario-{example_code}.txt"
        with level_path.open() as infile:
            i = 0
            for line in infile:
                Temporal["{}".format(i)] = list(line.split()[0])
                i += 1
        Complete_Level = pd.concat([Complete_Level, Temporal], axis=0)

    Complete_Level["Structures"] = Complete_Level.apply(Structure, axis=1)
    Complete_Level["Landings"] = Complete_Level.apply(Landing_Search, axis=1)
    Complete_Level["Colliders"] = Complete_Level.apply(Colliders_Search, axis=1)
    Complete_Level["Reward"] = Complete_Level.apply(Rewards_Search, axis=1)
    Complete_Level["Hazard"] = Complete_Level.apply(Hazard_Search, axis=1)

    # Deduplicate and assign integer token ids
    Keys_Structures = pd.DataFrame(columns=["token", "Structures", "Landings", "Colliders", "Reward", "Hazard"])
    Keys_Structures[["Structures", "Landings", "Colliders", "Reward", "Hazard"]] = (
        Complete_Level[["Structures", "Landings", "Colliders", "Reward", "Hazard"]].drop_duplicates()
    )
    Keys_Structures = Keys_Structures.reset_index(drop=True)
    for index in Keys_Structures.index:
        Keys_Structures.loc[index, "token"] = index

    return Keys_Structures


def load_structures_df(
    output_path: Optional[Path] = None,
    example_codes: Optional[List[str]] = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load the structures DataFrame, building it from level files if needed.

    This is the primary integration point consumed by
    :mod:`~world_generation.Performance_Evaluation` and
    :class:`~world_generation.Map_Grammars_Elite.Map_Elite`.

    Args:
        output_path: CSV file to load from / save to.
            Defaults to ``world_generation/Universal_Structures.txt``.
        example_codes: Level IDs used when the file is missing or *rebuild* is True.
            Defaults to :data:`DEFAULT_EXAMPLE_CODES`.
        rebuild: When ``True``, always rebuild from source level files even if
            the CSV already exists.

    Returns:
        DataFrame with columns: token, Structures, Landings, Colliders, Reward, Hazard.
    """
    if output_path is None:
        output_path = _DEFAULT_OUTPUT
    output_path = Path(output_path)

    if not rebuild and output_path.exists():
        return pd.read_csv(output_path)

    if example_codes is None:
        example_codes = DEFAULT_EXAMPLE_CODES

    df = build_structures_df(example_codes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Legacy entry point – kept for backward compatibility with direct-script use
# ---------------------------------------------------------------------------

def Save_Mix_Structures(Ex_Levels_List: List[str]) -> pd.DataFrame:
    """Legacy function: build the structures CSV and save to structures dir.

    Preserved for scripts that call this directly.  New code should use
    :func:`load_structures_df` or :func:`build_structures_df` instead.
    """
    _STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    df = build_structures_df(Ex_Levels_List)
    out_path = _STRUCTURES_DIR / "Structures_{}.txt".format(Ex_Levels_List)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df = load_structures_df(rebuild=True)
    print(f"Built structures DataFrame: {len(df)} rows")
    print(df.head())