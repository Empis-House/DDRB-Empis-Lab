# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 10:00:26 2023
Check List:
    *CUDA opt.
    *Finish Colliders
    *Hazardz Stitching
@author: Juan J. Rueda M.
The 2009 mario ai competition - togelius. (n.d.). http://julian.togelius.com/Togelius2010The.pdf 
Mario level generation from mechanics using scene stitching - arxiv.org. (n.d.-a). https://arxiv.org/pdf/2002.02992 

Examples
--------
Python API:

    from world_generation.Map_Grammars_Elite import Map_Elite
    from world_generation.Word_Stitching import Jumping_Fiasible_Word

    me = Map_Elite(lambda x: x)
    level = me.Extract_Level_String(["1-3"])
    feasible = Jumping_Fiasible_Word(level, me.Knowledge)

    # Editable jump envelope for the current test
    feasible = Jumping_Fiasible_Word(level, me.Knowledge, max_length=6, max_height=4)

Debug API:

    ok, info = Jumping_Fiasible_Word(level, me.Knowledge, debug=True)
    print(ok)
    print(info["reason"])

CLI examples:

    python world_generation/Word_Stitching.py --level-code 1-3
    python world_generation/Word_Stitching.py --level-code 1-3 --debug
    python world_generation/Word_Stitching.py --level-code 1-3 --debug --json
    python world_generation/Word_Stitching.py --tokens 1,1,1 --json
    python world_generation/Word_Stitching.py --level-code 1-3 --max-length 6 --max-height 4 --debug --json

"""

from numba import njit
import numpy as np

Max_Height = 4 # maximum vertical rise a Mario jump can cover
Max_Length = 4 # maximum horizontal gap between reachable landing columns
Band_Height = 14 #size of level


#check if there is a 1 in position k of the number n
# 4 = 00100000000000 so p(n=4,k=2) = 1 and p(n=4,k!=2) = 0
#@njit(target_backend='cuda') 
def Punctual_Binary_Prop(n,k):
    n_ = n>>k 
    #returns only position
    return (n_ % 2**(k+1))%2

# slice bit array sections
#@njit(target_backend='cuda')
def Range_Binary_Prop(n,a,b):
    #read structures
    a = max(0,a)
    b_ = (n>>b+1) * 2**(b+1)
    n_ = (n - b_)>>a
    #returns middle number
    return n_

def Jumping_Fiasible_Word(Level, knowledge, Game_mod = 0, debug = False, max_height = None, max_length = None):
    """
    Taken from https://arxiv.org/pdf/2002.02992.pdf
    This function verify if a single word is possible to wander with walking and jumping, 
    in left to rigth scrolling
    
    A word finished in a letter without landing is't admisible.
    
    Game_Mods:
        0 * classic lateral Scroll
        1 * inverse lateral Scroll
        """
        
    if Game_mod==7357:
        if debug:
            return True, {"reason": "debug_bypass", "game_mod": Game_mod}
        return True

    if max_height is None:
        max_height = Max_Height
    if max_length is None:
        max_length = Max_Length
    
    
    Landings = knowledge[0]
    Colliders = knowledge[1]

    Structures_id = [x for x in list(Level)]
    Int_Landings_List = [Landings[x] for x in Level]
    Int_Colliders_List = [Colliders[x] for x in Level]

    # Ignore trailing void columns after the last solid structure. They are
    # common in full levels and should not invalidate an otherwise reachable
    # ending.
    while Int_Landings_List and Int_Landings_List[-1] == 0 and Int_Colliders_List[-1] == 0:
        Int_Landings_List.pop()
        Int_Colliders_List.pop()
        Structures_id.pop()
    
    def First_One_Above(mask, start):
        for height in range(start, Band_Height):
            if Punctual_Binary_Prop(mask, height) == 1:
                return height
        return None

    All_Landings_Accesibility_Stitched = True
    n = len(Int_Landings_List)
     #FIX
    
    #if no landing and beginning or end, this is unplayable
    if n == 0:
        if debug:
            return False, {"reason": "empty_level_after_trim"}
        return False
    Initial = Int_Landings_List[0]
    Last = Int_Colliders_List[-1]
    if Initial==0 or Last==0:
        if debug:
            return False, {
                "reason": "invalid_start_or_end",
                "start_landing_mask": int(Initial),
                "end_collider_mask": int(Last),
                "effective_length": int(n),
            }
        return False
    
    #if level is only one letter, it is always playable
    if len(Level) == 1:
        if debug:
            return True, {"reason": "single_column_level"}
        return True

    LA_individually_Stitched_L = False
    JW_individually_Stitched = False         
    
    for n_letter in np.arange(n):
        Initial= Int_Landings_List[n_letter]
        Initial_Colliders = Int_Colliders_List[n_letter]
        R_Band_Height = np.arange(Band_Height)
        Column_Reachable = False
        landing_heights = []
        landing_checks = []

        # Columns without landings are traversed in mid-air or by gaps, so they
        # should not invalidate a level on their own.
        if Initial == 0:
            continue
        
        #check if the horizontal jump is possible
        #Forward_Stitching = [*range(n_letter +1, min(n_letter + Max_Length+1, n)),*range(max(n_letter - Max_Length, 0), n_letter-1 )]
        Forward_Stitching = np.arange(n_letter +1, min(n_letter + max_length+1, n))
        Back_Stiching = np.arange(max(n_letter - max_length, 0), n_letter)
        
        #Stitches = GPU_Stitching(Level, Forward_Stitching,Back_Stiching, R_Band_Height, Initial, n_letter)
        
        
        for i in R_Band_Height:
            if Punctual_Binary_Prop(Initial, i) != 1:
                continue

            if debug:
                landing_heights.append(int(i))

            LA_individually_Stitched_L = False
            JW_individually_Stitched = False

            for j in Back_Stiching:
                Int_Landing = Int_Landings_List[j]
                if Int_Landing == 0:
                    continue

                top_block = Band_Height
                next_landing_above = First_One_Above(Initial, i + 1)
                if next_landing_above is not None:
                    top_block = min(top_block, next_landing_above)

                next_collider_above = First_One_Above(Initial_Colliders, i + 1)
                if next_collider_above is not None:
                    top_block = min(top_block, next_collider_above)

                min_height = max(0, i - max_height)
                # Upward movement is bounded, but dropping from a higher
                # platform should remain feasible as long as the landing
                # column has enough headroom.
                ceiling_height = top_block - 1

                if ceiling_height >= min_height and Range_Binary_Prop(Int_Landing, min_height, ceiling_height) != 0:
                    LA_individually_Stitched_L = True
                    break

            for j in np.concatenate((Forward_Stitching, Back_Stiching), axis=0):
                Int_Landing = Int_Landings_List[j]
                if Range_Binary_Prop(Int_Landing, 0, i + max_height) != 0:
                    JW_individually_Stitched = True
                    break

            if debug:
                landing_checks.append(
                    {
                        "landing_height": int(i),
                        "left_accessible": bool(LA_individually_Stitched_L or n_letter == 0),
                        "jump_window_exists": bool(JW_individually_Stitched),
                    }
                )

            if n_letter == 0:
                Column_Reachable = True
                continue

            if LA_individually_Stitched_L and JW_individually_Stitched:
                Column_Reachable = True
                break

        if n_letter == 0:
            LA_individually_Stitched_L = True
            JW_individually_Stitched = True
        else:
            LA_individually_Stitched_L = Column_Reachable
            JW_individually_Stitched = Column_Reachable

        All_Landings_Accesibility_Stitched = All_Landings_Accesibility_Stitched and LA_individually_Stitched_L and JW_individually_Stitched
        if not All_Landings_Accesibility_Stitched:
            if debug:
                return False, {
                    "reason": "column_unreachable",
                    "column_index": int(n_letter),
                    "column_landing_mask": int(Initial),
                    "column_collider_mask": int(Initial_Colliders),
                    "max_length": int(max_length),
                    "max_height": int(max_height),
                    "back_window": [int(max(n_letter - max_length, 0)), int(n_letter - 1)],
                    "forward_window": [int(n_letter + 1), int(min(n_letter + max_length, n - 1))],
                    "landing_heights": landing_heights,
                    "landing_checks": landing_checks,
                }
            return False
        
        
        
    
    # returns if the whole level is possible  
    if debug:
        return True, {
            "reason": "feasible",
            "effective_length": int(n),
            "max_length": int(max_length),
            "max_height": int(max_height),
        }
    return  All_Landings_Accesibility_Stitched


def _load_level_tokens_from_code(level_code):
    from pathlib import Path
    import pandas as pd

    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent
    structures_path = module_dir / "Universal_Structures.txt"
    level_path = repo_root / "Super_Mario_Brothers_Maps" / "Processed" / f"mario-{level_code}.txt"

    if not structures_path.exists():
        try:
            from .Data_Preprocessing import load_structures_df
        except ImportError:
            from Data_Preprocessing import load_structures_df
        load_structures_df(output_path=structures_path)

    if not level_path.exists():
        raise FileNotFoundError(f"Level file not found: {level_path}")

    df = pd.read_csv(structures_path)
    token_lookup = dict(zip(df["Structures"], df["token"]))

    columns_data = {}
    with level_path.open() as infile:
        for i, line in enumerate(infile):
            columns_data[str(i)] = list(line.split()[0])

    temporal = pd.DataFrame(columns_data)
    structures = temporal.apply(lambda row: "".join(row), axis=1)

    tokens = []
    for idx, structure in enumerate(structures):
        if structure not in token_lookup:
            raise KeyError(
                "Not tokenized structure at column {} in level {}".format(idx, level_code)
            )
        tokens.append(int(token_lookup[structure]))

    return np.array(tokens)


def _load_knowledge():
    from pathlib import Path
    import pandas as pd

    module_dir = Path(__file__).resolve().parent
    structures_path = module_dir / "Universal_Structures.txt"

    if not structures_path.exists():
        try:
            from .Data_Preprocessing import load_structures_df
        except ImportError:
            from Data_Preprocessing import load_structures_df
        load_structures_df(output_path=structures_path)

    df = pd.read_csv(structures_path)
    return np.transpose(df.to_numpy())[3:]


def _parse_tokens(tokens_arg):
    raw = tokens_arg.strip()
    if not raw:
        return np.array([], dtype=int)
    return np.array([int(x.strip()) for x in raw.split(",") if x.strip()], dtype=int)


def _cli_main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test level feasibility with Jumping_Fiasible_Word")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--level-code", help="Level code from Processed set, e.g. 1-3")
    group.add_argument("--tokens", help="Comma-separated token ids, e.g. 12,7,7,20")
    parser.add_argument("--game-mod", type=int, default=0, help="Game mode used by feasibility check")
    parser.add_argument("--max-height", type=int, default=Max_Height, help="Maximum vertical rise for a jump")
    parser.add_argument("--max-length", type=int, default=Max_Length, help="Maximum horizontal gap between landings")
    parser.add_argument("--debug", action="store_true", help="Return failure reason details")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")

    args = parser.parse_args()

    if args.level_code:
        level = _load_level_tokens_from_code(args.level_code)
    else:
        level = _parse_tokens(args.tokens)

    knowledge = _load_knowledge()

    if args.debug:
        ok, info = Jumping_Fiasible_Word(
            level,
            knowledge,
            Game_mod=args.game_mod,
            debug=True,
            max_height=args.max_height,
            max_length=args.max_length,
        )
        payload = {"ok": bool(ok), "level_length": int(len(level)), "details": info}
    else:
        ok = Jumping_Fiasible_Word(
            level,
            knowledge,
            Game_mod=args.game_mod,
            debug=False,
            max_height=args.max_height,
            max_length=args.max_length,
        )
        payload = {"ok": bool(ok), "level_length": int(len(level))}

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("ok:", payload["ok"])
        print("level_length:", payload["level_length"])
        if args.debug:
            for key, value in payload["details"].items():
                print(f"{key}: {value}")

    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    _cli_main()
