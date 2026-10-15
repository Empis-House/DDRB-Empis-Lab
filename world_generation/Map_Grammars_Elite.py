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

import random
import threading
import json
import logging
import pandas as pd
import numpy as np
from itertools import product
from collections import deque
import time
from datetime import datetime, timezone
from scipy.spatial import distance
import sys
import os
from pathlib import Path

logger = logging.getLogger(__name__)

from ml_profiler import profile
from level_formats import build_token_structure_lookup, level_string_to_tilemap
try:
    # Prefer the package-relative import so the exception types line up with
    # world_generation.API's `except QueueBackpressureError` handling; fall back to
    # the sys.path-based absolute import for standalone/script execution.
    from .request_queue import RequestQueueManager, QueueBackpressureError, DifficultyPoolManager
except ImportError:
    from request_queue import RequestQueueManager, QueueBackpressureError, DifficultyPoolManager

sys.path.insert(0, os.path.dirname(__file__))

#CReate Map Elite Class
#Good explanation on https://arxiv.org/abs/2205.05834
class Map_Elite:
    all = []
    
    def __init__(self, function, alphas =[0], Variety_Domains = [], Grid_points =[], state_path=None, grid_mode="fixed", instance_id=""):
        # function must acept a Level as input
        # fuction must have Image the real interval [0,1]
        # The Filtter will capting the optimal value (min)
        self.__N = len(alphas)
        self.__f = function
        self.__alphas = alphas
        self.__Variety_Domains = Variety_Domains
        self.__state_path = Path(state_path) if state_path else None
        self.__grid_mode = grid_mode
        self.__instance_id = str(instance_id or "")
        
        self.__Opt_Values = None
        self.__Optimal_strings = None
        
        self.__Grid_points = self.__normalize_grid_points(Grid_points, grid_mode)
        
        if not(self.__Variety_Domains):
            self.__Opt_Values = np.ones(self.__N)
            self.__Optimal_strings = [""]*self.__N
            
        elif len(self.__Variety_Domains) == len(self.__Grid_points):
            self.__Opt_Values, self.__Optimal_strings = self.transform_lists_to_dict(self.__Grid_points,self.__N)
            
        else:
            print("The Variety Grid failse, the Map have set as Default")
            self.__Opt_Values = np.ones(self.__N)
            self.__Optimal_strings = [""]*self.__N

        # knowledge base – path resolved relative to this module so it works
        # regardless of working directory (local dev, Docker, etc.)
        _structures_path = Path(__file__).resolve().parent / "Universal_Structures.txt"
        if not _structures_path.exists():
            from .Data_Preprocessing import load_structures_df  # build on first use
            load_structures_df(output_path=_structures_path)
        self.df = pd.read_csv(_structures_path)
        self._token_to_structure = build_token_structure_lookup(self.df)
        self._structure_to_token = {structure: token for token, structure in self._token_to_structure.items()}
        dfnp = np.transpose(self.df.to_numpy())
        self.Knowledge = dfnp[3:]

        # Data Manager
        self.__manager_memory    = {}         # { agent_name: { 'strings': [...], 'generation': int, 'request_signature': str } }
        self.__current_generation = 0         # monotonic counter; the "flag"
        self.__refresh_lock      = threading.Lock()
        self.__queue_manager      = RequestQueueManager()  # backpressure + priority + dead-letter capture
        self.__difficulty_pool    = DifficultyPoolManager(capacity=32)  # opt-in producer/consumer human-lane pool
        self.__difficulty_recent_served = {}  # difficulty -> deque of recently served level strings (avoid quick repeats)
        self.__difficulty_refill_state_lock = threading.Lock()
        self.__difficulty_refills = set()
        if self.__state_path is not None:
            self.__load_manager_state()

    def get_queue_snapshot(self, queue_key):
        return self.__queue_manager.snapshot(queue_key)

    def get_dead_letters(self, queue_key=None):
        return self.__queue_manager.dead_letters(queue_key)
    
    def __repr__(self):
        return str([self.__Optimal_strings,self.__Opt_Values])

    def get_function_name(self):
        return getattr(self.__f, "__name__", str(self.__f))

    def get_alphas(self):
        return list(self.__alphas)

    def get_variety_domain_names(self):
        return [getattr(fn, "__name__", str(fn)) for fn in self.__Variety_Domains]

    def get_grid_points(self):
        return [list(point) for point in self.__Grid_points]

    def get_grid_mode(self):
        return self.__grid_mode

    def get_optimal_count(self):
        if not self.__Variety_Domains:
            return sum(1 for s in self.__Optimal_strings if s != "")
        return sum(
            1
            for values in self.__Optimal_strings.values()
            for s in values
            if s != ""
        )

    def get_current_generation(self):
        return int(self.__current_generation)

    def get_instance_id(self):
        return self.__instance_id

    def get_state_path(self):
        return str(self.__state_path) if self.__state_path is not None else None

    def refresh_generated_mapping(self, clear_manager_memory=False, **request):
        self.Generate_Mapping(**request)
        self.__current_generation += 1
        if clear_manager_memory:
            self.__manager_memory = {}
        self.__save_manager_state()

    def __manager_state_path(self):
        return str(self.__state_path) if self.__state_path is not None else None

    def __load_manager_state(self):
        path = self.__manager_state_path()
        if path is None or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.__current_generation = int(data.get("generation", self.__current_generation))
            self.__manager_memory = {}
            metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
            if metadata:
                self.__instance_id = str(metadata.get("instance_id", self.__instance_id))
            for agent_name, payload in data.get("manager_memory", {}).items():
                strings = payload.get("strings", [])
                generation = int(payload.get("generation", self.__current_generation))
                if not isinstance(strings, list):
                    strings = []
                self.__manager_memory[agent_name] = {
                    "strings": strings,
                    "generation": generation,
                    "request_signature": payload.get("request_signature", ""),
                }
            self.__queue_manager.load_state(data.get("queue_state", {}))
        except Exception as exc:
            print(f"[Map_Elite] Warning: failed to load manager state from {path}: {exc}")

    def __save_manager_state(self):
        path = self.__manager_state_path()
        if path is None:
            return
        try:
            state_dir = os.path.dirname(path) or "."
            os.makedirs(state_dir, exist_ok=True)
            tmp_path = f"{path}.tmp"
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            updated_at = datetime.now(timezone.utc).isoformat()
            data = {
                "schema_version": 2,
                "metadata": {
                    "instance_id": self.__instance_id,
                    "updated_at": updated_at,
                    "function": self.get_function_name(),
                    "alphas": self.get_alphas(),
                    "variety_domains": self.get_variety_domain_names(),
                    "grid_mode": self.get_grid_mode(),
                    "grid_points": self.get_grid_points(),
                    "state_path": self.get_state_path(),
                },
                "generation": self.__current_generation,
                "manager_memory": self.__manager_memory,
                # Real per-request lineage (run_id/split/seed/params/timestamps) is
                # recorded by world_generation.lineage from the API layer, not here.
                "queue_state": self.__queue_manager.to_state(),
            }
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp_path, path)
        except Exception as exc:
            print(f"[Map_Elite] Warning: failed to save manager state to {path}: {exc}")
    
    def __len__(self):
        return self.__N

    def __normalize_grid_points(self, grid_points, grid_mode):
        normalized_points = [list(point) for point in grid_points]

        if grid_mode == "fixed":
            return normalized_points

        if grid_mode == "percentile":
            percentile_points = []
            for point in normalized_points:
                if len(point) == 0:
                    percentile_points.append([])
                elif len(point) == 1:
                    percentile_points.append([float(point[0])])
                else:
                    quantiles = np.linspace(0.0, 1.0, len(point) + 2)[1:-1]
                    percentile_points.append(np.quantile(np.asarray(point, dtype=float), quantiles).tolist())
            return percentile_points

        raise ValueError(f"Unknown grid_mode '{grid_mode}'. Expected 'fixed' or 'percentile'.")
    
    def transform_lists_to_dict(self, lists, N):
        # Using itertools.product to get all combinations and converting to a list
        keys = list(product(*lists))
        
        # Creating the dictionaries
        transformed_dict = {tuple(key): np.ones(N) for key in keys}
        transformed_dict2 = {tuple(key): ['']*N for key in keys}
        
        return transformed_dict, transformed_dict2

    def get_nearest_label(self,input_tuple, dictionary):
        # Initialize the minimum distance to a large number
        min_dist = float('inf')
        nearest_key = None
        
        # Iterate through the dictionary keys
        for key in dictionary.keys():
            # Calculate the Euclidean distance between input_tuple and the current key
            dist = distance.euclidean(input_tuple, key)
            
            # Update nearest_key if the current distance is smaller than the minimum distance
            if dist < min_dist:
                min_dist = dist
                nearest_key = key
        
        return nearest_key

    def apply_functions(self,functions, x):
        """
        Apply a list of functions to an input x and return the results as a tuple.

        :param functions: List of functions [f1, f2, f3]
        :param x: The input to the functions
        :return: A tuple of results (f1(x), f2(x), f3(x))
        """
        return tuple(f(x,df=self.df) for f in functions)   
            
    def Quest(self, Level):
        y = self.__f(Level,df=self.df)
        
        if not(self.__Variety_Domains):
            for i in range(self.__N):
                if self.__Optimal_strings[i]=="" or np.abs(self.__alphas[i]-y) < np.abs(self.__alphas[i]-self.__Opt_Values[i]):
                    self.__Opt_Values[i] = y
                    self.__Optimal_strings[i] = Level
            return
        

        input_tuple = self.get_nearest_label(self.apply_functions(self.__Variety_Domains, Level), self.__Opt_Values)
        for i in range(self.__N):
            if self.__Optimal_strings[input_tuple][i]=="" or np.abs(self.__alphas[i]-y) < np.abs(self.__alphas[i]-self.__Opt_Values[input_tuple][i]):
                self.__Opt_Values[input_tuple][i] = y
                self.__Optimal_strings[input_tuple][i] = Level
        return 
    
    def format_array(self,arr, precision=2):
        return np.round(arr, precision)

    def printable_formatted_dict(self,d, precision=2):
        report = []
        for key, value in d.items():
            rounded_value = self.format_array(value, precision)
            report.append({key: rounded_value})
        return report
    
    def Report(self, Type = "Values"):
        if Type == "Values":
            return self.__Opt_Values
        if Type == "Strings":
            return self.__Optimal_strings
        
    def Structure(self, row):
        return "".join(row)

    def Key_substraction(self, x):
        "-------------X"
        "---------XXXXX"
        "--------------"
        "-------XXXXXXX"
        if len(self.df[self.df["Structures"] ==  x]) == 0:
            raise IndexError("Not tokenized Structure")
        return self.df[self.df["Structures"] ==  x]["token"]

    def Extract_Level_String(self, examples_codes):
        all_temporals = []

        for example_code in examples_codes:
            columns_data = {}  # Collect column data here first
            level_path = Path(__file__).resolve().parent.parent / "Super_Mario_Brothers_Maps" / "Processed" / f"mario-{example_code}.txt"
            with level_path.open("r", encoding="utf-8") as infile:
                for i, line in enumerate(infile):
                    columns_data[str(i)] = list(line.split()[0])

            # Build the DataFrame all at once — no fragmentation
            Temporal = pd.DataFrame(columns_data)
            all_temporals.append(Temporal)

        Level = pd.concat(all_temporals, axis=0, ignore_index=True)

        Level["Structures"] = Level.apply(self.Structure, axis=1)
        try:
            tokens = [self._structure_to_token[structure] for structure in Level["Structures"]]
        except KeyError as exc:
            raise IndexError(f"Not tokenized Structure: {exc.args[0]}") from exc

        return np.asarray(tokens, dtype=np.int64)
    
    def str_to_tilemap(self, Level):
        from Word_Stitching import Jumping_Fiasible_Word

        # print(Jumping_Fiasible_Word(Level,knowledge = Knowledge))
        return level_string_to_tilemap(Level, self._token_to_structure)

    def __generate_constrained_level(self,
                                     grammar,
                                     level_len,
                                     module,
                                     game_mode,
                                     start_token=None,
                                     allowed_end_tokens=None,
                                     seed=None,
                                     max_attempts=256):
        from Word_Stitching import Jumping_Fiasible_Word

        sequence = grammar.N_Level_Generator_Constrained(
            level_len,
            from_key=(start_token or ''),
            allowed_end_keys=allowed_end_tokens,
            seed=seed,
            module=module,
            Gen_Game_mod=game_mode,
            max_attempts=max_attempts,
        )._Sequence

        if not Jumping_Fiasible_Word(sequence, knowledge=self.Knowledge, Game_mod=game_mode):
            raise RuntimeError("Constrained generator produced a non-feasible sequence")
        return sequence

    def __grammar_degree_sets(self, grammar, min_degree=2):
        degree = max(1, int(min_degree))
        start_tokens = grammar.inner_degree_tokens(min_degree=degree)
        if not start_tokens:
            start_tokens = grammar.out_degree_tokens(min_degree=degree)
        end_tokens = grammar.out_degree_tokens(min_degree=degree)
        if not end_tokens:
            end_tokens = list(start_tokens)
        return start_tokens, end_tokens

    
    
    def Generate_Mapping(self,
                         example_code,
                         Level_Len,
                         module=None,
                         batch_size=1,
                         epochs=256,
                         seed=None,
                         Game_Mode = 0,
                         observability=False,
                         use_grammar_degree_grid=False,
                         grammar_min_degree=2,
                         grammar_end_out_degree=2,
                         grammar_bulk_repeats=1,
                         grammar_max_attempts=256):
        if not(module):
            module = Level_Len

        from Grammar_Notation import Grammar
        from plotly.subplots import make_subplots

        example_codes = [example_code] if isinstance(example_code, str) else list(example_code)
        if not example_codes:
            raise ValueError("At least one example_code is required")
        level_sequences = [self.Extract_Level_String([code]) for code in example_codes]
        level_sec = level_sequences[0]

        G = Grammar(level_sec, knowledge=self.Knowledge, additional_examples=level_sequences[1:])
        start_tokens = []
        end_tokens = []
        if use_grammar_degree_grid:
            start_tokens, end_tokens = self.__grammar_degree_sets(G, min_degree=grammar_min_degree)
            enforced_end_tokens = G.out_degree_tokens(min_degree=max(1, int(grammar_end_out_degree)))
            if enforced_end_tokens:
                end_tokens = enforced_end_tokens

        if use_grammar_degree_grid and start_tokens and end_tokens:
            pre_Level = self.__generate_constrained_level(
                grammar=G,
                level_len=Level_Len,
                module=module,
                game_mode=Game_Mode,
                start_token=start_tokens[0],
                allowed_end_tokens=end_tokens,
                seed=seed,
                max_attempts=grammar_max_attempts,
            )
        else:
            pre_Level = self.__generate_constrained_level(
                grammar=G,
                level_len=Level_Len,
                module=module,
                game_mode=Game_Mode,
                seed=seed,
                max_attempts=grammar_max_attempts,
            )
        self.Quest(pre_Level)

        if use_grammar_degree_grid and start_tokens and end_tokens:
            per_start_repeats = max(1, int(grammar_bulk_repeats))
            for _ in range(per_start_repeats):
                for start_token in start_tokens:
                    try:
                        seeded_level = self.__generate_constrained_level(
                            grammar=G,
                            level_len=Level_Len,
                            module=module,
                            game_mode=Game_Mode,
                            start_token=start_token,
                            allowed_end_tokens=end_tokens,
                            seed=None if seed is None else int(seed) + len(start_token),
                            max_attempts=grammar_max_attempts,
                        )
                        self.Quest(seeded_level)
                    except RuntimeError:
                        # Keep search robust when one start token is too restrictive.
                        continue


        #fig = make_subplots(rows=1, cols=batch_size, subplot_titles=[f"Level {i}" for i in range(batch_size)])

        with profile.context("data_prefetch"):
            for j in range(batch_size):
                
                base_time = time.time()
                t=0
                T=0
                for i in range(epochs):
                    if use_grammar_degree_grid and start_tokens and end_tokens:
                        start_token = start_tokens[(i + j) % len(start_tokens)]
                        pre_Level = self.__generate_constrained_level(
                            grammar=G,
                            level_len=Level_Len,
                            module=module,
                            game_mode=Game_Mode,
                            start_token=start_token,
                            allowed_end_tokens=end_tokens,
                            seed=None if seed is None else int(seed) + i + (j * max(1, epochs)),
                            max_attempts=grammar_max_attempts,
                        )
                        T += 1
                    else:
                        candidate_seed = None if seed is None else int(seed) + i + (j * max(1, epochs))
                        pre_Level = self.__generate_constrained_level(
                            grammar=G,
                            level_len=Level_Len,
                            module=module,
                            game_mode=Game_Mode,
                            seed=candidate_seed,
                            max_attempts=grammar_max_attempts,
                        )
                        T+=1
                    
                    t+=1
                    self.Quest(pre_Level)
            self.Report()
        self.__save_manager_state()
        #fig.show()
                    
        


        #profile.print_summary() 

    #display and save final level if name is included
    def Display_Level(self, Level, level_name=None):

        Display = self.str_to_tilemap(Level)
        if level_name != None:
            Display.to_csv(r'Super_Mario_Brothers_Maps/generated_levels/Run_Search/{}.txt'.format(level_name), index=False,header=False,sep=",")
        
        print(Display)   

    def find_exact_matches(self, quest):
        # Create a dictionary to map each value in the reference list to its index
        reference_index_map = {value: idx for idx, value in enumerate(self.__alphas)}
        
        # List to store the matches
        matches = []
        
        # Iterate over each element in the quest list
        for q in quest:
            # Check if the element is in the reference map and append its index if it is
            if q in reference_index_map:
                matches.append(reference_index_map[q])
        
        return matches

    def __request_signature(self, request):
        compact = {
            "mapping": self.__mapping_request(request),
            "target_alpha": (request or {}).get("target_alpha") if isinstance(request, dict) else None,
            "alpha_window_size": (request or {}).get("alpha_window_size") if isinstance(request, dict) else None,
            "alpha_window": (request or {}).get("alpha_window") if isinstance(request, dict) else None,
        }
        try:
            return json.dumps(compact, sort_keys=True, default=str)
        except Exception:
            return str(compact)

    def __mapping_request(self, request):
        if not isinstance(request, dict):
            return {}

        allowed_keys = {
            "example_code",
            "Level_Len",
            "module",
            "batch_size",
            "epochs",
            "seed",
            "Game_Mode",
            "observability",
            "use_grammar_degree_grid",
            "grammar_min_degree",
            "grammar_end_out_degree",
            "grammar_bulk_repeats",
            "grammar_max_attempts",
        }
        return {key: request[key] for key in allowed_keys if key in request}

    def __resolve_requested_alpha_indexes(self, request):
        if self.__N <= 0:
            return []

        if not isinstance(request, dict):
            return list(range(self.__N))

        target_alpha = request.get("target_alpha")
        if target_alpha is None:
            return list(range(self.__N))

        try:
            target_alpha = float(target_alpha)
        except (TypeError, ValueError):
            return list(range(self.__N))

        alpha_window = request.get("alpha_window")
        if isinstance(alpha_window, (list, tuple)) and len(alpha_window) == 2:
            try:
                low = float(alpha_window[0])
                high = float(alpha_window[1])
            except (TypeError, ValueError):
                low = high = target_alpha
        else:
            try:
                width = max(0.0, float(request.get("alpha_window_size", 0.0)))
            except (TypeError, ValueError):
                width = 0.0
            low = target_alpha - (width / 2.0)
            high = target_alpha + (width / 2.0)

        indexes = [
            idx for idx, alpha in enumerate(self.__alphas)
            if low <= float(alpha) <= high
        ]
        if indexes:
            return indexes

        nearest = min(
            range(self.__N),
            key=lambda idx: abs(float(self.__alphas[idx]) - target_alpha),
        )
        return [nearest]

    def Plot_Coordinate(self, cartesian_list, level_curves,level_name=None):
        conflictive_level_curves = [x for x in level_curves if x not in self.__alphas]
        for x in conflictive_level_curves: print(f"level curve {x} is not an acumulation point")
        indexes = self.find_exact_matches(level_curves)
        if len(indexes) == 0 : print("Level curve without matches")
        for x in cartesian_list:
            near_x = self.get_nearest_label(x,self.__Optimal_strings)
            for i in indexes:
                try:
                    self.Display_Level(self.__Optimal_strings[near_x][i], level_name + str(x))
                    original_val = [f(Level=self.__Optimal_strings[near_x][i],df=self.df) for f in self.__Variety_Domains]
                    print("variety class: ",near_x," // original_vals: ", original_val, " // Performance: ", self.__f(self.__Optimal_strings[near_x][i],df=self.df))

                except:
                    print(f"{near_x} might has no density in C = {self.__alphas[i]}")

    def __extract_valid_levels(self, request=None):
        indexes = self.__resolve_requested_alpha_indexes(request)
        if not self.__Variety_Domains:
            return [self.__Optimal_strings[i] for i in indexes if self.__Optimal_strings[i] != ""]

        import itertools
        cartesian_list = list(itertools.product(*self.__Grid_points))
        valid_levels = []

        for x in cartesian_list:
            near_x = self.get_nearest_label(x, self.__Optimal_strings)
            for i in indexes:
                if self.__Optimal_strings[near_x][i] != '':
                    valid_levels.append(self.__Optimal_strings[near_x][i])

        return valid_levels

    def _hasher(self):
        return
    
    def pool_manager(self, agent_name, request, batch_size=1, queue_key=None, max_pending=8, priority=0,
                      difficulty=None, refill_threshold=4):
        """Deliver pre-generated level strings/tilemap to RL agents.

        Each agent owns an isolated batch built from a snapshot of
        __Optimal_strings. When any agent exhausts its batch the first
        caller triggers Generate_Mapping(request); every later caller that
        exhausted in the same generation skips the call and just refills
        from the already-refreshed snapshot.

        queue_key tracks per-run/per-case contention. Requests are admitted
        up to max_pending concurrently; once at capacity, a higher-priority
        request preempts the lowest-priority pending one (which is recorded
        as a dead letter) and otherwise QueueBackpressureError is raised.
        Any request that fails after being admitted (e.g. generation could
        not produce levels) is also recorded as a dead letter instead of
        being silently dropped.

        If `difficulty` is given, requests are served from a producer/consumer
        pool keyed by that label instead of the per-agent snapshot above: the
        pool is topped up (not cleared) whenever it drops below
        `refill_threshold`, so gameplay only blocks on generation when the
        pool is fully drained rather than once per exhausted agent batch.
        """
        agent_name = str(agent_name or "").strip()
        batch_size = max(1, int(batch_size or 1))
        if not agent_name:
            raise ValueError("agent_name is required")
        request_signature = self.__request_signature(request)
        mapping_request = self.__mapping_request(request)
        queue_key = str(queue_key or f"agent:{agent_name}").strip() or f"agent:{agent_name}"
        request_id = self.__queue_manager.admit(
            queue_key, agent_name, priority=priority, max_pending=max_pending,
        )
        try:
            if difficulty is not None:
                tilemaps = self.__pool_manager_fill_difficulty(
                    str(difficulty), request, batch_size, refill_threshold,
                )
            else:
                chosen_strings = self.__pool_manager_fill(agent_name, request, batch_size, request_signature, mapping_request)
                tilemaps = [self.str_to_tilemap(level) for level in chosen_strings]
        except Exception as exc:
            self.__queue_manager.fail(queue_key, request_id, reason=str(exc))
            raise
        self.__save_manager_state()
        self.__queue_manager.release(queue_key, request_id)
        return tilemaps

    def __pool_manager_fill_difficulty(self, difficulty, request, batch_size, refill_threshold):
        refill_threshold = max(1, int(refill_threshold or 1))
        target = max(refill_threshold, batch_size)
        if self.__difficulty_pool.pending(difficulty) == 0:
            with self.__refresh_lock:
                if self.__difficulty_pool.pending(difficulty) == 0:
                    self.__refill_difficulty_pool(difficulty, request)
        elif self.__difficulty_pool.pending(difficulty) < target:
            self.__schedule_difficulty_refill(difficulty, request, target)

        recent = self.__difficulty_recent_served.setdefault(difficulty, deque(maxlen=64))
        chosen = []
        for _ in range(batch_size):
            try:
                level_string = self.__difficulty_pool.consume(difficulty)
            except RuntimeError:
                with self.__refresh_lock:
                    if self.__difficulty_pool.pending(difficulty) == 0:
                        self.__refill_difficulty_pool(difficulty, request)
                level_string = self.__difficulty_pool.consume(difficulty)
            recent.append(level_string)
            chosen.append(self.str_to_tilemap(level_string))

        if self.__difficulty_pool.pending(difficulty) < target:
            self.__schedule_difficulty_refill(difficulty, request, target)
        return chosen

    def __schedule_difficulty_refill(self, difficulty, request, target):
        with self.__difficulty_refill_state_lock:
            if difficulty in self.__difficulty_refills:
                return
            self.__difficulty_refills.add(difficulty)
            worker = threading.Thread(
                target=self.__run_difficulty_refill,
                args=(difficulty, request, target),
                name=f"level-refill-{difficulty}",
                daemon=True,
            )
            worker.start()

    def __run_difficulty_refill(self, difficulty, request, target):
        try:
            with self.__refresh_lock:
                if self.__difficulty_pool.pending(difficulty) < target:
                    self.__refill_difficulty_pool(difficulty, request)
        except Exception:
            logger.exception("Background refill failed for difficulty %s", difficulty)
        finally:
            with self.__difficulty_refill_state_lock:
                self.__difficulty_refills.discard(difficulty)

    def __refill_difficulty_pool(self, difficulty, request):
        recent = self.__difficulty_recent_served.setdefault(difficulty, deque(maxlen=64))
        queued = self.__difficulty_pool.preview_difficulty(difficulty)

        def _hashable_key(value):
            if isinstance(value, (str, bytes, int, float, bool, type(None))):
                return value
            try:
                return json.dumps(value, sort_keys=True, default=str)
            except Exception:
                return str(value)

        excluded = {_hashable_key(v) for v in recent} | {_hashable_key(v) for v in queued}

        strings = [s for s in self.__extract_valid_levels(request) if _hashable_key(s) not in excluded]
        if not strings:
            # Every known candidate for this difficulty is already queued or was served
            # recently; regenerate so the pool gets genuinely new content instead of
            # recycling the same static archive slice.
            self.refresh_generated_mapping(clear_manager_memory=False, **self.__mapping_request(request))
            strings = [s for s in self.__extract_valid_levels(request) if _hashable_key(s) not in excluded]
        if not strings:
            # Archive still has no fresh alternative (e.g. tiny grid) — fall back to the
            # full candidate set rather than blocking gameplay indefinitely.
            strings = self.__extract_valid_levels(request)
        if not strings:
            raise RuntimeError(f"Difficulty pool '{difficulty}' could not produce any valid levels.")
        random.shuffle(strings)
        self.__difficulty_pool.refill(difficulty, strings)

    def __pool_manager_fill(self, agent_name, request, batch_size, request_signature, mapping_request):
        agent_state = self.__manager_memory.get(agent_name)

        # 1. Initialise agent batch on first visit or when batch is empty
        if (
            agent_state is None
            or agent_state.get("request_signature") != request_signature
            or not agent_state["strings"]
        ):
            strings = self.__extract_valid_levels(request)
            if not strings:
                self.refresh_generated_mapping(clear_manager_memory=False, **mapping_request)
                strings = self.__extract_valid_levels(request)
            if not strings:
                raise RuntimeError("Pool manager could not produce any valid levels. Check Generate_Mapping parameters.")
            self.__manager_memory[agent_name] = {
                "strings": strings,
                "generation": self.__current_generation,
                "request_signature": request_signature,
            }

        batch = self.__manager_memory[agent_name]["strings"]
        chosen_strings = []
        while len(chosen_strings) < batch_size:
            if not batch:
                agent_gen = self.__manager_memory[agent_name]["generation"]
                if agent_gen == self.__current_generation:
                    with self.__refresh_lock:
                        if agent_gen == self.__current_generation:
                            self.refresh_generated_mapping(clear_manager_memory=False, **mapping_request)
                self.__manager_memory[agent_name] = {
                    "strings": self.__extract_valid_levels(request),
                    "generation": self.__current_generation,
                    "request_signature": request_signature,
                }
                batch = self.__manager_memory[agent_name]["strings"]
                if not batch:
                    raise RuntimeError("Pool manager could not produce any valid levels after regeneration.")
            chosen_strings.append(batch.pop(random.randrange(len(batch))))

        self.__manager_memory[agent_name]["strings"] = batch
        return chosen_strings
                


