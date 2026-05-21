# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:00:20 2023
Check List:
    *More Functions
    *Mapping Functions frome Grids
@author: Juan J. Rueda M.
"""
""""""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mario_rl_env.run_agent import run as agent_run
import pandas as pd
import itertools


API_BASE_URL = "http://127.0.0.1:8000"


def api_post(path, payload):
    url = f"{API_BASE_URL}{path}"
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request) as response:
            return json.load(response)
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API request failed: {exc.code} {exc.reason} {message}")
    except URLError as exc:
        raise RuntimeError(f"API connection failed: {exc.reason}")


example_codes = ["1-1"]
level_len = 40
module = level_len  # Temp: module != Level_Len could generate buged but solvable levels
grid_points = [[0, 0.5], [0, 1]]
alphas = [0, 0.1, 1]  # These are the system acumulation points on the interval [0,1]


def main() -> None:
    create_payload = {
        "function": "Performance",
        "alphas": alphas,
        "variety_domains": ["calculate_slope", "colliders_score"],
        "grid_points": grid_points,
    }
    instance = api_post("/instances", create_payload)
    instance_id = instance["instance_id"]

    generate_payload = {
        "example_code": example_codes,
        "Level_Len": level_len,
        "module": module,
        "batch_size": 5,
        "epochs": 10,
    }
    api_post(f"/instances/{instance_id}/generate_mapping", generate_payload)

    def agent_runner():
        response = api_post(
            f"/instances/{instance_id}/pool_manager",
            {
                "agent_hash": "12345",
                "request": generate_payload,
            },
        )
        return pd.DataFrame(response["tilemap"])

    agent_run(
        map_path=agent_runner,
        render=True,
        obs_mode="features",
        n_episodes=20,
        batch_size=4,
        agent_name="right_jump",
    )


if __name__ == "__main__":
    main()
