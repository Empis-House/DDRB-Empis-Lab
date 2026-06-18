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
import os
import time
import argparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mario_rl_env.run_agent import run as agent_run
from mario_rl_env.env import N_ACTIONS
from mario_rl_env.ml_agents import AGENT_MANAGER, RLAgent
import pandas as pd
import itertools


API_BASE_URL = "http://127.0.0.1:8000"


def wait_for_api(url: str, retries: int = 10, delay: float = 1.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)


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


def test_agents() -> None:
    """Test available RL agents to verify they load and run correctly."""
    print("\n" + "="*60)
    print("TESTING AVAILABLE AGENTS")
    print("="*60)
    
    available_agents = AGENT_MANAGER.list_agents()
    print(f"Available agents: {available_agents}\n")
    
    for agent_name in available_agents:
        print(f"Testing agent: {agent_name}")
        try:
            agent = AGENT_MANAGER.create(agent_name, action_size=N_ACTIONS)
            print(f"  ✓ {agent_name} created successfully")
            print(f"  ✓ Agent type: {type(agent).__name__}")
        except Exception as e:
            print(f"  ✗ Failed to create {agent_name}: {e}")
    
    print("="*60 + "\n")


def test_checkpoint_persistence() -> None:
    """Test that torch_dqn agent saves and loads checkpoints correctly."""
    print("\n" + "="*60)
    print("TESTING CHECKPOINT PERSISTENCE (torch_dqn)")
    print("="*60)
    
    try:
        from mario_rl_env.ml_agents import TorchDQNAgent
        
        # Create agent
        agent1 = AGENT_MANAGER.create("torch_dqn", action_size=N_ACTIONS)
        print(f"✓ Created initial agent")
        print(f"  Initial exploration_rate: {agent1.exploration_rate}")
        print(f"  Initial curr_step: {agent1.curr_step}")
        
        # Simulate some training steps
        import numpy as np
        for i in range(5):
            agent1.curr_step += 100
            agent1.exploration_rate *= agent1.exploration_decay ** 100
        
        print(f"\n✓ Simulated training:")
        print(f"  Updated exploration_rate: {agent1.exploration_rate:.6f}")
        print(f"  Updated curr_step: {agent1.curr_step}")
        
        # Save checkpoint
        checkpoint_path = "checkpoints/test_torch_dqn_persistence.pt"
        agent1.save(checkpoint_path)
        print(f"\n✓ Saved checkpoint to {checkpoint_path}")
        
        # Load checkpoint using TorchDQNAgent.load()
        agent2 = TorchDQNAgent.load(checkpoint_path)
        print(f"✓ Loaded checkpoint")
        print(f"  Loaded exploration_rate: {agent2.exploration_rate:.6f}")
        print(f"  Loaded curr_step: {agent2.curr_step}")
        
        # Verify persistence
        if agent2.curr_step == agent1.curr_step and abs(agent2.exploration_rate - agent1.exploration_rate) < 1e-6:
            print(f"\n✓ PERSISTENCE VERIFIED: Training state properly restored!")
            print(f"  Network weights restored: {agent2.net is not None}")
            return True
        else:
            print(f"\n✗ PERSISTENCE FAILED: State mismatch")
            return False
            
    except Exception as e:
        print(f"✗ Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("="*60 + "\n")


def parse_args() -> argparse.Namespace:
    """Minimal CLI for running agent training from main.py."""
    parser = argparse.ArgumentParser(description="Run Mario RL training through API map generation")
    parser.add_argument("--agent", default="smart_heuristic", help="Agent name to run")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for map regeneration")
    parser.add_argument("--render", action="store_true", help="Render all episodes")
    parser.add_argument("--render-last", action="store_true", help="Render only last try of each batch")
    parser.add_argument("--checkpoint-path", default="", help="Checkpoint path for trainable agents")
    parser.add_argument("--skip-self-tests", action="store_true", help="Skip startup agent/self persistence checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_self_tests:
        # Test available agents first
        test_agents()

        # Test checkpoint persistence for torch_dqn
        test_checkpoint_persistence()
    
    # Original main flow with smart_heuristic
    wait_for_api(f"{API_BASE_URL}/")

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

    has_checkpoint = bool(args.checkpoint_path and os.path.exists(args.checkpoint_path))
    persistence_mode = "resume" if has_checkpoint else "fresh"
    trace_tag = f"{args.agent}|persist:{persistence_mode}"

    def agent_runner():
        response = api_post(
            f"/instances/{instance_id}/pool_manager",
            {
                "agent_hash": trace_tag,
                "request": generate_payload,
            },
        )
        return pd.DataFrame(response["tilemap"])

    print(f"[Trace] tag={trace_tag}")

    result = agent_run(
        map_path=agent_runner,
        render=args.render,
        render_last=args.render_last,
        n_episodes=args.episodes,
        batch_size=args.batch_size,
        agent_name=args.agent,
        verbose=True,
        checkpoint_path=args.checkpoint_path,
    )



if __name__ == "__main__":
    main()
