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
from statistics import mean

from world_generation.tile_pattern_kl import compare_level_sets_tile_pattern_kl, load_level_set, domain_paths


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
grid_points = [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]
alphas = [0, 0.1, 1]  # These are the system acumulation points on the interval [0,1]
DOMAIN_WINDOW = (2, 2)
DOMAIN_STRIDE = (1, 1)


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
    parser.add_argument("--report-file", default="", help="Optional JSON file to save run report")
    parser.add_argument("--skip-self-tests", action="store_true", help="Skip startup agent/self persistence checks")
    return parser.parse_args()


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _summarize_training_history(history):
    if not history:
        return {
            "episodes": 0,
            "mean_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "mean_steps": 0.0,
            "wins": 0,
            "win_rate": 0.0,
            "mean_max_x": 0.0,
        }

    rewards = [_to_float(row.get("total_reward")) for row in history]
    steps = [_to_float(row.get("steps")) for row in history]

    def _won(row):
        info = row.get("info", {}) if isinstance(row, dict) else {}
        episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
        return bool(episode_info.get("won", info.get("won", False)))

    max_x_values = []
    for row in history:
        info = row.get("info", {}) if isinstance(row, dict) else {}
        episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
        max_x_values.append(_to_float(episode_info.get("max_x", info.get("max_x", 0.0))))

    wins = sum(1 for row in history if _won(row))
    n = len(history)
    return {
        "episodes": n,
        "mean_reward": mean(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_steps": mean(steps),
        "wins": wins,
        "win_rate": wins / n if n > 0 else 0.0,
        "mean_max_x": mean(max_x_values) if max_x_values else 0.0,
    }


def _format_agent_report(context, result):
    report = {
        "context": context,
        "raw_result": result,
    }

    if isinstance(result, dict) and "history" in result:
        history = result.get("history", [])
        report["mode"] = "training"
        report["metrics"] = _summarize_training_history(history)
    else:
        report["mode"] = "evaluation"
        report["metrics"] = result if isinstance(result, dict) else {"result": result}

    variety_summaries = context.get("level_variety_summaries", [])
    report["level_variety_summaries"] = variety_summaries
    if variety_summaries:
        domain_names = list(variety_summaries[0].get("domains", {}).keys())
        aggregate = {}
        for domain_name in domain_names:
            similarities = [entry["domains"][domain_name]["similarity"] for entry in variety_summaries if domain_name in entry.get("domains", {})]
            jsds = [entry["domains"][domain_name]["jsd"] for entry in variety_summaries if domain_name in entry.get("domains", {})]
            aggregate[domain_name] = {
                "mean_similarity": mean(similarities) if similarities else 0.0,
                "mean_jsd": mean(jsds) if jsds else 0.0,
            }
        report["level_variety_aggregate"] = aggregate

    return report


def _print_agent_report(report):
    ctx = report.get("context", {})
    metrics = report.get("metrics", {})
    variety_summaries = report.get("level_variety_summaries", [])
    aggregate = report.get("level_variety_aggregate", {})

    print("\n" + "=" * 70)
    print("AGENT RUN REPORT")
    print("=" * 70)
    print(f"agent: {ctx.get('agent_name')}")
    print(f"mode: {report.get('mode')}")
    print(f"episodes: {ctx.get('episodes')}  batch_size: {ctx.get('batch_size')}")
    print(f"domain_descriptors: {ctx.get('variety_domains')}")
    print(f"grid_points: {ctx.get('grid_points')}")
    print(f"instance_id: {ctx.get('instance_id')}")
    print("-" * 70)

    if report.get("mode") == "training":
        print(f"episodes_seen: {metrics.get('episodes', 0)}")
        print(f"reward mean/min/max: {metrics.get('mean_reward', 0.0):.3f} / {metrics.get('min_reward', 0.0):.3f} / {metrics.get('max_reward', 0.0):.3f}")
        print(f"wins: {metrics.get('wins', 0)}  win_rate: {metrics.get('win_rate', 0.0):.2%}")
        print(f"mean_steps: {metrics.get('mean_steps', 0.0):.1f}  mean_max_x: {metrics.get('mean_max_x', 0.0):.1f}")
    else:
        print(json.dumps(metrics, indent=2))

    if variety_summaries:
        print("-" * 70)
        print("level variety summaries:")
        for entry in variety_summaries:
            level_index = entry.get("level_index")
            closest_domain = entry.get("closest_domain")
            domain_bits = []
            for domain_name, domain_stats in entry.get("domains", {}).items():
                domain_bits.append(
                    f"{domain_name}=sim:{domain_stats['similarity']:.3f} jsd:{domain_stats['jsd']:.3f}"
                )
            print(f"  level {level_index}: closest={closest_domain} | " + "  ".join(domain_bits))

        print("-" * 70)
        print("variety aggregate:")
        for domain_name, stats in aggregate.items():
            print(
                f"  {domain_name}: mean_similarity={stats['mean_similarity']:.3f} "
                f"mean_jsd={stats['mean_jsd']:.3f}"
            )

    print("=" * 70 + "\n")


def _build_level_variety_summary(level_frame, level_index, domain_cache):
    level_grid = level_frame.values.tolist()
    domain_summaries = {}

    for domain_name, domain_levels in domain_cache.items():
        stats = compare_level_sets_tile_pattern_kl(
            [level_grid],
            domain_levels,
            window=DOMAIN_WINDOW,
            stride=DOMAIN_STRIDE,
        )
        domain_summaries[domain_name] = {
            "jsd": float(stats["jsd"]),
            "kld_symmetric": float(stats["kld_symmetric"]),
            "similarity": 1.0 / (1.0 + float(stats["jsd"])),
        }

    closest_domain = min(domain_summaries, key=lambda name: domain_summaries[name]["jsd"])
    return {
        "level_index": level_index,
        "closest_domain": closest_domain,
        "domains": domain_summaries,
    }


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
        "variety_domains": ["overworld", "underground", "athletic"],
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
    generate_result = api_post(f"/instances/{instance_id}/generate_mapping", generate_payload)
    domain_root_paths = domain_paths()
    cached_domain_levels = {
        domain_name: load_level_set(domain_path, recursive=True)
        for domain_name, domain_path in domain_root_paths.items()
    }
    level_variety_summaries = []
    level_counter = 0

    has_checkpoint = bool(args.checkpoint_path and os.path.exists(args.checkpoint_path))
    persistence_mode = "resume" if has_checkpoint else "fresh"
    trace_tag = f"{args.agent}|persist:{persistence_mode}"

    def agent_runner():
        nonlocal level_counter
        response = api_post(
            f"/instances/{instance_id}/pool_manager",
            {
                "agent_hash": trace_tag,
                "request": generate_payload,
            },
        )
        level_counter += 1
        level_frame = pd.DataFrame(response["tilemap"])
        try:
            summary = _build_level_variety_summary(level_frame, level_counter, cached_domain_levels)
            level_variety_summaries.append(summary)
            bits = []
            for domain_name, stats in summary["domains"].items():
                bits.append(f"{domain_name}: sim={stats['similarity']:.3f} jsd={stats['jsd']:.3f}")
            print(f"[Level Variety:{trace_tag}] level={level_counter} closest={summary['closest_domain']} | " + " | ".join(bits))
        except Exception as exc:
            print(f"[Level Variety:{trace_tag}] level={level_counter} summary unavailable: {exc}")
        return level_frame

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

    context = {
        "agent_name": args.agent,
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "instance_id": instance_id,
        "variety_domains": create_payload["variety_domains"],
        "grid_points": create_payload["grid_points"],
        "generate_status": generate_result.get("status") if isinstance(generate_result, dict) else None,
        "level_variety_summaries": level_variety_summaries,
    }
    report = _format_agent_report(context=context, result=result)
    _print_agent_report(report)

    if args.report_file:
        with open(args.report_file, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Saved report to: {args.report_file}")



if __name__ == "__main__":
    main()
