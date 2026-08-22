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
import sys
import time
import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Windows consoles default stdout/stderr to cp1252, which can't encode the
# unicode checkmarks (✓/✗) used in self-test output below; reconfigure to
# utf-8 so those prints don't crash the process with UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from mario_rl_env.artifact_registry import ArtifactRegistry, DEFAULT_REGISTRY_PATH, ExecutionConfig
from mario_rl_env.run_agent import run as agent_run
from mario_rl_env.env import N_ACTIONS
from mario_rl_env.ml_agents import AGENT_MANAGER, RLAgent
import pandas as pd
import numpy as np
import itertools
from statistics import mean

from world_generation.tile_pattern_kl import compare_level_sets_tile_pattern_kl, load_level_set, domain_paths
from world_generation.level_formats import rows_to_tilemap, tilemap_to_rows


API_BASE_URL = "http://127.0.0.1:8000"
LOG_USE_COLOR = True

# Palette based on colorblind-friendly hues (Okabe-Ito inspired) mapped to ANSI 256 codes.
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_BLUE = "\033[38;5;33m"
ANSI_SKY = "\033[38;5;39m"
ANSI_GREEN = "\033[38;5;35m"
ANSI_ORANGE = "\033[38;5;214m"
ANSI_MAGENTA = "\033[38;5;134m"
ANSI_GRAY = "\033[38;5;245m"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return hasattr(__import__("sys").stdout, "isatty") and __import__("sys").stdout.isatty()


def _paint(text: str, color: str = "", bold: bool = False, dim: bool = False) -> str:
    if not LOG_USE_COLOR:
        return text
    prefix = ""
    if dim:
        prefix += ANSI_DIM
    if bold:
        prefix += ANSI_BOLD
    if color:
        prefix += color
    return f"{prefix}{text}{ANSI_RESET}"


def _scope_color(scope: str, message: str) -> str:
    lowered = f"{scope} {message}".lower()
    if "error" in lowered or "failed" in lowered:
        return ANSI_ORANGE
    if scope.startswith("api"):
        return ANSI_SKY
    if scope in {"training", "trace"}:
        return ANSI_GREEN
    if scope in {"vars", "report"}:
        return ANSI_MAGENTA
    if scope in {"run.config", "instance", "domains", "selftest"}:
        return ANSI_BLUE
    return ANSI_GRAY


def _compact(value, max_len: int = 180) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    else:
        text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _log(run_id: str, scope: str, message: str, quiet: bool = False, **fields) -> None:
    if quiet:
        return
    ts = _paint(_timestamp(), dim=True)
    run = _paint(f"run={run_id}", color=ANSI_BLUE)
    scope_block = _paint(f"{scope:<12}", color=_scope_color(scope, message), bold=True)
    msg_block = _paint(message, bold=True)
    prefix = f"{ts} | {run} | {scope_block} | {msg_block}"
    if fields:
        # Multi-line field rendering keeps dense logs readable in narrow terminal panes.
        if len(fields) >= 4:
            print(prefix)
            for key, value in fields.items():
                key_block = _paint(f"{key:<14}", color=ANSI_GRAY)
                print(f"    - {key_block}: {_compact(value)}")
        else:
            details = "  |  ".join(f"{_paint(k, color=ANSI_GRAY)}={_compact(v)}" for k, v in fields.items())
            print(f"{prefix} | {details}")
    else:
        print(prefix)


def _log_section(run_id: str, title: str, quiet: bool = False) -> None:
    if quiet:
        return
    bar = _paint("=" * 90, color=ANSI_SKY)
    print(f"\n{bar}")
    header = f"{_paint(_timestamp(), dim=True)} | {_paint(f'run={run_id}', color=ANSI_BLUE)} | {_paint('SECTION', color=ANSI_SKY, bold=True):<12} | {_paint(title, color=ANSI_SKY, bold=True)}"
    print(header)
    print(bar)
    print()


def _log_general_variables(run_id: str, phase: str, quiet: bool = False) -> None:
    _log(
        run_id,
        "vars",
        f"general variables ({phase})",
        quiet=quiet,
        example_codes=example_codes,
        level_len=level_len,
        module=module,
        grid_mode=grid_mode,
        requested_grid_points=grid_points,
        alphas=alphas,
        domain_window=DOMAIN_WINDOW,
        domain_stride=DOMAIN_STRIDE,
    )


def wait_for_api(url: str, retries: int = 10, delay: float = 1.0, run_id: str = "-", quiet: bool = False) -> None:
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                if response.status == 200:
                    _log(run_id, "api.wait", "API ready", quiet=quiet, url=url, attempt=attempt)
                    return
        except Exception as exc:
            if attempt == retries:
                _log(run_id, "api.wait", "API not reachable", quiet=quiet, url=url, attempt=attempt, error=exc)
                raise
            _log(run_id, "api.wait", "retrying API health check", quiet=quiet, url=url, attempt=attempt, retries=retries)
            time.sleep(delay)


def api_post(path, payload, run_id: str = "-", trace_tag: str = "-", request_id: str = "-", quiet: bool = False):
    url = f"{API_BASE_URL}{path}"
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    payload_keys = list(payload.keys()) if isinstance(payload, dict) else []
    _log(
        run_id,
        "api.post",
        "request",
        quiet=quiet,
        request_id=request_id,
        trace=trace_tag,
        path=path,
        payload_keys=payload_keys,
        payload_type=type(payload).__name__,
    )
    try:
        with urlopen(request) as response:
            result = json.load(response)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _log(
                run_id,
                "api.post",
                "response",
                quiet=quiet,
                request_id=request_id,
                status=response.status,
                elapsed_ms=elapsed_ms,
                keys=list(result.keys()) if isinstance(result, dict) else type(result).__name__,
            )
            return result
    except HTTPError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        message = exc.read().decode("utf-8", errors="ignore")
        _log(
            run_id,
            "api.post",
            "http_error",
            quiet=quiet,
            request_id=request_id,
            status=exc.code,
            reason=exc.reason,
            elapsed_ms=elapsed_ms,
            body=message,
        )
        raise RuntimeError(f"API request failed: {exc.code} {exc.reason} {message}")
    except URLError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _log(
            run_id,
            "api.post",
            "connection_error",
            quiet=quiet,
            request_id=request_id,
            reason=exc.reason,
            elapsed_ms=elapsed_ms,
        )
        raise RuntimeError(f"API connection failed: {exc.reason}")

### variety aggregate:
#  overworld: mean_similarity=0.691 mean_CoD=0.449
#  underground: mean_similarity=0.644 mean_CoD=0.553
#  athletic: mean_similarity=0.638 mean_CoD=0.569

# '1-1','1-2','1-3','2-1','3-1','3-3','4-1','4-2','5-1','5-3','6-1','6-2','6-3','7-1','8-1'

example_codes = ['1-1','1-2','4-2','5-1']
level_len = 40
module = level_len  # Temp: module != Level_Len could generate buged but solvable levels
grid_points = [[0.4, 0.449, 0.49], [0.53, 0.553, 0.58], [0.53, 0.569, 0.59]]
grid_mode = "percentile"
alphas = [0, 0.1, 1]  # These are the system acumulation points on the interval [0,1]
DOMAIN_WINDOW = (6, 6)
DOMAIN_STRIDE = (3, 3)


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
    
    import tempfile

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
        
        # Self-test scratch file: use a temp dir, not the checkpoints/ lineage tree,
        # so it never collides with or pollutes real agent_name-keyed checkpoints.
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "test_torch_dqn_persistence.pt")
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
    parser.add_argument(
        "--mode",
        default="training",
        choices=["training", "testing", "validation", "inference"],
        help="Agent execution mode (default: training). Non-training modes force a fixed/deterministic seed.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Optional identifier grouping this run with others for comparison/reporting "
        "(e.g. same seed run with different agents). Leave unset for a standalone run.",
    )
    parser.add_argument("--skip-self-tests", action="store_true", help="Skip startup agent/self persistence checks")
    parser.add_argument("--quiet", action="store_true", help="Reduce custom logging output")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in logs")
    return parser.parse_args()


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _descriptive_stats(values):
    if not values:
        return {
            "n": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "max": 0.0,
            "std": 0.0,
        }

    arr = np.asarray([_to_float(v) for v in values], dtype=float)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


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
        closest_counts = {}
        for entry in variety_summaries:
            closest = entry.get("closest_domain")
            if closest:
                closest_counts[closest] = closest_counts.get(closest, 0) + 1

        for domain_name in domain_names:
            similarities = [entry["domains"][domain_name]["similarity"] for entry in variety_summaries if domain_name in entry.get("domains", {})]
            CoDs = [entry["domains"][domain_name]["CoD"] for entry in variety_summaries if domain_name in entry.get("domains", {})]
            klds = [entry["domains"][domain_name]["kld_symmetric"] for entry in variety_summaries if domain_name in entry.get("domains", {})]
            aggregate[domain_name] = {
                "samples": len(similarities),
                "similarity": _descriptive_stats(similarities),
                "CoD": _descriptive_stats(CoDs),
                "kld_symmetric": _descriptive_stats(klds),
            }
        report["level_variety_aggregate"] = aggregate
        report["variety_overview"] = {
            "levels_evaluated": len(variety_summaries),
            "closest_domain_counts": closest_counts,
        }

    return report


def _write_run_manifest(context: dict, report_path: str, result: dict | None = None) -> None:
    manifest = {
        "schema_version": 2,
        "kind": "agent_run_manifest",
        "created_at": datetime.now().isoformat(),
        "context": context,
        "result_summary": {
            "mode": report_path.split("/")[-1] if report_path else None,
            "history_entries": len(result.get("history", [])) if isinstance(result, dict) and isinstance(result.get("history"), list) else 0,
        },
    }
    directory = os.path.dirname(report_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _print_agent_report(report):
    ctx = report.get("context", {})
    metrics = report.get("metrics", {})
    aggregate = report.get("level_variety_aggregate", {})
    overview = report.get("variety_overview", {})
    run_id = ctx.get("run_id", "-")

    _log_section(run_id, "FINAL REPORT", quiet=False)
    _log(
        run_id,
        "report",
        "run summary",
        quiet=False,
        agent=ctx.get("agent_name"),
        mode=report.get("mode"),
        episodes=ctx.get("episodes"),
        batch_size=ctx.get("batch_size"),
        instance_id=ctx.get("instance_id"),
    )

    requested_grid = ctx.get("requested_grid_points")
    effective_grid = ctx.get("effective_grid_points")
    if requested_grid == effective_grid:
        _log(
            run_id,
            "report",
            "grid configuration",
            quiet=False,
            grid_mode=ctx.get("grid_mode"),
            effective_grid_points=effective_grid,
        )
    else:
        _log(
            run_id,
            "report",
            "grid configuration",
            quiet=False,
            grid_mode=ctx.get("grid_mode"),
            requested_grid_points=requested_grid,
            effective_grid_points=effective_grid,
        )

    if report.get("mode") == "training":
        _log(
            run_id,
            "report",
            "training metrics",
            quiet=False,
            episodes_seen=metrics.get("episodes", 0),
            reward_mean=round(metrics.get("mean_reward", 0.0), 3),
            reward_min=round(metrics.get("min_reward", 0.0), 3),
            reward_max=round(metrics.get("max_reward", 0.0), 3),
            wins=metrics.get("wins", 0),
            win_rate_pct=round(100.0 * metrics.get("win_rate", 0.0), 2),
            mean_steps=round(metrics.get("mean_steps", 0.0), 1),
            mean_max_x=round(metrics.get("mean_max_x", 0.0), 1),
        )
    else:
        _log(run_id, "report", "evaluation metrics", quiet=False, metrics=metrics)

    if aggregate:
        _log(
            run_id,
            "report",
            "variety overview",
            quiet=False,
            levels_evaluated=overview.get("levels_evaluated", 0),
            closest_domain_counts=overview.get("closest_domain_counts", {}),
        )
        for domain_name, stats in aggregate.items():
            sim = stats.get("similarity", {})
            cod = stats.get("CoD", {})
            kld = stats.get("kld_symmetric", {})
            _log(
                run_id,
                "report",
                f"variety aggregate :: {domain_name}",
                quiet=False,
                samples=stats.get("samples", 0),
                sim_mean=round(sim.get("mean", 0.0), 3),
                sim_median=round(sim.get("median", 0.0), 3),
                sim_std=round(sim.get("std", 0.0), 3),
                sim_range=f"[{sim.get('min', 0.0):.3f}, {sim.get('max', 0.0):.3f}]",
                cod_mean=round(cod.get("mean", 0.0), 3),
                cod_median=round(cod.get("median", 0.0), 3),
                cod_std=round(cod.get("std", 0.0), 3),
                cod_iqr=f"[{cod.get('p25', 0.0):.3f}, {cod.get('p75', 0.0):.3f}]",
                kld_mean=round(kld.get("mean", 0.0), 3),
            )


def _build_level_variety_summary(level_frame, level_index, domain_cache):
    level_grid = tilemap_to_rows(level_frame)
    domain_summaries = {}

    for domain_name, domain_levels in domain_cache.items():
        stats = compare_level_sets_tile_pattern_kl(
            [level_grid],
            domain_levels,
            window=DOMAIN_WINDOW,
            stride=DOMAIN_STRIDE,
        )
        domain_summaries[domain_name] = {
            "CoD": float(stats["CoD"]),
            "kld_symmetric": float(stats["kld_symmetric"]),
            "similarity": 1.0 / (1.0 + float(stats["CoD"])),
        }

    closest_domain = min(domain_summaries, key=lambda name: domain_summaries[name]["CoD"])
    return {
        "level_index": level_index,
        "closest_domain": closest_domain,
        "domains": domain_summaries,
    }


def main() -> None:
    global LOG_USE_COLOR
    args = parse_args()
    LOG_USE_COLOR = _supports_color() and not args.no_color
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    main_started = time.perf_counter()
    _log_section(run_id, "RUN START", quiet=args.quiet)
    _log(
        run_id,
        "run.config",
        "parsed arguments",
        quiet=args.quiet,
        agent=args.agent,
        episodes=args.episodes,
        batch_size=args.batch_size,
        render=args.render,
        render_last=args.render_last,
        checkpoint_path=args.checkpoint_path or "<none>",
        report_file=args.report_file or "<none>",
        skip_self_tests=args.skip_self_tests,
    )
    _log_general_variables(run_id, phase="start", quiet=args.quiet)

    if not args.skip_self_tests:
        # Test available agents first
        test_agents()

        # Test checkpoint persistence for torch_dqn
        test_checkpoint_persistence()
    else:
        _log(run_id, "selftest", "startup self tests skipped", quiet=args.quiet)
    
    # Original main flow with smart_heuristic
    wait_for_api(f"{API_BASE_URL}/", run_id=run_id, quiet=args.quiet)

    create_payload = {
        "function": "Performance",
        "alphas": alphas,
        "variety_domains": ["overworld", "underground", "athletic"],
        "grid_points": grid_points,
        "grid_mode": grid_mode,
        "instance_id":args.agent,
    }
    api_sequence = 0

    def next_request_id(prefix: str) -> str:
        nonlocal api_sequence
        api_sequence += 1
        return f"{prefix}-{api_sequence:04d}"

    bootstrap_trace = f"bootstrap|{run_id}"
    instance = api_post(
        "/instances",
        create_payload,
        run_id=run_id,
        trace_tag=bootstrap_trace,
        request_id=next_request_id("instances.create"),
        quiet=args.quiet,
    )
    instance_id = instance["instance_id"]
    _log(run_id, "instance", "instance created", quiet=args.quiet, instance_id=instance_id)
    _log(
        run_id,
        "grid",
        "grid configuration resolved",
        quiet=args.quiet,
        mode=instance.get("grid_mode", create_payload["grid_mode"]),
        requested=create_payload["grid_points"],
        effective=instance.get("grid_points", create_payload["grid_points"]),
    )

    generate_payload = {
        "example_code": example_codes,
        "Level_Len": level_len,
        "module": module,
        "batch_size": 5,
        "epochs": 10,
    }
    # queue_key isolates this run's pool_manager queue from other concurrent runs;
    # case_id is the (optional) user-supplied grouping id used by the artifact
    # registry for reporting/A-B comparison and must not be conflated with it.
    agent_request_payload = {
        "agent_name": bootstrap_trace,
        "request": generate_payload,
        "batch_size": max(1, args.batch_size),
        "case_id": args.case_id,
        "queue_key": f"run:{run_id}|agent:{args.agent}",
        "priority": 0,
    }
    generate_result = api_post(
        f"/instances/{instance_id}/generate_mapping",
        generate_payload,
        run_id=run_id,
        trace_tag=bootstrap_trace,
        request_id=next_request_id("mapping.generate"),
        quiet=args.quiet,
    )
    domain_root_paths = domain_paths()
    _log(run_id, "domains", "loading domain exemplars", quiet=args.quiet, domains=list(domain_root_paths.keys()))
    cached_domain_levels = {
        domain_name: load_level_set(domain_path, recursive=True)
        for domain_name, domain_path in domain_root_paths.items()
    }
    level_variety_summaries = []
    level_counter = 0

    if args.checkpoint_path:
        has_checkpoint = os.path.exists(args.checkpoint_path)
    else:
        # No explicit --checkpoint-path: run_agent.run() will resolve the
        # agent_name-keyed lineage itself (case-tagged if --case-id given, else
        # `latest`); check the same lineage here purely for logging.
        from mario_rl_env.artifact_registry import CheckpointManager
        has_checkpoint = bool(
            CheckpointManager(ArtifactRegistry(DEFAULT_REGISTRY_PATH)).resolve_resume_path(args.agent, args.case_id)
        )
    persistence_mode = "resume" if has_checkpoint else "fresh"
    trace_tag = f"run:{run_id}|agent:{args.agent}|persist:{persistence_mode}"
    execution_config = ExecutionConfig.resolve(
        mode=args.mode,
        overrides={
            "agent_name": args.agent,
            "case_id": args.case_id,
            "run_id": run_id,
            "seed": None,
            "deterministic": False,
        },
    )
    execution_config.apply_seed()
    registry = ArtifactRegistry(DEFAULT_REGISTRY_PATH)
    registry.record_run_start(
        agent_name=args.agent,
        mode=execution_config.mode,
        run_id=run_id,
        case_id=args.case_id,
        config_hash=execution_config.config_hash(),
        metadata={
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "checkpoint_path": args.checkpoint_path or None,
            "persistence_mode": persistence_mode,
            "seed_used": execution_config.seed_used,
        },
    )

    # Client-side batch cache + background prefetch: pool_manager already returns a
    # batch (args.batch_size levels) per HTTP round-trip; previously agent_runner
    # fetched a *fresh* batch on every call and threw away everything but the first
    # tilemap, so most of the API latency (and the variety-summary work) was pure
    # waste. Now the whole batch is cached client-side and a refill is prefetched in
    # the background once the cache runs low, overlapping API latency with episode
    # execution instead of stalling on it.
    level_cache: "deque" = deque()
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetch_future: Optional[Future] = None
    refill_threshold = max(1, max(1, args.batch_size) // 2)

    def fetch_batch():
        return api_post(
            f"/instances/{instance_id}/pool_manager",
            agent_request_payload,
            run_id=run_id,
            trace_tag=trace_tag,
            request_id=next_request_id("pool.pull"),
            quiet=args.quiet,
        )

    def _extract_tilemaps(response):
        tilemaps = response.get("tilemaps") or []
        if not tilemaps and response.get("tilemap") is not None:
            tilemaps = [response["tilemap"]]
        return tilemaps

    def agent_runner():
        nonlocal level_counter, prefetch_future
        if not level_cache:
            if prefetch_future is None:
                prefetch_future = prefetch_executor.submit(fetch_batch)
            response = prefetch_future.result()
            prefetch_future = None
            tilemaps = _extract_tilemaps(response)
            if not tilemaps:
                raise RuntimeError("pool_manager returned no tilemaps")
            level_cache.extend(tilemaps)

        if len(level_cache) <= refill_threshold and prefetch_future is None:
            prefetch_future = prefetch_executor.submit(fetch_batch)

        tilemap = level_cache.popleft()
        level_counter += 1
        level_frame = rows_to_tilemap(tilemap)
        try:
            summary = _build_level_variety_summary(level_frame, level_counter, cached_domain_levels)
            level_variety_summaries.append(summary)
            bits = []
            for domain_name, stats in summary["domains"].items():
                bits.append(f"{domain_name}: sim={stats['similarity']:.3f} CoD={stats['CoD']:.3f}")
            _log(
                run_id,
                "variety",
                "level summary",
                quiet=args.quiet,
                trace=trace_tag,
                level=level_counter,
                closest=summary["closest_domain"],
                stats=" | ".join(bits),
            )
        except Exception as exc:
            _log(
                run_id,
                "variety",
                "summary unavailable",
                quiet=args.quiet,
                trace=trace_tag,
                level=level_counter,
                error=exc,
            )

        return level_frame

    _log_section(run_id, "TRAINING START", quiet=args.quiet)
    _log(run_id, "trace", "training trace initialized", quiet=args.quiet, trace_tag=trace_tag)

    result = agent_run(
        map_path=agent_runner,
        render=args.render,
        render_last=args.render_last,
        n_episodes=args.episodes,
        batch_size=args.batch_size,
        agent_name=args.agent,
        verbose=not args.quiet,
        checkpoint_path=args.checkpoint_path,
        registry=registry,
        execution_config=execution_config,
    )
    prefetch_executor.shutdown(wait=False, cancel_futures=True)
    _log(
        run_id,
        "training",
        "agent run finished",
        quiet=args.quiet,
        elapsed_s=round(time.perf_counter() - main_started, 2),
        levels_requested=level_counter,
    )

    context = {
        "run_id": run_id,
        "trace_tag": trace_tag,
        "agent_name": args.agent,
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "instance_id": instance_id,
        "variety_domains": create_payload["variety_domains"],
        "grid_mode": instance.get("grid_mode", create_payload["grid_mode"]),
        "requested_grid_points": create_payload["grid_points"],
        "effective_grid_points": instance.get("grid_points", create_payload["grid_points"]),
        "generate_status": generate_result.get("status") if isinstance(generate_result, dict) else None,
        "level_variety_summaries": level_variety_summaries,
        "case_id": args.case_id,
    }
    report = _format_agent_report(context=context, result=result)
    _print_agent_report(report)

    registry.record_run_complete(
        agent_name=args.agent,
        mode=execution_config.mode,
        run_id=run_id,
        case_id=args.case_id,
        config_hash=execution_config.config_hash(),
        metrics=report.get("metrics", {}),
    )

    if args.report_file:
        with open(args.report_file, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        _write_run_manifest(context=context, report_path=os.path.splitext(args.report_file)[0] + ".manifest.json", result=result)
        _log(run_id, "report", "saved run report", quiet=args.quiet, path=args.report_file)

    _log_general_variables(run_id, phase="end", quiet=args.quiet)

    _log_section(run_id, "RUN END", quiet=args.quiet)



if __name__ == "__main__":
    main()
