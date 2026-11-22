"""Train on classic SMB stages and compare checkpoints on a shared ROM benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import uuid
from itertools import combinations, count
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .artifact_registry import ArtifactRegistry, DEFAULT_REGISTRY_PATH, ExecutionConfig
from .benchmark_protocol import (
    ACTION_MEANINGS,
    PROTOCOL_METADATA,
    make_classic_env,
)

DEFAULT_STAGES = ("1-1", "1-2", "2-1", "4-2")


def _pairs(values: Iterable[str], argument: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for value in values:
        label, separator, content = value.partition("=")
        if not separator or not label or not content:
            raise ValueError(f"{argument} entries must use LABEL=VALUE, got {value!r}")
        if label in pairs:
            raise ValueError(f"Duplicate {argument} label: {label}")
        pairs[label] = content
    return pairs


def _load_dqn_checkpoint(path: str):
    from .ml_agents import TORCH_AVAILABLE, TorchDQNAgent

    if not TORCH_AVAILABLE:
        raise RuntimeError("Checkpoint benchmarking requires torch, torchvision, tensordict, and torchrl")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    agent = TorchDQNAgent.load(path)
    if agent.action_size != len(ACTION_MEANINGS):
        raise ValueError(
            f"Checkpoint has {agent.action_size} actions; benchmark protocol requires "
            f"{len(ACTION_MEANINGS)}. Train a fresh checkpoint with this protocol."
        )
    agent.exploration_rate = 0.0
    return agent


def _runtime_metadata() -> dict:
    from importlib.metadata import version

    return {
        "python": platform.python_version(),
        "gymnasium": version("gymnasium"),
        "numpy": version("numpy"),
        "torch": version("torch"),
        "gym_super_mario_bros": version("gym-super-mario-bros"),
        "nes_py": version("nes-py"),
    }


def _summarize(records: List[dict]) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    return {
        "episodes": len(records),
        "mean_reward": statistics.fmean(row["total_reward"] for row in records),
        "std_reward": statistics.pstdev(row["total_reward"] for row in records),
        "mean_progress_tiles": statistics.fmean(row["max_progress_tiles"] for row in records),
        "std_progress_tiles": statistics.pstdev(row["max_progress_tiles"] for row in records),
        "win_rate": statistics.fmean(float(row["won"]) for row in records),
        "death_rate": statistics.fmean(float(row["died"]) for row in records),
        "mean_steps": statistics.fmean(row["steps"] for row in records),
    }


def _evaluate_checkpoint(
    *,
    checkpoint_path: str,
    training_environment: str,
    stages: Tuple[str, ...],
    episodes: int,
    seed: int,
    render: bool,
    agent_name: str,
    experiment_store=None,
) -> dict:
    from .schema.integration import (
        make_experiment_run, make_player_snapshot, record_episode,
    )

    agent = _load_dqn_checkpoint(checkpoint_path)
    run_id = f"evaluation-{agent_name}-{uuid.uuid4()}"
    evaluation_stage = (
        "stage_1_classic_baseline"
        if training_environment == "gym-super-mario-bros"
        else "stage_2_cross_origin"
    )
    active_snapshot = None
    total_env_steps = 0
    if experiment_store is not None:
        experiment_store.upsert_run(make_experiment_run(
            run_id=run_id,
            case_id=None,
            evaluation_stage=evaluation_stage,
            training_env_backend=training_environment,
            agent_init_seed=seed,
            env_seed=seed,
        ))
        active_snapshot = make_player_snapshot(
            agent,
            agent_name=agent_name,
            agent_id=f"{agent_name}:{training_environment}",
            source_environment=training_environment,
            episodes_completed=0,
            total_env_steps=int(getattr(agent, "curr_step", 0)),
            seed=seed,
            trigger="initial",
            parent_snapshot_id=None,
        )
        experiment_store.upsert_snapshot(active_snapshot)
    per_stage = {}
    all_records = []

    for stage_index, stage in enumerate(stages):
        env = make_classic_env(
            stage,
            render_mode="human" if render else None,
            seed=seed + stage_index * episodes,
        )
        stage_records = []
        try:
            for episode in range(episodes):
                episode_started = time.perf_counter()
                env_seed = seed + stage_index * episodes + episode
                observation, _ = env.reset(seed=env_seed)
                terminated = truncated = False
                total_reward = 0.0
                info = {}
                steps = 0
                while not (terminated or truncated):
                    action = agent.act(observation)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                episode_info = info["episode"]
                record = {
                    "stage": stage,
                    "episode": episode + 1,
                    "total_reward": float(total_reward),
                    "steps": steps,
                    "max_progress_tiles": float(episode_info["max_progress_tiles"]),
                    "won": bool(episode_info["won"]),
                    "died": bool(episode_info["died"]),
                    "termination_reason": episode_info["termination_reason"],
                    "coins": int(episode_info["coins"]),
                }
                stage_records.append(record)
                all_records.append(record)
                if experiment_store is not None and active_snapshot is not None:
                    record_episode(
                        experiment_store,
                        run_id=run_id,
                        agent_name=agent_name,
                        evaluation_stage=evaluation_stage,
                        mode="testing",
                        split_role="test",
                        snapshot=active_snapshot,
                        env=env,
                        info=info.get("episode", info),
                        episode_index=total_env_steps,
                        total_env_steps_start=total_env_steps,
                        total_env_steps_end=total_env_steps + steps,
                        episodes_completed_end=len(all_records),
                        episode_seconds=time.perf_counter() - episode_started,
                        total_reward=total_reward,
                        steps=steps,
                        terminated=terminated,
                        truncated=truncated,
                        seed=env_seed,
                        batch_index=stage_index,
                    )
                total_env_steps += steps
        finally:
            env.close()
        per_stage[stage] = {
            "metrics": _summarize(stage_records),
            "episodes": stage_records,
        }

    return {
        "training_environment": training_environment,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "overall": _summarize(all_records),
        "stages": per_stage,
    }


def compare_checkpoints(
    *,
    models: Dict[str, str],
    training_environments: Dict[str, str],
    stages: Tuple[str, ...] = DEFAULT_STAGES,
    episodes: int = 5,
    seed: int = 0,
    render: bool = False,
    experiment_store=None,
) -> dict:
    if set(models) != set(training_environments):
        missing_training = sorted(set(models) - set(training_environments))
        missing_models = sorted(set(training_environments) - set(models))
        raise ValueError(
            f"Each model needs a training-environment label. Missing metadata for "
            f"{missing_training}; metadata without a model: {missing_models}"
        )
    if not models:
        raise ValueError("At least one model checkpoint is required")
    if not stages or episodes <= 0:
        raise ValueError("At least one stage and a positive episode count are required")

    results = {
        label: _evaluate_checkpoint(
            checkpoint_path=path,
            training_environment=training_environments[label],
            stages=stages,
            episodes=episodes,
            seed=seed,
            render=render,
            agent_name=label,
            experiment_store=experiment_store,
        )
        for label, path in models.items()
    }
    episodes_by_training_environment: Dict[str, List[dict]] = {}
    for model_result in results.values():
        environment = model_result["training_environment"]
        episodes_by_training_environment.setdefault(environment, []).extend(
            episode
            for stage_result in model_result["stages"].values()
            for episode in stage_result["episodes"]
        )
    environment_summaries = {
        environment: _summarize(episodes)
        for environment, episodes in episodes_by_training_environment.items()
    }
    pairwise_differences = []
    for left, right in combinations(sorted(environment_summaries), 2):
        pairwise_differences.append({
            "training_environment_a": left,
            "training_environment_b": right,
            "mean_reward_b_minus_a": (
                environment_summaries[right]["mean_reward"]
                - environment_summaries[left]["mean_reward"]
            ),
            "mean_progress_tiles_b_minus_a": (
                environment_summaries[right]["mean_progress_tiles"]
                - environment_summaries[left]["mean_progress_tiles"]
            ),
            "win_rate_b_minus_a": (
                environment_summaries[right]["win_rate"]
                - environment_summaries[left]["win_rate"]
            ),
        })
    report = {
        "schema_version": 1,
        "kind": "classic_level_cross_environment_comparison",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": PROTOCOL_METADATA,
        "runtime": _runtime_metadata(),
        "evaluation": {
            "levels": list(stages),
            "episodes_per_level": episodes,
            "seed": seed,
            "paired_seeds_across_models": True,
            "rendered": render,
        },
        "models": results,
        "training_environment_summaries": environment_summaries,
        "pairwise_differences": pairwise_differences,
        "limitations": [
            "Both backends share actions, episode limits, reward shaping, and progress units; their native physics and level geometry differ.",
            "The protocol uses one emulator frame per action and does not add frame stacking or frame skipping.",
            "This comparison supports pixel-based DQN checkpoints with the shared eight-action output layer.",
        ],
    }
    if experiment_store is not None:
        from .schema import two_stage_evaluation_report

        report["two_stage_evaluation"] = two_stage_evaluation_report(
            experiment_store,
            metric="progress_ratio",
            split_role="test",
        )
    return report


def _write_json(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def _train_classic(args: argparse.Namespace) -> None:
    if args.agent != "torch_dqn":
        raise ValueError("Classic benchmark training currently supports --agent torch_dqn only")
    if not args.stages or args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("Stages, episodes, and batch size must be positive")

    from .ml_agents import AGENT_MANAGER, run_training_loop

    stage_cycle = count()

    def environment_factory():
        stage_index = next(stage_cycle)
        return make_classic_env(
            args.stages[stage_index % len(args.stages)],
            render_mode="human" if args.render else None,
            seed=args.seed + stage_index,
        )

    run_id = f"classic-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    execution_config = ExecutionConfig.resolve(
        mode="training",
        overrides={
            "agent_name": args.agent,
            "run_id": run_id,
            "seed": args.seed,
            "extra": {
                "training_environment": "gym-super-mario-bros",
                "stages": list(args.stages),
                "benchmark_protocol": PROTOCOL_METADATA,
            },
        },
    )
    execution_config.apply_seed()
    agent = AGENT_MANAGER.create(args.agent, action_size=len(ACTION_MEANINGS))
    registry = ArtifactRegistry(DEFAULT_REGISTRY_PATH)
    registry.record_run_start(
        agent_name=args.agent,
        mode="training",
        run_id=run_id,
        config_hash=execution_config.config_hash(),
        metadata={
            "training_environment": "gym-super-mario-bros",
            "stages": list(args.stages),
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "seed_used": execution_config.seed_used,
            "benchmark_protocol": PROTOCOL_METADATA,
        },
    )
    from .schema import SQLiteExperimentStore

    with SQLiteExperimentStore(args.schema_store) as experiment_store:
        history = run_training_loop(
            environment_factory,
            agent,
            n_episodes=args.episodes,
            batch_size=args.batch_size,
            render=args.render,
            verbose=True,
            history_path=args.history,
            checkpoint_path=args.checkpoint,
            trace_tag=run_id,
            registry=registry,
            execution_config=execution_config,
            experiment_store=experiment_store,
            evaluation_stage="stage_1_classic_baseline",
        )
    payload = {
        "schema_version": 1,
        "kind": "classic_level_training",
        "run_id": run_id,
        "training_environment": "gym-super-mario-bros",
        "stages": list(args.stages),
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "seed": execution_config.seed_used,
        "benchmark_protocol": PROTOCOL_METADATA,
        "runtime": _runtime_metadata(),
        "training_metrics": _summarize([
            {
                "total_reward": row["total_reward"],
                "steps": row["steps"],
                "won": row["info"]["episode"]["won"],
                "died": row["info"]["episode"]["died"],
                "max_progress_tiles": row["info"]["episode"]["max_progress_tiles"],
            }
            for row in history
        ]),
        "episode_levels": [
            {
                "episode": row["episode"],
                "level_id": row["info"]["episode"]["level_id"],
            }
            for row in history
        ],
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    _write_json(args.report, payload)
    registry.record_run_complete(
        agent_name=args.agent,
        mode="training",
        run_id=run_id,
        config_hash=execution_config.config_hash(),
        metrics=payload["training_metrics"],
    )
    print(f"Saved classic-training report to {args.report}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train against classic SMB stages and compare checkpoints on a common ROM benchmark"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-classic", help="Train a fresh DQN on rotating classic SMB stages")
    train.add_argument("--agent", default="torch_dqn", choices=["torch_dqn"])
    train.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    train.add_argument("--episodes", type=int, default=500)
    train.add_argument("--batch-size", type=int, default=10)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--history", default="")
    train.add_argument("--report", default="benchmark/classic_training.json")
    train.add_argument("--schema-store", default="checkpoints/experiments.sqlite3")
    train.add_argument("--render", action="store_true")

    compare = commands.add_parser("compare", help="Evaluate models from different training environments on classic stages")
    compare.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=CHECKPOINT",
        help="Checkpoint to compare; repeat for each model",
    )
    compare.add_argument(
        "--training-env",
        action="append",
        required=True,
        metavar="LABEL=DESCRIPTION",
        help="Training environment for each model label, e.g. procedural_custom_simulator",
    )
    compare.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    compare.add_argument("--episodes", type=int, default=5)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--report", default="benchmark/classic_comparison.json")
    compare.add_argument("--schema-store", default="checkpoints/experiments.sqlite3")
    compare.add_argument("--render", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "train-classic":
        _train_classic(args)
        return

    models = _pairs(args.model, "--model")
    training_environments = _pairs(args.training_env, "--training-env")
    from .schema import SQLiteExperimentStore

    with SQLiteExperimentStore(args.schema_store) as experiment_store:
        report = compare_checkpoints(
            models=models,
            training_environments=training_environments,
            stages=tuple(args.stages),
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            experiment_store=experiment_store,
        )
    _write_json(args.report, report)
    print(f"Saved classic-level comparison to {args.report}")


if __name__ == "__main__":
    main()
