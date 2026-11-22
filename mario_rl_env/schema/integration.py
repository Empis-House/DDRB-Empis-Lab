"""Adapters from the existing Gymnasium runners to the canonical record models."""

from __future__ import annotations

import hashlib
import platform
import uuid
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .models import (
    EnvPerformance, EnvironmentSpec, ExecutionContext, ExperimentRun, InlinePlayerState,
    Level, LevelOrigin, Performance, PosterioriDDRBDelta, PosterioriOutcome,
    RLPlayerSnapshot, SnapshotCadenceInfo, TrainingProvenance,
    TrainingState,
)
from .storage import SQLiteExperimentStore


def backend_name(env: Any) -> str:
    backend = getattr(env, "backend", None)
    if backend == "nes":
        return "gym-super-mario-bros"
    if backend == "custom":
        return "procedural_custom_simulator"
    return str(backend or "procedural_custom_simulator")


def runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("gymnasium", "numpy", "torch", "gym-super-mario-bros"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def make_experiment_run(
    *, run_id: str, case_id: str | None, evaluation_stage: str,
    training_env_backend: str, agent_init_seed: int | None, env_seed: int | None,
) -> ExperimentRun:
    return ExperimentRun(
        run_id=run_id,
        case_id=case_id,
        evaluation_stage=evaluation_stage,
        training_env_backend=training_env_backend,
        agent_init_seed=agent_init_seed,
        env_seed=env_seed,
        python_version=platform.python_version(),
        library_versions=runtime_versions(),
    )


def make_player_snapshot(
    agent: Any, *, agent_name: str, agent_id: str, source_environment: str,
    episodes_completed: int, total_env_steps: int, seed: int | None,
    trigger: str, parent_snapshot_id: str | None, checkpoint_uri: str | None = None,
    checkpoint_hash: str | None = None, light: bool = False,
) -> RLPlayerSnapshot:
    metadata = agent.get_snapshot_metadata() if hasattr(agent, "get_snapshot_metadata") else {}
    family = metadata.get("agent_family")
    if family is None:
        family = "value_based_drl" if agent_name == "torch_dqn" else (
            "tabular_rl" if agent_name == "tabular_q" else "heuristic"
        )
    trainable = bool(metadata.get("is_trainable", agent_name in {"tabular_q", "torch_dqn"}))
    return RLPlayerSnapshot(
        player_snapshot_id=str(uuid.uuid4()),
        agent_id=agent_id,
        parent_snapshot_id=parent_snapshot_id,
        agent_name=agent_name,
        agent_family=family,
        is_trainable=trainable,
        framework_backend=metadata.get(
            "framework_backend", "pytorch" if agent_name == "torch_dqn" else "pure_python"
        ),
        observation_contract=metadata.get("observation_contract", {"obs_mode": "features"}),
        action_space=metadata.get("action_space", {"num_actions": getattr(agent, "action_size", 8)}),
        hyperparameters=metadata.get("hyperparameters", {}),
        network_architecture=metadata.get("network_architecture"),
        agent_init_seed=seed,
        training_state=TrainingState(
            total_env_steps=total_env_steps,
            episodes_completed=episodes_completed,
            current_epsilon=metadata.get("current_epsilon"),
            total_gradient_updates=getattr(agent, "curr_step", None),
            replay_buffer_size=metadata.get("replay_buffer_size"),
            q_table_entries=len(getattr(agent, "q_table", {})) if hasattr(agent, "q_table") else None,
        ),
        snapshot_cadence_info=SnapshotCadenceInfo(
            snapshot_trigger=trigger,
            is_light_save_point=light,
            cycle_index=episodes_completed // 1000,
            checkpoint_uri=checkpoint_uri,
            checkpoint_hash=checkpoint_hash,
        ),
        training_provenance=TrainingProvenance(
            source_environment=source_environment,
            is_frozen=not trainable,
        ),
    )


def _level_for_episode(env: Any, info: dict[str, Any], run_id: str, episode_index: int) -> Level:
    level_id = info.get("level_id") or getattr(env, "level_id", None)
    backend = backend_name(env)
    if backend == "gym-super-mario-bros" and level_id:
        prefix = "SuperMarioBros-"
        stage = str(level_id)
        if stage.startswith(prefix):
            stage = stage[len(prefix):].removesuffix("-v0")
        return Level.classic_stage("SuperMarioBros", stage, rom_mode="gymnasium-v0")

    raw_env = getattr(env, "unwrapped", env)
    tilemap = getattr(raw_env, "_tilemap_source", None)
    if tilemap is not None and getattr(tilemap, "tiles", None) is not None:
        return Level.from_tilemap(
            tilemap.tiles,
            origin=LevelOrigin(
                source_type="pcg_qd_map_elites" if level_id else "handcrafted",
                instance_id=str(level_id) if level_id else None,
            ),
        )
    identity = f"{run_id}:{episode_index}:{level_id or 'unknown'}"
    return Level(
        level_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        classic_stage_ref={"game": backend, "id": level_id},
        origin=LevelOrigin(source_type="handcrafted"),
    )


def record_episode(
    store: SQLiteExperimentStore, *, run_id: str, agent_name: str,
    evaluation_stage: str, mode: str, split_role: str, snapshot: RLPlayerSnapshot,
    env: Any, info: dict[str, Any], episode_index: int, total_env_steps_start: int,
    total_env_steps_end: int, episodes_completed_end: int, episode_seconds: float,
    total_reward: float, steps: int, terminated: bool, truncated: bool,
    seed: int | None = None, batch_index: int | None = None,
    performance_id: str | None = None, trajectory_sidecar_uri: str | None = None,
    termination_reason: str | None = None,
) -> Performance:
    level = _level_for_episode(env, info, run_id, episode_index)
    store.upsert_level(level)
    won = bool(info.get("won") or info.get("clear") or info.get("flag_get"))
    died = bool(info.get("died") or info.get("death"))
    if termination_reason is not None:
        reason = termination_reason
    elif won:
        reason = "win_flag"
    elif died:
        reason = "death_enemy"
    elif truncated:
        reason = "truncated_step_cap"
    else:
        reason = "unknown"
    progress = float(info.get("progress_tiles", info.get("progress_max_tiles", 0.0)) or 0.0)
    max_progress = float(info.get("progress_max_tiles", progress) or 0.0)
    width = level.representation.width if level.representation else None
    ratio = min(1.0, max(0.0, max_progress / width)) if width else None
    actual_steps = int(info.get("steps", info.get("step", steps)) or steps)
    protocol_reward = float(info.get("total_reward", info.get("benchmark_reward", total_reward)) or 0.0)
    coins = info.get("coins")
    performance = Performance(
        performance_id=performance_id or str(uuid.uuid4()),
        run_id=run_id,
        episode_index=episode_index,
        batch_index=batch_index,
        player_snapshot_id=snapshot.player_snapshot_id,
        inline_player_state=InlinePlayerState(
            total_env_steps_start=total_env_steps_start,
            total_env_steps_end=total_env_steps_end,
            episodes_completed_end=episodes_completed_end,
            epsilon_at_episode=snapshot.training_state.current_epsilon,
        ),
        level_id=level.level_id,
        execution_context=ExecutionContext(
            mode=mode, split_role=split_role,
            env_reset_seed=seed,
        ),
        environment_spec=EnvironmentSpec(
            env_backend=backend_name(env),
            pixels_per_tile=16 if backend_name(env) == "gym-super-mario-bros" else 64,
        ),
        posteriori_outcome=PosterioriOutcome(
            termination_reason=reason,
            terminated=bool(terminated),
            truncated=bool(truncated),
            clear=won,
            death=died,
        ),
        env_performance=EnvPerformance(
            steps_taken=actual_steps,
            wall_clock_seconds=max(0.0, episode_seconds),
            ms_per_step=(max(0.0, episode_seconds) * 1000.0 / actual_steps) if actual_steps else 0.0,
            progress_tiles=progress,
            progress_max_tiles=max_progress,
            progress_ratio=ratio,
            protocol_reward_total=protocol_reward,
            protocol_reward_components={
                "progress": max_progress,
                "coins": 2.0 * int(coins or 0),
                "step_penalty": -0.01 * actual_steps,
                "win_bonus": 100.0 if won else 0.0,
                "death_penalty": -15.0 if died else 0.0,
            },
            native_reward_total=float(info["native_reward"]) if info.get("native_reward") is not None else None,
            coins_collected=int(coins) if coins is not None else None,
        ),
        posteriori_ddrb_delta=PosterioriDDRBDelta(
            reward_efficiency=protocol_reward / max(actual_steps, 1),
        ),
        trajectory_sidecar_uri=trajectory_sidecar_uri,
    )
    store.insert_performance(
        performance,
        agent_name=agent_name,
        evaluation_stage=evaluation_stage,
        seed=seed,
    )
    return performance
