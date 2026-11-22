"""Episode-granular exports and optional rliable metric calculation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .storage import SQLiteExperimentStore


def episode_score_matrix(
    store: SQLiteExperimentStore,
    *,
    evaluation_stage: str,
    split_role: str = "test",
    metric: str = "progress_ratio",
) -> tuple[list[str], list[str], np.ndarray]:
    """Build agent x level score matrices, averaging repeat episodes per cell."""
    if metric not in {"progress_ratio", "protocol_reward_total"}:
        raise ValueError("metric must be progress_ratio or protocol_reward_total")
    rows = store.query_episode_rows(evaluation_stage=evaluation_stage, split_role=split_role)
    agents = sorted({row["agent_name"] for row in rows})
    levels = sorted({row["level_id"] for row in rows})
    if not agents or not levels:
        return agents, levels, np.empty((len(agents), len(levels)), dtype=np.float64)
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row[metric]
        if value is not None and np.isfinite(value):
            samples[(row["agent_name"], row["level_id"])].append(float(value))
    matrix = np.full((len(agents), len(levels)), np.nan, dtype=np.float64)
    for ai, agent in enumerate(agents):
        for li, level in enumerate(levels):
            values = samples.get((agent, level), [])
            if values:
                matrix[ai, li] = float(np.mean(values))
    return agents, levels, matrix


def episode_score_matrices(
    store: SQLiteExperimentStore,
    *,
    evaluation_stage: str,
    split_role: str = "test",
    metric: str = "progress_ratio",
) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    """Return one run x episode-task matrix per agent without averaging episodes."""
    if metric not in {"progress_ratio", "protocol_reward_total"}:
        raise ValueError("metric must be progress_ratio or protocol_reward_total")
    rows = store.query_episode_rows(
        evaluation_stage=evaluation_stage, split_role=split_role,
    )
    agents = sorted({row["agent_name"] for row in rows})
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["agent_name"]][row["run_id"]].append(row)

    run_samples: dict[str, dict[str, dict[tuple[str, int], float]]] = {}
    task_keys: set[tuple[str, int]] = set()
    for agent in agents:
        run_samples[agent] = {}
        for run_id, run_rows in grouped[agent].items():
            occurrence: dict[str, int] = defaultdict(int)
            samples: dict[tuple[str, int], float] = {}
            for row in sorted(run_rows, key=lambda item: item["episode_index"]):
                level_id = row["level_id"]
                key = (level_id, occurrence[level_id])
                occurrence[level_id] += 1
                task_keys.add(key)
                value = row[metric]
                if value is not None and np.isfinite(value):
                    samples[key] = float(value)
            run_samples[agent][run_id] = samples

    ordered_tasks = sorted(task_keys, key=lambda item: (item[0], item[1]))
    task_labels = [f"{level_id}#episode_{index}" for level_id, index in ordered_tasks]
    matrices: dict[str, np.ndarray] = {}
    for agent in agents:
        run_ids = sorted(run_samples[agent])
        matrix = np.full((len(run_ids), len(ordered_tasks)), np.nan, dtype=np.float64)
        for row_index, run_id in enumerate(run_ids):
            samples = run_samples[agent][run_id]
            for column_index, task in enumerate(ordered_tasks):
                if task in samples:
                    matrix[row_index, column_index] = samples[task]
        matrices[agent] = matrix
    return agents, task_labels, matrices


def two_stage_evaluation_report(
    store: SQLiteExperimentStore,
    *,
    metric: str = "progress_ratio",
    split_role: str = "test",
    random_agent_name: str = "random",
) -> dict[str, Any]:
    """Summarize Stage 1/2 episode matrices and per-agent zero-shot gaps."""
    stages: dict[str, dict[str, Any]] = {}
    means: dict[str, dict[str, float]] = {}
    for stage in ("stage_1_classic_baseline", "stage_2_cross_origin"):
        agents, levels, matrix = episode_score_matrix(
            store, evaluation_stage=stage, split_role=split_role, metric=metric,
        )
        agent_means: dict[str, float] = {}
        for index, agent in enumerate(agents):
            valid = matrix[index][np.isfinite(matrix[index])]
            if valid.size:
                agent_means[agent] = float(np.mean(valid))
        stages[stage] = {
            "agents": agents, "level_ids": levels, "scores": matrix,
            "mean_scores": agent_means,
        }
        means[stage] = agent_means
    stage1 = means["stage_1_classic_baseline"]
    stage2 = means["stage_2_cross_origin"]
    zsg_gap = {
        agent: stage1[agent] - stage2[agent]
        for agent in sorted(stage1.keys() & stage2.keys())
    }
    return {
        "metric": metric,
        "split_role": split_role,
        "stages": stages,
        "zero_shot_generalization_gap_stage1_minus_stage2": zsg_gap,
        "stage2_random_baseline_floor": stage2.get(random_agent_name),
    }


def compute_rliable_metrics(
    scores: np.ndarray | dict[str, np.ndarray],
    *,
    score_range: tuple[float, float] | None = None,
    num_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Compute IQM, optimality gap, and performance profiles via optional rliable."""
    try:
        from rliable import library as rly
        from rliable import metrics
    except ImportError as exc:
        raise ImportError(
            "rliable metrics require the optional package; install requirements-benchmark.txt"
        ) from exc
    if isinstance(scores, dict):
        score_dict = {
            str(name): np.asarray(values, dtype=np.float64)
            for name, values in scores.items()
        }
    else:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("scores must have shape [agents, runs, tasks]")
        score_dict = {f"agent_{index}": values[index] for index in range(values.shape[0])}
    if not score_dict or any(value.ndim != 2 for value in score_dict.values()):
        raise ValueError("scores must contain [runs, tasks] matrices for each agent")
    if any(value.shape[0] == 0 or value.shape[1] == 0 for value in score_dict.values()):
        raise ValueError("each agent matrix must contain at least one run and task")
    if any(not np.isfinite(value).all() for value in score_dict.values()):
        raise ValueError("scores must be finite; align or filter missing run/task cells first")
    if score_range is None:
        minimum = min(float(np.min(value)) for value in score_dict.values())
        maximum = max(float(np.max(value)) for value in score_dict.values())
        score_range = (0.0, 1.0) if minimum >= 0.0 and maximum <= 1.0 else (minimum, maximum)
    if not np.isfinite(score_range).all() or score_range[0] >= score_range[1]:
        raise ValueError("score_range must contain finite values with minimum < maximum")

    thresholds = np.linspace(score_range[0], score_range[1], 51)

    def aggregate(agent_scores: np.ndarray) -> np.ndarray:
        return np.asarray([
            metrics.aggregate_iqm(agent_scores),
            metrics.aggregate_optimality_gap(agent_scores, gamma=score_range[1]),
        ])

    estimates, intervals = rly.get_interval_estimates(
        score_dict,
        aggregate,
        reps=num_bootstrap,
        task_bootstrap=True,
        confidence_interval_size=0.95,
    )
    point_estimates = {
        name: {
            "iqm": float(values[0]),
            "optimality_gap": float(values[1]),
        }
        for name, values in estimates.items()
    }
    probability_of_improvement: dict[str, float] = {}
    probability_intervals: dict[str, np.ndarray] = {}
    for left in score_dict:
        for right in score_dict:
            if left == right:
                continue
            comparison = f"{left}_vs_{right}"
            comparison_estimates, comparison_intervals = rly.get_interval_estimates(
                {comparison: [score_dict[left], score_dict[right]]},
                lambda scores_x, scores_y: np.asarray([
                    metrics.probability_of_improvement(scores_x, scores_y)
                ]),
                reps=num_bootstrap,
                confidence_interval_size=0.95,
            )
            probability_of_improvement[comparison] = float(
                comparison_estimates[comparison][0]
            )
            probability_intervals[comparison] = comparison_intervals[comparison]
    profiles, profile_intervals = rly.create_performance_profile(
        score_dict,
        thresholds,
        reps=num_bootstrap,
        task_bootstrap=True,
        confidence_interval_size=0.95,
    )
    return {
        "point_estimates": point_estimates,
        "bootstrap_estimates": estimates,
        "confidence_intervals_95": intervals,
        "performance_profile_thresholds": thresholds,
        "performance_profiles": profiles,
        "performance_profile_confidence_intervals_95": profile_intervals,
        "probability_of_improvement": probability_of_improvement,
        "probability_of_improvement_confidence_intervals_95": probability_intervals,
    }
