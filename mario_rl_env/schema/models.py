"""Validated, versionable records for DDRB RL experiments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


AgentFamily = Literal[
    "heuristic", "tabular_rl", "value_based_drl", "policy_gradient_drl",
    "actor_critic_drl", "planning_search", "human",
]
EvaluationStage = Literal[
    "stage_1_classic_baseline", "stage_2_cross_origin", "exploratory",
]
SplitRole = Literal["train", "val", "test"]


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExperimentRun(SchemaModel):
    run_id: str
    case_id: str | None = None
    evaluation_stage: EvaluationStage = "exploratory"
    training_env_backend: str
    agent_init_seed: int | None = None
    env_seed: int | None = None
    python_version: str
    library_versions: dict[str, str] = Field(default_factory=dict)
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str | None = None


class ObservationContract(SchemaModel):
    obs_mode: Literal["features", "features_layers", "tiles", "rgb"] = "features"
    shape: list[int] | None = None
    preprocessing: str | None = None


class ActionSpace(SchemaModel):
    space_type: Literal["discrete_8", "custom"] = "discrete_8"
    num_actions: int = Field(default=8, ge=1)
    frame_skip: int = Field(default=1, ge=1)


class TrainingState(SchemaModel):
    total_env_steps: int = Field(default=0, ge=0)
    episodes_completed: int = Field(default=0, ge=0)
    current_epsilon: float | None = None
    total_gradient_updates: int | None = Field(default=None, ge=0)
    replay_buffer_size: int | None = Field(default=None, ge=0)
    q_table_entries: int | None = Field(default=None, ge=0)


class SnapshotCadenceInfo(SchemaModel):
    snapshot_trigger: Literal[
        "initial", "periodic_500_ep", "checkpoint_1000_ep",
        "light_cycle_end", "final", "manual",
    ] = "initial"
    is_light_save_point: bool = False
    cycle_index: int = Field(default=0, ge=0)
    checkpoint_uri: str | None = None
    checkpoint_hash: str | None = None


class TrainingProvenance(SchemaModel):
    source_environment: str
    stages_or_instances_seen: list[str] = Field(default_factory=list)
    is_frozen: bool = False


class RLPlayerSnapshot(SchemaModel):
    player_snapshot_id: str
    agent_id: str
    parent_snapshot_id: str | None = None
    agent_name: str
    agent_family: AgentFamily
    is_trainable: bool
    framework_backend: str
    observation_contract: ObservationContract = Field(default_factory=ObservationContract)
    action_space: ActionSpace = Field(default_factory=ActionSpace)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    network_architecture: dict[str, Any] | None = None
    agent_init_seed: int | None = None
    training_state: TrainingState = Field(default_factory=TrainingState)
    snapshot_cadence_info: SnapshotCadenceInfo = Field(default_factory=SnapshotCadenceInfo)
    training_provenance: TrainingProvenance


class LevelRepresentation(SchemaModel):
    format: Literal["flattened_column_tokens"] = "flattened_column_tokens"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    column_tokens: list[int]
    alphabet_id: str
    column_order: Literal["bottom_to_top", "top_to_bottom"] = "bottom_to_top"

    @model_validator(mode="after")
    def token_count_matches_dimensions(self) -> "LevelRepresentation":
        if len(self.column_tokens) != self.width * self.height:
            raise ValueError("column_tokens must contain width * height values")
        return self


class LevelOrigin(SchemaModel):
    source_type: Literal[
        "original_smb", "pcg_qd_map_elites", "pcg_grammar", "ml_gan",
        "ml_llm", "ml_diffusion", "hybrid_mutated", "handcrafted",
    ]
    generator_name: str | None = None
    instance_id: str | None = None
    generation_seed: int | None = None
    parent_level_ids: list[str] = Field(default_factory=list)
    generation_params: dict[str, Any] = Field(default_factory=dict)


class DiversityQDMetrics(SchemaModel):
    cod_overworld: float | None = None
    cod_underground: float | None = None
    cod_athletic: float | None = None
    variety_domain_coords: list[float | int] | None = None
    kld_set_a_to_set_b: float | None = None
    kld_set_b_to_set_a: float | None = None
    kld_symmetric: float | None = None
    window: tuple[int, int] | None = None
    stride: tuple[int, int] | None = None


class PlayabilityOracle(SchemaModel):
    oracle_name: str = "world_stitching_oracle"
    is_solvable: bool | None = None
    oracle_score: float | None = None
    execution_device: Literal["cuda", "cpu", "unknown"] = "unknown"
    runtime_ms: float | None = Field(default=None, ge=0)
    oracle_version: str | None = None


class DDRBEconomyChallenge(SchemaModel):
    max_theoretical_coins: int | None = Field(default=None, ge=0)
    max_theoretical_progress_tiles: int | None = Field(default=None, ge=0)
    max_theoretical_protocol_reward: float | None = None
    enemy_count: int | None = Field(default=None, ge=0)
    reward_to_enemy_ratio: float | None = None
    metric_definitions_version: str = "1"


class Level(SchemaModel):
    level_id: str
    representation: LevelRepresentation | None = None
    classic_stage_ref: dict[str, Any] | None = None
    origin: LevelOrigin
    diversity_qd_metrics: DiversityQDMetrics = Field(default_factory=DiversityQDMetrics)
    performance_evaluation_metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    era_topology_metrics: dict[str, float | int | None] = Field(default_factory=dict)
    playability_oracle: PlayabilityOracle = Field(default_factory=PlayabilityOracle)
    ddrb_economy_challenge: DDRBEconomyChallenge = Field(default_factory=DDRBEconomyChallenge)

    @classmethod
    def from_tilemap(
        cls,
        tilemap: Any,
        *,
        alphabet_id: str = "mario_rl_tiletype_v1",
        column_order: Literal["bottom_to_top", "top_to_bottom"] = "bottom_to_top",
        origin: LevelOrigin | None = None,
    ) -> "Level":
        """Create a stable level record from top-to-bottom rows x columns."""
        if hasattr(tilemap, "values"):
            rows = tilemap.values.tolist()
        else:
            rows = tilemap.tolist() if hasattr(tilemap, "tolist") else tilemap
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            raise ValueError("tilemap must be a non-empty two-dimensional sequence")
        normalized = [list(row) for row in rows]
        width = len(normalized[0])
        if width == 0 or any(len(row) != width for row in normalized):
            raise ValueError("tilemap rows must have the same non-zero width")
        height = len(normalized)
        tokens: list[int] = []
        for column in range(width):
            column_values = []
            for row in range(height):
                value = normalized[row][column]
                if isinstance(value, str) and not value.strip().lstrip("-").isdigit():
                    from mario_rl_env.tilemap import CHAR_TO_TILE

                    try:
                        value = CHAR_TO_TILE[value.strip()]
                    except KeyError as exc:
                        raise ValueError(f"Unknown tile symbol: {value!r}") from exc
                column_values.append(int(value))
            if column_order == "bottom_to_top":
                column_values.reverse()
            tokens.extend(column_values)
        representation = LevelRepresentation(
            width=width, height=height, column_tokens=tokens,
            alphabet_id=alphabet_id, column_order=column_order,
        )
        identity = {
            "alphabet_id": alphabet_id, "height": height, "width": width,
            "column_tokens": tokens,
        }
        level_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            level_id=level_id,
            representation=representation,
            origin=origin or LevelOrigin(source_type="handcrafted"),
        )

    @classmethod
    def classic_stage(
        cls, game: str, stage: str, *, rom_mode: str = "default",
        task_id: str | None = None, linked_vglc_path: str | None = None,
    ) -> "Level":
        """Create a deterministic identity for a ROM-backed opaque stage."""
        world, separator, level = stage.partition("-")
        if not separator or not world.isdigit() or not level.isdigit():
            raise ValueError(f"stage must have the form WORLD-STAGE, got {stage!r}")
        identity = f"{game}:{world}-{level}:{rom_mode}"
        return cls(
            level_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            classic_stage_ref={
                "game": game, "world": int(world), "stage": int(level),
                "task_id": task_id, "linked_vglc_path": linked_vglc_path,
            },
            origin=LevelOrigin(source_type="original_smb"),
        )

    def to_tilemap(self) -> list[list[int]]:
        """Return rows x columns in the top-to-bottom convention used by API.py."""
        if self.representation is None:
            raise ValueError("opaque classic stages do not have a token tilemap")
        rep = self.representation
        rows = [[0 for _ in range(rep.width)] for _ in range(rep.height)]
        for column in range(rep.width):
            values = rep.column_tokens[column * rep.height:(column + 1) * rep.height]
            if rep.column_order == "bottom_to_top":
                values = list(reversed(values))
            for row, value in enumerate(values):
                rows[row][column] = value
        return rows


class InlinePlayerState(SchemaModel):
    total_env_steps_start: int = Field(ge=0)
    total_env_steps_end: int = Field(ge=0)
    episodes_completed_end: int = Field(ge=0)
    epsilon_at_episode: float | None = None


class ExecutionContext(SchemaModel):
    mode: Literal["training", "validation", "testing", "inference"]
    split_role: SplitRole
    level_selection_policy: Literal[
        "fixed_set", "uniform_random", "pool_manager_ddrb", "sequential_rotation",
    ] = "fixed_set"
    distribution_split: Literal["in_distribution", "held_out_iid", "ood_cross_origin"] = "in_distribution"
    env_reset_seed: int | None = None


class EnvironmentSpec(SchemaModel):
    env_backend: str
    pixels_per_tile: int = Field(ge=1)
    max_steps_cap: int = Field(default=500000, ge=1)
    reward_protocol_version: str = "mario-rl-benchmark-v1"


class PrioriConditions(SchemaModel):
    predicted_score: float | None = None
    predicted_win_prob: float | None = Field(default=None, ge=0, le=1)
    predictor_model_version: str | None = None
    mario_initial_status: Literal["small", "tall", "fireball"] | None = None
    ddrb_adjustment_action: dict[str, Any] = Field(default_factory=lambda: {
        "strategy_type": "none",
    })


class PosterioriOutcome(SchemaModel):
    termination_reason: Literal[
        "win_flag", "death_enemy", "death_pit", "death_time",
        "truncated_step_cap", "manual_abort", "unknown",
    ] = "unknown"
    terminated: bool = False
    truncated: bool = False
    clear: bool = False
    death: bool = False


class EnvPerformance(SchemaModel):
    steps_taken: int = Field(ge=0)
    wall_clock_seconds: float = Field(ge=0)
    ms_per_step: float = Field(ge=0)
    progress_tiles: float = 0
    progress_max_tiles: float = 0
    progress_ratio: float | None = Field(default=None, ge=0, le=1)
    protocol_reward_total: float = 0
    protocol_reward_components: dict[str, float] = Field(default_factory=dict)
    native_reward_total: float | None = None
    native_score: int | None = None
    coins_collected: int | None = Field(default=None, ge=0)
    time_left: int | None = None
    mario_final_status: str | None = None
    player_behavior_metrics: dict[str, float | int | None] = Field(default_factory=dict)
    training_step_diagnostics: dict[str, float | None] | None = None


class PosterioriDDRBDelta(SchemaModel):
    prediction_residual: float | None = None
    economy_utilization: float | None = None
    reward_efficiency: float | None = None
    flow_proxy_band: Literal[
        "boredom_too_easy", "flow_channel", "frustration_too_hard", "unknown",
    ] = "unknown"


class Performance(SchemaModel):
    performance_id: str
    run_id: str
    episode_index: int = Field(ge=0)
    batch_index: int | None = Field(default=None, ge=0)
    player_snapshot_id: str
    inline_player_state: InlinePlayerState
    level_id: str
    execution_context: ExecutionContext
    environment_spec: EnvironmentSpec
    priori_conditions: PrioriConditions = Field(default_factory=PrioriConditions)
    posteriori_outcome: PosterioriOutcome = Field(default_factory=PosterioriOutcome)
    env_performance: EnvPerformance
    posteriori_ddrb_delta: PosterioriDDRBDelta = Field(default_factory=PosterioriDDRBDelta)
    trajectory_sidecar_uri: str | None = None


class TrajectorySidecar(SchemaModel):
    performance_id: str
    uri: str
    format: Literal["jsonl.gz"] = "jsonl.gz"
    step_count: int = Field(ge=0)


TOP_LEVEL_MODELS: tuple[type[SchemaModel], ...] = (
    ExperimentRun, RLPlayerSnapshot, Level, Performance, TrajectorySidecar,
)


def export_json_schemas(directory: str) -> list[str]:
    """Write the JSON Schema document for each top-level record model."""
    from pathlib import Path

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for model in TOP_LEVEL_MODELS:
        path = output / f"{model.__name__}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(str(path))
    return paths


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
