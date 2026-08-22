from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from .Map_Grammars_Elite import Map_Elite
from .request_queue import GenerationQueueRouter, GenerationRequestContract, QueueBackpressureError
from .lineage import record_job as _record_lineage
from .Data_Preprocessing import load_structures_df as _load_structures_df
from .level_formats import tilemap_to_rows
from .Performance_Evaluation import (
    Performance,
    Performance_KL_Progression,
    get_performance_model_status,
    performance,
    train_performance_predictor,
    update_agent_history,
)
from .Variety_Domains import (
    athletic_domain_score,
    overworld_domain_score,
    underground_domain_score,
)

app = FastAPI(title="World Generation API", version="1.0")

_FUNCTION_MAP = {
    "Performance": Performance,
    "Performance_KL_Progression": Performance_KL_Progression,
    "performance": performance,
}

_VARIETY_MAP = {
    "overworld": overworld_domain_score,
    "underground": underground_domain_score,
    "athletic": athletic_domain_score,
}

_map_elite_registry: Dict[str, Map_Elite] = {}
_instance_id_counter = itertools.count(1)
_structures_df: Optional[pd.DataFrame] = None
_GENERATION_QUEUE_ROUTER = GenerationQueueRouter()


class MapEliteCreateRequest(BaseModel):
    function: str = Field("Performance", description="Evaluation function to use")
    alphas: List[float] = Field([0.0], description="Target alpha values")
    variety_domains: List[str] = Field(
        default_factory=lambda: ["overworld", "underground", "athletic"],
        description="Names of variety-domain functions to apply",
    )
    grid_points: List[List[float]] = Field(
        default_factory=lambda: [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
        description="Grid points used by the map elites search",
    )
    grid_mode: Literal["fixed", "percentile"] = Field(
        "fixed",
        description="How grid points are interpreted when building the archive",
    )
    instance_id: str = Field(default=None)


class GenerateMappingRequest(BaseModel):
    schema_version: int = Field(1, ge=1, description="Version of the world-generation request contract")
    queue_lane: str = Field("agentic", description="Traffic lane for this request: agentic or human")
    workflow: str = Field("train", description="High-level workflow such as train, test, validation, or live_pool")
    split: str = Field("train", description="Train/test/validation split for the run")
    idempotency_key: Optional[str] = Field(default=None, description="Stable key used to prevent duplicate generation jobs")
    correlation_id: Optional[str] = Field(default=None, description="Trace ID propagated across generation and result storage")
    priority: int = Field(0, description="Priority used for queue admission and preemption")
    run_id: Optional[str] = Field(default=None, description="Caller-supplied run identifier persisted in the lineage log")
    example_code: List[str]
    Level_Len: int
    module: Optional[int] = None
    batch_size: int = 1
    epochs: int = 256
    seed: Optional[int] = None
    Game_Mode: int = 0
    observability: bool = False
    use_grammar_degree_grid: bool = False
    grammar_min_degree: int = 2
    grammar_end_out_degree: int = 2
    grammar_bulk_repeats: int = 1
    grammar_max_attempts: int = 256


class PoolManagerRequest(BaseModel):
    agent_name: Optional[str] = None
    agent_hash: Optional[str] = None
    request: Dict[str, Any]
    batch_size: int = Field(1, ge=1, description="Number of levels to pull in one round-trip")
    case_id: Optional[str] = Field(default=None, description="Optional reproducibility tag for this request")
    queue_key: Optional[str] = Field(default=None, description="Optional queue identifier for per-run or per-case workload tracking")
    max_pending: int = Field(8, ge=1, description="Backpressure threshold for the queue key")
    priority: int = Field(0, description="Higher values preempt lower-priority requests when the queue is at capacity")
    queue_lane: str = Field("human", description="Lane for workload segregation: human or agentic")
    workflow: str = Field("live_pool", description="Workflow high-level name for this request")
    split: str = Field("live", description="Live gameplay split label")
    idempotency_key: Optional[str] = Field(default=None, description="Stable deduplication key for the pool request")
    correlation_id: Optional[str] = Field(default=None, description="Trace ID for the request lifecycle")
    run_id: Optional[str] = Field(default=None, description="Caller-supplied run identifier persisted in the lineage log")
    difficulty: Optional[str] = Field(default=None, description="Opt-in producer/consumer difficulty pool label; falls back to per-agent snapshot pooling when unset")
    refill_threshold: int = Field(4, ge=1, description="Difficulty pool floor that triggers a top-up refill")


class StrToTilemapRequest(BaseModel):
    level: str


class MapEliteResponse(BaseModel):
    instance_id: str
    function: str
    alphas: List[float]
    variety_domains: List[str]
    grid_points: List[List[float]]
    grid_mode: str
    optimal_count: int = Field(0, description="Number of discovered level candidates")
    generation: int = Field(0, description="Current generation counter")


class GenerateMappingResponse(BaseModel):
    status: str
    instance_id: str
    optimal_count: int
    generation: int
    job_id: Optional[str] = None
    job_status: Optional[str] = None
    queue_lane: Optional[str] = None
    correlation_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    schema_version: Optional[int] = Field(default=1, ge=1, description="Version of the request/response lineage contract")
    workflow: Optional[str] = None
    split: Optional[str] = None
    run_id: Optional[str] = None


class PoolManagerResponse(BaseModel):
    instance_id: str
    agent_name: str
    agent_hash: Optional[str] = None
    generation: int
    batch_size: int
    tilemap: Optional[List[List[Any]]] = None
    tilemaps: List[List[List[Any]]] = Field(default_factory=list)
    case_id: Optional[str] = None
    queue_key: Optional[str] = None
    pending: int = 0
    max_pending: int = 8
    dead_letter_count: int = 0
    job_id: Optional[str] = None
    job_status: Optional[str] = None
    queue_lane: Optional[str] = None
    correlation_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    schema_version: Optional[int] = Field(default=1, ge=1, description="Version of the request/response lineage contract")
    workflow: Optional[str] = None
    split: Optional[str] = None
    run_id: Optional[str] = None


class PerformanceHistoryRequest(BaseModel):
    agent_name: Optional[str] = None
    agent_id: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    append: bool = True


class PerformanceTrainRequest(BaseModel):
    epochs: Optional[int] = None
    history_window: Optional[int] = None


class PerformancePredictRequest(BaseModel):
    level: Any
    agent_name: Optional[str] = None
    agent_id: Optional[str] = None
    non_landing_weight: float = 5.0 / 3.0
    clip: bool = True
    history_window: int = 32
    auto_train: bool = True


class PerformancePredictResponse(BaseModel):
    score: float
    agent_name: Optional[str] = None
    agent_id: Optional[str] = None
    used_model: bool


def _resolve_agent_name(*names: Optional[str]) -> str:
    for value in names:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    raise HTTPException(status_code=400, detail="agent_name is required")


def _resolve_function(name: str):
    if name not in _FUNCTION_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown function '{name}'")
    return _FUNCTION_MAP[name]


def _resolve_variety_functions(names: List[str]):
    functions = []
    for name in names:
        if name not in _VARIETY_MAP:
            raise HTTPException(status_code=400, detail=f"Unknown variety function '{name}'")
        functions.append(_VARIETY_MAP[name])
    return functions


def _get_instance(instance_id: str) -> Map_Elite:
    if instance_id not in _map_elite_registry:
        raise HTTPException(status_code=404, detail=f"Map_Elite instance '{instance_id}' not found")
    return _map_elite_registry[instance_id]


def _serialize_tilemap(tilemap: Any) -> List[List[Any]]:
    try:
        return tilemap_to_rows(tilemap)
    except TypeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _as_mapping_dict(model: BaseModel) -> Dict[str, Any]:
    data = model.model_dump() if hasattr(model, "model_dump") else model.dict()
    return dict(data or {})


def _issue_generation_contract(payload: BaseModel, *, default_lane: str, default_workflow: str, default_split: str) -> GenerationRequestContract:
    data = _as_mapping_dict(payload)
    lane = str(data.get("queue_lane") or default_lane or "agentic")
    workflow = str(data.get("workflow") or default_workflow or "train")
    split = str(data.get("split") or default_split or "train")
    request = GenerationRequestContract(
        queue_lane=lane,
        workflow=workflow,
        split=split,
        seed=data.get("seed"),
        params={key: value for key, value in data.items() if key not in {"schema_version", "queue_lane", "workflow", "split", "idempotency_key", "correlation_id", "priority"}},
        idempotency_key=data.get("idempotency_key"),
        correlation_id=data.get("correlation_id"),
        priority=int(data.get("priority") or 0),
    )
    return request


def _get_structures_df() -> pd.DataFrame:
    global _structures_df
    if _structures_df is not None:
        return _structures_df

    try:
        _structures_df = _load_structures_df()
        return _structures_df
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load structures dataframe: {exc}")


@app.on_event("startup")
def create_instance_store() -> None:
    state_dir = Path(__file__).resolve().parent / "instances"
    state_dir.mkdir(parents=True, exist_ok=True)


@app.get("/")
def read_root() -> Dict[str, Any]:
    return {
        "message": "World Generation API",
        "available_functions": list(_FUNCTION_MAP),
        "available_variety_domains": list(_VARIETY_MAP),
        "performance_endpoints": [
            "/performance/history",
            "/performance/train",
            "/performance/predict",
        ],
        "instances": len(_map_elite_registry),
    }


@app.post("/instances", response_model=MapEliteResponse)
def create_map_elite(payload: MapEliteCreateRequest) -> MapEliteResponse:
    function = _resolve_function(payload.function)
    variety_functions = _resolve_variety_functions(payload.variety_domains)

    if payload.instance_id:
        instance_id=payload.instance_id
    else:
        instance_id = str(next(_instance_id_counter))

    state_dir = Path(__file__).resolve().parent / "instances"
    state_path = state_dir / f"map_elite_{instance_id}.json"

    instance = Map_Elite(
        function,
        alphas=payload.alphas,
        Variety_Domains=variety_functions,
        Grid_points=payload.grid_points,
        state_path=str(state_path),
        grid_mode=payload.grid_mode,
        instance_id=instance_id,
    )

    _map_elite_registry[instance_id] = instance

    return MapEliteResponse(
        instance_id=instance_id,
        function=instance.get_function_name(),
        alphas=instance.get_alphas(),
        variety_domains=instance.get_variety_domain_names(),
        grid_points=instance.get_grid_points(),
        grid_mode=instance.get_grid_mode(),
        optimal_count=instance.get_optimal_count(),
        generation=instance.get_current_generation(),
    )


@app.get("/instances", response_model=List[MapEliteResponse])
def list_instances() -> List[MapEliteResponse]:
    return [
        MapEliteResponse(
            instance_id=instance_id,
            function=instance.get_function_name(),
            alphas=instance.get_alphas(),
            variety_domains=instance.get_variety_domain_names(),
            grid_points=instance.get_grid_points(),
            grid_mode=instance.get_grid_mode(),
            optimal_count=instance.get_optimal_count(),
            generation=instance.get_current_generation(),
        )
        for instance_id, instance in _map_elite_registry.items()
    ]


@app.get("/instances/{instance_id}", response_model=MapEliteResponse)
def get_instance(instance_id: str) -> MapEliteResponse:
    instance = _get_instance(instance_id)
    return MapEliteResponse(
        instance_id=instance_id,
        function=instance.get_function_name(),
        alphas=instance.get_alphas(),
        variety_domains=instance.get_variety_domain_names(),
        grid_points=instance.get_grid_points(),
        grid_mode=instance.get_grid_mode(),
        optimal_count=instance.get_optimal_count(),
        generation=instance.get_current_generation(),
    )


@app.delete("/instances/{instance_id}")
def delete_instance(instance_id: str) -> Dict[str, Any]:
    instance = _get_instance(instance_id)
    state_path = instance.get_state_path()
    _map_elite_registry.pop(instance_id)
    if state_path:
        try:
            Path(state_path).unlink()
        except OSError:
            pass
    return {"status": "deleted", "instance_id": instance_id}


@app.post("/instances/{instance_id}/generate_mapping", response_model=GenerateMappingResponse)
def generate_mapping(instance_id: str, payload: GenerateMappingRequest) -> GenerateMappingResponse:
    instance = _get_instance(instance_id)
    contract = _issue_generation_contract(
        payload,
        default_lane="agentic",
        default_workflow="train",
        default_split="train",
    )
    job = _GENERATION_QUEUE_ROUTER.submit(contract)
    if job.job_id != contract.request_id:
        return GenerateMappingResponse(
            status="duplicate",
            instance_id=instance_id,
            optimal_count=instance.get_optimal_count(),
            generation=instance.get_current_generation(),
            job_id=job.job_id,
            job_status=job.status,
            queue_lane=job.lane,
            correlation_id=job.request.correlation_id,
            idempotency_key=job.request.idempotency_key,
            schema_version=job.request.schema_version,
            workflow=job.request.workflow,
            split=job.request.split,
            run_id=job.request.params.get("run_id"),
        )

    try:
        instance.refresh_generated_mapping(
            clear_manager_memory=True,
            example_code=payload.example_code,
            Level_Len=payload.Level_Len,
            module=payload.module,
            batch_size=payload.batch_size,
            epochs=payload.epochs,
            seed=payload.seed,
            Game_Mode=payload.Game_Mode,
            observability=payload.observability,
            use_grammar_degree_grid=payload.use_grammar_degree_grid,
            grammar_min_degree=payload.grammar_min_degree,
            grammar_end_out_degree=payload.grammar_end_out_degree,
            grammar_bulk_repeats=payload.grammar_bulk_repeats,
            grammar_max_attempts=payload.grammar_max_attempts,
        )
    except Exception as exc:
        _GENERATION_QUEUE_ROUTER.mark_failed(job.job_id, str(exc), lane=job.lane)
        _record_lineage(instance_id, job, status="failed")
        raise
    _GENERATION_QUEUE_ROUTER.mark_done(job.job_id, lane=job.lane)
    _record_lineage(instance_id, job, status="completed")
    return GenerateMappingResponse(
        status="generated",
        instance_id=instance_id,
        optimal_count=instance.get_optimal_count(),
        generation=instance.get_current_generation(),
        job_id=job.job_id,
        job_status="completed",
        queue_lane=job.lane,
        correlation_id=job.request.correlation_id,
        idempotency_key=job.request.idempotency_key,
        schema_version=job.request.schema_version,
        workflow=job.request.workflow,
        split=job.request.split,
        run_id=job.request.params.get("run_id"),
    )


@app.post("/instances/{instance_id}/pool_manager", response_model=PoolManagerResponse)
def pool_manager(instance_id: str, payload: PoolManagerRequest) -> PoolManagerResponse:
    # Runs in-process/synchronously by design (see request_queue.py for the tracked
    # follow-up on a cross-process durable queue). What this endpoint now provides:
    # admission control with a documented backpressure response (HTTP 429), priority
    # preemption, and dead-letter capture for failed/preempted requests instead of a
    # bare pending counter that silently dropped anything that went wrong.
    instance = _get_instance(instance_id)
    agent_name = _resolve_agent_name(payload.agent_name, payload.agent_hash)
    batch_size = max(1, int(payload.batch_size or 1))
    queue_key = payload.queue_key or f"case:{payload.case_id or 'default'}|agent:{agent_name}"
    contract = _issue_generation_contract(
        payload,
        default_lane=str(payload.queue_lane or "human"),
        default_workflow=str(payload.workflow or "live_pool"),
        default_split=str(payload.split or "live"),
    )
    contract.params.setdefault("queue_key", queue_key)
    contract.params.setdefault("agent_name", agent_name)
    job = _GENERATION_QUEUE_ROUTER.submit(contract)
    try:
        tilemaps = instance.pool_manager(
            agent_name,
            payload.request,
            batch_size=batch_size,
            queue_key=queue_key,
            max_pending=payload.max_pending,
            priority=payload.priority,
            difficulty=payload.difficulty,
            refill_threshold=payload.refill_threshold,
        )
    except QueueBackpressureError as exc:
        _GENERATION_QUEUE_ROUTER.mark_failed(job.job_id, str(exc), lane=job.lane)
        _record_lineage(instance_id, job, status="failed")
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        _GENERATION_QUEUE_ROUTER.mark_failed(job.job_id, str(exc), lane=job.lane)
        _record_lineage(instance_id, job, status="failed")
        raise
    _GENERATION_QUEUE_ROUTER.mark_done(job.job_id, lane=job.lane)
    _record_lineage(instance_id, job, status="completed")
    queue_state = instance.get_queue_snapshot(queue_key)
    serialized_tilemaps = [_serialize_tilemap(tilemap) for tilemap in tilemaps]
    return PoolManagerResponse(
        instance_id=instance_id,
        agent_name=agent_name,
        agent_hash=agent_name,
        generation=instance.get_current_generation(),
        batch_size=batch_size,
        tilemap=serialized_tilemaps[0] if serialized_tilemaps else None,
        tilemaps=serialized_tilemaps,
        case_id=payload.case_id,
        queue_key=queue_key,
        pending=int(queue_state.get("pending", 0)),
        max_pending=int(queue_state.get("max_pending", payload.max_pending)),
        dead_letter_count=int(queue_state.get("dead_letter_count", 0)),
        job_id=job.job_id,
        job_status="completed",
        queue_lane=job.lane,
        correlation_id=job.request.correlation_id,
        idempotency_key=job.request.idempotency_key,
        schema_version=job.request.schema_version,
        workflow=job.request.workflow,
        split=job.request.split,
        run_id=job.request.params.get("run_id"),
    )


@app.get("/instances/{instance_id}/dead_letters")
def dead_letters(instance_id: str, queue_key: Optional[str] = None) -> Dict[str, Any]:
    """Inspect requests that failed or were preempted instead of silently dropped."""
    instance = _get_instance(instance_id)
    return {"instance_id": instance_id, "queue_key": queue_key, "dead_letters": instance.get_dead_letters(queue_key)}


@app.post("/str_to_tilemap")
def str_to_tilemap(payload: StrToTilemapRequest) -> Dict[str, Any]:
    instance = Map_Elite(Performance, alphas=[0.0])
    tilemap = instance.str_to_tilemap(payload.level)
    return {"tilemap": _serialize_tilemap(tilemap)}


@app.post("/performance/history")
def performance_history(payload: PerformanceHistoryRequest) -> Dict[str, Any]:
    agent_name = _resolve_agent_name(payload.agent_name, payload.agent_id)
    try:
        update_agent_history(
            agent_id=agent_name,
            history=payload.history,
            append=payload.append,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not update history: {exc}")

    return {
        "status": "ok",
        "agent_name": agent_name,
        "agent_id": agent_name,
        "entries_received": len(payload.history),
        "append": payload.append,
    }


@app.post("/performance/train")
def performance_train(payload: PerformanceTrainRequest) -> Dict[str, Any]:
    try:
        result = train_performance_predictor(
            epochs=payload.epochs,
            history_window=payload.history_window,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not train performance model: {exc}")
    return result


@app.post("/performance/predict", response_model=PerformancePredictResponse)
def performance_predict(payload: PerformancePredictRequest) -> PerformancePredictResponse:
    df = _get_structures_df()
    agent_name = None
    if payload.agent_name is not None or payload.agent_id is not None:
        agent_name = _resolve_agent_name(payload.agent_name, payload.agent_id)
    status = get_performance_model_status(agent_id=agent_name)
    try:
        score = performance(
            payload.level,
            df=df,
            non_landing_weight=payload.non_landing_weight,
            clip=payload.clip,
            agent_id=agent_name,
            history_window=payload.history_window,
            auto_train=payload.auto_train,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not predict performance: {exc}")

    return PerformancePredictResponse(
        score=float(score),
        agent_name=agent_name,
        agent_id=agent_name,
        used_model=bool(status.get("model_ready") and status.get("has_agent_history")),
    )

