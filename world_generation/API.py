from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from .Map_Grammars_Elite import Map_Elite
from .Data_Preprocessing import load_structures_df as _load_structures_df
from .level_formats import tilemap_to_rows
from .Performance_Evaluation import (
    Performance,
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


class GenerateMappingRequest(BaseModel):
    example_code: List[str]
    Level_Len: int
    module: Optional[int] = None
    batch_size: int = 1
    epochs: int = 256
    seed: Optional[int] = None
    Game_Mode: int = 0
    observability: bool = False


class PoolManagerRequest(BaseModel):
    agent_hash: str
    request: Dict[str, Any]


class StrToTilemapRequest(BaseModel):
    level: str


class MapEliteResponse(BaseModel):
    instance_id: str
    function: str
    alphas: List[float]
    variety_domains: List[str]
    grid_points: List[List[float]]
    optimal_count: int = Field(0, description="Number of discovered level candidates")
    generation: int = Field(0, description="Current generation counter")


class GenerateMappingResponse(BaseModel):
    status: str
    instance_id: str
    optimal_count: int
    generation: int


class PoolManagerResponse(BaseModel):
    instance_id: str
    agent_hash: str
    tilemap: List[List[Any]]


class PerformanceHistoryRequest(BaseModel):
    agent_id: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    append: bool = True


class PerformanceTrainRequest(BaseModel):
    epochs: Optional[int] = None
    history_window: Optional[int] = None


class PerformancePredictRequest(BaseModel):
    level: Any
    agent_id: Optional[str] = None
    non_landing_weight: float = 5.0 / 3.0
    clip: bool = True
    history_window: int = 32
    auto_train: bool = True


class PerformancePredictResponse(BaseModel):
    score: float
    agent_id: Optional[str] = None
    used_model: bool


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

    instance_id = str(next(_instance_id_counter))
    state_dir = Path(__file__).resolve().parent / "instances"
    state_path = state_dir / f"map_elite_{instance_id}.json"

    instance = Map_Elite(
        function,
        alphas=payload.alphas,
        Variety_Domains=variety_functions,
        Grid_points=payload.grid_points,
        state_path=str(state_path),
    )

    _map_elite_registry[instance_id] = instance

    return MapEliteResponse(
        instance_id=instance_id,
        function=instance.get_function_name(),
        alphas=instance.get_alphas(),
        variety_domains=instance.get_variety_domain_names(),
        grid_points=instance.get_grid_points(),
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
    instance.Generate_Mapping(
        example_code=payload.example_code,
        Level_Len=payload.Level_Len,
        module=payload.module,
        batch_size=payload.batch_size,
        epochs=payload.epochs,
        seed=payload.seed,
        Game_Mode=payload.Game_Mode,
        observability=payload.observability,
    )
    return GenerateMappingResponse(
        status="generated",
        instance_id=instance_id,
        optimal_count=instance.get_optimal_count(),
        generation=instance.get_current_generation(),
    )


@app.post("/instances/{instance_id}/pool_manager", response_model=PoolManagerResponse)
def pool_manager(instance_id: str, payload: PoolManagerRequest) -> PoolManagerResponse:
    instance = _get_instance(instance_id)
    tilemap = instance.pool_manager(payload.agent_hash, payload.request)
    return PoolManagerResponse(
        instance_id=instance_id,
        agent_hash=payload.agent_hash,
        tilemap=_serialize_tilemap(tilemap),
    )


@app.post("/str_to_tilemap")
def str_to_tilemap(payload: StrToTilemapRequest) -> Dict[str, Any]:
    instance = Map_Elite(Performance, alphas=[0.0])
    tilemap = instance.str_to_tilemap(payload.level)
    return {"tilemap": _serialize_tilemap(tilemap)}


@app.post("/performance/history")
def performance_history(payload: PerformanceHistoryRequest) -> Dict[str, Any]:
    try:
        update_agent_history(
            agent_id=payload.agent_id,
            history=payload.history,
            append=payload.append,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not update history: {exc}")

    return {
        "status": "ok",
        "agent_id": payload.agent_id,
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
    status = get_performance_model_status(agent_id=payload.agent_id)
    try:
        score = performance(
            payload.level,
            df=df,
            non_landing_weight=payload.non_landing_weight,
            clip=payload.clip,
            agent_id=payload.agent_id,
            history_window=payload.history_window,
            auto_train=payload.auto_train,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not predict performance: {exc}")

    return PerformancePredictResponse(
        score=float(score),
        agent_id=payload.agent_id,
        used_model=bool(status.get("model_ready") and status.get("has_agent_history")),
    )

