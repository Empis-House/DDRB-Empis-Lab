from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .Map_Grammars_Elite import Map_Elite
from .Performance_Evaluation import Performance
from .Variety_Domains import calculate_slope, landings_score, colliders_score

app = FastAPI(title="World Generation API")

_FUNCTION_MAP = {
    "Performance": Performance,
}

_VARIETY_MAP = {
    "calculate_slope": calculate_slope,
    "landings_score": landings_score,
    "colliders_score": colliders_score,
}

_map_elite_registry: Dict[str, Map_Elite] = {}
_next_instance_id = 1


class MapEliteCreateRequest(BaseModel):
    function: str = Field("Performance", description="Evaluation function to use")
    alphas: List[float] = Field([0.0], description="Target alpha values")
    variety_domains: List[str] = Field(
        default_factory=list,
        description="Names of variety-domain functions to apply",
    )
    grid_points: List[List[float]] = Field(
        default_factory=list,
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
        values = tilemap.values.tolist()
    except AttributeError:
        if isinstance(tilemap, list):
            return tilemap
        raise HTTPException(status_code=500, detail="Unsupported tilemap format")
    return values


@app.get("/")
def read_root() -> Dict[str, Any]:
    return {
        "message": "World Generation API",
        "available_functions": list(_FUNCTION_MAP),
        "available_variety_domains": list(_VARIETY_MAP),
    }


@app.post("/instances", response_model=MapEliteResponse)
def create_map_elite(payload: MapEliteCreateRequest) -> MapEliteResponse:
    global _next_instance_id

    function = _resolve_function(payload.function)
    variety_functions = _resolve_variety_functions(payload.variety_domains)

    instance = Map_Elite(
        function,
        alphas=payload.alphas,
        Variety_Domains=variety_functions,
        Grid_points=payload.grid_points,
    )

    instance_id = str(_next_instance_id)
    _next_instance_id += 1
    _map_elite_registry[instance_id] = instance

    return MapEliteResponse(
        instance_id=instance_id,
        function=payload.function,
        alphas=payload.alphas,
        variety_domains=payload.variety_domains,
        grid_points=payload.grid_points,
    )


@app.get("/instances")
def list_instances() -> List[MapEliteResponse]:
    response: List[MapEliteResponse] = []
    for instance_id, instance in _map_elite_registry.items():
        response.append(
                MapEliteResponse(
                instance_id=instance_id,
                function="Performance",
                alphas=list(instance._Map_Elite__alphas),
                variety_domains=[
                    name for name, fn in _VARIETY_MAP.items()
                    if fn in instance._Map_Elite__Variety_Domains
                ],
                grid_points=[list(point) for point in instance._Map_Elite__Grid_points],
            )
        )
    return response


@app.post("/instances/{instance_id}/generate_mapping")
def generate_mapping(instance_id: str, payload: GenerateMappingRequest) -> Dict[str, Any]:
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
    return {
        "status": "generated",
        "instance_id": instance_id,
        "optimal_count": len(getattr(instance, "_Map_Elite__Optimal_strings", [])),
        "generation": getattr(instance, "_Map_Elite__current_generation", None),
    }


@app.post("/instances/{instance_id}/pool_manager")
def pool_manager(instance_id: str, payload: PoolManagerRequest) -> Dict[str, Any]:
    instance = _get_instance(instance_id)
    tilemap = instance.pool_manager(payload.agent_hash, payload.request)
    return {
        "instance_id": instance_id,
        "agent_hash": payload.agent_hash,
        "tilemap": _serialize_tilemap(tilemap),
    }


@app.post("/str_to_tilemap")
def str_to_tilemap(payload: StrToTilemapRequest) -> Dict[str, Any]:
    instance = Map_Elite(Performance, alphas=[0.0])
    tilemap = instance.str_to_tilemap(payload.level)
    return {"tilemap": _serialize_tilemap(tilemap)}

