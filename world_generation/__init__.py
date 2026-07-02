
from world_generation.Word_Stitching import Jumping_Fiasible_Word
from world_generation.Grammar_Notation import Grammar
from world_generation.Performance_Evaluation import (
    Performance,
    performance,
    train_performance_predictor,
    update_agent_history,
)
from world_generation.Map_Grammars_Elite import Map_Elite
from world_generation.Variety_Domains import (
    athletic_domain_score,
    overworld_domain_score,
    tile_pattern_kl_diversity_between_sets,
    tile_pattern_kl_diversity_between_domains,
    underground_domain_score,
)


def compare_level_sets_tile_pattern_kl(*args, **kwargs):
    from world_generation.tile_pattern_kl import compare_level_sets_tile_pattern_kl as _impl
    return _impl(*args, **kwargs)


def evaluate_level_folders(*args, **kwargs):
    from world_generation.tile_pattern_kl import evaluate_level_folders as _impl
    return _impl(*args, **kwargs)


def evaluate_named_domains(*args, **kwargs):
    from world_generation.tile_pattern_kl import evaluate_named_domains as _impl
    return _impl(*args, **kwargs)

__all__ = [
    "Jumping_Fiasible_Word",
    "Grammar",
    "Map_Elite",
    "Performance",
    "performance",
    "update_agent_history",
    "train_performance_predictor",
    "overworld_domain_score", "underground_domain_score", "athletic_domain_score",
    "tile_pattern_kl_diversity_between_sets",
    "tile_pattern_kl_diversity_between_domains",
    "compare_level_sets_tile_pattern_kl", "evaluate_level_folders", "evaluate_named_domains",
]