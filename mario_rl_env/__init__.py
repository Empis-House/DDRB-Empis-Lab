"""
mario_rl – A Super Mario Bros.–style gymnasium environment for ML research.

Quick start
-----------
    from mario_rl.env import MarioEnv

    env = MarioEnv("map.txt", obs_mode="features", render_mode=None)
    obs, _ = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            obs, _ = env.reset()

    env.close()
"""

from .env import MarioEnv, ACTION_MAP, N_ACTIONS
from .tilemap import TileMap, TileType, parse_tilemap
from .game_state import GameState
from .metrics import MetricsTracker
from .ml_agents import AGENT_MANAGER, MLAgentManager, RandomAgent, TabularQAgent, RLAgent, run_training_loop, evaluate_agent

__all__ = [
    "MarioEnv", "ACTION_MAP", "N_ACTIONS",
    "TileMap", "TileType", "parse_tilemap",
    "GameState", "MetricsTracker",
    "AGENT_MANAGER", "MLAgentManager", "RandomAgent", "TabularQAgent",
    "RLAgent", "run_training_loop", "evaluate_agent",
]
