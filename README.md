# DDRB Empis Lab

A research project for Super Mario Bros. level generation, evaluation, and reinforcement learning in a custom gymnasium-style environment.

## Overview

This repository combines two main systems:

- `world_generation/`: A FastAPI-backed procedural generation pipeline for Mario-like levels using map elites and variety-domain evaluation.
- `mario_rl_env/`: A custom Super Mario Bros.-style reinforcement learning environment with built-in baseline agents and support for training/evaluation.

The root `main.py` file demonstrates how to use the world generation API to create level maps and then run a Mario RL agent on the generated levels.

## Repository structure

- `main.py` - Example runner that creates a world generation instance, generates levels, and runs an RL agent.
- `requirements.txt` - Python dependency list.
- `mario_rl_env/` - Mario RL environment package and agent infrastructure.
  - `env.py` - Gymnasium-compatible Mario environment.
  - `ml_agents.py` - Built-in agents and training loop support.
  - `run_agent.py` - CLI entrypoint to execute agents on level maps.
  - `history/` - Training history outputs.
  - `checkpoints/` - Saved agent checkpoints.
- `world_generation/` - World generation API and supporting algorithms.
  - `API.py` - FastAPI service for creating map-elite instances and generating tilemaps.
  - `Map_Grammars_Elite.py` - Core map-elite implementation.
  - `Performance_Evaluation.py`, `Variety_Domains.py` - Evaluation and variety scoring.
- `Super_Mario_Brothers_Maps/` - Stored map data and generated levels.

## Requirements

Install the Python dependencies from `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

The project expects a Python 3.12-compatible environment.

## Running the world generation API

Start the FastAPI application from the repository root:

```bash
uvicorn world_generation.API:app --reload --host 0.0.0.0 --port 8000
```

Available endpoints:

- `GET /` - API status and available functions.
- `POST /instances` - Create a map elite generation instance.
- `POST /instances/{instance_id}/generate_mapping` - Generate level mappings for an instance.
- `POST /instances/{instance_id}/pool_manager` - Request a generated tilemap for an agent.
- `POST /str_to_tilemap` - Convert a level string into a tilemap.

## Example usage

A typical workflow is:

1. Start the FastAPI server.
2. Create a generation instance with target evaluation and variety domains.
3. Generate mappings for levels.
4. Use `mario_rl_env.run_agent.run()` or `run_agent.py` to run an agent on the generated map.

Example in `main.py`:

- Creates a `Performance` instance with `alphas=[0.0, 0.1, 1]`.
- Uses variety domains `calculate_slope` and `colliders_score`.
- Generates a batch of levels.
- Runs a tabular Q-learning agent with rendering enabled.

## Running agents

The `mario_rl_env` package supports both heuristic and learning agents.

```bash
python -m mario_rl_env.run_agent <map_path> --agent random --episodes 50
```

Built-in agents:

- `random` - uniform random actions.
- `right` - always move right.
- `right_jump` - move right and jump over obstacles.
- learning agents registered in `mario_rl_env.ml_agents` such as `tabular_q`.

## Notes

- The environment exposes a Gymnasium-compatible API, so it is easy to integrate with RL libraries like Stable-Baselines3.
- The `Super_Mario_Brothers_Maps/` folder contains original and processed map data plus generated levels.
- `main.py` is an example integration point; adjust payload values, agent settings, and checkpoint/history paths as needed.

## Contact

This repository is maintained by the DDRB Empis Lab research project.
