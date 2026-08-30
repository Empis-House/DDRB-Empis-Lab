# Level Up: Dinamic Difficulty-Reward Balancing 
A research about learning optimization on the AI Super Mario Bros. Framework.

## Abstract

Difficulty adjustment is a core design decision in games and education alike. The video game industry has studied how to balance difficulty to maintain player engagement for decades, leading to a specialized field focused on automatic adjustment: Dynamic Difficulty Adjustment (DDA). Although DDA is extensively studied in games, it originated in cognitive science as Csikszentmihalyi's flow theory (1970). However, the transition from flow theory to DDA has lost both applicability to other contexts (such as education) and explicit consideration of factors beyond difficulty that are equally relevant to dynamic adjustment.

This project reconceptualizes DDA as an abstract learning problem grounded in flow theory, proposing it as a new paradigm for AI-supported educational systems. Specifically, we present a Dynamic Difficulty-Reward Balancing (DDRB) pipeline that encourages flow-theoretic conditions and extends automatic adjustment to consider both task challenge and environmental economy. By doing so, the dynamic balancing system implicitly addresses more complex aspects of frustration and boredom, moving beyond binary task feasibility to evaluate whether a task is worthwhile in the learner's context.

As an initial experiment, we use the Super Mario Bros framework (a widely studied computational benchmark) reframing it not as an engaging game but as a learning task with an optimal learning curve. Rather than training from human gameplay, which would be prohibitively ambitious, we leverage the extensive literature on SMB agents to train and test our model. Our core hypothesis is that DDRB improves agent robustness and convergence compared to existing methods. We additionally hypothesize that agent behaviour provides sufficient diversity to correlate with human learning experience.

Look at WorkPaper.md 

# DDRB Lab

DDRB Lab is a research workspace for Super Mario Bros. procedural content generation, level evaluation, and reinforcement learning. The repository combines a FastAPI-based world-generation service with a custom Gymnasium-style Mario environment for training, benchmarking, and human play.

## What this repository contains

- `world_generation/`: Map-Elites style level generation, evaluation functions, diversity analysis, and the HTTP API.
- `mario_rl_env/`: A pure-Python Mario environment with built-in heuristic and trainable agents.
- `Super_Mario_Brothers_Maps/`: Source maps, processed maps, annotated paths, and generated output sets.
- `main.py`: End-to-end example that requests levels from the API and runs an agent on them.
- `PLAYME.md`: Quick start for launching the human-play mode.

## Quick start

Create a virtual environment and install the root dependencies.

### PowerShell

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Bash

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Main workflows

### 1. Start the world-generation API

The root `main.py` workflow expects the API to be running on `http://127.0.0.1:8000`.

```bash
uvicorn world_generation.API:app --reload --host 0.0.0.0 --port 8000
```

You can also start it with Docker Compose:

```bash
docker compose up --build world-gen-api
```

Container versioning and editability:

- Set `WORLD_GEN_API_VERSION` in `.env` to control the local image tag used by Compose.
- The `world_generation/` folder is bind-mounted into `/app/world_generation`, so code edits on the host are reflected in the running container.
- The API runs with `--reload` in Compose for live-reload behavior during development.

Core endpoints:

- `GET /`: health check plus available functions and variety domains
- `POST /instances`: create a generation instance
- `GET /instances`: list active instances
- `GET /instances/{instance_id}`: inspect one instance
- `POST /instances/{instance_id}/generate_mapping`: populate a search batch
- `POST /instances/{instance_id}/pool_manager`: fetch a level tilemap for an agent
- `POST /str_to_tilemap`: convert a serialized level string into a tilemap
- `POST /performance/history`: append agent history for the performance model
- `POST /performance/train`: train the performance predictor
- `POST /performance/predict`: score a level with the performance model

### 2. Run the integrated generation plus agent pipeline

With the API running, execute the root runner:

```bash
python main.py --agent smart_heuristic --episodes 50 --batch-size 10
```

What `main.py` does:

1. Waits for the API.
2. Creates a `Performance` generation instance.
3. Generates level mappings.
4. Pulls levels from `pool_manager`.
5. Runs the selected agent in `mario_rl_env`.
6. Prints a report with run metrics and level-diversity summaries.

Supported agent names for `main.py`:

- Heuristic agents: `random`, `right`, `right_jump`, `smart_heuristic`
- Trainable agents: `tabular_q`, `torch_dqn`

Notes for trainable agents:

- `tabular_q` uses the exposed Q-learning hyperparameters and works with feature-style observations.
- `torch_dqn` is a PyTorch DDQN agent and is only registered when the PyTorch stack is installed successfully.
- `torch_dqn` requires image observations; the runner automatically switches it to `rgb` mode when needed.
- Existing checkpoint files are resumed automatically when passed through `--checkpoint-path` in `main.py` or `--save-agent` in `run_agent.py`.

Useful options:

- `--agent`: agent name to run
- `--episodes`: number of episodes
- `--batch-size`: episodes to run before regenerating levels
- `--render`: render every episode
- `--render-last`: render only the last episode in each batch
- `--checkpoint-path`: resume a trainable agent from disk
- `--report-file`: save the final report as JSON
- `--skip-self-tests`: skip agent-load and checkpoint persistence checks

### 3. Run an agent on an existing map

To evaluate or train directly against a fixed map file, use the agent runner:

```bash
python -m mario_rl_env.run_agent Super_Mario_Brothers_Maps/generated_levels/Performance_opt=0.1.txt --agent right_jump --episodes 20
```

Built-in heuristic agents include:

- `random`
- `right`
- `right_jump`
- `smart_heuristic`

Trainable agents registered through `AGENT_MANAGER` in `mario_rl_env/ml_agents.py`:

- `tabular_q`: tabular Q-learning with `alpha`, `gamma`, `epsilon`, `epsilon_decay`, and `min_epsilon`
- `torch_dqn`: PyTorch DDQN with CNN-based pixel input, replay buffer, target network syncing, and checkpoint persistence for model weights plus exploration state

Example training commands:

```bash
python -m mario_rl_env.run_agent Super_Mario_Brothers_Maps/generated_levels/Performance_opt=0.1.txt --agent tabular_q --episodes 200 --history-file history/tabular_q_history.json --save-agent checkpoints/tabular_q.pkl

python -m mario_rl_env.run_agent Super_Mario_Brothers_Maps/generated_levels/Performance_opt=0.1.txt --agent torch_dqn --episodes 200 --obs rgb --history-file history/agent_performance_history.json --save-agent checkpoints/performance_transformer.pt
```

Checkpoint behavior:

- `--save-agent FILE` writes the trained agent checkpoint.
- If `FILE` already exists, training resumes from it.
- For `torch_dqn`, the checkpoint restores network weights, exploration rate, current step, and initialization parameters.

Useful options:

- `--obs features|features_layers|tiles|rgb`
- `--render`
- `--render-last`
- `--save-csv`
- `--save-replay`
- `--history-file`
- `--save-agent`
- `--alpha`
- `--gamma`
- `--epsilon`
- `--epsilon-decay`
- `--min-epsilon`
- `--analyse replay.json`

### 4. Play as a human

Standard mode on a fixed map:

```bash
python mario_rl_env/run_human.py Super_Mario_Brothers_Maps/generated_levels/Performance_opt=0.1.txt
```

Infinite mode backed by the API:

```bash
python mario_rl_env/run_human.py --infinite --api-url http://127.0.0.1:8000
```

Controls:

- `Left` / `A`: move left
- `Right` / `D`: move right
- `Z` / `Space`: jump
- `X` / `Shift`: run
- `R`: restart
- `Q` / `Esc`: quit

## Diversity analysis

The repository includes tile-pattern KL divergence utilities for comparing generated level distributions against a reference corpus.

```bash
python -m world_generation.tile_pattern_kl \
  --set-a Super_Mario_Brothers_Maps/Original \
  --set-b Super_Mario_Brothers_Maps/generated_levels/Performance \
  --window 2 2 \
  --stride 1 1
```

The command reports metrics such as:

- `kld_set_a_to_set_b`
- `kld_set_b_to_set_a`
- `kld_symmetric`
- `CoD`

Use this when you want to quantify how far generated levels drift from the original level distribution.

## Repository layout

```text
.
├── main.py
├── requirements.txt
├── docker-compose.yml
├── mario_rl_env/
│   ├── env.py
│   ├── ml_agents.py
│   ├── run_agent.py
│   ├── run_human.py
│   └── README.md
├── world_generation/
│   ├── API.py
│   ├── Map_Grammars_Elite.py
│   ├── Performance_Evaluation.py
│   ├── Variety_Domains.py
│   ├── tile_pattern_kl.py
│   └── instances/
├── Super_Mario_Brothers_Maps/
├── checkpoints/
└── history/
```

## Outputs and artifacts

- `checkpoints/`: saved trainable-agent checkpoints
- `history/`: training histories and reports
- `world_generation/instances/`: persisted generation-instance state
- `results.json`: generated experiment output used by local analysis

## Extending the project

To add a new trainable agent, implement an `RLAgent` subclass and register it in `mario_rl_env.ml_agents.AGENT_MANAGER`. Once registered, it becomes available to `main.py` and `mario_rl_env/run_agent.py` by name.

The environment is Gymnasium-compatible, so you can also integrate external RL libraries directly against `mario_rl_env.env`.
