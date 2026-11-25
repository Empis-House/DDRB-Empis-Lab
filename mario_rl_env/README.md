# Mario RL Environment

A Super Mario Bros.–style gymnasium environment written in pure Python.  
Designed for ML research: trains agents, benchmarks performance, and records  
full episode replays for offline analysis — all from a single `.txt` tile map.

For the repository-wide experiment sequence, use the
[phased experiment guide](../docs/EXPERIMENTS.md). This page is the detailed
reference for using and extending the Mario environment.

## Follow the project phases

1. **Verify the environment:** run the fixed-map examples in
   [Phase 1](../docs/EXPERIMENTS.md#phase-1-fixed-map-baseline).
2. **Train and evaluate:** follow [Phases 2–5](../docs/EXPERIMENTS.md#phase-2-procedural-pipeline)
   for procedural runs, cross-environment comparisons, and checkpoint resumes.
3. **Extend the player:** follow
   [Adding a player-agent kind](#adding-a-player-agent-kind), then verify the
   agent on a fixed map before using it in experiments.

---

## Quick start

```bash
pip install pygame gymnasium numpy

# Human play from the repository root
python -m mario_rl_env.run_human MAP.txt

# AI agent (headless, fast)
python -m mario_rl_env.run_agent MAP.txt --agent right_jump --episodes 50

# Analyse a saved replay
python -m mario_rl_env.run_agent MAP.txt --analyse replay.json
```

---

## Architecture

```
mario_rl_env/
│
├── constants.py      ← All physics / reward / screen tuning knobs
├── tilemap.py        ← CSV tile-map parser → TileMap dataclass
├── entities.py       ← Mario + Goomba dataclasses (pure data, no logic)
├── physics.py        ← AABB collision engine (per-axis separation)
├── game_state.py     ← Mutable world state + step() loop
├── renderer.py       ← Pygame renderer (sprite-free, headless-capable)
├── env.py            ← Gymnasium Env (4 obs modes, discrete action space)
├── metrics.py        ← Episode stats tracker, CSV + JSON replay logger
│
├── run_human.py      ← Human-playable entry point (keyboard controls)
└── run_agent.py      ← Built-in agents + CLI runner + replay analyser
```

Data flows in one direction: `tilemap → game_state → env → agent`.  
The renderer is entirely optional and never affects physics.

---

## Tile map format

Comma-separated `.txt` files, one row per line:

| Char | Tile            | Behaviour                                  |
|------|-----------------|--------------------------------------------|
| `-`  | AIR             | Passable                                   |
| `o`  | COIN            | Passable; collected on touch               |
| `X`  | GROUND          | Solid                                      |
| `S`/`B`/`b` | BRICK    | Solid; broken by Big Mario from below      |
| `Q`/`?` | QUESTION BLOCK | Solid; yields a coin on first hit       |
| `E`  | ENEMY SPAWN     | Spawns a Goomba; tile becomes AIR          |
| `<`/`>` | PIPE TOP   | Solid (decorative pipe head)               |
| `[`/`]` | PIPE BODY  | Solid (pipe shaft)                         |
| `r`/`R`/`u`/`U` | RED PIPE | Solid; hold Down to transport in human mode |
| `v`/`V`/`w`/`W` | BLUE PIPE | Solid; hold Down to transport in human mode |

Any `.txt` map with this schema works — just pass `--map your_map.txt`.

---

## Gymnasium API

```python
from mario_rl_env.env import MarioEnv

env = MarioEnv(
    map_path   = "Performance_opt=0.1.txt",
    obs_mode   = "features",   # "rgb" | "tiles" | "features" | "features_layers"
    render_mode= None,          # None | "human" | "rgb_array"
)

obs, info = env.reset()
for _ in range(10_000):
    action = env.action_space.sample()          # Discrete(8)
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        obs, info = env.reset()

env.close()
```

### Observation modes

| Mode       | Shape            | dtype   | Notes                                        |
|------------|------------------|---------|----------------------------------------------|
| `features` | `(252,)`         | float32 | Mario state + tile window + 5 nearest enemies|
| `features_layers` | `(700,)` | float32 | Features with separate landing and collider tile layers |
| `tiles`    | `(14, 16)`       | int8    | Tile-type grid centred on Mario              |
| `rgb`      | `(448, 512, 3)`  | uint8   | Full pixel frame (also used by renderer)     |

### Action space — `Discrete(8)`

| Index | Name             | Keys pressed                  |
|-------|------------------|-------------------------------|
| 0     | NOOP             | —                             |
| 1     | RIGHT            | →                             |
| 2     | LEFT             | ←                             |
| 3     | JUMP             | Z                             |
| 4     | RIGHT + JUMP     | → + Z                         |
| 5     | LEFT  + JUMP     | ← + Z                         |
| 6     | RUN RIGHT        | → + X                         |
| 7     | RUN RIGHT + JUMP | → + X + Z                     |

### Reward structure

| Event                     | Reward                           |
|---------------------------|----------------------------------|
| New furthest-right tile   | `+1.0 × tiles_advanced`          |
| Moving backwards          | `–0.5 × tiles_regressed`         |
| Stomping a Goomba         | `+5.0`                           |
| Hitting a Question Block  | `+2.0`                           |
| Taking damage / dying     | `–15.0`                          |
| Falling into a pit        | `–15.0`                          |
| Reaching end of level     | `+100.0`                         |
| Every step (time penalty) | `–0.01`                          |

### Info dict (every step)

```
step, mario_x, mario_y, mario_vx, mario_vy, on_ground,
coins, score, kills, mario_state, total_reward, n_goombas
```

At episode end the `info['episode']` key contains:

```
total_reward, steps, coins, score, kills, max_x, won
```

---

## Physics

- **Gravity** `0.50 px/frame²`, terminal velocity `12.0 px/frame`  
- **AABB per-axis separation** — X resolved first, then Y, eliminating corner sticking  
- **Mario walk** `2.5 px/frame`, **run** `4.5 px/frame`, **jump** `vy = –12.5 px/frame`  
- **Goomba** walks at `1.2 px/frame`, reverses on walls, falls off ledges, dies permanently below map  
- **Stomp detection** — Mario falling (`vy > 0`) and his bottom was above goomba's midpoint last frame  
- **Invincibility** — 90 frames (1.5 s) after taking damage while Big Mario  
- All values are in `constants.py`; edit there to retune without touching engine code

---

## Metrics & Replay

```python
from mario_rl_env.metrics import MetricsTracker

tracker = MetricsTracker(keep_steps=True)

obs, _ = env.reset()
tracker.new_episode()

while True:
    action = agent.act(obs)
    obs, reward, done, trunc, info = env.step(action)
    tracker.record(action, reward, info)
    if done or trunc:
        tracker.end_episode(info)
        break

# Aggregate stats
tracker.print_summary()          # pretty-print to console
tracker.save_csv("metrics.csv")  # per-episode CSV

# Full step-level replay (useful for imitation learning, visualisation)
tracker.save_replay("replay.json")

# Replay analysis (action distribution, reward curve, progress quarters)
python -m mario_rl_env.run_agent map.txt --analyse replay.json
```

### CSV columns

`episode_id, total_reward, steps, coins, score, kills, max_x, won, duration_sec`

### Replay JSON schema

```json
{
  "episode_id": 7,
  "total_reward": 250.44,
  "steps": 1446,
  "won": true,
  "max_x": 4993.0,
  "actions":    [6, 6, 7, 6, ...],
  "rewards":    [0.15, 0.13, ...],
  "mario_x":    [132.0, 164.0, ...],
  "mario_y":    [388.0, 388.0, ...],
  "on_ground":  [true, true, ...]
}
```

---

## Stable-Baselines3 integration (one-liner)

```python
from stable_baselines3 import PPO
from mario_rl_env.env import MarioEnv

env = MarioEnv("map.txt", obs_mode="features")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("mario_ppo")
```

## Adding a player-agent kind

A player agent can be a lightweight heuristic or a trainable `RLAgent`
subclass. Keep agent logic separate from the game engine: agents choose an
action from observations, while the environment owns physics, maps, rewards,
rendering, and episode termination.

### Contract every agent must follow

- Implement `act(obs)` and return exactly one Python `int` in the range
  `0 <= action < env.action_space.n`. The current action space is
  `Discrete(8)`:

  | Index | Action |
  |---:|---|
  | 0 | No-op |
  | 1 | Right |
  | 2 | Left |
  | 3 | Jump |
  | 4 | Right + jump |
  | 5 | Left + jump |
  | 6 | Right + run |
  | 7 | Right + run + jump |

- Support and document the observation mode the agent uses. `rgb` observations
  are pixel frames, `tiles` are a 2-D tile grid, and `features` and
  `features_layers` are 1-D feature vectors. Do not treat those shapes as
  interchangeable.
- Keep episode state resettable. Implement `reset()` when the agent has
  counters, cooldowns, recurrent state, or other state that must not leak
  between episodes. Ensure the relevant runner calls it at each episode start.
- Keep randomness reproducible from the configured seed. Do not use a separate
  unseeded random source.
- Do not control Pygame, mutate the environment or map, or alter the physics,
  reward, or termination rules.
- Do not retain live environment objects or Pygame surfaces in state that may be
  checkpointed. Only depend on an attached environment if the agent explicitly
  needs it and follows the runner's attach/detach lifecycle.
- Add focused tests for construction, action bounds, reset behavior, supported
  observation shapes, and checkpoint round trips when applicable.

### Trainable or checkpointed agent

Subclass `RLAgent` in `mario_rl_env/ml_agents.py`. Implement `act`; override
`observe(...)` for transition learning and `reset()` for episode-specific state
when needed. Register the new agent name with `AGENT_MANAGER`:

```python
import numpy as np

from mario_rl_env.ml_agents import AGENT_MANAGER, RLAgent


class MyAgent(RLAgent):
    def __init__(self, action_size: int) -> None:
        super().__init__(action_size)

    def act(self, obs: np.ndarray) -> int:
        return 0


AGENT_MANAGER.register(
    "my_agent",
    lambda action_size, **kwargs: MyAgent(action_size),
)
```

The `action_size` constructor argument must size the agent's action outputs.
Registration makes the name available to the registry-backed agent runner. Add
special constructor options or observation-mode selection to `run_agent.run()`
when necessary, and update the integrated `main.py` path if it applies agent-
specific constraints. Document any optional dependency or supported observation
mode.

For checkpointed agents, verify that `save()` and `load()` preserve all state
needed to resume. The inherited `RLAgent.save()` uses atomic pickle writes; keep
checkpoint state serializable. If experiment snapshots are used, ensure
`get_snapshot_metadata()` returns lightweight serializable metadata.
`observe(obs, action, reward, next_obs, done)` receives terminal state after
termination or truncation in the training loop; do not train on transitions
outside the runner's environment loop.

### Lightweight heuristic agent

Heuristics do not need to inherit `RLAgent`. Add the class in
`mario_rl_env/run_agent.py`, then:

1. Add an explicit construction branch in `run_agent.run()`.
2. Add the name to `agent_choices` in `run_agent.main()`.
3. Reset episode-local state at the start of each episode in the runner.
4. Document supported observations and any mode-specific behavior.
5. Add the agent to `main.py` or benchmark selection only if it should be
   available through those entry points.

Registered agent names are generated from `AGENT_MANAGER`; heuristic names need
an explicit CLI choice. In either case, verify the agent on a fixed map before
running long or procedural experiments.

---

## Human controls

| Key             | Action        |
|-----------------|---------------|
| ← / A           | Move left     |
| → / D           | Move right    |
| Z / Space       | Jump          |
| X / Shift       | Run (hold)    |
| R               | Restart level |
| Q / Esc         | Quit          |

---

## Baseline benchmark (50 episodes, `features` obs mode)

| Agent        | Win rate | Mean reward | Mean max_x | Steps/sec |
|--------------|----------|-------------|------------|-----------|
| Random       | ~20%     | ~+32        | ~2590 px   | ~1 900    |
| right_jump   | ~40%     | ~+96        | ~2179 px   | ~1 800    |
| PPO (trained)| TBD      | TBD         | TBD        | —         |

*Simulated headlessly on a single CPU core, no GPU required.*

---

## Tuning

Edit `constants.py` to change:

- Physics (`GRAVITY`, `MARIO_JUMP_VY`, `GOOMBA_SPEED`, …)
- Rewards (`REWARD_KILL_ENEMY`, `REWARD_WIN`, …)
- Screen resolution (`SCREEN_TILES_W/H`, `TILE_SIZE`)
- Observation window (`OBS_TILES_W/H`)
- Episode length (`MAX_STEPS_PER_EPISODE`)

---

## File list

| File | Purpose |
|------|---------|
| `constants.py`  | All tunable parameters |
| `tilemap.py`    | Tile map parser |
| `entities.py`   | Mario & Goomba dataclasses |
| `physics.py`    | AABB physics engine |
| `game_state.py` | World state & step loop |
| `renderer.py`   | Pygame renderer |
| `env.py`        | Gymnasium environment |
| `metrics.py`    | Stats tracker & replay logger |
| `run_human.py`  | Human play entry point |
| `run_agent.py`  | AI agent runner & analyser |
| `requirements.txt` | `pygame`, `gymnasium`, `numpy` |
