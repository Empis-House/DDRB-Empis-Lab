# Play this Project — Quick Start

Welcome! This repository contains a Super Mario Bros research and playground environment. This `PLAYME.md` is a short, friendly guide for someone who just wants to play the game or try the human-run demo.

**Quick start (Windows)**

Prerequisites:

- Python 3.9+ (recommended). A virtual environment is highly recommended.
- The repository `requirements.txt` has the Python dependencies.

1. Open a terminal in the repository root.
2. Create and activate a virtual environment (PowerShell example):

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Run the human-play demo (this launches the in-game renderer):

```powershell
python mario_rl_env/run_human.py --infinite --checkpoint --checkpoint-dir checkpoints
```

This will start an interactive play session. The `--checkpoint` flag enables automatic saving/loading of checkpoints in the `checkpoints/` folder.

Controls and configuration

- Keyboard controls and exact key mappings depend on the renderer in `mario_rl_env/run_human.py`. For the authoritative mapping, see [mario_rl_env/run_human.py](mario_rl_env/run_human.py#L1).
- If rendering is headless or fails, try disabling the renderer flags in the same script or run the agent demos instead.

Want to try an agent instead of playing?

- Run the tabular Q-learning demo or training pipeline via `main.py` or the agent runner scripts. Example (simple run):

```powershell
& .venv\Scripts\python.exe main.py
```

Or run the agent runner directly:

```powershell
python mario_rl_env/run_agent.py
```

Files to look at next

- `main.py` — high-level pipeline for generation, validation and agent runs.
- `mario_rl_env/run_human.py` — human-play entry point and renderer glue.
- `mario_rl_env/run_agent.py` — agent runner for training/evaluation.
- `world_generation/` — generation, Map-Elites, and solvability routines.


Contributing and experiments

- This project supports research workflows (generation → validation → training). If you want to modify reward balancing or generation parameters, check `main.py` and `world_generation/`.
- For reproducible experiments, use the `checkpoints/` and `history/` folders to store runs.

Have fun!
