# DDRB Lab Experiment Guide

All commands below use Python 3.13 and assume you are at the repository root.

## 1. Setup and Quick Verification

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\verify_experiments.py --smoke
```

The smoke run uses temporary files and headless mocks where display or ROM APIs are unavailable. Successful output starts with `{"status": "passed" ...}` and exercises fixed-map, API, benchmark, and diversity surfaces.

## 2. Fixed-Map Evaluation and Training

```bash
python -m mario_rl_env.run_agent Super_Mario_Brothers_Maps\Processed\mario-1-1.txt --agent random --episodes 10 --obs features --save-csv history\random.csv --save-replay history\random-replay.json
python -m mario_rl_env.run_agent Super_Mario_Brothers_Maps\Processed\mario-1-1.txt --agent right_jump --episodes 10 --obs features
python -m mario_rl_env.run_agent Super_Mario_Brothers_Maps\Processed\mario-1-1.txt --agent smart_heuristic --episodes 10 --obs features
python -m mario_rl_env.run_agent Super_Mario_Brothers_Maps\Processed\mario-1-1.txt --agent tabular_q --episodes 100 --obs tiles --save-agent checkpoints\tabular-q.pt --history-file history\tabular-q.json --save-csv history\tabular-q.csv
python -m mario_rl_env.run_agent Super_Mario_Brothers_Maps\Processed\mario-1-1.txt --agent torch_dqn --episodes 100 --obs rgb --save-agent checkpoints\dqn-fixed.pt --history-file history\dqn-fixed.json --save-csv history\dqn-fixed.csv --save-replay history\dqn-fixed-replay.json
```

Expected artifacts: episode CSV/replay/history files and a `.pt` agent checkpoint. `tabular_q` supports `features` and `tiles`; `torch_dqn` expects `rgb`.

## 3. Stage 1 Classic Baseline

```bash
python -m pip install -r requirements-benchmark.txt
python -m mario_rl_env.benchmark train-classic --stages 1-1 1-2 2-1 4-2 --episodes 500 --batch-size 10 --seed 1 --checkpoint checkpoints\classic-seed1.pt --report benchmark\classic-seed1.json
python -m mario_rl_env.benchmark train-classic --stages 1-1 1-2 2-1 4-2 --episodes 500 --batch-size 10 --seed 2 --checkpoint checkpoints\classic-seed2.pt --report benchmark\classic-seed2.json
python -m mario_rl_env.benchmark compare --model random=random --training-env classic-seed1=gym-super-mario-bros --stages 1-1 1-2 2-1 4-2 --episodes 5 --seed 100 --report benchmark\classic-comparison.json
```

The comparison JSON includes per-stage episodes, win/death rates, progress, environment summaries, pairwise differences, and optional `rliable` aggregation.

## 4. Procedural DDRB Loop

```bash
uvicorn world_generation.API:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
python main.py --mode training --agent smart_heuristic --episodes 10 --batch-size 5 --case-id smoke-train --seed 10 --report-file reports\procedural-training.json
python main.py --mode validation --agent smart_heuristic --episodes 5 --batch-size 5 --case-id smoke-validation --seed 11 --report-file reports\procedural-validation.json
python main.py --mode testing --agent smart_heuristic --episodes 5 --batch-size 5 --case-id smoke-testing --seed 12 --report-file reports\procedural-testing.json
python main.py --mode inference --agent smart_heuristic --episodes 5 --batch-size 5 --case-id smoke-inference --seed 13 --report-file reports\procedural-inference.json
```

Docker alternative:

```bash
docker compose up --build world-gen-api
```

Expected output: level tiles from `pool_manager`, performance predictions, world-stitching oracle metadata, episode records in `checkpoints\experiments.sqlite3`, and JSON reports for non-empty `--report-file` paths.

## 5. Partitioned and Resumable Training

```bash
python -m mario_rl_env.run_agent Super_Mario_Brothers_Maps\Processed\mario-1-1.txt --agent tabular_q --mode training --case-id partitioned --episodes 500 --obs features --epsilon 0.4 --save-agent checkpoints\tabular-partition.pt --history-file history\tabular-partition-1.json
python -m mario_rl_env.run_agent Super_Mario_Brothers_Maps\Processed\mario-1-1.txt --agent tabular_q --mode training --case-id partitioned --episodes 500 --obs features --save-agent checkpoints\tabular-partition.pt --history-file history\tabular-partition-2.json
```

`--save-agent` loads an existing checkpoint and saves a new completion checkpoint. The structured store records periodic 500-episode snapshots and 1,000-episode checkpoint snapshots. A cycle ending between boundaries writes a lightweight cycle-end save point.

## 6. Stage 2 Cross-Origin Generalization

```bash
python main.py --mode training --agent torch_dqn --episodes 500 --batch-size 10 --seed 20 --checkpoint-path checkpoints\procedural-dqn.pt --report-file reports\procedural-dqn.json
python -m mario_rl_env.benchmark train-classic --agent torch_dqn --stages 1-1 1-2 2-1 4-2 --episodes 500 --batch-size 10 --seed 21 --checkpoint checkpoints\classic-dqn.pt --report benchmark\classic-dqn.json
python -m mario_rl_env.benchmark compare --model procedural=checkpoints\procedural-dqn.pt --training-env procedural=procedural_custom_simulator --model classic=checkpoints\classic-dqn.pt --training-env classic=gym-super-mario-bros --model random=random --training-env random=random_floor --stages 1-1 1-2 2-1 4-2 --procedural-level-set Super_Mario_Brothers_Maps\Processed\mario-1-1.txt Super_Mario_Brothers_Maps\Processed\mario-4-2.txt --episodes 5 --seed 200 --report benchmark\cross-origin-comparison.json
```

The report separates classic and `procedural:` held-out levels and includes cross-origin summaries and `rliable` metrics when `rliable` is installed.

## 7. Human Play and Diversity

Fixed map:

```bash
python -m mario_rl_env.run_human Super_Mario_Brothers_Maps\Processed\mario-1-1.txt --fps 60
```

Infinite procedural mode:

```bash
python -m mario_rl_env.run_human --infinite --api-url http://127.0.0.1:8000 --instance-id 1 --agent-name human-player
```

Diversity analysis:

```bash
python -m world_generation.tile_pattern_kl --set-a Super_Mario_Brothers_Maps\Original --set-b Super_Mario_Brothers_Maps\Processed --window 2 2 --stride 1 1
```

Tile-pattern output is JSON with forward/reverse KL and symmetric coefficient-of-determination fields. Human play requires a display in normal use; automated tests mock Pygame input and rendering.
