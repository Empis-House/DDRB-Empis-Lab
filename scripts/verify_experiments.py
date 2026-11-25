"""Fast end-to-end smoke verification for DDRB Lab experiment workflows."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fixed_map_smoke(output: Path) -> dict:
    from mario_rl_env.run_agent import run

    map_path = ROOT / "Super_Mario_Brothers_Maps" / "Processed" / "mario-1-1.txt"
    result = run(
        map_path=str(map_path),
        agent_name="random",
        n_episodes=1,
        obs_mode="features",
        verbose=False,
        save_csv=str(output / "episodes.csv"),
        save_replay=str(output / "replay.json"),
        history_path=str(output / "history.json"),
    )
    return {"result": result, "artifacts": sorted(p.name for p in output.iterdir())}


def api_smoke(output: Path) -> dict:
    from world_generation.API import app

    with TestClient(app) as client:
        health = client.get("/")
        health.raise_for_status()
        instance = client.post(
            "/instances",
            json={
                "function": "Performance",
                "alphas": [0.0],
                "variety_domains": ["overworld"],
                "grid_points": [[0.0, 0.5, 1.0]],
                "grid_mode": "fixed",
                "instance_id": "smoke",
            },
        )
        instance.raise_for_status()
        predicted = client.post(
            "/performance/predict",
            json={"level": "XXXXXXXX\n--------", "agent_name": "smoke"},
        )
        predicted.raise_for_status()
        report = {
            "health": health.json(),
            "instance": instance.json(),
            "prediction": predicted.json(),
        }
    (output / "api-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def diversity_smoke(output: Path) -> dict:
    from world_generation.tile_pattern_kl import evaluate_level_folders

    original = ROOT / "Super_Mario_Brothers_Maps" / "Original"
    generated = ROOT / "Super_Mario_Brothers_Maps" / "Domains"
    if not original.exists() or not any(original.rglob("*.txt")):
        original = ROOT / "Super_Mario_Brothers_Maps" / "Processed"
    if not generated.exists():
        generated = original
    result = evaluate_level_folders(original, generated, window=(2, 2), stride=(1, 1))
    (output / "diversity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def benchmark_smoke(output: Path) -> dict:
    from mario_rl_env import benchmark
    import gymnasium as gym
    import numpy as np

    class SmokeEnv(gym.Env):
        action_space = gym.spaces.Discrete(8)
        observation_space = gym.spaces.Box(0, 255, shape=(2, 2, 3), dtype=np.uint8)

        def reset(self, *, seed=None, options=None):
            return np.zeros((2, 2, 3), dtype=np.uint8), {}

        def step(self, action):
            obs = np.zeros((2, 2, 3), dtype=np.uint8)
            return obs, 1.0, True, False, {
                "episode": {
                    "max_progress_tiles": 1.0,
                    "won": True,
                    "died": False,
                    "termination_reason": "win",
                    "coins": 0,
                },
                "coins": 0,
            }

    def fake_make_classic_env(stage, *, render_mode=None, seed=None):
        return SmokeEnv()

    benchmark.make_classic_env = fake_make_classic_env

    report = benchmark.compare_checkpoints(
        models={"random": "random"},
        training_environments={"random": "random_floor"},
        stages=("1-1",),
        episodes=1,
        procedural_levels=(),
    )
    (output / "benchmark-smoke.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run fast headless verification")
    args = parser.parse_args()
    if not args.smoke:
        parser.print_help()
        return 2

    with tempfile.TemporaryDirectory(prefix="ddrb-smoke-") as directory:
        output = Path(directory)
        checks = {
            "fixed_map": fixed_map_smoke(output),
            "api": api_smoke(output),
            "benchmark": benchmark_smoke(output),
            "diversity": diversity_smoke(output),
        }
        print(json.dumps({
            "status": "passed",
            "checks": list(checks),
            "output_temporary": str(output),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
