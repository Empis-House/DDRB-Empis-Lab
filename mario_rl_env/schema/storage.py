"""SQLite and gzip JSONL persistence for experiment records."""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from .models import ExperimentRun, Level, Performance, RLPlayerSnapshot


class SQLiteExperimentStore:
    """Small SQLite store; payload JSON preserves the complete validated models."""

    def __init__(self, path: str | Path = "checkpoints/experiments.sqlite3") -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            self.connection.execute("PRAGMA journal_mode = WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY,
                case_id TEXT,
                evaluation_stage TEXT NOT NULL,
                training_env_backend TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS player_snapshots (
                player_snapshot_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                episodes_completed INTEGER NOT NULL,
                payload TEXT NOT NULL,
                parent_snapshot_id TEXT,
                FOREIGN KEY(parent_snapshot_id) REFERENCES player_snapshots(player_snapshot_id)
                    DEFERRABLE INITIALLY DEFERRED
            );
            CREATE TABLE IF NOT EXISTS levels (
                level_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS performances (
                performance_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                evaluation_stage TEXT NOT NULL,
                episode_index INTEGER NOT NULL,
                evaluation_mode TEXT NOT NULL,
                split_role TEXT NOT NULL,
                level_id TEXT NOT NULL,
                player_snapshot_id TEXT NOT NULL,
                seed INTEGER,
                progress_ratio REAL,
                protocol_reward_total REAL NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(run_id, episode_index),
                FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id),
                FOREIGN KEY(player_snapshot_id) REFERENCES player_snapshots(player_snapshot_id),
                FOREIGN KEY(level_id) REFERENCES levels(level_id)
            );
            CREATE INDEX IF NOT EXISTS idx_perf_run_episode ON performances(run_id, episode_index);
            CREATE INDEX IF NOT EXISTS idx_perf_agent_stage ON performances(agent_name, evaluation_stage);
            CREATE INDEX IF NOT EXISTS idx_perf_stage_split ON performances(evaluation_stage, split_role);
            CREATE INDEX IF NOT EXISTS idx_perf_level_seed ON performances(level_id, seed);
            CREATE INDEX IF NOT EXISTS idx_snapshot_agent_episode
                ON player_snapshots(agent_id, episodes_completed);
        """)
        self.connection.commit()

    @staticmethod
    def _payload(model: Any) -> str:
        return model.model_dump_json()

    def upsert_run(self, run: ExperimentRun) -> None:
        self.connection.execute(
            """INSERT INTO experiment_runs VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET case_id=excluded.case_id,
               evaluation_stage=excluded.evaluation_stage,
               training_env_backend=excluded.training_env_backend, payload=excluded.payload""",
            (run.run_id, run.case_id, run.evaluation_stage, run.training_env_backend, self._payload(run)),
        )
        self.connection.commit()

    def upsert_snapshot(self, snapshot: RLPlayerSnapshot) -> None:
        self.connection.execute(
            """INSERT INTO player_snapshots
               (player_snapshot_id, agent_id, agent_name, episodes_completed, payload, parent_snapshot_id)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(player_snapshot_id) DO UPDATE SET agent_id=excluded.agent_id,
               agent_name=excluded.agent_name, episodes_completed=excluded.episodes_completed,
               payload=excluded.payload, parent_snapshot_id=excluded.parent_snapshot_id""",
            (snapshot.player_snapshot_id, snapshot.agent_id, snapshot.agent_name,
             snapshot.training_state.episodes_completed, self._payload(snapshot),
             snapshot.parent_snapshot_id),
        )
        self.connection.commit()

    def upsert_level(self, level: Level) -> None:
        self.connection.execute(
            """INSERT INTO levels VALUES (?, ?)
               ON CONFLICT(level_id) DO UPDATE SET payload=excluded.payload""",
            (level.level_id, self._payload(level)),
        )
        self.connection.commit()

    def latest_snapshot(self, agent_id: str) -> RLPlayerSnapshot | None:
        row = self.connection.execute(
            """SELECT payload FROM player_snapshots WHERE agent_id = ?
               ORDER BY episodes_completed DESC, rowid DESC LIMIT 1""",
            (agent_id,),
        ).fetchone()
        return RLPlayerSnapshot.model_validate_json(row["payload"]) if row else None

    def insert_performance(
        self, performance: Performance, *, agent_name: str,
        evaluation_stage: str, seed: int | None = None,
    ) -> None:
        self.connection.execute(
            """INSERT INTO performances
               (performance_id, run_id, agent_name, evaluation_stage, episode_index,
                evaluation_mode, split_role, level_id, seed, progress_ratio,
                protocol_reward_total, payload, player_snapshot_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (performance.performance_id, performance.run_id, agent_name, evaluation_stage,
             performance.episode_index, performance.execution_context.mode,
             performance.execution_context.split_role, performance.level_id, seed,
             performance.env_performance.progress_ratio,
             performance.env_performance.protocol_reward_total, self._payload(performance),
             performance.player_snapshot_id),
        )
        self.connection.commit()

    def list_performances(
        self, *, evaluation_stage: str | None = None,
        agent_name: str | None = None, split_role: str | None = None,
    ) -> list[Performance]:
        clauses: list[str] = []
        values: list[str] = []
        for column, value in (
            ("evaluation_stage", evaluation_stage), ("agent_name", agent_name),
            ("split_role", split_role),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT payload FROM performances{where} ORDER BY run_id, episode_index", values
        ).fetchall()
        return [Performance.model_validate_json(row["payload"]) for row in rows]

    def query_episode_rows(
        self, *, evaluation_stage: str, split_role: str = "test",
        agent_name: str | None = None, run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["p.evaluation_stage = ?", "p.split_role = ?"]
        values: list[Any] = [evaluation_stage, split_role]
        if agent_name is not None:
            clauses.append("p.agent_name = ?")
            values.append(agent_name)
        if run_id is not None:
            clauses.append("p.run_id = ?")
            values.append(run_id)
        rows = self.connection.execute(
            "SELECT p.agent_name, p.run_id, p.episode_index, p.level_id, p.seed, p.progress_ratio, "
            "p.protocol_reward_total, p.payload FROM performances p WHERE "
            + " AND ".join(clauses)
            + " ORDER BY p.agent_name, p.run_id, p.episode_index",
            values,
        ).fetchall()
        return [dict(row) for row in rows]

    def summarize_run(self, run_id: str) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT payload FROM performances WHERE run_id = ? ORDER BY episode_index",
            (run_id,),
        ).fetchall()
        episodes = [Performance.model_validate_json(row["payload"]) for row in rows]
        if not episodes:
            return {
                "episodes": 0, "mean_reward": 0.0, "min_reward": 0.0,
                "max_reward": 0.0, "mean_steps": 0.0, "wins": 0,
                "win_rate": 0.0, "mean_max_x": 0.0,
                "mean_max_progress_tiles": 0.0,
            }
        rewards = [item.env_performance.protocol_reward_total for item in episodes]
        steps = [item.env_performance.steps_taken for item in episodes]
        progress = [item.env_performance.progress_max_tiles for item in episodes]
        wins = sum(item.posteriori_outcome.clear for item in episodes)
        return {
            "episodes": len(episodes),
            "mean_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "mean_steps": sum(steps) / len(steps),
            "wins": wins,
            "win_rate": wins / len(episodes),
            "mean_max_x": sum(progress) / len(progress),
            "mean_max_progress_tiles": sum(progress) / len(progress),
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteExperimentStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class GzipTrajectoryWriter:
    """Write one trajectory step per compressed JSONL record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = gzip.open(self.path, "wt", encoding="utf-8")
        self.step_count = 0

    def write(self, step: dict[str, Any]) -> None:
        self._file.write(json.dumps(step, separators=(",", ":"), allow_nan=False) + "\n")
        self.step_count += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "GzipTrajectoryWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def read_trajectory(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("trajectory JSONL records must be JSON objects")
                yield value
