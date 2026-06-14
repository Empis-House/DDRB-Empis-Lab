from __future__ import annotations

import abc
import csv
import json
import os
import pickle
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np

StateKey = Tuple[int, ...]


class RLAgent(abc.ABC):
    """Base interface for Mario RL agents."""

    def __init__(self, action_size: int) -> None:
        self.action_size = action_size
        self.env: Optional[gym.Env] = None

    @abc.abstractmethod
    def act(self, obs: np.ndarray) -> int:
        """Choose an action for the given observation."""

    def attach_env(self, env: gym.Env) -> None:
        """Attach a live environment to the agent for use during training."""
        self.env = env

    def detach_env(self) -> None:
        """Detach the current environment from the agent."""
        self.env = None

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Receive transition feedback after taking an action."""
        return None

    def reset(self) -> None:
        """Reset any episode-specific state."""
        return None

    def save(self, path: str) -> None:
        """Serialize the agent to disk."""
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Temporarily detach environment (pygame.Surface not serializable)
        saved_env = self.env if hasattr(self, "env") else None
        if saved_env is not None:
            self.env = None
        # Use atomic write: write to temp file then replace to avoid leaving
        # truncated empty files if serialization fails midway.
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(self, f)
            # Remove target if it exists (Windows file lock workaround)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            os.replace(tmp_path, path)
            print(f"[Agent] Saved checkpoint to {path}")
        except Exception as e:
            # clean up temp file on error
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            print(f"[Agent] ERROR saving checkpoint to {path}: {e}")
            raise
        finally:
            # Restore environment so training can continue
            if saved_env is not None:
                self.env = saved_env

    @classmethod
    def load(cls, path: str) -> "RLAgent":
        """Load an agent previously saved with save()."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, RLAgent):
            raise TypeError(f"Loaded object is not an RLAgent: {type(obj)}")
        return obj


class TrainingHistory:
    """One-line history recorder for batch training."""

    def __init__(self, records: Optional[List[Dict[str, Any]]] = None) -> None:
        self.records = records or []

    def append(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def save(self, path: str) -> None:
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            if path.lower().endswith(".csv"):
                self._save_csv(path)
            else:
                self._save_json(path)
            print(f"[History] Saved {len(self.records)} records to {path}")
        except Exception as e:
            print(f"[History] ERROR saving to {path}: {e}")
            raise

    def _save_json(self, path: str) -> None:
        def _default(obj):
            # numpy types
            try:
                import numpy as _np
                if isinstance(obj, _np.generic):
                    return obj.item()
            except Exception:
                pass
            # enums
            try:
                from enum import Enum as _Enum
                if isinstance(obj, _Enum):
                    return obj.name
            except Exception:
                pass
            # fallback to string
            return str(obj)

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, default=_default)
        # Remove target if it exists (Windows file lock workaround)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        os.replace(tmp_path, path)

    def _save_csv(self, path: str) -> None:
        if not self.records:
            # create empty file atomically
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write("")
            # Remove target if it exists (Windows file lock workaround)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            os.replace(tmp_path, path)
            return
        keys = list(self.records[0].keys())
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
        # Remove target if it exists (Windows file lock workaround)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        os.replace(tmp_path, path)


class RandomAgent(RLAgent):
    """Baseline agent that selects actions uniformly at random."""

    def act(self, obs: np.ndarray) -> int:
        return int(np.random.randint(self.action_size))


def default_state_encoder(obs: np.ndarray) -> StateKey:
    arr = np.asarray(obs, dtype=np.float32).flatten()
    if arr.size == 0:
        return tuple()

    if np.issubdtype(arr.dtype, np.integer):
        return tuple(int(x) for x in arr.tolist())

    clipped = np.tanh(arr)
    bins = np.floor(clipped * 5.0).astype(int)
    return tuple(int(x) for x in bins.tolist())


class TabularQAgent(RLAgent):
    """Simple tabular Q-learning agent for discrete or encoded observations."""

    def __init__(
        self,
        action_size: int,
        state_encoder: Callable[[np.ndarray], StateKey] = default_state_encoder,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ) -> None:
        super().__init__(action_size=action_size)
        self.state_encoder = state_encoder
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)
        self.q_table: Dict[StateKey, np.ndarray] = {}

    def _ensure_state(self, state: StateKey) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[state]

    def resize_action_space(self, new_action_size: int) -> None:
        """Resize internal Q-table vectors to match a new action space size.

        Existing Q-values are preserved for the overlapping actions. If the
        action space grows, new entries are zero-initialized.
        """
        new_size = int(new_action_size)
        if new_size == self.action_size:
            return
        for state, q in list(self.q_table.items()):
            new_q = np.zeros(new_size, dtype=np.float32)
            copy_len = min(len(q), new_size)
            if copy_len > 0:
                new_q[:copy_len] = q[:copy_len]
            self.q_table[state] = new_q
        self.action_size = new_size

    def attach_env(self, env: gym.Env) -> None:
        """Attach an environment and derive env-dependent parameters.

        - Stores `observation_space` on the agent for encoder use.
        - If `env.action_space.n` is present and different from the current
          `action_size`, resize internal Q vectors to match.
        """
        super().attach_env(env)
        # store observation space for possible encoder use
        self.observation_space = getattr(env, "observation_space", None)
        # if discrete action space with `n`, adapt action size
        action_n = None
        try:
            action_n = getattr(env.action_space, "n", None)
        except Exception:
            action_n = None
        if action_n is not None and action_n != self.action_size:
            print(f"[Agent] Adapting TabularQAgent action_size {self.action_size} -> {action_n}")
            self.resize_action_space(int(action_n))

    def act(self, obs: np.ndarray) -> int:
        state = self.state_encoder(obs)
        q_values = self._ensure_state(state)
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
        return int(int(np.argmax(q_values)))

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        state = self.state_encoder(obs)
        next_state = self.state_encoder(next_obs)
        q_values = self._ensure_state(state)
        q_next = self._ensure_state(next_state)
        target = reward + self.gamma * np.max(q_next) * (0.0 if done else 1.0)
        q_values[action] += self.alpha * (target - q_values[action])

    def reset(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)


class MLAgentManager:
    """Registry and factory for Mario RL agents."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., RLAgent]] = {}

    def register(self, name: str, builder: Callable[..., RLAgent]) -> None:
        self._registry[name] = builder

    def create(self, name: str, *args: Any, **kwargs: Any) -> RLAgent:
        if name not in self._registry:
            raise KeyError(f"Unknown agent '{name}'. Available: {list(self._registry)}")
        return self._registry[name](*args, **kwargs)

    def list_agents(self) -> List[str]:
        return list(self._registry.keys())


AGENT_MANAGER = MLAgentManager()
AGENT_MANAGER.register("random", lambda action_size, **kwargs: RandomAgent(action_size))
AGENT_MANAGER.register(
    "tabular_q",
    lambda action_size, **kwargs: TabularQAgent(action_size, **kwargs),
)


def run_training_loop(
    env: Union[gym.Env, Callable[[], gym.Env]],
    agent: RLAgent,
    n_episodes: int = 100,
    max_steps: Optional[int] = None,
    render: bool = False,
    render_last: bool = False,
    verbose: bool = False,
    batch_size: int = 1,
    history_path: str = "",
    checkpoint_path: str = "",
    debug: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    history = TrainingHistory()
    if batch_size <= 0:
        batch_size = 1

    if history_path:
        print(f"[Training] Will save history to {history_path}")
    if checkpoint_path:
        print(f"[Training] Will save checkpoints to {checkpoint_path}")

    is_factory = callable(env)
    current_env = env() if is_factory else env
    episode_env = current_env

    for episode in range(1, n_episodes + 1):
        if debug and episode % 50 == 0:
            # periodically print some stats about Q-table size
            try:
                q_count = len(getattr(agent, "q_table", {}))
                print(f"[Debug] Episode {episode}: Q-table states={q_count}")
            except Exception:
                pass
        if episode > 1 and is_factory and (episode - 1) % batch_size == 0:
            if hasattr(episode_env, "close"):
                try:
                    episode_env.close()
                except Exception:
                    pass
            episode_env = env()

        agent.attach_env(episode_env)
        obs, _ = episode_env.reset()
        agent.reset()
        total_reward = 0.0
        steps = 0
        done = False
        truncated = False

        # debug tracking state for first few steps
        debug_tracked = None

        while not (done or truncated):
            action = agent.act(np.asarray(obs))

            # capture pre-update Q-values for debug
            if debug and debug_tracked is None:
                try:
                    state_enc = None
                    if hasattr(agent, "state_encoder"):
                        state_enc = agent.state_encoder(np.asarray(obs))
                    q_before = None
                    if state_enc is not None and hasattr(agent, "q_table"):
                        q_before = np.copy(agent.q_table.get(state_enc, np.zeros(agent.action_size, dtype=np.float32)))
                    debug_tracked = (state_enc, q_before)
                    print(f"[Debug] Episode {episode} step {steps}: obs_shape={np.asarray(obs).shape} encoded={state_enc}")
                except Exception as e:
                    print(f"[Debug] Pre-update debug failed: {e}")

            next_obs, reward, done, truncated, info = episode_env.step(action)
            agent.observe(np.asarray(obs), action, float(reward), np.asarray(next_obs), done or truncated)

            # capture post-update Q-values and print delta
            if debug and debug_tracked is not None:
                try:
                    state_enc, q_before = debug_tracked
                    if state_enc is not None and hasattr(agent, "q_table"):
                        q_after = agent.q_table.get(state_enc, None)
                        if q_after is not None and q_before is not None:
                            delta = q_after - q_before
                            print(f"[Debug] Q-delta for state {state_enc} action {action}: {delta}")
                            # only track first state's first delta
                            debug_tracked = None
                except Exception as e:
                    print(f"[Debug] Post-update debug failed: {e}")

            total_reward += float(reward)
            steps += 1
            obs = next_obs

            if render and hasattr(episode_env, "render"):
                if not render_last or episode == n_episodes:
                    episode_env.render()

            if max_steps is not None and steps >= max_steps:
                break

        episode_record = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "done": done,
            "truncated": truncated,
            "info": info,
        }
        results.append(episode_record)
        history.append(episode_record)

        if history_path:
            history.save(history_path)

        if checkpoint_path and episode % batch_size == 0:
            agent.save(checkpoint_path)

        if verbose:
            print(
                f"Episode {episode}/{n_episodes}: reward={total_reward:.2f}, "
                f"steps={steps}, done={done}, truncated={truncated}"
            )

    if checkpoint_path:
        agent.save(checkpoint_path)

    agent.detach_env()
    return results


def evaluate_agent(env: gym.Env, agent: RLAgent, n_episodes: int = 10) -> Dict[str, float]:
    history = run_training_loop(env, agent, n_episodes=n_episodes, render=False, verbose=False)
    rewards = [entry["total_reward"] for entry in history]
    return {
        "episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
    }
