from __future__ import annotations

import abc
import csv
import json
import os
import pickle
import warnings
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


# ── PyTorch DDQN Agent (from mario_rl_tutorial.ipynb) ─────────────────────

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torchvision import transforms as T
    from tensordict import TensorDict
    from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MarioNet(nn.Module):
    """Mini CNN structure for Mario RL (from notebook).
    
    Architecture: input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim: Tuple[int, int, int], output_dim: int):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self._build_cnn(c, output_dim)
        self.target = self._build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input: torch.Tensor, model: str = "online") -> torch.Tensor:
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        else:
            raise ValueError(f"Unknown model: {model}")

    def _build_cnn(self, c: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class TorchDQNAgent(RLAgent):
    """Double Deep Q-Network (DDQN) agent following mario_rl_tutorial.ipynb."""

    def __init__(
        self,
        action_size: int,
        state_dim: Tuple[int, int, int] = (4, 84, 84),
        learning_rate: float = 0.00025,
        gamma: float = 0.9,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.99999975,
        exploration_min: float = 0.1,
        batch_size: int = 32,
        memory_size: int = 10000,
        burnin: int = 10000,
        learn_every: int = 3,
        sync_every: int = 10000,
        save_every: int = 500000,
    ) -> None:
        super().__init__(action_size=action_size)
        
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch, torchvision, tensordict, and torchrl are required for TorchDQNAgent. "
                "Install them with: pip install torch torchvision tensordict torchrl"
            )

        self.state_dim = state_dim
        self.gamma = float(gamma)
        self.exploration_rate = float(exploration_rate)
        self.exploration_decay = float(exploration_decay)
        self.exploration_min = float(exploration_min)
        self.batch_size = int(batch_size)
        self.burnin = int(burnin)
        self.learn_every = int(learn_every)
        self.sync_every = int(sync_every)
        self.save_every = int(save_every)
        self.curr_step = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network
        self.net = MarioNet(state_dim, action_size).float()
        self.net = self.net.to(device=self.device)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # Replay buffer
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(memory_size, device=torch.device("cpu"))
        )

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert an observation to a compact uint8 4x84x84 tensor layout."""
        arr = np.asarray(obs)

        if arr.ndim == 1:
            raise ValueError(
                "TorchDQNAgent received a 1D feature vector observation. "
                "This CNN agent requires pixel observations (use obs_mode='rgb')."
            )

        if arr.ndim == 2:
            gray = arr.astype(np.float32)
        elif arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                rgb = arr[..., :3].astype(np.float32)
                gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            elif arr.shape[-1] == 1:
                gray = arr[..., 0].astype(np.float32)
            elif arr.shape[0] in (3, 4):
                chw = arr.astype(np.float32)
                gray = 0.299 * chw[0] + 0.587 * chw[1] + 0.114 * chw[2]
            elif arr.shape[0] == 1:
                gray = arr[0].astype(np.float32)
            else:
                raise ValueError(
                    f"Unsupported 3D observation shape for TorchDQNAgent: {arr.shape}. "
                    "Expected HxWxC or CxHxW image tensors."
                )
        else:
            raise ValueError(
                f"Unsupported observation rank for TorchDQNAgent: ndim={arr.ndim}, shape={arr.shape}"
            )

        frame = torch.tensor(gray, dtype=torch.float32)
        if frame.max().item() <= 1.0:
            frame = frame * 255.0

        # Resize to the Atari-style 84x84 expected by MarioNet.
        frame = F.interpolate(
            frame.unsqueeze(0).unsqueeze(0),
            size=(84, 84),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        # Duplicate grayscale frame into a 4-frame stack shape expected by the model.
        stacked = frame.unsqueeze(0).repeat(4, 1, 1).clamp(0, 255).to(torch.uint8)
        return stacked.detach().cpu().numpy()

    def act(self, obs: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy."""
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_size)
        # EXPLOIT
        else:
            obs_arr = self._preprocess_obs(obs)
            obs_tensor = torch.tensor(obs_arr, device=self.device, dtype=torch.float32).unsqueeze(0) / 255.0
            action_values = self.net(obs_tensor, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

        # Increment step
        self.curr_step += 1
        return int(action_idx)

    def cache(self, state: np.ndarray, next_state: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Store experience in replay buffer."""
        state = torch.tensor(state, dtype=torch.uint8)
        next_state = torch.tensor(next_state, dtype=torch.uint8)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)

        self.memory.add(
            TensorDict(
                {
                    "state": state,
                    "next_state": next_state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                },
                batch_size=[],
            )
        )

    def recall(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample batch from replay buffer."""
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done")
        )
        state = state.float() / 255.0
        next_state = next_state.float() / 255.0
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute TD estimate: Q_online(s, a)."""
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute TD target: r + γ * Q_target(s', argmax_a Q_online(s', a))."""
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate: torch.Tensor, td_target: torch.Tensor) -> float:
        """Update online network with loss: MSE(TD_estimate, TD_target)."""
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def sync_Q_target(self) -> None:
        """Sync target network parameters with online network."""
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self) -> Tuple[Optional[float], Optional[float]]:
        """Perform one learning step."""
        # Sync networks
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Burnin period: collect experiences before learning
        if self.curr_step < self.burnin:
            return None, None

        # Only learn every N steps
        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample batch
        state, next_state, action, reward, done = self.recall()

        # Compute TD estimate and target
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        # Update online network
        loss = self.update_Q_online(td_est, td_tgt)

        return (float(td_est.mean().item()), loss)

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience and learn from it."""
        try:
            state = self._preprocess_obs(obs)
            next_state = self._preprocess_obs(next_obs)
        except ValueError as exc:
            warnings.warn(str(exc), RuntimeWarning)
            raise

        self.cache(state, next_state, action, reward, done)
        _, _ = self.learn()

    def reset(self) -> None:
        """Reset any episode-specific state."""
        pass

    def save(self, path: str) -> None:
        """Save agent checkpoint including network state, exploration rate, and init params."""
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        checkpoint = {
            "model": self.net.state_dict(),
            "exploration_rate": self.exploration_rate,
            "curr_step": self.curr_step,
            # Store init parameters for proper reconstruction
            "init_params": {
                "action_size": self.action_size,
                "state_dim": self.state_dim,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "gamma": self.gamma,
                "exploration_decay": self.exploration_decay,
                "exploration_min": self.exploration_min,
                "batch_size": self.batch_size,
                "burnin": self.burnin,
                "learn_every": self.learn_every,
                "sync_every": self.sync_every,
                "save_every": self.save_every,
            }
        }
        
        tmp_path = f"{path}.tmp"
        try:
            torch.save(checkpoint, tmp_path)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            os.replace(tmp_path, path)
            print(f"[Agent] Saved DDQN checkpoint to {path} at step {self.curr_step}")
        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            print(f"[Agent] ERROR saving checkpoint to {path}: {e}")
            raise

    @classmethod
    def load(cls, path: str) -> "TorchDQNAgent":
        """Load agent from checkpoint with all parameters restored."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location="cpu")
        init_params = checkpoint.get("init_params", {})
        
        # Reconstruct agent with saved parameters
        agent = cls(**init_params)
        
        # Restore network weights
        if "model" in checkpoint:
            agent.net.load_state_dict(checkpoint["model"])
        
        # Restore training state
        if "exploration_rate" in checkpoint:
            agent.exploration_rate = float(checkpoint["exploration_rate"])
        if "curr_step" in checkpoint:
            agent.curr_step = int(checkpoint["curr_step"])
        
        print(f"[Agent] Loaded DDQN checkpoint from {path} (step {agent.curr_step})")
        return agent


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

# Register DDQN agent if PyTorch is available
if TORCH_AVAILABLE:
    AGENT_MANAGER.register(
        "torch_dqn",
        lambda action_size, **kwargs: TorchDQNAgent(action_size, **kwargs),
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
    trace_tag: str = "",
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

        if hasattr(episode_env, "set_auto_render"):
            if render_last:
                episode_env.set_auto_render(episode % batch_size == 0)
            else:
                episode_env.set_auto_render(render)

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
        if trace_tag:
            episode_record["trace_tag"] = trace_tag
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
