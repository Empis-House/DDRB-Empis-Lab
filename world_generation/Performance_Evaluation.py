# -*- coding: utf-8 -*-
"""Performance evaluation with transformer-based agent-history prediction.

This module keeps a legacy structural performance score for backward compatibility,
and adds a learned predictor that estimates an agent-conditioned performance score
for unplayed level samples using recent episode history.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .evaluation_utils import metric_values, normalize_level_tokens
except ImportError:  # pragma: no cover - keeps direct script execution working
    from evaluation_utils import metric_values, normalize_level_tokens


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _extract_episode_payload(episode: Dict[str, Any]) -> Dict[str, Any]:
    info = episode.get("info", {}) if isinstance(episode, dict) else {}
    episode_info = info.get("episode", {}) if isinstance(info, dict) else {}

    total_reward = _safe_float(episode.get("total_reward", episode.get("reward", 0.0)))
    steps = max(1.0, _safe_float(episode.get("steps", 1.0), 1.0))
    max_x = _safe_float(episode_info.get("max_x", info.get("max_x", 0.0)))
    coins = _safe_float(episode_info.get("coins", info.get("coins", 0.0)))
    kills = _safe_float(episode_info.get("kills", info.get("kills", 0.0)))
    won = _safe_bool(episode_info.get("won", info.get("won", False)))
    done = _safe_bool(episode.get("done", False))
    truncated = _safe_bool(episode.get("truncated", False))

    # Normalize into stable ranges expected by the model.
    reward_norm = math.tanh(total_reward / 1000.0)
    steps_norm = min(steps / 2000.0, 1.0)
    progress_norm = min(max_x / 3200.0, 1.0)
    coins_norm = min(coins / 30.0, 1.0)
    kills_norm = min(kills / 20.0, 1.0)

    # Supervision target: generic episode performance scalar [0,1].
    # This is intentionally agent-agnostic so different training styles can mix.
    target = _clip01(
        0.45 * won
        + 0.30 * progress_norm
        + 0.10 * coins_norm
        + 0.05 * kills_norm
        + 0.10 * _clip01((reward_norm + 1.0) * 0.5)
    )

    features = [
        reward_norm,
        steps_norm,
        progress_norm,
        coins_norm,
        kills_norm,
        won,
        done,
        truncated,
    ]
    return {"features": features, "target": target}


def _legacy_structural_score(level, df, non_landing_weight: float = 5.0 / 3.0, clip: bool = True) -> float:
    tokens = normalize_level_tokens(level)
    if not tokens:
        return 0.0

    values = metric_values(tokens, df, "Landings")
    if not values:
        return 0.0

    count = sum(non_landing_weight for value in values if value == 0)
    score = float(count / len(values))
    return _clip01(score) if clip else score


@dataclass
class _PerfConfig:
    feature_dim: int = 8
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    history_window: int = 32
    train_epochs: int = 12
    learning_rate: float = 1e-3
    min_samples: int = 24


class _HistoryTransformer(nn.Module):
    def __init__(self, config: _PerfConfig, num_agents: int):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.feature_dim, config.d_model)
        self.agent_embed = nn.Embedding(max(1, num_agents), config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.capability_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F), mask: (B, T) where 1 marks valid tokens.
        h = self.input_proj(x)
        h = h + self.agent_embed(agent_ids).unsqueeze(1)

        key_padding_mask = ~mask.bool()
        enc = self.encoder(h, src_key_padding_mask=key_padding_mask)

        # Mean-pool over valid steps only.
        valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (enc * mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        return self.capability_head(pooled).squeeze(-1)


_CONFIG = _PerfConfig()
_DEVICE = torch.device("cpu")
_HISTORY_DIR = Path("history")
_HISTORY_PATH = _HISTORY_DIR / "agent_performance_history.json"
_MODEL_PATH = Path("checkpoints") / "performance_transformer.pt"

_AGENT_HISTORY: Dict[str, List[Dict[str, Any]]] = {}
_AGENT_TO_INDEX: Dict[str, int] = {}
_MODEL: Optional[_HistoryTransformer] = None
_MODEL_READY = False
_LAST_TRAIN_SAMPLE_COUNT = 0


def _ensure_agent_index(agent_id: str) -> int:
    if agent_id not in _AGENT_TO_INDEX:
        _AGENT_TO_INDEX[agent_id] = len(_AGENT_TO_INDEX)
    return _AGENT_TO_INDEX[agent_id]


def _encode_sequence(agent_id: str, episodes: List[Dict[str, Any]], history_window: int) -> tuple[np.ndarray, np.ndarray, int]:
    parsed = [_extract_episode_payload(ep) for ep in episodes[-history_window:]]
    if not parsed:
        features = np.zeros((history_window, _CONFIG.feature_dim), dtype=np.float32)
        mask = np.zeros((history_window,), dtype=np.float32)
    else:
        feat = np.array([p["features"] for p in parsed], dtype=np.float32)
        seq_len = len(feat)
        features = np.zeros((history_window, _CONFIG.feature_dim), dtype=np.float32)
        mask = np.zeros((history_window,), dtype=np.float32)
        features[-seq_len:] = feat
        mask[-seq_len:] = 1.0

    agent_index = _ensure_agent_index(agent_id)
    return features, mask, agent_index


def _build_dataset(history_window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    agents: List[int] = []
    targets: List[float] = []

    for agent_id, episodes in _AGENT_HISTORY.items():
        if len(episodes) < 3:
            continue

        for end in range(2, len(episodes) + 1):
            window_eps = episodes[max(0, end - history_window):end - 1]
            label_ep = episodes[end - 1]
            parsed_label = _extract_episode_payload(label_ep)

            feat, mask, agent_index = _encode_sequence(agent_id, window_eps, history_window)
            xs.append(feat)
            masks.append(mask)
            agents.append(agent_index)
            targets.append(parsed_label["target"])

    if not xs:
        return (
            np.zeros((0, history_window, _CONFIG.feature_dim), dtype=np.float32),
            np.zeros((0, history_window), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.stack(xs).astype(np.float32),
        np.stack(masks).astype(np.float32),
        np.array(agents, dtype=np.int64),
        np.array(targets, dtype=np.float32),
    )


def _maybe_init_model() -> None:
    global _MODEL
    if _MODEL is None:
        _MODEL = _HistoryTransformer(_CONFIG, num_agents=max(1, len(_AGENT_TO_INDEX))).to(_DEVICE)


def _resize_agent_embeddings_if_needed() -> None:
    global _MODEL
    _maybe_init_model()
    assert _MODEL is not None

    required = max(1, len(_AGENT_TO_INDEX))
    current = _MODEL.agent_embed.num_embeddings
    if required <= current:
        return

    old = _MODEL.agent_embed
    new = nn.Embedding(required, _CONFIG.d_model).to(_DEVICE)
    with torch.no_grad():
        new.weight[:current] = old.weight
        nn.init.normal_(new.weight[current:], mean=0.0, std=0.02)
    _MODEL.agent_embed = new


def _save_state() -> None:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _HISTORY_PATH.open("w", encoding="utf-8") as fh:
        json.dump({"history": _AGENT_HISTORY, "agent_index": _AGENT_TO_INDEX}, fh, indent=2)

    if _MODEL is not None and _MODEL_READY:
        torch.save(
            {
                "state_dict": _MODEL.state_dict(),
                "agent_index": _AGENT_TO_INDEX,
                "config": _CONFIG.__dict__,
                "last_train_sample_count": _LAST_TRAIN_SAMPLE_COUNT,
            },
            _MODEL_PATH,
        )


def _load_state() -> None:
    global _AGENT_HISTORY, _AGENT_TO_INDEX, _MODEL_READY, _LAST_TRAIN_SAMPLE_COUNT
    if _HISTORY_PATH.exists():
        try:
            payload = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
            _AGENT_HISTORY = payload.get("history", {})
            _AGENT_TO_INDEX = {k: int(v) for k, v in payload.get("agent_index", {}).items()}
        except Exception:
            _AGENT_HISTORY = {}
            _AGENT_TO_INDEX = {}

    if _MODEL_PATH.exists():
        try:
            checkpoint = torch.load(_MODEL_PATH, map_location=_DEVICE)
            _AGENT_TO_INDEX = {k: int(v) for k, v in checkpoint.get("agent_index", _AGENT_TO_INDEX).items()}
            _maybe_init_model()
            _resize_agent_embeddings_if_needed()
            assert _MODEL is not None
            _MODEL.load_state_dict(checkpoint["state_dict"], strict=False)
            _MODEL_READY = True
            _LAST_TRAIN_SAMPLE_COUNT = int(checkpoint.get("last_train_sample_count", 0))
        except Exception:
            _MODEL_READY = False


_load_state()


def update_agent_history(agent_id: str, history: Iterable[Dict[str, Any]], append: bool = True) -> None:
    """Register latest episode history for an agent.

    Args:
        agent_id: Stable identifier of the policy/agent variant.
        history: Iterable of episode dictionaries from training/evaluation loops.
        append: If True append to existing history, else replace it.
    """
    if not agent_id:
        raise ValueError("agent_id is required")

    _ensure_agent_index(agent_id)
    incoming = [h for h in history if isinstance(h, dict)]
    if append:
        _AGENT_HISTORY.setdefault(agent_id, []).extend(incoming)
    else:
        _AGENT_HISTORY[agent_id] = incoming

    _save_state()


def train_performance_predictor(epochs: Optional[int] = None, history_window: Optional[int] = None) -> Dict[str, Any]:
    """Train or refresh the transformer model from all recorded agent histories."""
    global _MODEL_READY, _LAST_TRAIN_SAMPLE_COUNT
    if history_window is None:
        history_window = _CONFIG.history_window
    if epochs is None:
        epochs = _CONFIG.train_epochs

    x, mask, agent_ids, y = _build_dataset(history_window=history_window)
    sample_count = int(len(y))
    if sample_count < _CONFIG.min_samples:
        _MODEL_READY = False
        return {"trained": False, "samples": sample_count, "reason": "insufficient_samples"}

    _resize_agent_embeddings_if_needed()
    assert _MODEL is not None
    _MODEL.train()

    x_t = torch.tensor(x, dtype=torch.float32, device=_DEVICE)
    m_t = torch.tensor(mask, dtype=torch.float32, device=_DEVICE)
    a_t = torch.tensor(agent_ids, dtype=torch.int64, device=_DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=_DEVICE)

    optimizer = optim.Adam(_MODEL.parameters(), lr=_CONFIG.learning_rate)
    criterion = nn.MSELoss()

    batch_size = min(64, sample_count)
    losses: List[float] = []
    for _ in range(int(epochs)):
        perm = torch.randperm(sample_count, device=_DEVICE)
        epoch_loss = 0.0
        for start in range(0, sample_count, batch_size):
            idx = perm[start:start + batch_size]
            pred = _MODEL(x_t[idx], m_t[idx], a_t[idx])
            loss = criterion(pred, y_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu()) * len(idx)
        losses.append(epoch_loss / sample_count)

    _MODEL.eval()
    _MODEL_READY = True
    _LAST_TRAIN_SAMPLE_COUNT = sample_count
    _save_state()
    return {"trained": True, "samples": sample_count, "final_loss": losses[-1] if losses else None}


def _predict_agent_capability(agent_id: str, history_window: int) -> Optional[float]:
    if not _MODEL_READY or _MODEL is None:
        return None

    episodes = _AGENT_HISTORY.get(agent_id, [])
    if not episodes:
        return None

    feat, mask, agent_index = _encode_sequence(agent_id, episodes, history_window)
    x_t = torch.tensor(feat[None, ...], dtype=torch.float32, device=_DEVICE)
    m_t = torch.tensor(mask[None, ...], dtype=torch.float32, device=_DEVICE)
    a_t = torch.tensor([agent_index], dtype=torch.int64, device=_DEVICE)
    with torch.no_grad():
        pred = _MODEL(x_t, m_t, a_t).item()
    return _clip01(pred)


def get_performance_model_status(agent_id: Optional[str] = None) -> Dict[str, Any]:
    """Return current model readiness and optional agent-history availability."""
    status: Dict[str, Any] = {
        "model_ready": bool(_MODEL_READY and _MODEL is not None),
        "known_agents": len(_AGENT_HISTORY),
    }
    if agent_id is not None:
        status["agent_id"] = agent_id
        status["has_agent_history"] = bool(_AGENT_HISTORY.get(agent_id))
    return status


def Performance(
    level,
    df,
    non_landing_weight: float = 5.0 / 3.0,
    clip: bool = True,
    agent_id: Optional[str] = None,
    history_window: int = 32,
    auto_train: bool = True,
):
    """Predict level performance for an unplayed sample.

    If `agent_id` has enough recorded history and a trained transformer model
    is available, prediction is agent-conditioned:

        score = 0.35 * structural_score + 0.65 * transformer_capability

    Otherwise it falls back to the legacy structural score.

    This preserves old call sites that only pass `(level, df)`.

    Args:
        level: Sequence or string-repr of token ids.
        df: DataFrame with at least token and Landings columns.
        non_landing_weight: Weight applied to non-landing tokens.
        clip: If True, clamps final score to [0, 1].
        agent_id: Agent identifier for personalized prediction.
        history_window: Number of recent episodes used by the transformer.
        auto_train: If True, retrains when enough new samples are available.
    """
    structural = _legacy_structural_score(level, df=df, non_landing_weight=non_landing_weight, clip=True)

    if not agent_id:
        return structural if clip else _legacy_structural_score(level, df=df, non_landing_weight=non_landing_weight, clip=False)

    if auto_train:
        x, _, _, y = _build_dataset(history_window=history_window)
        sample_count = int(len(y))
        # Train only if model is missing/stale and enough samples exist.
        if sample_count >= _CONFIG.min_samples and (not _MODEL_READY or sample_count - _LAST_TRAIN_SAMPLE_COUNT >= 16):
            train_performance_predictor(history_window=history_window)

    capability = _predict_agent_capability(agent_id=agent_id, history_window=history_window)
    if capability is None:
        return structural

    score = 0.35 * structural + 0.65 * capability
    return _clip01(score) if clip else float(score)


def performance(
    level,
    df,
    non_landing_weight: float = 5.0 / 3.0,
    clip: bool = True,
    agent_id: Optional[str] = None,
    history_window: int = 32,
    auto_train: bool = True,
):
    """snake_case alias for Performance with identical behavior."""
    return Performance(
        level,
        df=df,
        non_landing_weight=non_landing_weight,
        clip=clip,
        agent_id=agent_id,
        history_window=history_window,
        auto_train=auto_train,
    )


Levels_Performance = pd.DataFrame(columns=["Level_String", "Added_Wordlist", "Performance"])


def Add_Level_To_Performance(level, df, WordsList=None, Levels_Performance=Levels_Performance):
    """Generate a sample level from a grammar and append its performance."""
    if WordsList is None:
        WordsList = []

    try:
        from .Grammar_Notation import Grammar
    except ImportError:  # pragma: no cover
        from Grammar_Notation import Grammar

    knowledge = df.to_numpy().T[3:]
    grammar = Grammar(level, knowledge=knowledge, WordsList=WordsList)
    generated = grammar.N_Level_Generator(100)._Sequence

    new_row = [level, WordsList, performance(generated, df=df)]
    Levels_Performance.loc[len(Levels_Performance)] = new_row
    return Levels_Performance


    
