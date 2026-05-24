"""Option-D true U-shaped split-learning TimeMixer backend with dynamic clients.

Split implemented here:
    Client front: scaling + multiscale downsampling + input embedding
    Client -> Server: embedded multiscale states, not raw demand values
    Server: dynamic aggregation, personalized redistribution, PDM/FMM feature mixing
    Server -> Client: personalized mixed multiscale hidden states, not final forecasts
    Client tail: local future heads, final forecast, inverse-scaling, and label-side loss

This is a true U-shaped split: the forward pass starts on the client, continues on
one shared server, and returns to the client for the prediction head. No additional
privacy projection is used, so the embeddings and returned hidden states should be
understood as learned representations rather than a formal privacy guarantee.

Expected project imports are kept close to your current backend. If your package paths
differ, adapt only the imports at the top.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from helpers.helpers import create_dataset, select_gpu
from .base import ForecastingBackend
from . import register_backend

logger = logging.getLogger("logger")


# ---------------------------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------------------------


def _resize_sequence(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resize sequence length of [B, L, D] tensor with linear interpolation."""
    if x.size(1) == target_len:
        return x
    return F.interpolate(
        x.transpose(1, 2), size=target_len, mode="linear", align_corners=False
    ).transpose(1, 2)


class MovingAverageDecomposition(nn.Module):
    """Non-trainable TimeMixer-style moving-average decomposition.

    seasonal = x - moving_average(x)
    trend    = moving_average(x)
    """

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = int(kernel_size)
        self.pad = self.kernel_size // 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, D]
        x_t = x.transpose(1, 2)  # [B, D, L]
        trend = F.avg_pool1d(
            F.pad(x_t, (self.pad, self.pad), mode="replicate"),
            kernel_size=self.kernel_size,
            stride=1,
        ).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class MixerBlock(nn.Module):
    """MLP mixer over time tokens and feature channels.

    This is a compact, TimeMixer-style MLP-only block. It has a fixed sequence
    length for the scale it operates on. Dynamic client counts are handled by
    the server's aggregation rule, not by changing this block's shape.
    """

    def __init__(self, d_model: int, seq_len: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.norm_time = nn.LayerNorm(d_model)
        self.token_mixer = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.seq_len, self.seq_len),
            nn.Dropout(dropout),
        )
        self.norm_channel = nn.LayerNorm(d_model)
        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.seq_len:
            x = _resize_sequence(x, self.seq_len)

        # Token/time mixing: [B, L, D] -> [B, D, L] -> MLP over L.
        residual = x
        t = self.norm_time(x).transpose(1, 2)
        t = self.token_mixer(t).transpose(1, 2)
        x = residual + t

        # Channel/feature mixing.
        x = x + self.channel_mixer(self.norm_channel(x))
        return x


# ---------------------------------------------------------------------------
# Client: front embedding + tail prediction head
# ---------------------------------------------------------------------------


class TimeMixerEmbeddingClient(nn.Module):
    """Client-side module for the true U-shaped Option-D split.

    The same client module owns two local parts:
    1. front/encoder: creates multiscale embeddings that are sent to the server;
    2. tail/decoder: receives personalized mixed hidden states from the server and
       computes the final forecast locally.

    Thus, the server never owns the final future heads in this version.
    """

    def __init__(
        self,
        input_dim: int = 1,
        sequence_length: int = 24,
        scales: Sequence[int] = (1, 2, 4),
        d_model: int = 32,
        ff_dim: int = 64,
        dropout: float = 0.1,
        horizon: int = 1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if not scales:
            raise ValueError("scales must contain at least one downsampling factor")
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.scales = tuple(int(s) for s in scales)
        self.scale_lengths = tuple((self.sequence_length + s - 1) // s for s in self.scales)
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)

        # Client front: local TimeMixer-style value embeddings.
        self.embeddings = nn.ModuleList(
            [nn.Linear(self.input_dim, self.d_model) for _ in self.scales]
        )
        self.embedding_norms = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in self.scales]
        )

        # Client tail: local future multipredictor heads. The server returns a
        # fused seasonal/trend state with feature size 2 * d_model for each scale.
        self.future_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(2 * self.d_model),
                    nn.Linear(2 * self.d_model, int(ff_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(ff_dim), self.horizon * self.output_dim),
                )
                for _ in self.scales
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))

    def _make_multiscale(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, L, input_dim]
        xs: List[torch.Tensor] = []
        x_t = x.transpose(1, 2)
        for factor in self.scales:
            if factor == 1:
                xs.append(x)
            else:
                pooled = F.avg_pool1d(
                    x_t, kernel_size=factor, stride=factor, ceil_mode=True
                ).transpose(1, 2)
                xs.append(pooled)
        return xs

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Client front: local sequence -> multiscale embeddings sent to server."""
        multiscale = self._make_multiscale(x)
        embedded: List[torch.Tensor] = []
        for scale_id, x_s in enumerate(multiscale):
            h = self.embeddings[scale_id](x_s)
            h = self.embedding_norms[scale_id](h)
            embedded.append(h)
        return embedded

    def decode(self, server_states: Sequence[torch.Tensor]) -> torch.Tensor:
        """Client tail: server-returned hidden states -> final local forecast.

        Args:
            server_states: list over scales, each tensor [B, L_s, 2 * d_model].

        Returns:
            Local forecast [B, horizon, output_dim].
        """
        if len(server_states) != len(self.future_heads):
            raise ValueError("server scale count does not match client scale count")

        preds: List[torch.Tensor] = []
        for scale_id, state in enumerate(server_states):
            pooled = state[:, -1, :]
            pred = self.future_heads[scale_id](pooled).view(
                pooled.size(0), self.horizon, self.output_dim
            )
            preds.append(pred)

        weights = torch.softmax(self.scale_logits, dim=0)
        forecast = torch.zeros_like(preds[0])
        for weight, pred in zip(weights, preds):
            forecast = forecast + weight * pred
        return forecast

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Kept for compatibility with the older code path. In the true U-shaped
        # path, call encode(...) before the server and decode(...) after it.
        return self.encode(x)


# ---------------------------------------------------------------------------
# Server: aggregation + redistribution + TimeMixer feature processing
# ---------------------------------------------------------------------------


class OptionDTimeMixerServer(nn.Module):
    """Server-side middle part for the true U-shaped Option-D split.

    The server receives client embeddings, aggregates them dynamically, injects
    global context back into each channel, and applies the TimeMixer PDM feature
    processing. It does NOT compute final forecasts. Instead, it returns one
    personalized list of mixed hidden states per channel back to the clients.
    """

    def __init__(
        self,
        scale_lengths: Sequence[int],
        d_model: int = 32,
        ff_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
        decomp_kernel: int = 3,
        horizon: int = 1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if not scale_lengths:
            raise ValueError("scale_lengths must not be empty")
        self.scale_lengths = tuple(int(x) for x in scale_lengths)
        self.num_scales = len(self.scale_lengths)
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)

        # Dynamic aggregation of arbitrary number of client/channel inputs.
        self.client_gates = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(self.num_scales)]
        )

        # Distribution of global aggregated context back into each channel's
        # personalized server-side representation before PDM.
        self.entry_distributors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(2 * d_model),
                    nn.Linear(2 * d_model, ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, d_model),
                )
                for _ in range(self.num_scales)
            ]
        )

        self.decomposition = MovingAverageDecomposition(decomp_kernel)

        # PDM: bottom-up seasonal mixing.
        self.seasonal_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        MixerBlock(d_model, self.scale_lengths[i], ff_dim, dropout)
                        for _ in range(n_layers)
                    ]
                )
                for i in range(self.num_scales)
            ]
        )
        self.seasonal_down_projections = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(d_model, d_model) for _ in range(1, self.num_scales)]
        )

        # PDM: top-down trend mixing.
        self.trend_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        MixerBlock(d_model, self.scale_lengths[i], ff_dim, dropout)
                        for _ in range(n_layers)
                    ]
                )
                for i in range(self.num_scales)
            ]
        )
        self.trend_up_projections = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(self.num_scales)]
        )

    def _aggregate_by_scale(
        self, embedded_by_channel: Sequence[Sequence[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Return one global aggregated representation per scale."""
        if not embedded_by_channel:
            raise ValueError("embedded_by_channel must not be empty")
        if len(embedded_by_channel[0]) != self.num_scales:
            raise ValueError("input scale count does not match server num_scales")

        global_by_scale: List[torch.Tensor] = []
        for scale_id in range(self.num_scales):
            target_len = self.scale_lengths[scale_id]
            states = [
                _resize_sequence(channel_embeds[scale_id], target_len)
                for channel_embeds in embedded_by_channel
            ]
            stacked = torch.stack(states, dim=1)  # [B, N, L, D]
            scores = self.client_gates[scale_id](stacked).squeeze(-1)  # [B, N, L]
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, N, L, 1]
            global_state = (weights * stacked).sum(dim=1)  # [B, L, D]
            global_by_scale.append(global_state)
        return global_by_scale

    def _redistribute_to_channels(
        self,
        embedded_by_channel: Sequence[Sequence[torch.Tensor]],
        global_by_scale: Sequence[torch.Tensor],
    ) -> List[List[torch.Tensor]]:
        """Combine each local embedding with the aggregated global state."""
        personalized_by_channel: List[List[torch.Tensor]] = []
        for channel_embeds in embedded_by_channel:
            personalized_scales: List[torch.Tensor] = []
            for scale_id, local_state in enumerate(channel_embeds):
                global_state = global_by_scale[scale_id]
                if global_state.size(1) != local_state.size(1):
                    global_state = _resize_sequence(global_state, local_state.size(1))
                personalized = self.entry_distributors[scale_id](
                    torch.cat([local_state, global_state], dim=-1)
                )
                personalized_scales.append(local_state + personalized)
            personalized_by_channel.append(personalized_scales)
        return personalized_by_channel

    def _process_one_channel(self, personalized_scales: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Run server-side PDM and return fused hidden states for the client tail.

        Returns:
            list over scales, each tensor [B, L_s, 2 * d_model], where the last
            dimension is concat([seasonal_mixed, trend_mixed]).
        """
        if len(personalized_scales) != self.num_scales:
            raise ValueError("personalized scale count does not match server num_scales")

        seasonal_raw: List[torch.Tensor] = []
        trend_raw: List[torch.Tensor] = []
        for scale_id, h in enumerate(personalized_scales):
            h = _resize_sequence(h, self.scale_lengths[scale_id])
            seasonal, trend = self.decomposition(h)
            seasonal_raw.append(seasonal)
            trend_raw.append(trend)

        # Bottom-up seasonal mixing: fine -> coarse.
        seasonal_mixed: List[torch.Tensor] = []
        for scale_id, seasonal in enumerate(seasonal_raw):
            if scale_id == 0:
                current = seasonal
            else:
                prev = _resize_sequence(seasonal_mixed[-1], seasonal.size(1))
                current = seasonal + self.seasonal_down_projections[scale_id](prev)
            seasonal_mixed.append(self.seasonal_mixers[scale_id](current))

        # Top-down trend mixing: coarse -> fine.
        trend_mixed: List[Optional[torch.Tensor]] = [None] * self.num_scales
        trend_mixed[-1] = self.trend_mixers[-1](trend_raw[-1])
        for scale_id in reversed(range(self.num_scales - 1)):
            coarse = _resize_sequence(trend_mixed[scale_id + 1], trend_raw[scale_id].size(1))
            current = trend_raw[scale_id] + self.trend_up_projections[scale_id](coarse)
            trend_mixed[scale_id] = self.trend_mixers[scale_id](current)

        # The server returns mixed features; the client tail owns the forecast heads.
        fused_states: List[torch.Tensor] = []
        for scale_id in range(self.num_scales):
            trend_state = trend_mixed[scale_id]
            assert trend_state is not None
            seasonal_state = seasonal_mixed[scale_id]
            if trend_state.size(1) != seasonal_state.size(1):
                trend_state = _resize_sequence(trend_state, seasonal_state.size(1))
            fused_states.append(torch.cat([seasonal_state, trend_state], dim=-1))
        return fused_states

    def forward(
        self, embedded_by_channel: Sequence[Sequence[torch.Tensor]]
    ) -> Tuple[List[List[torch.Tensor]], Dict[str, Any]]:
        """Return personalized server hidden states, not predictions.

        Args:
            embedded_by_channel: list with length N_channels. Each element is a
                list over scales, each tensor [B, L_s, D].

        Returns:
            states_by_channel: list with length N_channels. Each element is a
                list over scales, each tensor [B, L_s, 2 * D] to be sent back to
                the matching client tail.
        """
        global_by_scale = self._aggregate_by_scale(embedded_by_channel)
        personalized_by_channel = self._redistribute_to_channels(
            embedded_by_channel, global_by_scale
        )
        states_by_channel = [
            self._process_one_channel(channel_scales)
            for channel_scales in personalized_by_channel
        ]
        aux = {
            "num_channels": len(embedded_by_channel),
            "num_scales": self.num_scales,
        }
        return states_by_channel, aux


# ---------------------------------------------------------------------------
# Forecasting wrapper saved on each supply-chain agent
# ---------------------------------------------------------------------------


@dataclass
class OptionDSplitTimeMixerForecastingModel:
    client_models: nn.ModuleList
    server_model: OptionDTimeMixerServer
    scaler: StandardScaler
    device: torch.device
    scales: Tuple[int, ...]
    scale_lengths: Tuple[int, ...]
    horizon: int = 1


@dataclass(frozen=True)
class ChannelSpec:
    agent_idx: int
    retailer_idx: int
    loader_idx: int


class SplitEarlyStopping:
    """Early stopping that stores best client/server state_dicts."""

    def __init__(self, patience: int = 100, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_state: Optional[Dict[str, Any]] = None

    def __call__(
        self,
        val_loss: float,
        agent_client_models: Sequence[nn.ModuleList],
        server_model: nn.Module,
        scalers: Sequence[StandardScaler],
    ) -> None:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = float(val_loss)
            self.counter = 0
            self.best_state = {
                "server": copy.deepcopy(server_model.state_dict()),
                "clients": [copy.deepcopy(models.state_dict()) for models in agent_client_models],
                "scalers": list(scalers),
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@register_backend("split_timemixer")
class TimeMixerOptionDSplitBackend(ForecastingBackend):
    """True U-shaped split TimeMixer backend.

    Client front:
        scaling, multiscale downsampling, trainable embeddings.

    Server middle:
        dynamic aggregation, redistribution, decomposition, seasonal/trend mixing.

    Client tail:
        receives personalized mixed states from the server, computes the final
        forecast locally, and can compute the label-side loss locally.
    """

    name = "split_timemixer_option_d"

    @property
    def collaborative_level(self) -> Optional[int]:
        return 1

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[List[float]]:
        return _train_split_timemixer_option_d(
            simulation, market, supply_chain, sc_agent_list, self.cfg
        )

    def collaborative_predict(self, level_agents, demand_t, sc_agent_list, t) -> List[List[float]]:
        if not level_agents:
            return []

        first_model: OptionDSplitTimeMixerForecastingModel = level_agents[0].get_forecasting_model()
        device = first_model.device
        server = first_model.server_model.to(device)
        server.eval()

        embedded_by_channel: List[List[torch.Tensor]] = []
        channel_to_agent_retailer: List[Tuple[int, int]] = []

        for agent_idx, agent in enumerate(level_agents):
            fm: OptionDSplitTimeMixerForecastingModel = agent.get_forecasting_model()
            raw_demand_data = agent.demand_by_retailer_history
            data = [np.array(raw_demand_data[r]) for r in range(agent.num_retailer)]
            df = pd.DataFrame(data).transpose()
            df = df.iloc[-(agent.sequence_length - 1):, :]

            # Same convention as the original backend: demand_t has retailer rows
            # and one column per level-1 agent.
            df.loc[len(df)] = demand_t[:, agent_idx]
            scaled = fm.scaler.transform(df.to_numpy())

            for retailer_idx in range(agent.num_retailer):
                x = scaled[-agent.sequence_length :, retailer_idx].reshape(
                    1, agent.sequence_length, 1
                )
                input_tensor = torch.tensor(x, dtype=torch.float32, device=device)
                client_model = fm.client_models[retailer_idx].to(device)
                client_model.eval()
                embedded = client_model.encode(input_tensor)
                embedded_by_channel.append(embedded)
                channel_to_agent_retailer.append((agent_idx, retailer_idx))

        with torch.no_grad():
            # Server returns personalized hidden states, one list of scale states
            # per channel. Each matching client tail finishes the prediction.
            server_states_by_channel, _aux = server(embedded_by_channel)
            flat_predictions: List[float] = []
            for channel_idx, (agent_idx, retailer_idx) in enumerate(channel_to_agent_retailer):
                fm = level_agents[agent_idx].get_forecasting_model()
                client_model = fm.client_models[retailer_idx].to(device)
                client_model.eval()
                output = client_model.decode(server_states_by_channel[channel_idx])
                value_scaled = output[:, -1, 0].item()
                value = value_scaled * fm.scaler.scale_[retailer_idx] + fm.scaler.mean_[retailer_idx]
                flat_predictions.append(float(value))

        predictions: List[List[float]] = []
        cursor = 0
        for agent in level_agents:
            n = agent.num_retailer
            predictions.append(flat_predictions[cursor : cursor + n])
            cursor += n
        return predictions


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _cfg_get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get("timemixer", {}).get(key, default)


def _last_horizon_target(y: torch.Tensor, horizon: int) -> torch.Tensor:
    """Normalize dataset target shapes to [B, horizon, 1]."""
    if y.dim() == 2:
        y = y.unsqueeze(-1)
    if y.dim() != 3:
        raise ValueError(f"expected target tensor with 2 or 3 dims, got shape {tuple(y.shape)}")
    return y[:, -horizon:, :]


def _inverse_scale_tensor(
    y_scaled: torch.Tensor, scaler: StandardScaler, retailer_idx: int
) -> torch.Tensor:
    return y_scaled * float(scaler.scale_[retailer_idx]) + float(scaler.mean_[retailer_idx])


def _zero_all(optimizers: Iterable[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)


def _step_all(optimizers: Iterable[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.step()


def _set_train(models: Iterable[nn.Module], mode: bool) -> None:
    for model in models:
        model.train(mode)


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _train_split_timemixer_option_d(simulation, market, supply_chain, sc_agent_list, cfg):
    logger.info("Starting Option-D U-shaped split TimeMixer training")

    device = select_gpu()
    level = int(_cfg_get(cfg, "collaborative_level", 1))
    level_agents = list(sc_agent_list[level])
    if not level_agents:
        logger.warning("No agents found at collaborative level %s", level)
        return []

    epochs = int(cfg["sim"]["epochs"])
    train_size = int(simulation.train_size)
    val_size = int(simulation.val_size)
    horizon = int(_cfg_get(cfg, "horizon", 1))
    scales = tuple(int(s) for s in _cfg_get(cfg, "scales", [1, 2, 4]))
    d_model = int(_cfg_get(cfg, "d_model", 32))
    ff_dim = int(_cfg_get(cfg, "ff_dim", 64))
    server_layers = int(_cfg_get(cfg, "server_layers", 1))
    dropout = float(_cfg_get(cfg, "dropout", 0.1))
    decomp_kernel = int(_cfg_get(cfg, "decomp_kernel", 3))
    learning_rate = float(_cfg_get(cfg, "learning_rate", 1e-3))
    weight_decay = float(_cfg_get(cfg, "weight_decay", 1e-4))
    grad_clip = float(_cfg_get(cfg, "grad_clip", 1.0))
    patience = int(_cfg_get(cfg, "patience", 100))
    min_delta = float(_cfg_get(cfg, "min_delta", 0.0))
    num_workers = int(_cfg_get(cfg, "num_workers", 0))
    loss_cal = str(_cfg_get(cfg, "loss_cal", "aggregated"))
    common_batch_size = int(
        _cfg_get(cfg, "batch_size", min(int(a.batch_size) for a in level_agents))
    )
    loss_fn = nn.L1Loss()

    # The server uses the largest level-1 sequence length so it can accept a
    # dynamic set of clients with different local sequence lengths.
    max_sequence_length = max(int(a.sequence_length) for a in level_agents)
    server_scale_lengths = tuple((max_sequence_length + s - 1) // s for s in scales)

    server_model = OptionDTimeMixerServer(
        scale_lengths=server_scale_lengths,
        d_model=d_model,
        ff_dim=ff_dim,
        n_layers=server_layers,
        dropout=dropout,
        decomp_kernel=decomp_kernel,
        horizon=horizon,
        output_dim=1,
    ).to(device)

    agent_client_models: List[nn.ModuleList] = []
    client_optimizers: List[torch.optim.Optimizer] = []
    for agent_idx, agent in enumerate(level_agents):
        client_models = nn.ModuleList()
        for _ in range(agent.num_retailer):
            model = TimeMixerEmbeddingClient(
                input_dim=1,
                sequence_length=int(agent.sequence_length),
                scales=scales,
                d_model=d_model,
                ff_dim=ff_dim,
                dropout=dropout,
                horizon=horizon,
                output_dim=1,
            ).to(device)
            client_models.append(model)
            client_optimizers.append(
                torch.optim.AdamW(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            )
        agent_client_models.append(client_models)
        logger.info("Created %d embedding clients for agent %d", agent.num_retailer, agent_idx)

    server_optimizer = torch.optim.AdamW(
        server_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    all_optimizers = client_optimizers + [server_optimizer]

    server_params = _count_trainable_params(server_model)
    client_params = sum(
        _count_trainable_params(model)
        for models in agent_client_models
        for model in models
    )
    logger.info("Option-D server params: %d", server_params)
    logger.info("Option-D client front+tail params total: %d", client_params)
    logger.info("Option-D total trainable params: %d", server_params + client_params)

    # ------------------------------------------------------------------
    # Build data loaders dynamically from all current clients/channels.
    # ------------------------------------------------------------------
    trainloaders: List[DataLoader] = []
    val_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
    channel_specs: List[ChannelSpec] = []
    scalers: List[StandardScaler] = []

    loader_idx = 0
    for agent_idx, agent in enumerate(level_agents):
        raw_demand_data = agent.demand_by_retailer_history
        data = [np.array(raw_demand_data[r]) for r in range(agent.num_retailer)]
        df = pd.DataFrame(data).transpose()

        df_train = df.iloc[-(train_size + val_size) : -val_size, :]
        df_val = df.iloc[-val_size:, :]

        scaler = StandardScaler().fit(df_train.to_numpy())
        scalers.append(scaler)
        train_scaled = scaler.transform(df_train.to_numpy())
        val_scaled = scaler.transform(df_val.to_numpy())

        for retailer_idx in range(agent.num_retailer):
            train_series = train_scaled[:, retailer_idx].reshape(train_size, 1)
            val_series = val_scaled[:, retailer_idx].reshape(val_size, 1)
            x_train, y_train = create_dataset(train_series, lookback=agent.sequence_length)
            x_val, y_val = create_dataset(val_series, lookback=agent.sequence_length)

            trainloaders.append(
                DataLoader(
                    TensorDataset(x_train, y_train),
                    batch_size=common_batch_size,
                    shuffle=False,
                    drop_last=True,
                    num_workers=num_workers,
                )
            )
            val_data.append((x_val, y_val))
            channel_specs.append(ChannelSpec(agent_idx, retailer_idx, loader_idx))
            loader_idx += 1

    if not trainloaders:
        logger.warning("No trainloaders created; aborting Option-D TimeMixer training")
        return []

    logger.info(
        "Prepared %d dynamic channels from %d agents; batch_size=%d; scales=%s; d_model=%d",
        len(channel_specs), len(level_agents), common_batch_size, scales, d_model,
    )

    early_stopping = SplitEarlyStopping(patience=patience, min_delta=min_delta)
    val_loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Training loop with explicit split-gradient handoff.
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        _set_train([server_model], True)
        for models in agent_client_models:
            _set_train(models, True)

        batch_losses: List[float] = []
        batch_losses_per_group: List[np.ndarray] = []

        for batches in zip(*trainloaders):
            _zero_all(all_optimizers)

            features_by_channel: List[torch.Tensor] = []
            targets_by_channel: List[torch.Tensor] = []
            for batch in batches:
                features_by_channel.append(batch[0].to(device).float())
                targets_by_channel.append(
                    _last_horizon_target(batch[1].to(device).float(), horizon)
                )

            # Client front: local inputs -> embeddings sent to the server.
            embedded_original_by_channel: List[List[torch.Tensor]] = []
            embedded_to_server_by_channel: List[List[torch.Tensor]] = []
            for channel_idx, spec in enumerate(channel_specs):
                client = agent_client_models[spec.agent_idx][spec.retailer_idx]
                embedded_original = client.encode(features_by_channel[channel_idx])
                embedded_to_server = [
                    tensor.detach().requires_grad_(True)
                    for tensor in embedded_original
                ]
                embedded_original_by_channel.append(embedded_original)
                embedded_to_server_by_channel.append(embedded_to_server)

            # Server middle: aggregate -> redistribute -> TimeMixer feature mixing.
            # The server returns hidden states, not forecasts.
            server_original_by_channel, _aux = server_model(embedded_to_server_by_channel)

            # Server -> client boundary: returned hidden states are detached before
            # the client tail receives them. Their gradients are later sent back
            # manually to the server, matching the U-shaped split.
            server_to_client_by_channel: List[List[torch.Tensor]] = []
            for states in server_original_by_channel:
                server_to_client_by_channel.append(
                    [state.detach().requires_grad_(True) for state in states]
                )

            # Client tail: local future heads compute final forecasts. In a real
            # distributed deployment, targets stay here and loss is local.
            outputs_by_channel: List[torch.Tensor] = []
            for channel_idx, spec in enumerate(channel_specs):
                client = agent_client_models[spec.agent_idx][spec.retailer_idx]
                outputs_by_channel.append(client.decode(server_to_client_by_channel[channel_idx]))

            # Loss: individual per channel or aggregated per agent. This code is
            # still a single-process simulation, but architecturally this is now
            # client-side label loss after the server has returned hidden states.
            losses: List[torch.Tensor] = []
            if loss_cal == "individual":
                losses = [
                    loss_fn(outputs_by_channel[i], targets_by_channel[i])
                    for i in range(len(outputs_by_channel))
                ]
            elif loss_cal == "aggregated":
                for agent_idx, _agent in enumerate(level_agents):
                    idxs = [i for i, spec in enumerate(channel_specs) if spec.agent_idx == agent_idx]
                    outputs = torch.cat([outputs_by_channel[i] for i in idxs], dim=2)
                    targets = torch.cat([targets_by_channel[i] for i in idxs], dim=2)
                    losses.append(loss_fn(outputs, targets))
            else:
                raise ValueError("loss_cal must be 'individual' or 'aggregated'")

            total_loss = torch.stack(losses).sum()

            # Backward phase for true U-shaped split:
            # 1) client tail computes gradients wrt server-returned states;
            # 2) those gradients are sent to the server and backpropagated there;
            # 3) server gradients wrt client embeddings are sent back to client front.
            total_loss.backward(retain_graph=True)

            server_return_pairs: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
            for original_states, detached_states in zip(
                server_original_by_channel, server_to_client_by_channel
            ):
                for original, detached in zip(original_states, detached_states):
                    server_return_pairs.append((original, detached.grad))

            for pair_idx, (original, grad) in enumerate(server_return_pairs):
                if grad is not None:
                    original.backward(grad, retain_graph=True)

            embedding_pairs: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
            for original_scales, detached_scales in zip(
                embedded_original_by_channel, embedded_to_server_by_channel
            ):
                for original, detached in zip(original_scales, detached_scales):
                    embedding_pairs.append((original, detached.grad))

            for pair_idx, (original, grad) in enumerate(embedding_pairs):
                if grad is not None:
                    original.backward(grad, retain_graph=(pair_idx < len(embedding_pairs) - 1))

            trainable_modules: List[nn.Module] = [server_model]
            for models in agent_client_models:
                trainable_modules.extend(list(models))
            nn.utils.clip_grad_norm_(
                [p for m in trainable_modules for p in m.parameters() if p.grad is not None],
                max_norm=grad_clip,
            )
            _step_all(all_optimizers)

            loss_items = np.array([float(l.detach().cpu().item()) for l in losses])
            batch_losses.append(float(total_loss.detach().cpu().item()))
            batch_losses_per_group.append(loss_items)

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        if batch_losses_per_group:
            train_loss_per_group = np.mean(np.stack(batch_losses_per_group), axis=0).tolist()
        else:
            train_loss_per_group = []

        # --------------------------------------------------------------
        # Validation on original scale.
        # --------------------------------------------------------------
        _set_train([server_model], False)
        for models in agent_client_models:
            _set_train(models, False)

        with torch.no_grad():
            embedded_by_channel: List[List[torch.Tensor]] = []
            targets_by_channel: List[torch.Tensor] = []
            for channel_idx, spec in enumerate(channel_specs):
                x_val, y_val = val_data[channel_idx]
                x_val = x_val.to(device).float()
                y_val = _last_horizon_target(y_val.to(device).float(), horizon)
                client = agent_client_models[spec.agent_idx][spec.retailer_idx]
                embedded_by_channel.append(client.encode(x_val))
                targets_by_channel.append(y_val)

            server_states_by_channel, _aux = server_model(embedded_by_channel)
            outputs_by_channel: List[torch.Tensor] = []
            for channel_idx, spec in enumerate(channel_specs):
                client = agent_client_models[spec.agent_idx][spec.retailer_idx]
                outputs_by_channel.append(client.decode(server_states_by_channel[channel_idx]))

            val_losses_per_agent: List[float] = []
            for agent_idx, _agent in enumerate(level_agents):
                idxs = [i for i, spec in enumerate(channel_specs) if spec.agent_idx == agent_idx]
                outputs_rescaled = []
                targets_rescaled = []
                for i in idxs:
                    spec = channel_specs[i]
                    scaler = scalers[spec.agent_idx]
                    outputs_rescaled.append(
                        _inverse_scale_tensor(outputs_by_channel[i], scaler, spec.retailer_idx)
                    )
                    targets_rescaled.append(
                        _inverse_scale_tensor(targets_by_channel[i], scaler, spec.retailer_idx)
                    )
                outputs_agent = torch.cat(outputs_rescaled, dim=2)
                targets_agent = torch.cat(targets_rescaled, dim=2)
                val_losses_per_agent.append(float(loss_fn(outputs_agent, targets_agent).cpu().item()))

            val_loss_sum = float(np.sum(val_losses_per_agent))
            val_loss_history.append(val_loss_sum)

        logger.info(
            "Epoch %03d | train_loss=%.6f | train_per_group=%s | val_sum=%.6f | val_per_agent=%s",
            epoch,
            train_loss,
            np.round(train_loss_per_group, 6).tolist() if train_loss_per_group else [],
            val_loss_sum,
            np.round(val_losses_per_agent, 6).tolist(),
        )

        early_stopping(val_loss_sum, agent_client_models, server_model, scalers)
        if early_stopping.early_stop:
            logger.info(
                "Early stopping at epoch %d. Best validation loss: %.6f",
                epoch,
                early_stopping.best_loss,
            )
            break

    # ------------------------------------------------------------------
    # Restore best model and attach forecasting objects to agents.
    # ------------------------------------------------------------------
    if early_stopping.best_state is not None:
        server_model.load_state_dict(early_stopping.best_state["server"])
        for agent_idx, state in enumerate(early_stopping.best_state["clients"]):
            agent_client_models[agent_idx].load_state_dict(state)
        scalers = early_stopping.best_state["scalers"]

    for agent_idx, agent in enumerate(level_agents):
        fm = OptionDSplitTimeMixerForecastingModel(
            client_models=agent_client_models[agent_idx],
            server_model=server_model,
            scaler=scalers[agent_idx],
            device=device,
            scales=scales,
            scale_lengths=server_scale_lengths,
            horizon=horizon,
        )
        agent.set_forecasting_model(fm)

    return val_loss_history
