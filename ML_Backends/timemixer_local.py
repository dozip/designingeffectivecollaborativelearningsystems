"""Local independent TimeMixer backend.

This backend is the local/non-collaborative counterpart to the split TimeMixer
backends. It trains TimeMixer models locally for each agent, without a server,
without cross-agent aggregation, and without shared weights between agents.

Default behavior:
    - train agents at cfg['timemixer']['local_level'], default 1
    - one independent TimeMixer per retailer/channel of that agent
    - one StandardScaler per agent, fitted on that agent's local retailer matrix
    - attach a forecasting object with predict(data), so normal simulation can use
      agent.act(...) -> forecasting_model.predict(data)

To train every level instead of only level 1:
    timemixer:
      local_level: all
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    """Resize a [B, L, D] tensor to target_len along the time dimension."""
    if x.size(1) == target_len:
        return x
    return F.interpolate(
        x.transpose(1, 2), size=target_len, mode="linear", align_corners=False
    ).transpose(1, 2)


class MovingAverageDecomposition(nn.Module):
    """TimeMixer-style moving-average decomposition.

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
    """MLP mixer block over time tokens and hidden channels."""

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

        # Token/time mixing.
        residual = x
        t = self.norm_time(x).transpose(1, 2)
        t = self.token_mixer(t).transpose(1, 2)
        x = residual + t

        # Channel/feature mixing.
        x = x + self.channel_mixer(self.norm_channel(x))
        return x


# ---------------------------------------------------------------------------
# Local TimeMixer model for one retailer/channel
# ---------------------------------------------------------------------------


class LocalTimeMixer(nn.Module):
    """A compact, fully local TimeMixer-style forecaster for one univariate channel.

    Input:
        x: [B, sequence_length, 1]

    Output:
        forecast: [B, horizon, 1]

    The model performs:
        multiscale downsampling -> embedding -> decomposition -> seasonal/trend
        mixing -> per-scale forecast heads -> learned scale ensemble.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        sequence_length: int = 24,
        horizon: int = 1,
        scales: Sequence[int] = (1, 2, 4),
        d_model: int = 32,
        ff_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
        decomp_kernel: int = 3,
    ) -> None:
        super().__init__()
        if not scales:
            raise ValueError("scales must contain at least one downsampling factor")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.sequence_length = int(sequence_length)
        self.horizon = int(horizon)
        self.scales = tuple(int(s) for s in scales)
        self.scale_lengths = tuple((self.sequence_length + s - 1) // s for s in self.scales)
        self.num_scales = len(self.scales)
        self.d_model = int(d_model)

        self.embeddings = nn.ModuleList(
            [nn.Linear(self.input_dim, self.d_model) for _ in self.scales]
        )
        self.embedding_norms = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in self.scales]
        )
        self.decomposition = MovingAverageDecomposition(decomp_kernel)

        self.seasonal_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        MixerBlock(self.d_model, self.scale_lengths[i], ff_dim, dropout)
                        for _ in range(n_layers)
                    ]
                )
                for i in range(self.num_scales)
            ]
        )
        self.trend_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        MixerBlock(self.d_model, self.scale_lengths[i], ff_dim, dropout)
                        for _ in range(n_layers)
                    ]
                )
                for i in range(self.num_scales)
            ]
        )

        self.seasonal_down_projections = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(self.d_model, self.d_model) for _ in range(1, self.num_scales)]
        )
        self.trend_up_projections = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model) for _ in range(self.num_scales)]
        )

        self.future_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(2 * self.d_model),
                    nn.Linear(2 * self.d_model, ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, self.horizon * self.output_dim),
                )
                for _ in self.scales
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B, L, C], got {tuple(x.shape)}")
        if x.size(1) != self.sequence_length:
            raise ValueError(
                f"expected sequence_length={self.sequence_length}, got {x.size(1)}"
            )

        multiscale = self._make_multiscale(x)

        seasonal_raw: List[torch.Tensor] = []
        trend_raw: List[torch.Tensor] = []
        for scale_id, x_s in enumerate(multiscale):
            h = self.embedding_norms[scale_id](self.embeddings[scale_id](x_s))
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

        # Future multipredictor mixing: predict from every scale, then ensemble.
        preds: List[torch.Tensor] = []
        for scale_id in range(self.num_scales):
            trend_state = trend_mixed[scale_id]
            assert trend_state is not None
            seasonal_state = seasonal_mixed[scale_id]
            if trend_state.size(1) != seasonal_state.size(1):
                trend_state = _resize_sequence(trend_state, seasonal_state.size(1))
            fused = torch.cat([seasonal_state, trend_state], dim=-1)
            pooled = fused[:, -1, :]
            pred = self.future_heads[scale_id](pooled).view(
                pooled.size(0), self.horizon, self.output_dim
            )
            preds.append(pred)

        weights = torch.softmax(self.scale_logits, dim=0)
        forecast = torch.zeros_like(preds[0])
        for weight, pred in zip(weights, preds):
            forecast = forecast + weight * pred
        return forecast


# ---------------------------------------------------------------------------
# Forecasting wrapper attached to each agent
# ---------------------------------------------------------------------------


@dataclass
class LocalTimeMixerForecastingModel:
    """Forecasting object compatible with Agent._replenish_inv().

    Agent._replenish_inv() calls:
        d_est = self.forecasting_model.predict(data)

    Here, data is a list with one latest demand sequence per retailer.
    """

    models: nn.ModuleList
    scaler: StandardScaler
    device: torch.device
    horizon: int = 1

    def predict(self, data: Sequence[np.ndarray]) -> List[float]:
        if len(data) != len(self.models):
            raise ValueError(
                f"expected {len(self.models)} retailer series, got {len(data)}"
            )

        df = pd.DataFrame([np.asarray(series) for series in data]).transpose()
        scaled = self.scaler.transform(df.to_numpy())

        predictions: List[float] = []
        with torch.no_grad():
            for retailer_idx, model in enumerate(self.models):
                model = model.to(self.device)
                model.eval()
                seq_len = int(model.sequence_length)
                x = scaled[-seq_len:, retailer_idx].reshape(1, seq_len, 1)
                input_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
                pred_scaled = model(input_tensor)[:, -1, 0].item()
                pred = pred_scaled * self.scaler.scale_[retailer_idx] + self.scaler.mean_[retailer_idx]
                predictions.append(float(pred))
        return predictions


class LocalEarlyStopping:
    """Early stopping for one independently trained agent."""

    def __init__(self, patience: int = 100, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_state: Optional[Dict[str, Any]] = None

    def __call__(self, val_loss: float, models: nn.ModuleList, scaler: StandardScaler) -> None:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = float(val_loss)
            self.counter = 0
            self.best_state = {
                "models": copy.deepcopy(models.state_dict()),
                "scaler": scaler,
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@register_backend("local_timemixer")
class LocalTimeMixerBackend(ForecastingBackend):
    """Train TimeMixer models locally and independently for each selected agent."""

    name = "local_timemixer"

    @property
    def collaborative_level(self) -> Optional[int]:
        # Non-collaborative backend: normal simulation calls agent.act(...),
        # and agent.act(...) calls forecasting_model.predict(data).
        return None

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[List[float]]:
        return _train_local_timemixers(simulation, market, supply_chain, sc_agent_list, self.cfg)


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
        raise ValueError(f"expected target tensor with 2 or 3 dims, got {tuple(y.shape)}")
    return y[:, -horizon:, :]


def _inverse_scale_tensor(y_scaled: torch.Tensor, scaler: StandardScaler, retailer_idx: int) -> torch.Tensor:
    return y_scaled * float(scaler.scale_[retailer_idx]) + float(scaler.mean_[retailer_idx])


def _set_train(models: Iterable[nn.Module], mode: bool) -> None:
    for model in models:
        model.train(mode)


def _zero_all(optimizers: Iterable[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)


def _step_all(optimizers: Iterable[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.step()


def _selected_levels(sc_agent_list: list, cfg: Dict[str, Any]) -> List[int]:
    level_spec: Union[int, str] = _cfg_get(cfg, "local_level", 1)
    if isinstance(level_spec, str) and level_spec.lower() == "all":
        return list(range(len(sc_agent_list)))
    return [int(level_spec)]


def _build_agent_dataframe(agent) -> pd.DataFrame:
    raw_demand_data = agent.demand_by_retailer_history
    data = [np.asarray(raw_demand_data[r]) for r in range(agent.num_retailer)]
    return pd.DataFrame(data).transpose()


def _train_one_agent(agent, agent_label: str, simulation, cfg: Dict[str, Any], device: torch.device) -> float:
    tm_cfg = cfg.get("timemixer", {})

    epochs = int(cfg["sim"]["epochs"])
    train_size = int(simulation.train_size)
    val_size = int(simulation.val_size)
    horizon = int(tm_cfg.get("horizon", 1))
    scales = tuple(int(s) for s in tm_cfg.get("scales", [1, 2, 4]))
    d_model = int(tm_cfg.get("d_model", 32))
    ff_dim = int(tm_cfg.get("ff_dim", 64))
    n_layers = int(tm_cfg.get("n_layers", tm_cfg.get("server_layers", 1)))
    dropout = float(tm_cfg.get("dropout", 0.1))
    decomp_kernel = int(tm_cfg.get("decomp_kernel", 3))
    learning_rate = float(tm_cfg.get("learning_rate", 1e-3))
    weight_decay = float(tm_cfg.get("weight_decay", 1e-4))
    grad_clip = float(tm_cfg.get("grad_clip", 1.0))
    patience = int(tm_cfg.get("patience", 100))
    min_delta = float(tm_cfg.get("min_delta", 0.0))
    num_workers = int(tm_cfg.get("num_workers", 0))
    loss_cal = str(tm_cfg.get("loss_cal", "aggregated"))
    batch_size = int(tm_cfg.get("batch_size", int(agent.batch_size)))
    loss_fn = nn.L1Loss()

    logger.info("Training local TimeMixer for %s with %d retailer channels", agent_label, agent.num_retailer)

    # ----------------------------
    # Build local data.
    # ----------------------------
    df = _build_agent_dataframe(agent)
    if len(df) < train_size + val_size:
        raise ValueError(
            f"{agent_label}: not enough history for train_size + val_size. "
            f"have {len(df)}, need {train_size + val_size}"
        )

    df_train = df.iloc[-(train_size + val_size):-val_size, :] if val_size > 0 else df.iloc[-train_size:, :]
    df_val = df.iloc[-val_size:, :] if val_size > 0 else df.iloc[-train_size:, :]

    scaler = StandardScaler().fit(df_train.to_numpy())
    train_scaled = scaler.transform(df_train.to_numpy())
    val_scaled = scaler.transform(df_val.to_numpy())

    models = nn.ModuleList()
    optimizers: List[torch.optim.Optimizer] = []
    trainloaders: List[DataLoader] = []
    val_data: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for retailer_idx in range(agent.num_retailer):
        model = LocalTimeMixer(
            input_dim=1,
            output_dim=1,
            sequence_length=int(agent.sequence_length),
            horizon=horizon,
            scales=scales,
            d_model=d_model,
            ff_dim=ff_dim,
            n_layers=n_layers,
            dropout=dropout,
            decomp_kernel=decomp_kernel,
        ).to(device)
        models.append(model)
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay))

        train_series = train_scaled[:, retailer_idx].reshape(train_size, 1)
        val_series = val_scaled[:, retailer_idx].reshape(len(val_scaled), 1)

        x_train, y_train = create_dataset(train_series, lookback=agent.sequence_length)
        x_val, y_val = create_dataset(val_series, lookback=agent.sequence_length)

        trainloaders.append(
            DataLoader(
                TensorDataset(x_train, y_train),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )
        )
        val_data.append((x_val, y_val))

    early_stopping = LocalEarlyStopping(patience=patience, min_delta=min_delta)
    val_history: List[float] = []

    # ----------------------------
    # Train this agent only.
    # ----------------------------
    for epoch in range(epochs):
        _set_train(models, True)
        batch_losses: List[float] = []

        for batches in zip(*trainloaders):
            _zero_all(optimizers)

            features: List[torch.Tensor] = []
            targets: List[torch.Tensor] = []
            for batch in batches:
                features.append(batch[0].to(device).float())
                targets.append(_last_horizon_target(batch[1].to(device).float(), horizon))

            outputs = [models[r](features[r]) for r in range(agent.num_retailer)]

            if loss_cal == "individual":
                losses = [loss_fn(outputs[r], targets[r]) for r in range(agent.num_retailer)]
                loss = torch.stack(losses).sum()
            elif loss_cal == "aggregated":
                outputs_agent = torch.cat(outputs, dim=2)  # [B, horizon, num_retailer]
                targets_agent = torch.cat(targets, dim=2)  # [B, horizon, num_retailer]
                loss = loss_fn(outputs_agent, targets_agent)
            else:
                raise ValueError("timemixer.loss_cal must be 'individual' or 'aggregated'")

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for model in models for p in model.parameters() if p.grad is not None],
                max_norm=grad_clip,
            )
            _step_all(optimizers)
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

        # ----------------------------
        # Validate on original scale.
        # ----------------------------
        _set_train(models, False)
        with torch.no_grad():
            outputs_rescaled: List[torch.Tensor] = []
            targets_rescaled: List[torch.Tensor] = []
            for retailer_idx in range(agent.num_retailer):
                x_val, y_val = val_data[retailer_idx]
                x_val = x_val.to(device).float()
                y_val = _last_horizon_target(y_val.to(device).float(), horizon)
                pred = models[retailer_idx](x_val)
                outputs_rescaled.append(_inverse_scale_tensor(pred, scaler, retailer_idx))
                targets_rescaled.append(_inverse_scale_tensor(y_val, scaler, retailer_idx))

            outputs_agent = torch.cat(outputs_rescaled, dim=2)
            targets_agent = torch.cat(targets_rescaled, dim=2)
            val_loss = float(loss_fn(outputs_agent, targets_agent).detach().cpu().item())
            val_history.append(val_loss)

        logger.info(
            "%s | Epoch %03d | train_loss=%.6f | val_loss=%.6f",
            agent_label,
            epoch,
            train_loss,
            val_loss,
        )

        early_stopping(val_loss, models, scaler)
        if early_stopping.early_stop:
            logger.info(
                "%s | Early stopping at epoch %d. Best validation loss: %.6f",
                agent_label,
                epoch,
                early_stopping.best_loss,
            )
            break

    # Restore best local models for this agent.
    if early_stopping.best_state is not None:
        models.load_state_dict(early_stopping.best_state["models"])
        scaler = early_stopping.best_state["scaler"]

    agent.set_forecasting_model(
        LocalTimeMixerForecastingModel(
            models=models,
            scaler=scaler,
            device=device,
            horizon=horizon,
        )
    )

    return float(early_stopping.best_loss if early_stopping.best_state is not None else val_history[-1])


def _train_local_timemixers(simulation, market, supply_chain, sc_agent_list, cfg: Dict[str, Any]) -> List[float]:
    logger.info("Starting independent local TimeMixer training")
    device = select_gpu()

    level_ids = _selected_levels(sc_agent_list, cfg)
    logger.info("Training local TimeMixers for levels: %s", level_ids)

    val_losses: List[float] = []
    for level in level_ids:
        for agent_idx, agent in enumerate(sc_agent_list[level]):
            agent_label = f"level_{level}/agent_{agent_idx}"
            best_val = _train_one_agent(agent, agent_label, simulation, cfg, device)
            val_losses.append(best_val)

    logger.info("Finished local TimeMixer training. Validation losses: %s", val_losses)
    return val_losses
