"""Local independent PatchTST backend.

This backend is the local/non-collaborative counterpart to the split PatchTST
backend. It trains PatchTST models locally for each agent, without a server,
without cross-agent aggregation, and without shared weights between agents.

Default behavior:
    - train agents at cfg['patchtst']['local_level'], default 1
    - one independent PatchTST per retailer/channel of that agent
    - one StandardScaler per agent, fitted on that agent's local retailer matrix
    - attach a forecasting object with predict(data), so normal simulation can use
      agent.act(...) -> forecasting_model.predict(data)

To train every level instead of only level 1:
    patchtst:
      local_level: all

Model internals follow the official PatchTST repo (yuqinie98/PatchTST):
    - end-padding via nn.ReplicationPad1d((0, stride))
    - unfold patching along the time dimension
    - linear value embedding + learnable positional embedding (init zeros)
    - standard TransformerEncoder (norm_first, GELU)
    - LayerNorm + flatten + linear forecasting head

Per-retailer model instantiation gives channel-independence at the
codebase level, matching the existing LSTM/TimeMixer convention.
Paper-canonical channel-independence via [B*n_vars, ...] reshape is not
used here.
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
from helpers.helper_classes import EarlyStopping
from .base import ForecastingBackend
from . import register_backend

logger = logging.getLogger("logger")


# ---------------------------------------------------------------------------
# RevIN — reversible instance normalization (ported from yuqinie98/PatchTST)
# ---------------------------------------------------------------------------


class RevIN(nn.Module):
    """Reversible instance normalization.

    Stores per-instance statistics during forward(x, 'norm') and replays them
    during forward(x, 'denorm'). When affine=True, learns per-feature
    affine_weight (init ones) and affine_bias (init zeros).
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :]
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"RevIN: unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# PatchTST model for one univariate channel
# ---------------------------------------------------------------------------


def _compute_num_patches(sequence_length: int, patch_len: int, stride: int, padding_patch: str) -> Tuple[int, int]:
    """Return (num_patches, pre_pad) for the configured patching scheme.

    pre_pad > 0 means the input must be replicate-padded to patch_len before
    end-padding (handles sequence_length < patch_len).
    """
    if sequence_length < patch_len:
        pre_pad = patch_len - sequence_length
        effective_L = patch_len
    else:
        pre_pad = 0
        effective_L = sequence_length
    if padding_patch == "end":
        padded_L = effective_L + stride
    else:
        padded_L = effective_L
    num_patches = (padded_L - patch_len) // stride + 1
    return num_patches, pre_pad


class PatchTST(nn.Module):
    """Compact PatchTST for one univariate channel.

    Input:
        x: [B, sequence_length, 1]

    Output:
        forecast: [B, horizon, 1]
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        sequence_length: int = 24,
        horizon: int = 1,
        patch_len: int = 2,
        stride: int = 1,
        padding_patch: str = "end",
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.1,
        revin: bool = False,
        revin_affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.sequence_length = int(sequence_length)
        self.horizon = int(horizon)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding_patch = str(padding_patch)
        self.d_model = int(d_model)
        self.revin_enabled = bool(revin)

        self.num_patches, self.pre_pad = _compute_num_patches(
            self.sequence_length, self.patch_len, self.stride, self.padding_patch
        )
        if self.num_patches < 2:
            logger.warning(
                "PatchTST: num_patches=%d (sequence_length=%d, patch_len=%d, stride=%d, padding_patch=%s). "
                "Transformer collapses to a single/very few tokens.",
                self.num_patches,
                self.sequence_length,
                self.patch_len,
                self.stride,
                self.padding_patch,
            )

        if self.padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        else:
            self.padding_patch_layer = None

        if self.revin_enabled:
            self.revin_layer = RevIN(
                self.input_dim,
                affine=bool(revin_affine),
                subtract_last=bool(subtract_last),
            )

        self.W_P = nn.Linear(self.patch_len, self.d_model)
        self.W_pos = nn.Parameter(torch.zeros(self.num_patches, self.d_model))
        self.dropout = nn.Dropout(float(dropout))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))

        self.head_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.num_patches * self.d_model, self.horizon * self.output_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 1] -> [B, N, patch_len]
        z = x.transpose(1, 2)  # [B, 1, L]
        if self.pre_pad > 0:
            z = F.pad(z, (0, self.pre_pad), mode="replicate")
        if self.padding_patch_layer is not None:
            z = self.padding_patch_layer(z)
        patches = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # patches: [B, 1, N, patch_len] — squeeze channel dim (input_dim=1)
        return patches.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B, L, C], got {tuple(x.shape)}")
        if x.size(1) != self.sequence_length:
            raise ValueError(
                f"expected sequence_length={self.sequence_length}, got {x.size(1)}"
            )

        if self.revin_enabled:
            x = self.revin_layer(x, "norm")

        patches = self._patchify(x)  # [B, N, patch_len]
        if patches.size(1) != self.num_patches:
            raise RuntimeError(
                f"got {patches.size(1)} patches but expected {self.num_patches}"
            )

        u = self.W_P(patches) + self.W_pos.unsqueeze(0)  # [B, N, d_model]
        u = self.dropout(u)
        z = self.encoder(u)  # [B, N, d_model]
        z = self.head_norm(z)
        flat = z.reshape(z.size(0), -1)  # [B, N*d_model]
        out = self.head(flat).view(x.size(0), self.horizon, self.output_dim)

        if self.revin_enabled:
            out = self.revin_layer(out, "denorm")

        return out


# ---------------------------------------------------------------------------
# Forecasting wrapper attached to each agent
# ---------------------------------------------------------------------------


@dataclass
class LocalPatchTSTForecastingModel:
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




# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@register_backend("local_patchtst")
class LocalPatchTSTBackend(ForecastingBackend):
    """Train PatchTST models locally and independently for each selected agent."""

    name = "local_patchtst"

    @property
    def collaborative_level(self) -> Optional[int]:
        return None

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[List[float]]:
        return _train_local_patchtsts(simulation, market, supply_chain, sc_agent_list, self.cfg)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _cfg_get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """Read settings from YAML.

    Early-stopping settings are global for all trainable backends:
        early_stopping:
          patience: 100
          min_delta: 0.0

    Other model-specific hyperparameters are still read from:
        patchtst:
          ...
    """
    if isinstance(cfg, dict) and key in {"patience", "min_delta"}:
        es_cfg = cfg.get("early_stopping", {})
        if isinstance(es_cfg, dict) and key in es_cfg:
            return es_cfg[key]

    model_cfg = cfg.get("patchtst", {}) if isinstance(cfg, dict) else {}
    if isinstance(model_cfg, dict) and key in model_cfg:
        return model_cfg[key]
    return default


def _last_horizon_target(y: torch.Tensor, horizon: int) -> torch.Tensor:
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


def _train_one_agent(agent, agent_label: str, simulation, cfg: Dict[str, Any], device: torch.device) -> List[float]:
    pt_cfg = cfg.get("patchtst", {})

    epochs = int(cfg["sim"]["epochs"])
    train_size = int(simulation.train_size)
    val_size = int(simulation.val_size)
    horizon = int(pt_cfg.get("horizon", 1))
    patch_len = int(pt_cfg.get("patch_len", 2))
    stride = int(pt_cfg.get("stride", 1))
    padding_patch = str(pt_cfg.get("padding_patch", "end"))
    d_model = int(pt_cfg.get("d_model", 32))
    n_heads = int(pt_cfg.get("n_heads", 4))
    n_layers = int(pt_cfg.get("n_layers", 2))
    ff_dim = int(pt_cfg.get("ff_dim", 64))
    dropout = float(pt_cfg.get("dropout", 0.1))
    revin = bool(pt_cfg.get("revin", False))
    revin_affine = bool(pt_cfg.get("revin_affine", True))
    subtract_last = bool(pt_cfg.get("subtract_last", False))
    learning_rate = float(pt_cfg.get("learning_rate", 1e-3))
    weight_decay = float(pt_cfg.get("weight_decay", 1e-4))
    grad_clip = float(pt_cfg.get("grad_clip", 1.0))
    patience = int(_cfg_get(cfg, "patience", 100))
    min_delta = float(_cfg_get(cfg, "min_delta", 0.0))
    num_workers = int(pt_cfg.get("num_workers", 0))
    loss_cal = str(pt_cfg.get("loss_cal", "aggregated"))
    batch_size = int(pt_cfg.get("batch_size", int(agent.batch_size)))
    loss_fn = nn.L1Loss()

    logger.info("Training local PatchTST for %s with %d retailer channels", agent_label, agent.num_retailer)

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
        model = PatchTST(
            input_dim=1,
            output_dim=1,
            sequence_length=int(agent.sequence_length),
            horizon=horizon,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            revin=revin,
            revin_affine=revin_affine,
            subtract_last=subtract_last,
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

    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        delta=min_delta,
    )
    val_history: List[float] = []

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
                outputs_agent = torch.cat(outputs, dim=2)
                targets_agent = torch.cat(targets, dim=2)
                loss = loss_fn(outputs_agent, targets_agent)
            else:
                raise ValueError("patchtst.loss_cal must be 'individual' or 'aggregated'")

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for model in models for p in model.parameters() if p.grad is not None],
                max_norm=grad_clip,
            )
            _step_all(optimizers)
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

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

        logger.debug(
            "%s | Epoch %03d | train_loss=%.6f | val_loss=%.6f",
            agent_label,
            epoch,
            train_loss,
            val_loss,
        )

        early_stopping(
            val_loss,
            {
                "models": copy.deepcopy(models.state_dict()),
                "scaler": copy.deepcopy(scaler),
            },
        )
        if early_stopping.early_stop:
            logger.debug(
                "%s | Early stopping at epoch %d. Best validation loss: %.6f",
                agent_label,
                epoch,
                early_stopping.best_loss,
            )
            break

    if early_stopping.best_model is not None:
        best_snapshot = early_stopping.best_model
        models.load_state_dict(best_snapshot["models"])
        scaler = best_snapshot["scaler"]

    agent.set_forecasting_model(
        LocalPatchTSTForecastingModel(
            models=models,
            scaler=scaler,
            device=device,
            horizon=horizon,
        )
    )

    # Pad val_history to `epochs` with the last observed value so all agents
    # contribute the same-length per-epoch series for plotting.
    while len(val_history) < epochs:
        val_history.append(val_history[-1] if val_history else float("nan"))
    return val_history


def _train_local_patchtsts(simulation, market, supply_chain, sc_agent_list, cfg: Dict[str, Any]) -> List[float]:
    logger.info("Starting independent local PatchTST training")
    device = select_gpu()

    level_ids = _selected_levels(sc_agent_list, cfg)
    logger.info("Training local PatchTSTs for levels: %s", level_ids)

    val_loss_list: List[List[float]] = []
    for level in level_ids:
        for agent_idx, agent in enumerate(sc_agent_list[level]):
            agent_label = f"level_{level}/agent_{agent_idx}"
            history = _train_one_agent(agent, agent_label, simulation, cfg, device)
            val_loss_list.append(history)
            logger.info("%s | best val=%.6f", agent_label, float(np.min(history)))

    val_loss = np.sum(np.array(val_loss_list), axis=0)  # shape (epochs,)
    logger.info(
        "Finished local PatchTST training. Per-epoch summed val loss: first=%.6f last=%.6f",
        float(val_loss[0]), float(val_loss[-1]),
    )
    return val_loss.tolist()
