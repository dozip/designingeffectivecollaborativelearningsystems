"""U-shaped split-learning PatchTST backend with dynamic clients.

Split implemented here (analogous to the existing TimeMixer split backend,
with PatchTST as the per-client model internals):

    S_0  Client front   : RevIN normalize (optional) + replicate end-padding
                          + patch unfold + linear value embedding + learned
                          positional encoding
    Client -> Server     : token sequences [B, N, d_model], not raw demand
    S_1  Server middle   : cross-channel gated aggregation + redistribution +
                          shared TransformerEncoder backbone (the shared
                          backbone is the formal collaboration point — every
                          client's gradients train the same parameters)
    Server -> Client     : personalized hidden states [B, N, d_model] per client
    S_2  Client tail     : LayerNorm + flatten + linear forecast head +
                          RevIN denormalize (optional)

This is a true U-shaped split for horizontal collaboration across retailers:
the forward pass starts on the client, continues on one shared server, and
returns to the client for the prediction head. The shared server weights are
trained by every client's data through the explicit detach + manual
gradient handoff in the training loop.

Model internals follow yuqinie98/PatchTST (RevIN, end-padding, unfold,
learned positional encoding, standard transformer encoder). The cross-channel
aggregation/redistribution layers are kept from the TimeMixer-split design
because they are architecture-agnostic mechanisms for horizontal
collaboration over a dynamic set of clients.
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
from helpers.helper_classes import EarlyStopping
from .base import ForecastingBackend
from . import register_backend
from .patchtst_local import RevIN, _compute_num_patches

logger = logging.getLogger("logger")


# ---------------------------------------------------------------------------
# Client front + tail (S_0 + S_2)
# ---------------------------------------------------------------------------


class PatchTSTSplitClient(nn.Module):
    """Per-client (per-retailer-channel) module owning the client front and tail.

    encode(...) is S_0; decode(...) is S_2. The server S_1 sits between.
    """

    def __init__(
        self,
        input_dim: int = 1,
        sequence_length: int = 24,
        patch_len: int = 2,
        stride: int = 1,
        padding_patch: str = "end",
        d_model: int = 32,
        ff_dim: int = 64,
        dropout: float = 0.1,
        horizon: int = 1,
        output_dim: int = 1,
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
                "PatchTSTSplitClient: num_patches=%d (sequence_length=%d, patch_len=%d, stride=%d). "
                "Transformer collapses to a single/very few tokens.",
                self.num_patches,
                self.sequence_length,
                self.patch_len,
                self.stride,
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

        # S_0 front: patch embedding.
        self.W_P = nn.Linear(self.patch_len, self.d_model)
        self.W_pos = nn.Parameter(torch.zeros(self.num_patches, self.d_model))
        self.dropout = nn.Dropout(float(dropout))

        # S_2 tail: flatten head.
        self.head_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.num_patches * self.d_model, self.horizon * self.output_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        if self.pre_pad > 0:
            z = F.pad(z, (0, self.pre_pad), mode="replicate")
        if self.padding_patch_layer is not None:
            z = self.padding_patch_layer(z)
        patches = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return patches.squeeze(1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """S_0: local sequence -> token sequence [B, N, d_model] sent to server."""
        if self.revin_enabled:
            x = self.revin_layer(x, "norm")
        patches = self._patchify(x)
        u = self.W_P(patches) + self.W_pos.unsqueeze(0)
        return self.dropout(u)

    def decode(self, server_state: torch.Tensor) -> torch.Tensor:
        """S_2: server-returned hidden state -> final local forecast [B, horizon, output_dim]."""
        z = self.head_norm(server_state)
        flat = z.reshape(z.size(0), -1)
        out = self.head(flat).view(z.size(0), self.horizon, self.output_dim)
        if self.revin_enabled:
            out = self.revin_layer(out, "denorm")
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compatibility forward; the U-shaped path explicitly calls encode/decode.
        return self.encode(x)


# ---------------------------------------------------------------------------
# Server (S_1): shared backbone + cross-channel aggregation
# ---------------------------------------------------------------------------


class PatchTSTSharedServer(nn.Module):
    """Shared server-side backbone for U-shaped split learning.

    Receives client token sequences, aggregates dynamically across channels,
    injects global context back into each channel's tokens, and applies a
    shared TransformerEncoder. Returns one personalized hidden state per
    channel back to the matching client tail.

    The TransformerEncoder weights are the formal S_1 collaboration mechanism:
    every client's gradients train the same encoder parameters.
    """

    def __init__(
        self,
        num_patches: int,
        d_model: int = 32,
        n_heads: int = 4,
        ff_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.d_model = int(d_model)

        # Dynamic cross-channel aggregation: a gate over channels per token position.
        self.client_gate = nn.Linear(d_model, 1)

        # Redistribution of the global state back into each channel's local tokens.
        self.entry_distributor = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        # Shared TransformerEncoder backbone (the S_1 collaboration point).
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

    def _aggregate(self, embedded_by_channel: Sequence[torch.Tensor]) -> torch.Tensor:
        """Gated softmax across the N channels at each token position.

        Returns global state [B, num_patches, d_model].
        """
        if not embedded_by_channel:
            raise ValueError("embedded_by_channel must not be empty")
        stacked = torch.stack(list(embedded_by_channel), dim=1)  # [B, N_ch, num_patches, d_model]
        scores = self.client_gate(stacked).squeeze(-1)  # [B, N_ch, num_patches]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, N_ch, num_patches, 1]
        return (weights * stacked).sum(dim=1)  # [B, num_patches, d_model]

    def _redistribute(
        self,
        embedded_by_channel: Sequence[torch.Tensor],
        global_state: torch.Tensor,
    ) -> List[torch.Tensor]:
        personalized: List[torch.Tensor] = []
        for local in embedded_by_channel:
            fused = self.entry_distributor(torch.cat([local, global_state], dim=-1))
            personalized.append(local + fused)
        return personalized

    def forward(
        self, embedded_by_channel: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Apply aggregation/redistribution then the shared encoder per channel.

        Args:
            embedded_by_channel: list with length N_channels. Each element is a
                tensor [B, num_patches, d_model].

        Returns:
            states_by_channel: list with length N_channels. Each element is a
                tensor [B, num_patches, d_model] to be sent back to the matching
                client tail.
        """
        global_state = self._aggregate(embedded_by_channel)
        personalized = self._redistribute(embedded_by_channel, global_state)
        states_by_channel = [self.encoder(state) for state in personalized]
        aux = {"num_channels": len(embedded_by_channel)}
        return states_by_channel, aux


# ---------------------------------------------------------------------------
# Forecasting wrapper saved on each supply-chain agent
# ---------------------------------------------------------------------------


@dataclass
class SplitPatchTSTForecastingModel:
    client_models: nn.ModuleList
    server_model: PatchTSTSharedServer
    scaler: StandardScaler
    device: torch.device
    num_patches: int
    horizon: int = 1


@dataclass(frozen=True)
class ChannelSpec:
    agent_idx: int
    retailer_idx: int
    loader_idx: int




# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@register_backend("split_patchtst")
class PatchTSTSplitBackend(ForecastingBackend):
    """U-shaped split-learning PatchTST backend.

    Client front (S_0): RevIN (optional), patching, value+positional embedding.
    Server middle (S_1): cross-channel aggregation, redistribution, shared
        TransformerEncoder backbone (the formal collaboration point).
    Client tail (S_2): LayerNorm + flatten + linear head, RevIN denormalize.
    """

    name = "split_patchtst"

    @property
    def collaborative_level(self) -> Optional[int]:
        return int(self.cfg.get("patchtst", {}).get("collaborative_level", 1))

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[List[float]]:
        return _train_split_patchtst(simulation, market, supply_chain, sc_agent_list, self.cfg)

    def collaborative_predict(self, level_agents, demand_t, sc_agent_list, t) -> List[List[float]]:
        if not level_agents:
            return []

        first_model: SplitPatchTSTForecastingModel = level_agents[0].get_forecasting_model()
        device = first_model.device
        server = first_model.server_model.to(device)
        server.eval()

        embedded_by_channel: List[torch.Tensor] = []
        channel_to_agent_retailer: List[Tuple[int, int]] = []

        for agent_idx, agent in enumerate(level_agents):
            fm: SplitPatchTSTForecastingModel = agent.get_forecasting_model()
            raw_demand_data = agent.demand_by_retailer_history
            data = [np.array(raw_demand_data[r]) for r in range(agent.num_retailer)]
            df = pd.DataFrame(data).transpose()
            df = df.iloc[-(agent.sequence_length - 1):, :]

            df.loc[len(df)] = demand_t[:, agent_idx]
            scaled = fm.scaler.transform(df.to_numpy())

            for retailer_idx in range(agent.num_retailer):
                x = scaled[-agent.sequence_length:, retailer_idx].reshape(
                    1, agent.sequence_length, 1
                )
                input_tensor = torch.tensor(x, dtype=torch.float32, device=device)
                client_model = fm.client_models[retailer_idx].to(device)
                client_model.eval()
                embedded = client_model.encode(input_tensor)
                embedded_by_channel.append(embedded)
                channel_to_agent_retailer.append((agent_idx, retailer_idx))

        with torch.no_grad():
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
            predictions.append(flat_predictions[cursor: cursor + n])
            cursor += n
        return predictions


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


def _train_split_patchtst(simulation, market, supply_chain, sc_agent_list, cfg):
    logger.info("Starting U-shaped split PatchTST training")

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
    patch_len = int(_cfg_get(cfg, "patch_len", 2))
    stride = int(_cfg_get(cfg, "stride", 1))
    padding_patch = str(_cfg_get(cfg, "padding_patch", "end"))
    d_model = int(_cfg_get(cfg, "d_model", 32))
    n_heads = int(_cfg_get(cfg, "n_heads", 4))
    n_layers = int(_cfg_get(cfg, "n_layers", 2))
    server_layers = int(_cfg_get(cfg, "server_layers", n_layers))
    ff_dim = int(_cfg_get(cfg, "ff_dim", 64))
    dropout = float(_cfg_get(cfg, "dropout", 0.1))
    revin = bool(_cfg_get(cfg, "revin", False))
    revin_affine = bool(_cfg_get(cfg, "revin_affine", True))
    subtract_last = bool(_cfg_get(cfg, "subtract_last", False))
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
    use_onecycle = bool(_cfg_get(cfg, "use_onecycle", True))
    onecycle_total_epochs = int(_cfg_get(cfg, "onecycle_total_epochs", 100))
    onecycle_pct_start = float(_cfg_get(cfg, "onecycle_pct_start", 0.3))
    onecycle_div_factor = float(_cfg_get(cfg, "onecycle_div_factor", 25.0))
    onecycle_final_div_factor = float(_cfg_get(cfg, "onecycle_final_div_factor", 1e4))
    onecycle_anneal_strategy = str(_cfg_get(cfg, "onecycle_anneal_strategy", "cos"))
    loss_fn = nn.L1Loss()

    # The server uses the largest level sequence length so it accepts a dynamic
    # set of clients with possibly different local sequence lengths.
    max_sequence_length = max(int(a.sequence_length) for a in level_agents)
    server_num_patches, _ = _compute_num_patches(
        max_sequence_length, patch_len, stride, padding_patch
    )

    server_model = PatchTSTSharedServer(
        num_patches=server_num_patches,
        d_model=d_model,
        n_heads=n_heads,
        ff_dim=ff_dim,
        n_layers=server_layers,
        dropout=dropout,
    ).to(device)

    agent_client_models: List[nn.ModuleList] = []
    client_optimizers: List[torch.optim.Optimizer] = []
    for agent_idx, agent in enumerate(level_agents):
        client_models = nn.ModuleList()
        for _ in range(agent.num_retailer):
            model = PatchTSTSplitClient(
                input_dim=1,
                sequence_length=int(agent.sequence_length),
                patch_len=patch_len,
                stride=stride,
                padding_patch=padding_patch,
                d_model=d_model,
                ff_dim=ff_dim,
                dropout=dropout,
                horizon=horizon,
                output_dim=1,
                revin=revin,
                revin_affine=revin_affine,
                subtract_last=subtract_last,
            ).to(device)
            client_models.append(model)
            client_optimizers.append(
                torch.optim.AdamW(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            )
        agent_client_models.append(client_models)
        logger.info("Created %d split clients for agent %d", agent.num_retailer, agent_idx)

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
    logger.info("Split PatchTST server params: %d", server_params)
    logger.info("Split PatchTST client front+tail params total: %d", client_params)
    logger.info("Split PatchTST total trainable params: %d", server_params + client_params)

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

        df_train = df.iloc[-(train_size + val_size): -val_size, :]
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
        logger.warning("No trainloaders created; aborting split PatchTST training")
        return []

    logger.info(
        "Prepared %d dynamic channels from %d agents; batch_size=%d; patch_len=%d; stride=%d; d_model=%d",
        len(channel_specs), len(level_agents), common_batch_size, patch_len, stride, d_model,
    )

    # OneCycleLR (one per optimizer, sized over `onecycle_total_epochs`).
    schedulers: List[Optional[torch.optim.lr_scheduler.OneCycleLR]] = []
    onecycle_total_steps = 0
    if use_onecycle:
        steps_per_epoch = max(1, len(trainloaders[0]))
        onecycle_total_steps = steps_per_epoch * max(1, onecycle_total_epochs)
        for opt in all_optimizers:
            schedulers.append(
                torch.optim.lr_scheduler.OneCycleLR(
                    opt,
                    max_lr=learning_rate,
                    total_steps=onecycle_total_steps,
                    pct_start=onecycle_pct_start,
                    anneal_strategy=onecycle_anneal_strategy,
                    div_factor=onecycle_div_factor,
                    final_div_factor=onecycle_final_div_factor,
                )
            )
        logger.info(
            "Split PatchTST OneCycleLR enabled: max_lr=%g total_steps=%d "
            "pct_start=%g (warmup~%d steps / %d epochs); optimizers=%d",
            learning_rate, onecycle_total_steps, onecycle_pct_start,
            int(onecycle_total_steps * onecycle_pct_start),
            int(onecycle_total_steps * onecycle_pct_start / steps_per_epoch),
            len(all_optimizers),
        )
    else:
        schedulers = [None] * len(all_optimizers)

    global_step = 0
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        delta=min_delta,
    )
    val_loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Training loop with explicit U-shaped gradient handoff.
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

            # S_0 client front: local input -> token sequences.
            embedded_original_by_channel: List[torch.Tensor] = []
            embedded_to_server_by_channel: List[torch.Tensor] = []
            for channel_idx, spec in enumerate(channel_specs):
                client = agent_client_models[spec.agent_idx][spec.retailer_idx]
                embedded_original = client.encode(features_by_channel[channel_idx])
                embedded_to_server = embedded_original.detach().requires_grad_(True)
                embedded_original_by_channel.append(embedded_original)
                embedded_to_server_by_channel.append(embedded_to_server)

            # S_1 server: aggregate -> redistribute -> shared TransformerEncoder.
            server_original_by_channel, _aux = server_model(embedded_to_server_by_channel)

            # Server -> client boundary: detach so gradients are sent back manually.
            server_to_client_by_channel: List[torch.Tensor] = [
                state.detach().requires_grad_(True) for state in server_original_by_channel
            ]

            # S_2 client tail: local future head + RevIN denorm -> final forecast.
            outputs_by_channel: List[torch.Tensor] = []
            for channel_idx, spec in enumerate(channel_specs):
                client = agent_client_models[spec.agent_idx][spec.retailer_idx]
                outputs_by_channel.append(client.decode(server_to_client_by_channel[channel_idx]))

            # Loss: per-channel or per-agent aggregated.
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

            # U-shaped backward pass:
            # 1) client tails compute grads wrt server-returned states;
            # 2) those grads are sent to the server and backpropagated there;
            # 3) server grads wrt client embeddings are sent back to client fronts.
            total_loss.backward(retain_graph=True)

            server_return_pairs: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
            for original, detached in zip(
                server_original_by_channel, server_to_client_by_channel
            ):
                server_return_pairs.append((original, detached.grad))

            for pair_idx, (original, grad) in enumerate(server_return_pairs):
                if grad is not None:
                    original.backward(grad, retain_graph=True)

            embedding_pairs: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
            for original, detached in zip(
                embedded_original_by_channel, embedded_to_server_by_channel
            ):
                embedding_pairs.append((original, detached.grad))

            for pair_idx, (original, grad) in enumerate(embedding_pairs):
                if grad is not None:
                    original.backward(
                        grad, retain_graph=(pair_idx < len(embedding_pairs) - 1)
                    )

            trainable_modules: List[nn.Module] = [server_model]
            for models in agent_client_models:
                trainable_modules.extend(list(models))
            nn.utils.clip_grad_norm_(
                [p for m in trainable_modules for p in m.parameters() if p.grad is not None],
                max_norm=grad_clip,
            )
            _step_all(all_optimizers)
            if onecycle_total_steps > 0 and global_step < onecycle_total_steps - 1:
                for sch in schedulers:
                    if sch is not None:
                        sch.step()
            global_step += 1

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
            embedded_by_channel: List[torch.Tensor] = []
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

        early_stopping(
            val_loss_sum,
            {
                "server": copy.deepcopy(server_model.state_dict()),
                "clients": [
                    copy.deepcopy(models.state_dict())
                    for models in agent_client_models
                ],
                "scalers": copy.deepcopy(scalers),
            },
        )
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
    if early_stopping.best_model is not None:
        best_snapshot = early_stopping.best_model
        server_model.load_state_dict(best_snapshot["server"])
        for agent_idx, state in enumerate(best_snapshot["clients"]):
            agent_client_models[agent_idx].load_state_dict(state)
        scalers = best_snapshot["scalers"]

    for agent_idx, agent in enumerate(level_agents):
        fm = SplitPatchTSTForecastingModel(
            client_models=agent_client_models[agent_idx],
            server_model=server_model,
            scaler=scalers[agent_idx],
            device=device,
            num_patches=server_num_patches,
            horizon=horizon,
        )
        agent.set_forecasting_model(fm)

    return val_loss_history
