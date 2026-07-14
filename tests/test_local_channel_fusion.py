"""Local-baseline symmetry tests for `local_channel_fusion`.

The three local backends must implement the SAME baseline: an agent may jointly
model every channel it observes itself, but never a peer agent's channel.

Checks:
  1. flag resolution (per-section override beats global, default True, non-bool
     fails loudly)
  2. head input dimension: fusion -> num_channels * latent_dim, no fusion ->
     latent_dim, for all three architectures
  3. all three backends train end to end on a 2x3x3 multi-product config in both
     flag settings
  4. CONFIDENTIALITY: while an agent is being trained, no local backend reads
     `demand_by_retailer_history` of any other agent — in either mode

Run directly:
    python tests/test_local_channel_fusion.py
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ML_Backends.base import resolve_local_channel_fusion  # noqa: E402
from ML_Backends.lstm_local import LSTMLocalBackend  # noqa: E402
from ML_Backends.patchtst_local import LocalPatchTSTBackend, PatchTST  # noqa: E402
from ML_Backends.timemixer_local import LocalTimeMixerBackend, LocalTimeMixer  # noqa: E402
from ML_Backends.ma import NoOpBackend  # noqa: E402
from Simulation_Component.agent import Agent  # noqa: E402
from Simulation_Component.runner import run_simulation_phase  # noqa: E402
from helpers.helpers import reset_simulaltion_from_dict  # noqa: E402

SEED = 42
R, P, M = 2, 3, 3


def _cfg(lstm_fusion=None, patchtst_fusion=None, timemixer_fusion=None, global_fusion=None):
    cfg = {
        "sim": {"simulation_runs": 1, "run_start_id": 0, "convergence_time": 30,
                "simulation_time": 120, "training_time": 60, "testing_time": 10,
                "training_type": None, "train_size": 60, "val_size": 15, "epochs": 2,
                "learning_rate": 0.001, "momentum": 0.9, "batch_size": 16,
                "sequence_length": 6},
        "early_stopping": {"patience": 100, "min_delta": 0.0},
        "market": {
            "data_scource": None, "num_products": P,
            "primary_demand": 1000, "trend_magnitude": 2, "seasonality_magnitude": 20,
            "seasonality_frequncy": 52, "random_walk": {"mean": 0, "variance": 1},
            # MarketFactorModel partitions the total market: all R*P shares sum to 1.
            "demand_split": [0.2333, 0.2333, 0.2334, 0.1, 0.1, 0.1],
            "factor_model": {"lam": 0.75, "tau": 2, "idio_noise_scale": 1.0,
                             "shock_prob": 0.0, "shock_mag": 0.0, "seed": 0},
        },
        "supply_chain": {
            "agents_per_level": [R, M], "num_products": P,
            "product_to_manufacturer": {0: 0, 1: 1, 2: 2},
            "sc_levels": {
                "sc_level_0": {"adjaceny_list": [[1, 1, 1], [1, 1, 1]], "lead_time": 1,
                               "replenishment_strat": ["OUT"] * R,
                               "forecasting_strat": ["MA"] * R,
                               "inv_capacity": [5000000] * R, "init_inv": [0] * R,
                               "safety_risk_factor": [1] * R, "R": [1] * R},
                "sc_level_1": {"adjaceny_list": [[1]] * M, "lead_time": 1,
                               "replenishment_strat": ["OUT"] * M,
                               "forecasting_strat": ["MA"] * M,
                               "inv_capacity": [5000000] * M, "init_inv": [0] * M,
                               "safety_risk_factor": [1] * M, "R": [1] * M}}},
    }
    if global_fusion is not None:
        cfg["local_channel_fusion"] = global_fusion
    for section, value in (("lstm", lstm_fusion), ("patchtst", patchtst_fusion),
                           ("timemixer", timemixer_fusion)):
        if value is not None:
            cfg.setdefault(section, {})["local_channel_fusion"] = value
    return cfg


# ---------------------------------------------------------------------------
# 1. flag resolution
# ---------------------------------------------------------------------------

def test_flag_resolution():
    assert resolve_local_channel_fusion({}, "patchtst") is True, "default must be True"
    assert resolve_local_channel_fusion({"local_channel_fusion": False}, "patchtst") is False
    # per-section override wins over the global
    cfg = {"local_channel_fusion": True, "patchtst": {"local_channel_fusion": False}}
    assert resolve_local_channel_fusion(cfg, "patchtst") is False
    assert resolve_local_channel_fusion(cfg, "timemixer") is True
    # the legacy (asymmetric) configuration is expressible
    legacy = {"local_channel_fusion": False, "lstm": {"local_channel_fusion": True}}
    assert resolve_local_channel_fusion(legacy, "lstm") is True
    assert resolve_local_channel_fusion(legacy, "patchtst") is False
    assert resolve_local_channel_fusion(legacy, "timemixer") is False
    # non-bool fails loudly
    for bad in ("true", 1, None):
        try:
            resolve_local_channel_fusion({"local_channel_fusion": bad}, "patchtst")
        except ValueError:
            pass
        else:
            raise AssertionError(f"non-bool {bad!r} must raise")


# ---------------------------------------------------------------------------
# 2. head input dims + trainable parameter counts
# ---------------------------------------------------------------------------

def _n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_counts(num_channels=2, seq_len=13):
    """Return {(arch, fusion): (per_channel_params, head_in_dim)}."""
    from ML_Models.model import LSTM_Model, NetLocal2

    out = {}
    for fusion in (False, True):
        c = num_channels if fusion else 1

        # LSTM: latent = lstm_hidden_dim (48), head = NetLocal2
        lstm = LSTM_Model(n_input=1, n_output=24, n_hidden=48)
        dense = NetLocal2(n_input=48 * c, n_output=1)
        assert dense.lin1.in_features == 48 * c
        out[("lstm", fusion)] = (_n_params(lstm) + _n_params(dense), dense.lin1.in_features)

        pt = PatchTST(sequence_length=seq_len, fusion_channels=c)
        assert pt.head.in_features == pt.latent_dim * c
        out[("patchtst", fusion)] = (_n_params(pt), pt.head.in_features)

        tm = LocalTimeMixer(sequence_length=seq_len, fusion_channels=c)
        assert tm.head_in_dim == tm.latent_dim * c
        first_linear = tm.future_heads[0][2]
        assert first_linear.in_features == tm.scale_lengths[0] * tm.latent_dim * c
        out[("timemixer", fusion)] = (_n_params(tm), tm.head_in_dim)
    return out


def test_head_input_dims():
    counts = param_counts(num_channels=2)
    for arch in ("lstm", "patchtst", "timemixer"):
        _, dim_off = counts[(arch, False)]
        _, dim_on = counts[(arch, True)]
        assert dim_on == 2 * dim_off, (
            f"{arch}: fusion head must see num_channels x latent_dim; "
            f"got {dim_on} vs {dim_off}"
        )


def test_fusion_shape_mismatch_fails_loudly():
    """A wrong-width latent must raise, not broadcast."""
    pt = PatchTST(sequence_length=13, fusion_channels=2)
    try:
        pt.decode(torch.zeros(4, pt.latent_dim))  # one channel's latent, not two
    except ValueError:
        pass
    else:
        raise AssertionError("PatchTST.decode must reject an under-wide latent")

    tm = LocalTimeMixer(sequence_length=13, fusion_channels=2)
    latents = tm.encode(torch.zeros(4, 13, 1))  # single-channel width
    try:
        tm.decode(latents)
    except ValueError:
        pass
    else:
        raise AssertionError("LocalTimeMixer.decode must reject an under-wide latent")

    # forward() on a fusion-sized model is the wrong path and must say so
    for model, x in ((pt, torch.zeros(2, 13, 1)), (tm, torch.zeros(2, 13, 1))):
        try:
            model(x)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{type(model).__name__}.forward must reject fusion_channels>1")


# ---------------------------------------------------------------------------
# 3. + 4. end-to-end runs with a confidentiality probe
# ---------------------------------------------------------------------------

class _HistoryProbe:
    """Records which agent's demand_by_retailer_history is read, and when.

    Installs a data descriptor on the Agent class, which takes precedence over
    the per-instance attribute, so every read is observed.
    """

    def __init__(self):
        self.reads = []          # uids read since the last model attachment
        self.violations = []
        self._saved_set_model = Agent.set_forecasting_model

    def __enter__(self):
        probe = self

        for level_id, level in enumerate(self.levels):
            for agent in level:
                agent._probe_uid = f"L{level_id}A{agent.id}"
                agent._probe_hist = agent.__dict__.pop("demand_by_retailer_history")

        def _get_hist(agent):
            probe.reads.append(agent._probe_uid)
            return agent._probe_hist

        def _set_model(agent, forecasting_model):
            # everything read since the previous attachment belongs to this agent
            foreign = {uid for uid in probe.reads if uid != agent._probe_uid}
            if foreign:
                probe.violations.append((agent._probe_uid, sorted(foreign)))
            probe.reads = []
            return probe._saved_set_model(agent, forecasting_model)

        Agent.demand_by_retailer_history = property(_get_hist)
        Agent.set_forecasting_model = _set_model
        return self

    def __exit__(self, *exc):
        del Agent.demand_by_retailer_history
        Agent.set_forecasting_model = self._saved_set_model
        for level in self.levels:
            for agent in level:
                agent.demand_by_retailer_history = agent._probe_hist
        return False


def _warm_up(cfg):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    sim, market, chain, sc = reset_simulaltion_from_dict(cfg)
    warm_end = sim.conv_time + sim.sim_time
    run_simulation_phase(0, warm_end, sim, market, chain, sc, NoOpBackend(cfg))
    return sim, market, chain, sc


BACKENDS = {
    "lstm": ("local_multichannel", LSTMLocalBackend),
    "patchtst": ("local_patchtst", LocalPatchTSTBackend),
    "timemixer": ("local_timemixer", LocalTimeMixerBackend),
}


def test_backends_run_and_stay_local():
    for arch, (training_type, backend_cls) in BACKENDS.items():
        for fusion in (False, True):
            cfg = _cfg(**{f"{arch}_fusion": fusion})
            cfg["sim"]["training_type"] = training_type
            sim, market, chain, sc = _warm_up(cfg)

            probe = _HistoryProbe()
            probe.levels = sc
            with probe:
                backend_cls(cfg).train(sim, market, chain, sc)

            assert not probe.violations, (
                f"{arch} (fusion={fusion}) read another agent's history: "
                f"{probe.violations}"
            )

            # the trained models must actually predict through the simulation
            end = sim.conv_time + sim.sim_time + sim.testing_time
            run_simulation_phase(sim.conv_time + sim.sim_time, end, sim, market,
                                 chain, sc, backend_cls(cfg))
            for agent in sc[1]:
                forecasts = agent.forecast_by_retailer_history
                assert len(forecasts) == agent.num_retailer
                assert all(np.isfinite(f).all() for f in forecasts), (
                    f"{arch} (fusion={fusion}) produced non-finite forecasts"
                )


if __name__ == "__main__":
    test_flag_resolution()
    print("PASS: local_channel_fusion flag resolution (default True, per-section override)")

    test_head_input_dims()
    test_fusion_shape_mismatch_fails_loudly()
    print("PASS: head input dims (fusion = num_channels x latent_dim) + loud shape checks")

    counts = param_counts(num_channels=2, seq_len=13)
    print("\nTrainable parameters of one channel's local model "
          "(num_channels=2, sequence_length=13):")
    print(f"  {'arch':<11}{'fusion=false':>14}{'fusion=true':>14}{'head_in off/on':>20}")
    for arch in ("lstm", "patchtst", "timemixer"):
        p_off, d_off = counts[(arch, False)]
        p_on, d_on = counts[(arch, True)]
        print(f"  {arch:<11}{p_off:>14,}{p_on:>14,}{f'{d_off} / {d_on}':>20}")

    test_backends_run_and_stay_local()
    print("\nPASS: all three local backends train + predict in both modes, "
          "and never read a peer agent's history")
