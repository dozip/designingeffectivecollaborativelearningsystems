"""Single-product backward-compatibility regression test.

Acceptance requirement: with num_products = 1 and the committed config, the full
simulation must reproduce the original (pre-multi-product) results *exactly*
under the same seed, including the order/inventory trajectories that the
bullwhip / variance-ratio numbers are computed from.

The multi-product refactor turned retailers per-product and replaced the gamma
order split with a product routing map. With a single product and the default
routing (equal split across all connected suppliers) both reduce to the original
behaviour, so every agent-level history must be byte-identical.

This test runs the deterministic no-training (moving-average) path and compares a
sha256 of every reported agent history against a frozen golden captured from the
original code (tests/golden_p1_field_hashes.json). A mismatch points at the exact
(level, agent, field) that diverged.

KNOWN BASELINE PROPERTY — do not "fix": in this single-product baseline config,
manufacturers 0 and 2 (inv_capacity 10,000) are capacity-bound ~100% of the time
(their OUT orders clamp at max_order_size), which suppresses their bullwhip
relative to manufacturer 1 (inv_capacity 100,000). This clamp is part of the
original behaviour the golden documents; it is intentionally preserved here.
It is NOT a target property of the model — the sweep/REALWORLD configs use
non-binding manufacturer capacities and a saturation guard. Keep this config and
its golden untouched precisely so this historical behaviour stays documented.

Run directly:
    python tests/test_single_product_regression.py
Regenerate the golden after an intentional, reviewed change:
    python tests/test_single_product_regression.py --capture
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from helpers.helpers import reset_simulaltion_from_dict  # noqa: E402
from Simulation_Component.runner import run_simulation_phase  # noqa: E402
from ML_Backends.ma import NoOpBackend  # noqa: E402

# Config-checkpoint holds the committed single-product base configuration.
CONFIG_PATH = REPO / ".ipynb_checkpoints" / "config-checkpoint.yaml"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden_p1_field_hashes.json"

# Same seed the production entrypoint uses for experiment_id=0, run_id=0.
SEED = 42

# Agent histories that fully determine the reported results.
FIELDS = [
    "inventory_history",
    "order_history",
    "demand_sum_history",
    "forecast_history",
    "received_shipment_history",
    "order_per_supplier_history",
    "demand_by_retailer_history",
    "forecast_by_retailer_history",
]


def _hash(x) -> str:
    return hashlib.sha256(np.asarray(x, dtype=float).tobytes()).hexdigest()[:16]


def _build_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg.pop("experiments", None)
    cfg["sim"]["training_type"] = None  # deterministic moving-average path
    cfg["sim"]["simulation_runs"] = 1
    return cfg


def _run_and_hash() -> dict:
    np.random.seed(SEED)
    random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
    except Exception:
        pass

    cfg = _build_cfg()
    simulation, market, supply_chain, sc_agent_list = reset_simulaltion_from_dict(cfg)
    end = simulation.conv_time + simulation.sim_time + simulation.testing_time
    run_simulation_phase(0, end, simulation, market, supply_chain,
                         sc_agent_list, NoOpBackend(cfg))

    snapshot = {"market_split_demand_history": _hash(market.split_demand_history),
                "levels": []}
    for level in sc_agent_list:
        level_dump = []
        for agent in level:
            level_dump.append({
                "id": agent.id,
                "num_retailer": agent.num_retailer,
                "fields": {f: _hash(getattr(agent, f)) for f in FIELDS},
            })
        snapshot["levels"].append(level_dump)
    return snapshot


def _diff(golden: dict, current: dict) -> list[str]:
    diffs = []
    if golden["market_split_demand_history"] != current["market_split_demand_history"]:
        diffs.append("market_split_demand_history")
    for li, (lg, lc) in enumerate(zip(golden["levels"], current["levels"])):
        for ag, ac in zip(lg, lc):
            if ag["num_retailer"] != ac["num_retailer"]:
                diffs.append(f"level {li} agent {ag['id']}: num_retailer "
                             f"{ag['num_retailer']} != {ac['num_retailer']}")
            for f in ag["fields"]:
                if ag["fields"][f] != ac["fields"].get(f):
                    diffs.append(f"level {li} agent {ag['id']} field {f}")
    return diffs


def test_single_product_regression():
    """Assert the single-product run matches the frozen golden exactly."""
    golden = json.loads(GOLDEN_PATH.read_text())
    current = _run_and_hash()
    diffs = _diff(golden, current)
    assert not diffs, "Single-product regression drift in:\n  " + "\n  ".join(diffs)


if __name__ == "__main__":
    if "--capture" in sys.argv:
        snap = _run_and_hash()
        GOLDEN_PATH.write_text(json.dumps(snap, indent=1))
        print("captured golden ->", GOLDEN_PATH)
    else:
        test_single_product_regression()
        print("PASS: single-product regression reproduces the golden exactly")
