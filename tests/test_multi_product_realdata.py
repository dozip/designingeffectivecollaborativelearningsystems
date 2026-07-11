"""Real-world multi-product market test (Option A: one series per product).

Writes one CSV per product, points market.data_scource at them, and checks:
  - MarketDataSourceMultiProduct is selected for num_products > 1 with data
  - each product's demand is split across retailers by demand_split
    (order s = retailer*P + product), preserving the integer total
  - the simulation runs end to end and routing stays correct
"""
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from helpers.helpers import reset_simulaltion_from_dict  # noqa: E402
from Simulation_Component.runner import run_simulation_phase  # noqa: E402
from Simulation_Component.market import MarketDataSourceMultiProduct  # noqa: E402
from ML_Backends.ma import NoOpBackend  # noqa: E402


def _write_product_csvs(tmp: Path, n: int):
    series = {
        "A": (100 + np.arange(n)).astype(float),
        "B": (60 + (np.arange(n) % 7)).astype(float),
        "C": np.full(n, 40.0),
    }
    paths = {}
    for name, s in series.items():
        p = tmp / f"prod_{name}.csv"
        pd.DataFrame({"demand": s}).to_csv(p, index=False)
        paths[name] = str(p)
    return paths


def _cfg(paths):
    R, P, M = 2, 3, 3
    return {
        "sim": {"simulation_runs": 1, "run_start_id": 0, "convergence_time": 100,
                "simulation_time": 100, "training_time": 60, "testing_time": 10,
                "training_type": None, "train_size": 40, "val_size": 15, "epochs": 2,
                "learning_rate": 0.001, "momentum": 0.9, "batch_size": 16,
                "sequence_length": 4},
        "early_stopping": {"patience": 100, "min_delta": 0.0},
        "market": {
            "data_scource": [paths["A"], paths["B"], paths["C"]],
            "demand_column": "demand",
            "primary_demand": 1000, "trend_magnitude": 2, "seasonality_magnitude": 5,
            "seasonality_frequncy": 7, "random_walk": {"mean": 10, "variance": 1},
            # retailer shares 0.7/0.3 for every product; order s = r*P + p
            "demand_split": [0.7, 0.7, 0.7, 0.3, 0.3, 0.3], "num_products": P,
        },
        "supply_chain": {
            "agents_per_level": [R, M], "num_products": P,
            "product_to_manufacturer": {0: 0, 1: 1, 2: 2},
            "sc_levels": {
                "sc_level_0": {"adjaceny_list": [[1, 1, 1], [1, 1, 1]], "lead_time": 1,
                               "replenishment_strat": ["OUT", "OUT"],
                               "forecasting_strat": ["MA", "MA"],
                               "inv_capacity": [50000, 50000], "init_inv": [0, 0],
                               "safety_risk_factor": [1, 1], "R": [1, 1]},
                "sc_level_1": {"adjaceny_list": [[1], [1], [1]], "lead_time": 1,
                               "replenishment_strat": ["OUT"] * 3,
                               "forecasting_strat": ["MA"] * 3,
                               "inv_capacity": [10000, 100000, 10000], "init_inv": [0, 0, 0],
                               "safety_risk_factor": [1, 1, 1], "R": [1, 1, 1]}}},
    }


def test_multi_product_real_data():
    tmp = Path(tempfile.mkdtemp(prefix="mp_realdata_"))
    paths = _write_product_csvs(tmp, n=260)
    cfg = _cfg(paths)

    np.random.seed(42)
    random.seed(42)
    sim, market, sc_chain, sc = reset_simulaltion_from_dict(cfg)

    assert isinstance(market, MarketDataSourceMultiProduct)

    # A=100 -> 70/30, B=60 -> 42/18, C=40 -> 28/12 ; order s = r*P + p
    assert market.split_demand_on_time(0) == [70.0, 42.0, 28.0, 30.0, 18.0, 12.0]
    assert market.get_demand(0) == 200  # 100 + 60 + 40

    # full run (fresh instance so t=0 isn't already consumed)
    np.random.seed(42)
    random.seed(42)
    sim, market, sc_chain, sc = reset_simulaltion_from_dict(cfg)
    end = sim.conv_time + sim.sim_time + sim.testing_time
    run_simulation_phase(0, end, sim, market, sc_chain, sc, NoOpBackend(cfg))

    for r in sc[0]:
        ops = np.array(r.order_per_supplier_history)
        for p in range(3):
            assert np.array_equal(ops[:, p], np.array(r.order_history_by_product[p]))


if __name__ == "__main__":
    test_multi_product_real_data()
    print("PASS: multi-product real-data (Option A) split + routing")
