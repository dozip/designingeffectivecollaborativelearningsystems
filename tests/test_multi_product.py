"""Multi-product end-to-end and routing-correctness tests (2 retailers x 3
products x 3 manufacturers, M_p -> product p).

Covers:
  - the full simulation runs end to end under num_products > 1 (MA path)
  - routing is correct: each retailer's order to manufacturer m equals that
    retailer's product-m order (no cross-product leakage), and manufacturer m
    receives exactly R product-m channels
  - the flat aggregate histories equal the sum of the per-product histories
  - the fail-loud guards fire on invalid routing / unequal channel counts

Nothing is hardcoded to 2/3/3 beyond this fixture's own config.
"""
import random
import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # allow running without pytest installed
    pytest = None

import contextlib


@contextlib.contextmanager
def _raises(exc):
    """Minimal pytest.raises replacement so the file runs standalone too."""
    if pytest is not None:
        with pytest.raises(exc):
            yield
        return
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from helpers.helpers import reset_simulaltion_from_dict  # noqa: E402
from Simulation_Component.runner import run_simulation_phase  # noqa: E402
from ML_Backends.ma import NoOpBackend  # noqa: E402
from ML_Backends.base import assert_equal_channel_counts  # noqa: E402


def _cfg(R=2, P=3, M=3, product_to_manufacturer=None):
    split = [1.0 / (R * P)] * (R * P)
    if product_to_manufacturer is None:
        product_to_manufacturer = {p: p for p in range(P)}
    return {
        "sim": {"simulation_runs": 1, "run_start_id": 0, "convergence_time": 200,
                "simulation_time": 200, "training_time": 80, "testing_time": 20,
                "training_type": None, "train_size": 60, "val_size": 20, "epochs": 2,
                "learning_rate": 0.001, "momentum": 0.9, "batch_size": 16,
                "sequence_length": 4},
        "early_stopping": {"patience": 100, "min_delta": 0.0},
        "market": {"data_scource": None, "primary_demand": 1000, "trend_magnitude": 2,
                   "seasonality_magnitude": 5, "seasonality_frequncy": 7,
                   "random_walk": {"mean": 10, "variance": 1},
                   "demand_split": split, "num_products": P},
        "supply_chain": {
            "agents_per_level": [R, M], "num_products": P,
            "product_to_manufacturer": product_to_manufacturer,
            "sc_levels": {
                "sc_level_0": {"adjaceny_list": [[1] * M for _ in range(R)],
                               "lead_time": 1,
                               "replenishment_strat": ["OUT"] * R,
                               "forecasting_strat": ["MA"] * R,
                               "inv_capacity": [50000] * R, "init_inv": [0] * R,
                               "safety_risk_factor": [1] * R, "R": [1] * R},
                "sc_level_1": {"adjaceny_list": [[1] for _ in range(M)],
                               "lead_time": 1,
                               "replenishment_strat": ["OUT"] * M,
                               "forecasting_strat": ["MA"] * M,
                               "inv_capacity": [10000] * M, "init_inv": [0] * M,
                               "safety_risk_factor": [1] * M, "R": [1] * M}}},
    }


def _run(cfg):
    np.random.seed(42)
    random.seed(42)
    sim, market, supply_chain, sc = reset_simulaltion_from_dict(cfg)
    end = sim.conv_time + sim.sim_time + sim.testing_time
    run_simulation_phase(0, end, sim, market, supply_chain, sc, NoOpBackend(cfg))
    return sim, market, supply_chain, sc, end


def test_multi_product_runs_and_routes():
    R, P, M = 2, 3, 3
    sim, market, supply_chain, sc, end = _run(_cfg(R, P, M))

    assert supply_chain.product_to_suppliers == [[p] for p in range(P)]

    retailers, manufacturers = sc[0], sc[1]
    for r in retailers:
        assert r.num_products == P and r.num_retailer == P
        assert len(r.shipment_queues) == P and len(r.replenishment) == P
        # routing correctness: order to manufacturer p == product-p order
        ops = np.array(r.order_per_supplier_history)  # (T, M)
        for p in range(P):
            assert np.array_equal(ops[:, p], np.array(r.order_history_by_product[p]))
        # aggregate == sum over products
        agg = np.array(r.order_history)
        per = sum(np.array(r.order_history_by_product[p]) for p in range(P))
        assert np.allclose(agg, per)

    for m in manufacturers:
        assert m.num_products == 1 and m.num_retailer == R
        assert all(len(c) == end for c in m.demand_by_retailer_history)


def test_invalid_routing_fails_loud():
    # two products routed to the same manufacturer -> a manufacturer would
    # produce two products, which is rejected.
    with _raises(ValueError):
        reset_simulaltion_from_dict(_cfg(product_to_manufacturer={0: 0, 1: 0, 2: 2}))


def test_channel_count_guard():
    class _A:
        def __init__(self, n):
            self.num_retailer = n
    with _raises(ValueError):
        assert_equal_channel_counts([_A(2), _A(3)], "split X")
    assert assert_equal_channel_counts([_A(2), _A(2)], "split X") == 2


if __name__ == "__main__":
    test_multi_product_runs_and_routes()
    test_invalid_routing_fails_loud()
    test_channel_count_guard()
    print("PASS: multi-product end-to-end + routing + guards")
