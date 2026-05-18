from typing import Optional

from .base import ForecastingBackend


class MABackend(ForecastingBackend):
    """Moving-average backend.

    Agents already initialize with an ``MA`` forecasting model in their
    ``__init_forecasting_strat``, so there is nothing to train or attach.
    The whole warm-up + simulation + testing window runs as one continuous
    vanilla loop.
    """

    name = "ma"

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[list[float]]:
        return None

    @property
    def needs_training_phase(self) -> bool:
        return False


class NoOpBackend(MABackend):
    """Backend used to drive the warm-up phase.

    The warm-up phase is always vanilla per-agent inference regardless of the
    real backend, so it reuses the moving-average no-op behaviour.
    """

    name = "noop"
