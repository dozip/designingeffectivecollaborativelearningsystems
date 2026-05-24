from abc import ABC, abstractmethod
from typing import Optional


class ForecastingBackend(ABC):
    """Pluggable forecasting backend.

    A backend encapsulates:
      - any training logic (or no-op for zero-shot / no-training backends)
      - how to attach forecasting models to agents
      - whether inference uses the vanilla per-agent loop or a collaborative
        loop where forecasts at a specific level are produced jointly
    """

    name: str = "abstract"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    @abstractmethod
    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[list[float]]:
        """Train (or no-op) and return validation loss history, or None.

        Implementations should also attach a Forecasting model to each
        relevant agent here, so inference can call
        agent.forecasting_model.predict(...).
        """

    @property
    def needs_training_phase(self) -> bool:
        """Whether the simulation must pause between warm-up and testing so
        that train() can run. Backends that do real training or attach a
        model between phases return True; pure no-training backends (MA)
        return False and run warm-up + testing as one continuous loop."""
        return True

    @property
    def collaborative_level(self) -> Optional[int]:
        """If the backend needs a custom collaborative inference path at a
        specific supply-chain level, return that level. Otherwise return None
        to use the vanilla per-agent inference loop."""
        return None

    def collaborative_predict(self, level_agents, demand_t, sc_agent_list, t) -> list:
        """Override only if collaborative_level is not None.

        Given the full set of level agents and the current demand vector,
        return a list of predictions (one entry per agent at this level;
        each entry is the list of per-retailer forecasts the agent's
        `act_multichannel` expects).

        Default raises NotImplementedError; non-collaborative backends never
        call this.
        """
        raise NotImplementedError
