from abc import ABC, abstractmethod
from typing import Optional


def resolve_local_channel_fusion(cfg: dict, section: str) -> bool:
    """Resolve the `local_channel_fusion` flag for a local backend.

    Meaning (identical for all three local backends):
        True  — the agent jointly models every channel it observes *itself*:
                the per-channel encoders run as usual, their latent
                representations are concatenated along the feature axis, and
                every channel's prediction head consumes the concatenation.
                No server, no peer agent's data.
        False — every channel is a fully independent model: each head consumes
                only its own channel's latent.

    Resolution order (most specific wins):
        cfg[section]['local_channel_fusion']  ->  cfg['local_channel_fusion']  ->  True

    The per-section override exists because the pre-fusion (asymmetric) results
    need LSTM=True together with PatchTST/TimeMixer=False, which a single global
    flag cannot express.
    """
    for holder in (cfg.get(section) if isinstance(cfg, dict) else None, cfg):
        if isinstance(holder, dict) and "local_channel_fusion" in holder:
            value = holder["local_channel_fusion"]
            if not isinstance(value, bool):
                raise ValueError(
                    f"local_channel_fusion must be a bool, got {value!r} "
                    f"({type(value).__name__})."
                )
            return value
    return True


def assert_equal_channel_counts(level_agents, backend_name: str) -> int:
    """Assert every agent at a collaborative level has the same channel count.

    Split backends fuse the per-channel encodings of all agents at the
    collaborative level and size a shared server / dense head to the total
    channel count assuming a uniform per-agent count. Unequal counts would
    silently misalign the fusion, so this fails loudly instead.

    Returns the common channel count.
    """
    counts = [int(agent.num_retailer) for agent in level_agents]
    if len(set(counts)) > 1:
        raise ValueError(
            f"{backend_name}: all agents at the collaborative level must have the "
            f"same number of channels (num_retailer), got {counts}. The split "
            "server/dense heads assume a uniform per-agent channel count."
        )
    return counts[0] if counts else 0


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
