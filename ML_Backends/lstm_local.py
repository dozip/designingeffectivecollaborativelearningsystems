from typing import Optional

from .base import ForecastingBackend
from . import register_backend


@register_backend("local_multichannel")
class LSTMLocalBackend(ForecastingBackend):
    """Per-agent multi-channel LSTM, trained locally for each agent.

    Inference uses the vanilla per-agent loop: each agent's trained
    ``MultiChannel_LSTM`` is attached and queried through ``agent.act``.
    """

    name = "local_multichannel"

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[list[float]]:
        from main import local_training_multichannel_lstm
        return local_training_multichannel_lstm(
            simulation, market, supply_chain, sc_agent_list, self.cfg
        )
