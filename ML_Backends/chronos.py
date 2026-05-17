from typing import Optional

from .base import ForecastingBackend
from . import register_backend


@register_backend("Chronos_zero_shot")
class ChronosZeroShotBackend(ForecastingBackend):
    """Chronos foundation model, zero-shot (no training, no fine-tuning).

    ``train`` loads the pretrained pipeline and attaches a
    ``ChronosZeroShotForecaster`` to the target-level agents between the
    warm-up and testing phases. Inference uses the vanilla per-agent loop.
    """

    name = "Chronos_zero_shot"

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[list[float]]:
        from main import attach_chronos_zero_shot_models
        attach_chronos_zero_shot_models(sc_agent_list, self.cfg)
        return None
