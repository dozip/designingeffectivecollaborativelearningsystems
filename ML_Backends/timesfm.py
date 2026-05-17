from typing import Optional

from .base import ForecastingBackend
from . import register_backend


@register_backend("TimesFM_zero_shot")
class TimesFMZeroShotBackend(ForecastingBackend):
    """TimesFM foundation model, zero-shot (no training, no fine-tuning).

    ``train`` loads the pretrained model and attaches a
    ``TimesFMZeroShotForecaster`` to the target-level agents between the
    warm-up and testing phases. Inference uses the vanilla per-agent loop.
    """

    name = "TimesFM_zero_shot"

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[list[float]]:
        from main import attach_timesfm_zero_shot_models
        attach_timesfm_zero_shot_models(sc_agent_list, self.cfg)
        return None
