import logging
from typing import Optional

from helpers.helper_classes import ChronosZeroShotForecaster

from .base import ForecastingBackend
from . import register_backend

logger = logging.getLogger('logger')


@register_backend("Chronos_zero_shot")
class ChronosZeroShotBackend(ForecastingBackend):
    """Chronos foundation model, zero-shot (no training, no fine-tuning).

    ``train`` loads the pretrained pipeline and attaches a
    ``ChronosZeroShotForecaster`` to the target-level agents between the
    warm-up and testing phases. Inference uses the vanilla per-agent loop.
    """

    name = "Chronos_zero_shot"

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[list[float]]:
        _attach_chronos_zero_shot_models(sc_agent_list, self.cfg)
        return None


def _attach_chronos_zero_shot_models(sc_agent_list, cfg):
    from chronos import Chronos2Pipeline

    chronos_cfg = cfg.get("chronos", {})

    model_id = chronos_cfg.get("model_id", "amazon/chronos-2")
    device = chronos_cfg.get("device", "cuda")
    prediction_length = chronos_cfg.get("prediction_length", 1)
    quantile_levels = chronos_cfg.get("quantile_levels", [0.5])
    target_level = chronos_cfg.get("target_level", 1)

    pipeline = Chronos2Pipeline.from_pretrained(
        model_id,
        device_map=device,
    )

    logger.info(f"Chronos loaded. Attaching to level {target_level} agents.")

    for j, agent in enumerate(sc_agent_list[target_level]):
        logger.info(
            f"Attaching Chronos to level={target_level}, "
            f"agent_index={j}, agent_id={agent.id}, "
            f"num_retailer={agent.num_retailer}"
        )

        agent.set_forecasting_model(
            ChronosZeroShotForecaster(
                pipeline=pipeline,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
            )
        )

    logger.info(f"Chronos zero-shot forecasters attached to level {target_level}.")
