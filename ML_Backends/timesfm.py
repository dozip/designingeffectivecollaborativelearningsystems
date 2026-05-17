import logging
from typing import Optional

from huggingface_hub import hf_hub_download

from helpers.helper_classes import TimesFMZeroShotForecaster

from .base import ForecastingBackend
from . import register_backend

logger = logging.getLogger('logger')


@register_backend("TimesFM_zero_shot")
class TimesFMZeroShotBackend(ForecastingBackend):
    """TimesFM foundation model, zero-shot (no training, no fine-tuning).

    ``train`` loads the pretrained model and attaches a
    ``TimesFMZeroShotForecaster`` to the target-level agents between the
    warm-up and testing phases. Inference uses the vanilla per-agent loop.
    """

    name = "TimesFM_zero_shot"

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[list[float]]:
        _attach_timesfm_zero_shot_models(sc_agent_list, self.cfg)
        return None


def _attach_timesfm_zero_shot_models(sc_agent_list, cfg):
    """
    Load TimesFM once and attach it to agents of one selected supply-chain level.

    No training.
    No fine-tuning.
    Only inference.
    """

    import torch
    import timesfm

    timesfm_cfg = cfg.get("timesfm", {})

    model_id = timesfm_cfg.get("model_id", "google/timesfm-2.5-200m-pytorch")
    target_level = timesfm_cfg.get("target_level", 1)

    prediction_length = timesfm_cfg.get("prediction_length", 1)
    max_context = timesfm_cfg.get("max_context", 1024)
    max_horizon = timesfm_cfg.get("max_horizon", 256)

    normalize_inputs = timesfm_cfg.get("normalize_inputs", True)
    use_continuous_quantile_head = timesfm_cfg.get("use_continuous_quantile_head", True)
    force_flip_invariance = timesfm_cfg.get("force_flip_invariance", True)
    infer_is_positive = timesfm_cfg.get("infer_is_positive", True)
    fix_quantile_crossing = timesfm_cfg.get("fix_quantile_crossing", True)

    logger.info(f"Loading TimesFM zero-shot model: {model_id}")

    torch.set_float32_matmul_precision("high")

    weights_path = hf_hub_download(
        repo_id=model_id,
        filename=timesfm.TimesFM_2p5_200M_torch.WEIGHTS_FILENAME,
    )

    model = timesfm.TimesFM_2p5_200M_torch(
        torch_compile=False,
    )

    model.model.load_checkpoint(
        weights_path,
        torch_compile=model.torch_compile,
    )

    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            per_core_batch_size=1,
            normalize_inputs=normalize_inputs,
            use_continuous_quantile_head=use_continuous_quantile_head,
            force_flip_invariance=force_flip_invariance,
            infer_is_positive=infer_is_positive,
            fix_quantile_crossing=fix_quantile_crossing,
        )
    )

    logger.info(f"TimesFM loaded. Attaching to level {target_level} agents.")

    for j, agent in enumerate(sc_agent_list[target_level]):
        logger.info(
            f"Attaching TimesFM to level={target_level}, "
            f"agent_index={j}, agent_id={agent.id}, "
            f"num_retailer={agent.num_retailer}"
        )

        agent.set_forecasting_model(
            TimesFMZeroShotForecaster(
                model=model,
                prediction_length=prediction_length,
            )
        )

    logger.info(f"TimesFM zero-shot forecasters attached to level {target_level}.")
