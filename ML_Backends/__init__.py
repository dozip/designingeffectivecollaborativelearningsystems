from typing import Type
from .base import ForecastingBackend

BACKEND_REGISTRY: dict[str, Type[ForecastingBackend]] = {}


def register_backend(name: str):
    def decorator(cls):
        BACKEND_REGISTRY[name] = cls
        return cls
    return decorator


def build_backend(cfg: dict) -> ForecastingBackend:
    training_type = cfg["sim"]["training_type"]
    if training_type is None:
        from .ma import MABackend
        return MABackend(cfg)
    if training_type not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown training_type {training_type!r}. "
            f"Registered backends: {sorted(BACKEND_REGISTRY)}"
        )
    return BACKEND_REGISTRY[training_type](cfg)


# Eager imports so @register_backend decorators fire
from . import ma           # noqa: F401, E402
from . import lstm_local   # noqa: F401, E402
from . import lstm_split   # noqa: F401, E402
from . import chronos      # noqa: F401, E402
from . import timesfm      # noqa: F401, E402
from . import split_timemixer_option_d_backend
