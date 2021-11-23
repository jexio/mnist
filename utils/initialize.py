from distutils.util import strtobool

from dotenv import load_dotenv, dotenv_values
from fulmo.schedulers import SCHEDULER_DATACLASS_REGISTRY
from fulmo.models import MODEL_DATACLASS_REGISTRY
from hydra.core.config_store import ConfigStore

SPEAKER_TRAIN_CONFIGS = [
    "lr_scheduler",
    "model"
]


def initialize() -> None:
    """Initialize modules and load a configuration file."""
    load_dotenv()
    env_config = dotenv_values(".env")
    env_config = {key: strtobool(value.lower()) == 1 if value.lower() in ("true", "false") else value
                  for key, value in env_config.items()}

    cs = ConfigStore.instance()
    registries = {"lr_scheduler": SCHEDULER_DATACLASS_REGISTRY, "model": MODEL_DATACLASS_REGISTRY}

    for group in SPEAKER_TRAIN_CONFIGS:
        registry = registries[group]
        for k, v in registry.items():
            cs.store(group=group, name=k, node=v, provider="fulmo")


__all__ = ["initialize"]
