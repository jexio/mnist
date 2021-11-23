from datetime import datetime
from functools import reduce
from typing import List, Optional, Union

import torch
import hydra
from fulmo.core import BaseModule, BaseDataModule
from fulmo.utils import logging as log_utils
from fulmo.utils.seed import set_seed
from torch.utils.data import Sampler
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info
import pytorch_lightning as pl

from utils import initialize


logger = log_utils.get_logger(__name__)


def scale_lr(scalars: List[Union[float, List[int], List[float]]]) -> float:
    """Scale learning rate."""
    if len(scalars) == 1:
        return scalars[0]
    lr_coefficient, coefficient, batch_size, accumulate_grad_batches, num_gpus, num_nodes = scalars
    if isinstance(num_gpus, List):
        raise ValueError("`num_gpus` must be a number.")
    if num_gpus is None:
        num_gpus = 1
    if num_nodes is None:
        num_nodes = 1
    numerator = reduce(lambda x, y: x * y, [batch_size, accumulate_grad_batches, num_gpus, num_nodes])
    return round(lr_coefficient * (numerator / coefficient), 5)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    """Set seed and run experiment."""
    initialize()
    OmegaConf.register_new_resolver("to_tuple", lambda x: tuple(x))
    OmegaConf.register_new_resolver("scale_lr", scale_lr)
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    rank_zero_info(OmegaConf.to_yaml(config))
    set_seed(config.seed)
    run(config)


def run(config: DictConfig) -> None:
    """Run experiment."""
    log_utils.extras(config)
    # Init PyTorch Lightning datamodule
    sampler: Optional[Sampler[int]] = None
    if "sampler" in config:
        begin_time = datetime.now()
        sampler: Sampler[int] = hydra.utils.instantiate(config.sampler)
        end_time = datetime.now()
        elapsed = (end_time - begin_time).total_seconds()
        logger.info("Instantiating sampler <%s> in <%s> seconds", config.sampler._target_, elapsed)

    begin_time = datetime.now()
    datamodule: BaseDataModule = hydra.utils.instantiate(config.datamodule, _convert_="partial")
    datamodule.prepare_data()
    datamodule.setup()
    datamodule.set_sampler(sampler)
    end_time = datetime.now()
    elapsed = (end_time - begin_time).total_seconds()
    logger.info("Instantiating datamodule <%s> in <%s>", config.datamodule._target_, elapsed)

    # Init PyTorch Lightning pipeline
    begin_time = datetime.now()
    pl_module: pl.LightningModule = BaseModule(config=config)
    end_time = datetime.now()
    elapsed = (end_time - begin_time).total_seconds()
    logger.info("Instantiating module in <%s> seconds", elapsed)

    # Init PyTorch Lightning callbacks
    begin_time = datetime.now()
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                logger.info("Instantiating callback <%s>", cb_conf._target_)
                callbacks.append(hydra.utils.instantiate(cb_conf))
    end_time = datetime.now()
    elapsed = (end_time - begin_time).total_seconds()
    if callbacks:
        logger.info("Instantiating callbacks in <%s> seconds", elapsed)

    # Init PyTorch Lightning loggers
    begin_time = datetime.now()
    loggers: List[pl.loggers.base.LightningLoggerBase] = list()
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                logger.info("Instantiating logger <%s>", lg_conf._target_)
                loggers.append(hydra.utils.instantiate(lg_conf))
    end_time = datetime.now()
    elapsed = (end_time - begin_time).total_seconds()
    if loggers:
        logger.info("Instantiating loggers in <%s> seconds", elapsed)

    begin_time = datetime.now()
    trainer: pl.Trainer = hydra.utils.instantiate(config["trainer"], callbacks=callbacks, logger=loggers)
    end_time = datetime.now()
    elapsed = (end_time - begin_time).total_seconds()
    logger.info("Instantiating trainer <%s> in <%s> seconds", config.trainer._target_, elapsed)

    # Send some parameters from config to all lightning loggers
    logger.info("Logging hyperparameters!")
    log_utils.log_hyperparameters(config, pl_module, datamodule, trainer)
    # Train the model
    logger.info("Starting training!")
    trainer.fit(model=pl_module, datamodule=datamodule)

    # Print path to best checkpoint
    checkpoint_path = trainer.checkpoint_callback.best_model_path
    logger.info("Best checkpoint path :<%s>", checkpoint_path)

    if not config.trainer.get("fast_dev_run"):
        # Evaluate model on test set after training
        if datamodule.data_test is not None:
            logger.info("Starting testing!")
            trainer.test(dataloaders=datamodule.test_dataloader())

    log_utils.finish(
        config=config,
        model=pl_module,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )


if __name__ == "__main__":
    main()
