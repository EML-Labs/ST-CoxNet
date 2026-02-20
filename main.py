import random

import hydra
import torch
from omegaconf import DictConfig

from Utils.Logger import get_logger
from pipeline import build_pipeline


@hydra.main(config_path="conf", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    """
    Simple CLI that trains and validates using the shared pipeline.
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    logger = get_logger()
    pipeline = build_pipeline(cfg, logger=logger)
    trainer = pipeline.trainer
    validator = pipeline.validator

    for epoch in range(cfg.trainer.epochs):
        train_loss, train_losses = trainer.train_epoch(pipeline.train_loader)
        val_loss, val_losses = (None, None)
        if validator is not None and pipeline.val_loader is not None:
            val_loss, val_losses = validator.validation_epoch(pipeline.val_loader)

        msg = f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            msg += f", Val Loss: {val_loss:.4f}"
        logger.info(msg)


if __name__ == "__main__":
    main()
