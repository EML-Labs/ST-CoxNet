import random

import hydra
import torch
from omegaconf import DictConfig

from Utils.Logger import get_logger
from pipeline import build_pipeline


@hydra.main(config_path="conf", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    """
    Lightweight debug / quick-run entry that trains without WandB.
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    logger = get_logger()
    pipeline = build_pipeline(cfg, logger=logger)
    trainer = pipeline.trainer

    for epoch in range(cfg.trainer.epochs):
        avg_loss, avg_losses = trainer.train_epoch(pipeline.train_loader)
        logger.info(
            f"[DEBUG] Epoch {epoch+1}, "
            f"Total Train Loss: {avg_loss:.4f}, "
            + ", ".join(f"Pred {i}: {l:.4f}" for i, l in enumerate(avg_losses))
        )


if __name__ == "__main__":
    main()

