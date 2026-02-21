import random

import wandb  # Imported inside to avoid side effects when unused
import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from Utils.Logger import get_logger
from Utils.Pipeline import build_pipeline, Pipeline
from Train import Trainer


load_dotenv()


@hydra.main(config_path="conf", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    """
    Primary training entry point with WandB logging.
    """

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    run = None
    logger = None

    try:
        wandb.login()
        run = wandb.init(
            entity="eml-labs",
            project="PAF Prediction - CPC",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"CPCPreModel_Run_{wandb.util.generate_id()}",
            tags=["CPCPreModel", "Experiment"],
        )

        logger = get_logger(run=run)

        pipeline: Pipeline = build_pipeline(cfg, logger=logger)
        trainer: Trainer = pipeline.trainer
        validator = pipeline.validator

        for epoch in tqdm(range(cfg.trainer.epochs), desc="Training Epochs"):
            train_loss, train_losses = trainer.train_epoch(pipeline.train_loader)

            val_loss = None
            val_losses = None
            if validator is not None and pipeline.val_loader is not None:
                val_loss, val_losses = validator.validation_epoch(pipeline.val_loader)

            metrics = {
                "train/loss": train_loss,
                **{f"train/loss_pred_{i}": l for i, l in enumerate(train_losses)},
                "epoch": epoch + 1,
            }
            if val_loss is not None and val_losses is not None:
                metrics.update(
                    {
                        "val/loss": val_loss,
                        **{f"val/loss_pred_{i}": l for i, l in enumerate(val_losses)},
                    }
                )

            logger.log(metrics)

            if val_loss is not None:
                logger.info(
                    f"Epoch {epoch+1}, "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

        # Save final model and upload to WandB
        model_path = "model.pt"
        torch.save(pipeline.model.state_dict(), model_path)
        artifact = wandb.Artifact("CPCPreModel", type="model")
        artifact.add_file(model_path)
        if run is not None:
            run.log_artifact(artifact)
        if logger is not None:
            logger.info("Final model saved and logged to WandB as artifact.")

    except KeyboardInterrupt:
        # Handle manual interruption: save the current model state and upload it as an artifact
        if "pipeline" in locals():
            model_path = "model_interrupt.pt"
            torch.save(pipeline.model.state_dict(), model_path)
            artifact = wandb.Artifact("CPCPreModel-interrupt", type="model")
            artifact.add_file(model_path)
            if run is not None:
                run.log_artifact(artifact)
        if logger is not None:
            logger.info("Training interrupted by user (KeyboardInterrupt). Model checkpoint saved and logged to WandB.")
        # Do not re-raise to allow a clean shutdown

    except Exception as e:
        # Save model and log as artifact if something goes wrong
        if "pipeline" in locals():
            torch.save(pipeline.model.state_dict(), "model.pt")
            artifact = wandb.Artifact("CPCPreModel", type="model")
            artifact.add_file("model.pt")
            if run is not None:
                run.log_artifact(artifact)
        if logger is not None:
            logger.info(f"An error occurred: {str(e)}")
        if run is not None:
            run.finish()
        raise

    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
