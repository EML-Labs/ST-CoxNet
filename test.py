import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from Utils.Dataset.Splitter import split
from Metadata import SplitMetadata,FileLoaderMetadata
from Utils.Dataset.RRSequenceDataset import RRSequenceDataset
from torch.utils.data import DataLoader
from Utils.Logger import get_logger
from train import Trainer
from dotenv import load_dotenv
import wandb
from tqdm import tqdm
import torch
import numpy as np
import random
from Device import Device


load_dotenv()  # Load environment variables from .env file

@hydra.main(config_path="conf", config_name="conf", version_base=None)
def main(cfg: DictConfig):

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    try:
        wandb.login()
        run = wandb.init(
            entity="eml-labs",
            project="PAF Prediction - CPC",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"CPCPreModel_Run_{wandb.util.generate_id()}",
            tags=["CPCPreModel", "Experiment"]
        )

        logger = get_logger(run=run)

        file_loader:FileLoaderMetadata = instantiate(cfg.dataset.file_loader)
        splitt_metadata = SplitMetadata(**cfg.split)

        train_files, val_files, test_files = split(file_loader.file_names, splitt_metadata)

        logger.info(f"Files are split into Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)} | Total: {len(file_loader.file_names)}")

        train_file_loader_metadata = file_loader.model_copy(update={"file_names": train_files})
        val_file_loader_metadata = file_loader.model_copy(update={"file_names": val_files})
        test_file_loader_metadata = file_loader.model_copy(update={"file_names": test_files})

        sampling_rate = cfg.dataset.sampling_rate
        rr_sequence_config = instantiate(cfg.preprocessing.rr_sequence)
        feature_extractors_config = instantiate(cfg.feature_extractors)

        train_dataset = RRSequenceDataset(sampling_rate, rr_sequence_config, train_file_loader_metadata, feature_extractors_config)
        val_dataset = RRSequenceDataset(sampling_rate, rr_sequence_config, val_file_loader_metadata, feature_extractors_config)
        test_dataset = RRSequenceDataset(sampling_rate, rr_sequence_config, test_file_loader_metadata, feature_extractors_config)

        logger.info(f"Datasets created with lengths - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, **cfg.trainer.loader)
        val_loader = DataLoader(val_dataset, **cfg.validator.loader)
        # test_loader = DataLoader(test_dataset, **cfg.tester.loader)

        model:torch.nn.Module = instantiate(cfg.model)

        logger.info(f"Model instantiated: {model.__class__.__name__}")

        model.eval()
        device:Device = instantiate(cfg.device).device
        optimizer:torch.optim = instantiate(cfg.trainer.optimizer, params=model.parameters())
        loss_fn :torch.nn.Module = instantiate(cfg.trainer.loss)
        trainer = Trainer(model, optimizer, device.device, loss_fn, number_of_predictors=cfg.model.config.predictor.num_heads, loss_weights=cfg.trainer.loss_weights)
        
        for epoch in tqdm(range(cfg.trainer.epochs), desc="Training Epochs"):
            avg_loss, avg_losses = trainer.train_epoch(train_loader)
            logger.log({"train_loss": avg_loss, **{f"train_loss_pred_{i}": l for i, l in enumerate(avg_losses)}, "epoch": epoch+1})
            logger.info(f"Epoch {epoch+1}, Total Train Loss: {avg_loss:.4f}")
            tqdm.write(f"Epoch {epoch+1}, Total Train Loss: {avg_loss:.4f}")

    except Exception as e:
        torch.save(model.state_dict(), "model.pt")
        artifact = wandb.Artifact("CPCPreModel", type="model")
        artifact.add_file("model.pt")
        run.log_artifact(artifact)
        logger.info("Model saved successfully")
        if logger is not None:
            logger.info(f"An error occurred: {str(e)}")
        if run is not None:
            run.finish()
        raise e

    # print(f"File Loader Config: {file_loader_config}")
if __name__ == "__main__":
    main()
