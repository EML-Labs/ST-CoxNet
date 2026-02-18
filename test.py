import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from Utils.Dataset.Splitter import split
from Metadata import SplitMetadata,FileLoaderMetadata
from Utils.Dataset.RRSequenceDataset import RRSequenceDataset
from torch.utils.data import DataLoader

@hydra.main(config_path="conf", config_name="conf", version_base=None)
def main(cfg: DictConfig):

    file_loader_config = cfg.dataset.file_loader
    file_loader:FileLoaderMetadata = instantiate(file_loader_config)
    splitt_metadata = SplitMetadata(**cfg.split)
    train_files, val_files, test_files = split(file_loader.file_names, splitt_metadata)

    train_file_loader_metadata = file_loader.model_copy(update={"file_names": train_files})
    val_file_loader_metadata = file_loader.model_copy(update={"file_names": val_files})
    test_file_loader_metadata = file_loader.model_copy(update={"file_names": test_files})

    sampling_rate = cfg.dataset.sampling_rate
    rr_sequence_config = instantiate(cfg.preprocessing.rr_sequence)
    feature_extractors_config = instantiate(cfg.feature_extractors)

    train_dataset = RRSequenceDataset(sampling_rate, rr_sequence_config, train_file_loader_metadata, feature_extractors_config)
    val_dataset = RRSequenceDataset(sampling_rate, rr_sequence_config, val_file_loader_metadata, feature_extractors_config)
    test_dataset = RRSequenceDataset(sampling_rate, rr_sequence_config, test_file_loader_metadata, feature_extractors_config)

    train_loader = DataLoader(train_dataset, **cfg.trainer.loader)
    val_loader = DataLoader(val_dataset, **cfg.validator.loader)

    model = instantiate(cfg.model)

    model.eval()
    # optimizer = instantiate(cfg.trainer.optimizer)
    # loss_fn = instantiate(cfg.trainer.loss)
    # test_loader = DataLoader(test_dataset, **cfg.tester.loader)


    # print(f"File Loader Config: {file_loader_config}")
if __name__ == "__main__":
    main()
