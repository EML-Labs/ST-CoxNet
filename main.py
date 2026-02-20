from Metadata import DatasetMetadata, FeatureType,FileLoaderMetadata, RRSequenceMetadata
from Utils.Loader.FileLoader import FileLoader
from Utils.Dataset.RRSequenceDataset import RRSequenceDataset
from torch.utils.data import DataLoader
from Model.CPCPreModel import CPCPreModel
from train import Trainer
from Validator import Validator
import torch
import random

random.seed(42)

number_list = range(1,51)
validation_numbers = random.sample(number_list, 10)
training_numbers = [n for n in number_list if n not in validation_numbers]

validation_file_loader_metadata = FileLoaderMetadata(
    file_path="/Users/yasantha-mac/Downloads/paf-prediction-challenge-database-1.0.0",
    file_names=[f"p{p:02d}.dat" for p in validation_numbers],
    sample_needed=False
)
training_file_loader_metadata = FileLoaderMetadata(
    file_path="/Users/yasantha-mac/Downloads/paf-prediction-challenge-database-1.0.0",
    file_names=[f"p{p:02d}.dat" for p in training_numbers],
    sample_needed=False
)
rr_sequence_metadata = RRSequenceMetadata(
    window_size=50,
    stride=10,
    horizons=[4, 8, 16],
    seq_len=10
)

feature_types = [
    (FeatureType.SampleEntropy, {"m": 2, "r": 0.2}),
    (FeatureType.ApproximateEntropy, {"m": 2, "r": 0.2}),
    (FeatureType.RMSSD, {}),
    (FeatureType.LFHF, {}),
    (FeatureType.EctopicPercentage, {}),
    (FeatureType.Alpha1, {})
]
training_metadata = DatasetMetadata(
    name="PAF Training Dataset",
    sampling_frequency=128,
    file_loader=training_file_loader_metadata,
    rr_sequence=rr_sequence_metadata,
    feature_types=feature_types
)
validation_metadata = DatasetMetadata(
    name="PAF Validation Dataset",
    sampling_frequency=128,
    file_loader=validation_file_loader_metadata,
    rr_sequence=rr_sequence_metadata,
    feature_types=feature_types 
)

if __name__ == "__main__":
    training_dataset = RRSequenceDataset(training_metadata)
    validation_dataset = RRSequenceDataset(validation_metadata)
    train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True,pin_memory=True,num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False,pin_memory=True,num_workers=4)

    model = CPCPreModel(num_targets=6,num_predictors=3)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer = Trainer(model, optimizer, device,loss,number_of_predictors=3)
    validator = Validator(model, device, loss,number_of_predictors=3)

    for epoch in range(10):
        avg_loss, avg_losses = trainer.train_epoch(train_dataloader)
        val_avg_loss, val_avg_losses = validator.validation_epoch(validation_dataloader)

        print(f"Epoch {epoch+1}, Total Train Loss: {avg_loss:.4f}, Total Val Loss: {val_avg_loss:.4f}")
        for i, l in enumerate(avg_losses):
            print(f"  Predictor {i+1} Train Loss: {l:.4f}")
        for i, l in enumerate(val_avg_losses):
            print(f"  Predictor {i+1} Val Loss: {l:.4f}")