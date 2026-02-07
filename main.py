from Metadata import DatasetMetadata, FeatureType,FileLoaderMetadata, RRSequenceMetadata
from Utils.Loader.FileLoader import FileLoader
from Utils.Dataset.RRSequenceDataset import RRSequenceDataset
from torch.utils.data import DataLoader
from Model.CPCPreModel import CPCPreModel
from train import Trainer
import torch

dataset_metadata = DatasetMetadata(
    name="MIT-BIH Arrhythmia Dataset",
    sampling_frequency=128,
    file_loader=FileLoaderMetadata(
        file_path="/Users/yasantha-mac/Downloads/paf-prediction-challenge-database-1.0.0"
        ),
    rr_sequence=RRSequenceMetadata(
        window_size=50,
        stride=10,
        horizons=[4, 8, 16],
        seq_len=10
    ),
    feature_types=[
        (FeatureType.SampleEntropy, {"m": 2, "r": 0.2}),
        (FeatureType.ApproximateEntropy, {"m": 2, "r": 0.2}),
        (FeatureType.RMSSD, {}),
        (FeatureType.LFHF, {}),
        (FeatureType.EctopicPercentage, {}),
        (FeatureType.Alpha1, {})
    ]
)

if __name__ == "__main__":
    dataset = RRSequenceDataset(dataset_metadata)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,pin_memory=True,num_workers=4)
    model = CPCPreModel(num_targets=6)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer = Trainer(model, optimizer, device,loss,number_of_predictors=6)
    for epoch in range(2):
        avg_loss, avg_losses = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch+1}, Total Loss: {avg_loss:.4f}, " + 
              ", ".join([f"Loss_{i+1}: {l:.4f}" for i, l in enumerate(avg_losses)]))