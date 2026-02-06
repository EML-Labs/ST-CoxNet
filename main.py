from Metadata import DatasetMetadata, FeatureType,FileLoaderMetadata, RRSequenceMetadata
from Utils.Loader.FileLoader import FileLoader
from Utils.Dataset.RRSequenceDataset import RRSequenceDataset


dataset_metadata = DatasetMetadata(
    name="MIT-BIH Arrhythmia Dataset",
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
    print(f"Dataset Name: {dataset_metadata.name}")
    print(f"Total samples prepared: {len(dataset)}")