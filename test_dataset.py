CSV_PATH = "/Users/yasantha-mac/FYP/ST-CoxNet/Utils/Dataset/AFDB/CSV_Files"
import torch

from Utils.Dataset.RRSequenceCSVData import RRSequenceCSVData
from Utils.Dataset.RRSequenceCSVDataset import RRSequenceCSVDataset
from Utils.Dataset.Splitter import split
from torch.utils.data import DataLoader
import random
from Metadata import FeatureType, SplitMetadata,RRCSVDataMetadata

features = [
    FeatureType.LFHF,
    FeatureType.RMSSD,
    FeatureType.Alpha1,
    FeatureType.SampleEntropy,
    FeatureType.ApproximateEntropy
]
random.seed(42)

dataset = RRSequenceCSVData(
    metadata=RRCSVDataMetadata(
    rri_csv_path=f"{CSV_PATH}/rri_windows.csv",
    features_csv_path=f"{CSV_PATH}/hrv_features.csv",
    seq_len=10,
    horizons=[1, 4, 8],
    feature_types=features)
)
train_records, test_records, val_records = split(dataset.records, split_metadata=SplitMetadata(train=0.7, val=0.1, test=0.2))
train_dataset = RRSequenceCSVDataset(
    records=train_records,
    horizons=dataset.horizons,
    seq_len=dataset.seq_len
)

test_dataset = RRSequenceCSVDataset(
    records=test_records,
    horizons=dataset.horizons,
    seq_len=dataset.seq_len
)

val_dataset = RRSequenceCSVDataset(
    records=val_records,
    horizons=dataset.horizons,
    seq_len=dataset.seq_len
)
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Test Dataset Size: {len(test_dataset)}")
print(f"Val Dataset Size: {len(val_dataset)}")
# for key, rec in records.items():
#     print(f"Segment: {key}")
#     print(f"RRI Shape: {rec['rri'].shape}")
#     print(f"HRV Shape: {rec['hrv'].shape}")
#     print(f"Keys: {rec['keys'][:5]}")
#     print(f"Feature Names: {rec['feature_names']}")
#     print("-" * 50)

# for i in range(5):
#     rri_windows, current_hrv, future_hrvs = dataset[i]
#     print(rri_windows)
#     print(f"RRI Windows Shape: {rri_windows.shape}")
#     print(f"Current HRV Shape: {current_hrv.shape}")
#     print(f"Future HRVs Shape: {future_hrvs.shape}")

import torch
for rr_windows, hrv_targets, _ in train_dataset:
    print(hrv_targets)
    # if torch.isnan(hrv_targets).any():
    #     print("NaNs in hrv_targets")
    #     break