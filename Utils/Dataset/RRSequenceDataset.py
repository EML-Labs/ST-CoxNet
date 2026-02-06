from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Generator
from Metadata import DatasetMetadata, FeatureType
from Utils.Loader.FileLoader import FileLoader
from Utils.FeatureExtractor.HRVMetrics.NonLinear import SampleEntropy, ApproximateEntropy
from Utils.FeatureExtractor.HRVMetrics.ConventionalFeatures import RMSSD,LFHF, EctopicPercentage
from Utils.FeatureExtractor.HRVMetrics.FractalMeasures import Alpha1

class RRSequenceDataset(Dataset):
    
    def create_index(self):
        self.index = []
        history_len = self.window_size + self.stride * (self.seq_len - 1)
        future_len = self.window_size + self.stride * (max(self.horizons) - 1)
        total_len = history_len + future_len

        for rid, rr in enumerate(self.rr_records):
            max_start = len(rr) - total_len
            for start in range(0, max_start + 1, self.stride):
                self.index.append((rid, start))

    def load_feature_extractors(self):
        feature_extractors = []
        for feature_type, params in self.metadata.feature_types:
            if feature_type == FeatureType.SampleEntropy:
                extractor = SampleEntropy(**params)
            elif feature_type == FeatureType.ApproximateEntropy:
                extractor = ApproximateEntropy(**params)
            elif feature_type == FeatureType.RMSSD:
                extractor = RMSSD(**params)
            elif feature_type == FeatureType.LFHF:
                extractor = LFHF(**params)
            elif feature_type == FeatureType.EctopicPercentage:
                extractor = EctopicPercentage(**params)
            elif feature_type == FeatureType.Alpha1:
                extractor = Alpha1(**params)
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            feature_extractors.append(extractor)
        return feature_extractors
    
    def calculate_features(self, rr_window):
        return [ext.compute(rr_window) for ext in self.feature_extractors]

    def __getitem__(self, idx):
        rid, start = self.index[idx]
        rr = self.rr_records[rid]

        total_len = self.window_size + self.stride * (
            self.seq_len - 1 + max(self.horizons)
        )

        rr_seq = rr[start:start + total_len]

        # Input windows
        rr_windows = np.stack([
            rr_seq[j * self.stride : j * self.stride + self.window_size]
            for j in range(self.seq_len)
        ])

        # Targets
        hrvs = []
        for h in self.horizons:
            t_start = (self.seq_len - 1 + h) * self.stride
            t_end = t_start + self.window_size
            rr_target = rr_seq[t_start:t_end]
            hrvs.append(self.calculate_features(rr_target))

        hrvs = np.array(hrvs)
        current_hrv = self.calculate_features(rr_windows[-1])

        return (
            torch.tensor(rr_windows, dtype=torch.float32),
            torch.tensor(hrvs, dtype=torch.float32),
            torch.tensor(current_hrv, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.index)
    
    def __init__(self, metadata: DatasetMetadata):
        self.metadata = metadata
        self.window_size = metadata.rr_sequence.window_size
        self.stride = metadata.rr_sequence.stride
        self.horizons = metadata.rr_sequence.horizons
        self.seq_len = metadata.rr_sequence.seq_len
        self.samples = []

        self.feature_extractors = self.load_feature_extractors()

        self.rr_records = []
        file_loader = FileLoader(metadata.file_loader)

        for _, qrs in file_loader.load():
            if qrs is None:
                continue
            rr = np.diff(qrs.sample)
            self.rr_records.append(rr)

        self.create_index()
