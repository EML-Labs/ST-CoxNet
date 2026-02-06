from torch.utils.data import Dataset
import torch
import numpy as np
from Configs import WINDOW_SIZE, STRIDE, HORIZONS, SEQUENCE_LENGTH
from Metadata import DatasetMetadata, FeatureType
from Utils.Loader.FileLoader import FileLoader
from typing import Generator
from Utils.FeatureExtractor.HRVMetrics.NonLinear import SampleEntropy, ApproximateEntropy
from Utils.FeatureExtractor.HRVMetrics.ConventionalFeatures import RMSSD,LFHF, EctopicPercentage
from Utils.FeatureExtractor.HRVMetrics.FractalMeasures import Alpha1

class RRSequenceDataset(Dataset):

    def load_rr_sequences(self)-> Generator[np.ndarray, None, None]:
        file_loader = FileLoader(self.metadata.file_loader)
        return file_loader.load()
    
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
        features = []
        for extractor in self.feature_extractors:
            feature_value = extractor.compute(rr_window)
            features.append(feature_value)
        return features
    
    def prepare_samples(self):

        history_len = self.window_size + self.stride * (self.seq_len - 1)
        future_len = self.window_size + self.stride * (max(self.horizons) - 1)
        total_len_needed = history_len + future_len

        for record, qrs in self.load_rr_sequences():
            print(f"Processing record: {record.record_name}")
            print(f"Len of processed samples so far: {len(self.samples)}")
            if record is None or qrs is None:
                continue
            rr_intervals = np.diff(qrs.sample)
            seq_len = len(rr_intervals)
            if seq_len < total_len_needed:
                continue
            total_windows = (seq_len - self.window_size) // self.stride + 1
            for i in range(total_windows):
                start_idx = i * self.stride
                end_idx = start_idx + total_len_needed
                if end_idx > seq_len:
                    break
                rr_seq = rr_intervals[start_idx:end_idx]
                
                # Input windows
                rr_windows = []
                for j in range(self.seq_len):
                    w_start = j * self.stride
                    w_end = w_start + self.window_size
                    rr_windows.append(rr_seq[w_start:w_end])

                rr_windows = np.stack(rr_windows)  # (seq_len, window_size)

                # Feature targets
                hrvs = []
                for horizon in self.horizons:
                    t_start = (self.seq_len - 1 + horizon) * self.stride
                    t_end = t_start + self.window_size
                    rr_target = rr_seq[t_start:t_end]
                    hrv_target = self.calculate_features(rr_target)
                    hrvs.append(hrv_target)
                hrvs = np.stack(hrvs)  # (len(horizons), Number of HRV metrics)

                self.samples.append((rr_windows, hrvs))

    def __init__(self, metadata: DatasetMetadata):
        self.metadata = metadata
        self.window_size = metadata.rr_sequence.window_size
        self.stride = metadata.rr_sequence.stride
        self.horizons = metadata.rr_sequence.horizons
        self.seq_len = metadata.rr_sequence.seq_len
        self.samples = []

        self.feature_extractors = self.load_feature_extractors()
        self.prepare_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rr_windows, hrvs = self.samples[idx]
        current_hrv = self.calculate_features(rr_windows[-1])
        return torch.tensor(rr_windows, dtype=torch.float32), torch.tensor(hrvs, dtype=torch.float32), torch.tensor(current_hrv, dtype=torch.float32)