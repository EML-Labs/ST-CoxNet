from torch.utils.data import Dataset
import torch
import numpy as np
from Metadata import FileLoaderMetadata,RRSequenceMetadata
from Utils.Loader.FileLoader import FileLoader

class RRSequenceDataset(Dataset):
    def __init__(self, sampling_rate:int,rr_sequence: RRSequenceMetadata, file_loader: FileLoaderMetadata, feature_extractors:list[dict]):
        self.window_size = rr_sequence.window_size
        self.stride = rr_sequence.stride
        self.horizons = rr_sequence.horizons
        self.seq_len = rr_sequence.seq_len
        self.samples = []

        self.feature_extractors = {
            list(fe.keys())[0]:list(fe.values())[0] for fe in feature_extractors
        }

        self.rr_records = []
        file_loader = FileLoader(file_loader)
        self.patient_ids = []
        id = 0
        for _, qrs in file_loader.load():
            if qrs is None:
                continue
            rr = np.diff(qrs.sample) / sampling_rate
            self.rr_records.append(rr)
            self.patient_ids.append(id)
            id += 1


        self.create_index()

    def create_index(self):
        self.index = []
        history_len = self.window_size + self.stride * (self.seq_len - 1)
        future_len = self.window_size + self.stride * (max(self.horizons) - 1)
        total_len = history_len + future_len

        for rid, rr in enumerate(self.rr_records):
            max_start = len(rr) - total_len
            for start in range(0, max_start + 1, self.stride):
                self.index.append((rid, start))
    
    def calculate_features(self, rr_window):
        for name, ext in self.feature_extractors.items():
            print(f"Calculating {name} for window: {rr_window}")
        return {
            name:ext.compute(rr_window) for name, ext in self.feature_extractors.items()
        }

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
            hrvs.append(list(self.calculate_features(rr_target).values()))

        hrvs = np.array(hrvs)
        current_hrv = list(self.calculate_features(rr_windows[-1]).values())

        return (
            torch.tensor(rr_windows, dtype=torch.float32),
            torch.tensor(hrvs, dtype=torch.float32),
            torch.tensor(current_hrv, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.index)
    
