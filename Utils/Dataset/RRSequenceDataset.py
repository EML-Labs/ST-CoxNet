from torch.utils.data import Dataset
import torch
import numpy as np
from Metadata import FileLoaderMetadata, RRSequenceMetadata

from Utils.Dataset.rr_loader import load_rr_records
from Utils.Dataset.rr_windowing import build_index, build_rr_windows
from Utils.Dataset.hrv_targets import build_targets


class RRSequenceDataset(Dataset):
    def __init__(
        self,
        sampling_rate: int,
        rr_sequence: RRSequenceMetadata,
        file_loader: FileLoaderMetadata,
        feature_extractors: list[dict],
    ):
        self.window_size = rr_sequence.window_size
        self.stride = rr_sequence.stride
        self.horizons = rr_sequence.horizons
        self.seq_len = rr_sequence.seq_len

        self.feature_extractors = {
            list(fe.keys())[0]: list(fe.values())[0] for fe in feature_extractors
        }

        self.rr_records, self.patient_ids = load_rr_records(
            sampling_rate=sampling_rate,
            file_loader_metadata=file_loader,
        )

        self.index = build_index(
            rr_records=self.rr_records,
            window_size=self.window_size,
            stride=self.stride,
            horizons=self.horizons,
            seq_len=self.seq_len,
        )

    def __getitem__(self, idx):
        rid, start = self.index[idx]
        rr = self.rr_records[rid]

        total_len = self.window_size + self.stride * (
            self.seq_len - 1 + max(self.horizons)
        )
        rr_seq = rr[start : start + total_len]

        rr_windows = build_rr_windows(
            rr_seq=rr_seq,
            window_size=self.window_size,
            stride=self.stride,
            seq_len=self.seq_len,
        )

        hrvs, current_hrv = build_targets(
            rr_seq=rr_seq,
            window_size=self.window_size,
            stride=self.stride,
            horizons=self.horizons,
            seq_len=self.seq_len,
            feature_extractors=self.feature_extractors,
        )

        return (
            torch.tensor(rr_windows, dtype=torch.float32),
            torch.tensor(hrvs, dtype=torch.float32),
            torch.tensor(np.array(current_hrv), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.index)
    
