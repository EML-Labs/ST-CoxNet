from torch.utils.data import Dataset
import numpy as np
import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

class RRSequenceCoxDataset(Dataset):
    def __init__(
        self,
        patient_ids,          # list of main record names (e.g., ['p16', 'p18', ...])
        dataset_path,         # path to the .hea/.dat files
        csv_file_name,       # name of the CSV file containing patient metadata (e.g., 'patient_metadata.csv')
        limit_time_to_event=0.5, # Max time to event in hours (e.g., 0.5 hours = 30 minutes)
        window_size=200,
        stride=20,
        seq_len=10,
        rr_scaler=None       # Optional scaler for RR intervals (e.g., RobustScaler)
    ):
        if rr_scaler is None:
            self.rr_scaler = RobustScaler()
        else:
            self.rr_scaler = rr_scaler
        csv_path = os.path.join(dataset_path, csv_file_name)
        df = pd.read_csv(csv_path)
        self.samples = []
        sample_size = seq_len
        num_of_segments = [len(df[df['patient_id'] == patient]) for patient in patient_ids]
        total_segments = sum(num_of_segments) - len(patient_ids) * (sample_size - 1)
        pbar = tqdm(total=total_segments, desc="Processing patients", unit="segment")
        for patient_id in patient_ids:
            patient_df = df[df['patient_id'] == patient_id]
            total_segments = len(patient_df)
            for i in range(0, total_segments - sample_size + 1, 1):
                if patient_df.iloc[i]['EventType'] == 'AF':
                    continue  # Skip sequences that start with an AF event
                time_to_event = patient_df.iloc[i + sample_size - 1]['TimeToEvent']/3600 # Convert seconds to hours
                if time_to_event > limit_time_to_event:  # Skip if time to event exceeds the limit
                    continue  # Skip if time to event exceeds the limit
                event = patient_df.iloc[i + sample_size - 1]['Event']
                rr_window_indexes = range(i, i + sample_size)
                rr_windows = []
                for idx in rr_window_indexes:
                    file_name = f"{patient_id}_{patient_df.iloc[idx]['episode_id']}_{patient_df.iloc[idx]['segment_id']}.npy"
                    rr_window = np.load(os.path.join(dataset_path, file_name))
                    rr_window_scaled = self.rr_scaler.transform(rr_window.reshape(1, -1)).flatten()
                    rr_windows.append(rr_window_scaled)
                rr_windows = np.stack(rr_windows)  # [seq_len, window_size]
                self.samples.append((rr_windows, time_to_event, event))
                pbar.update(1)
        pbar.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rr_windows, time_to_event, event = self.samples[idx]
        return (
            torch.tensor(rr_windows, dtype=torch.float32),  # (T, W)
            torch.tensor(time_to_event, dtype=torch.float32),  # hours
            torch.tensor(event, dtype=torch.float32)        # 0/1
        )
