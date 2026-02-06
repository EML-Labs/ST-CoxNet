from torch.utils.data import Dataset
import torch
import numpy as np
from Configs import WINDOW_SIZE, STRIDE, HORIZONS, SEQUENCE_LENGTH

class RRSequenceDataset(Dataset):
    def __init__(self, rr_sequences, window_size=WINDOW_SIZE, stride=STRIDE, horizons=HORIZONS, seq_len=SEQUENCE_LENGTH):

        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons
        self.seq_len = seq_len
        self.samples = []

        history_len = window_size + stride * (seq_len - 1)
        future_len = window_size + stride * (max(horizons) - 1)

        total_len_needed = history_len + future_len

        for rr_seq in rr_sequences:
            rr_seq = np.asarray(rr_seq, dtype=np.float32)
            seq_total_len = len(rr_seq)

            total_windows = (seq_total_len - window_size) // stride + 1

            for i in range(total_windows):
                start_idx = i * stride
                end_idx = start_idx + total_len_needed
                if end_idx > seq_total_len:
                    break

                rr_subseq = rr_seq[start_idx:end_idx]

                # -------- INPUT WINDOWS (T=10, W=50) --------
                rr_windows = []
                for j in range(self.seq_len):
                    w_start = j * stride
                    w_end = w_start + window_size
                    rr_windows.append(rr_subseq[w_start:w_end])

                rr_windows = np.stack(rr_windows)  # (seq_len, window_size)
                # -------- HRV TARGETS --------

                hrvs = []
                for horizon in horizons:
                    t_start = (self.seq_len - 1 + horizon) * stride
                    t_end = t_start + window_size
                    rr_target = rr_subseq[t_start:t_end]
                    hrv_target = self.compute_hrv_metrics(rr_target)
                    hrvs.append(hrv_target)

                hrvs = np.stack(hrvs)  # (len(horizons),Number of HRV metrics)

                self.samples.append((rr_windows, hrvs))

    def compute_hrv_metrics(self, rr_window):
        rr_diff = np.diff(rr_window)
        rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) else 0.0
        sdnn = np.std(rr_window)
        sd1 = np.std(rr_diff) / np.sqrt(2) if len(rr_diff) else 0.0
        sd2_val = 2 * sdnn ** 2 - sd1 ** 2
        sd2 = np.sqrt(sd2_val) if sd2_val > 0 else 0.0

        hrv = np.array([rmssd, sdnn, sd1, sd2], dtype=np.float32)
        return np.log1p(hrv)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rr_windows, hrvs = self.samples[idx]
        current_hrv = self.compute_hrv_metrics(rr_windows[-1])
        return torch.tensor(rr_windows, dtype=torch.float32), torch.tensor(hrvs, dtype=torch.float32), torch.tensor(current_hrv, dtype=torch.float32)