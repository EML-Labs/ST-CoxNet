import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import wfdb
from tqdm import tqdm
from typing import List, Tuple


class DEDataset(Dataset):
    data:List
    def __init__(self, dataset_path:str,
                 window_size:int=36, overlap:int=9, sampling_rate=128,
                 transform:nn.Module=None):

        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.data = []
        self.dataset_path = dataset_path



        self.load_data()

    def load_data(self):
        pre_afib_path:str = os.path.join(self.dataset_path, 'afpdb_pre_af_wfdb')
        non_afib_path:str = os.path.join(self.dataset_path, 'afpdb_non_af_wfdb')
        afib_path:str = os.path.join(self.dataset_path, 'afpdb_af_wfdb')
        pre_afib_file_names = [os.path.join(pre_afib_path, file).replace('.hea','') for file in os.listdir(pre_afib_path) if file.endswith('.hea')]
        non_afib_file_names = [os.path.join(non_afib_path, file).replace('.hea','') for file in os.listdir(non_afib_path) if file.endswith('.hea')]
        afib_file_names = [os.path.join(afib_path, file).replace('.hea','') for file in os.listdir(afib_path) if file.endswith('.hea')]
        all_file_names = pre_afib_file_names + non_afib_file_names + afib_file_names
        pbar = tqdm(
            all_file_names,
            total=len(all_file_names),
            desc="Processing patients"
        )

        for file_name in pbar:

            qrs = wfdb.rdann(
                file_name,
                'qrs'
            ).sample

            for i in range(0, len(qrs) - self.window_size, self.overlap):
                window = qrs[i:i + self.window_size]
                rr_window = np.diff(window) / self.sampling_rate
                self.data.append(rr_window)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rri = torch.tensor(
            self.data[idx], dtype=torch.float32
        ).unsqueeze(0)  # (C=1, L)

        if self.transform:
            rri = self.transform(rri.unsqueeze(0)).squeeze(0)
        return rri
