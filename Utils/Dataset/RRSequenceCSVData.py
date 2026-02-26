import torch
import numpy as np
from typing import List, Dict

from Metadata import FeatureType, RRCSVDataMetadata
from Utils.Dataset.rr_loader import load_csv_records
from Utils.Dataset.rr_windowing import build_csv_index


class RRSequenceCSVData:
    def __init__(self, 
                 metadata: RRCSVDataMetadata
                 ):
        self.seq_len = metadata.seq_len
        self.horizons = metadata.horizons
        self.feature_types = metadata.feature_types
        
        self.records = load_csv_records(
            rri_csv_path=metadata.rri_csv_path,
            features_csv_path=metadata.features_csv_path,
            feature_types=self.feature_types
        )

