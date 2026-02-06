from pydantic import BaseModel
from enum import IntEnum
from typing import List,Tuple

class FileLoaderMetadata(BaseModel):
    file_path: str

class RRSequenceMetadata(BaseModel):
    window_size: int
    stride: int
    horizons: list[int]
    seq_len: int

class FeatureType(IntEnum):
    LFHF = 1
    RMSSD = 2
    EctopicPercentage = 3
    Alpha1 = 4
    SampleEntropy = 5
    ApproximateEntropy = 6

class DatasetMetadata(BaseModel):
    name: str
    file_loader: FileLoaderMetadata
    rr_sequence: RRSequenceMetadata
    feature_types: List[Tuple[FeatureType, dict]]  # List of (FeatureType, parameters) tuples


