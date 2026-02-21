from __future__ import annotations

from typing import List, Tuple

import numpy as np

from Metadata import FileLoaderMetadata
from Utils.Loader.FileLoader import FileLoader


def load_rr_records(
    sampling_rate: int,
    file_loader_metadata: FileLoaderMetadata,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load RR-interval sequences and associated patient IDs from raw QRS detections.
    """
    rr_records: List[np.ndarray] = []
    patient_ids: List[int] = []

    file_loader = FileLoader(file_loader_metadata)
    pid = 0
    for _, qrs in file_loader.load():
        if qrs is None:
            continue
        rr = np.diff(qrs.sample) / sampling_rate
        rr_records.append(rr)
        patient_ids.append(pid)
        pid += 1

    return rr_records, patient_ids

