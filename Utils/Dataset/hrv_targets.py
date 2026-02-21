from __future__ import annotations

from typing import Dict, List

import numpy as np


def calculate_features(rr_window: np.ndarray, feature_extractors: Dict[str, object]) -> Dict[str, float]:
    return {
        name: ext.compute(rr_window) for name, ext in feature_extractors.items()
    }


def build_targets(
    rr_seq: np.ndarray,
    window_size: int,
    stride: int,
    horizons: list[int],
    seq_len: int,
    feature_extractors: Dict[str, object],
) -> tuple[np.ndarray, List[float]]:
    """
    Compute HRV targets for each future horizon and the current-window HRV.
    Returns:
        hrvs: [H, num_metrics]
        current_hrv: list[num_metrics]
    """
    hrvs = []
    for h in horizons:
        t_start = (seq_len - 1 + h) * stride
        t_end = t_start + window_size
        rr_target = rr_seq[t_start:t_end]
        hrvs.append(list(calculate_features(rr_target, feature_extractors).values()))

    hrvs_arr = np.array(hrvs)
    current_hrv = list(
        calculate_features(
            rr_seq[(seq_len - 1) * stride : (seq_len - 1) * stride + window_size],
            feature_extractors,
        ).values()
    )
    return hrvs_arr, current_hrv

