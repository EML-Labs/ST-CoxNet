import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import neurokit2 as nk
import numpy as np
from Utils.FeatureExtractor.HRVMetrics.HRVTimeFeatures import HRVTimeFeatures

data = nk.data("bio_resting_5min_100hz")
data.head()  # Print first 5 rows
peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)
hrv_time = nk.hrv_time(peaks, sampling_rate=100, show=True)
print(hrv_time.columns.tolist())
print()
print(hrv_time.head())

print(peaks)

print("\nTesting HRVTimeFeatures class when the whole ECG is given")
hrv_time_features = HRVTimeFeatures(data["ECG"], fs=100, rri_given=False)
features = hrv_time_features.compute()
print("\nExtracted HRV Time-Domain Features:")
for key, value in features.items():
    print(f"{key}: {value}")

print("\nTesting HRVTimeFeatures class when QRS peaks are given")
r_peaks = peaks["ECG_R_Peaks"].values
peak_indices = np.where(r_peaks == 1)[0]
print(f"R-peak indices: {peak_indices} ")
rri = np.diff(peak_indices) * 1000 / 100 # Convert to miliseconds (assuming fs=100Hz)
print(f"RR intervals (s): {rri}")
hrv_time_features = HRVTimeFeatures(rri, fs=100, rri_given=True)
features = hrv_time_features.compute()
print("\nExtracted HRV Time-Domain Features:")
for key, value in features.items():
    print(f"{key}: {value}")

print("\nHelper information for HRV Time-Domain Features:")
helper_info = hrv_time_features.__helper__()
for key, value in helper_info.items():
    print(f"{key}: {value}")

