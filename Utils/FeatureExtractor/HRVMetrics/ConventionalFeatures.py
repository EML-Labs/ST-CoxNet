import numpy as np
from scipy.signal import welch

class LFHF:
    def __init__(self, fs=4.0):
        self.fs = fs

    def compute(self, rr_window) -> float:
        """
        Calculates LF/HF Ratio using Welch's method.
        Significant decrease observed before onset of AF[cite: 206].
        """
        # Study uses LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz) [cite: 58]
        freqs, psd = welch(rr_window, fs=self.fs, nperseg=len(rr_window))
        
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
        
        lf_power = np.trapezoid(psd[lf_mask], freqs[lf_mask])
        hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask])
        
        return lf_power / hf_power if hf_power > 0 else 0.0


class RMSSD:
    def compute(self, rr_window) -> float:
        """
        Calculates the Root Mean Square of Successive Differences (RMSSD).
        A reduction implies a change to a more random state[cite: 259].
        """
        rr_diff = np.diff(rr_window)
        return np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) else 0.0


class EctopicPercentage:
    def compute(self, rr_window) -> float:
        """
        Calculates the percentage of ectopic beats.
        Ectopic beats often correlate negatively with complexity[cite: 226].
        """
        if len(rr_window) == 0:
            return 0.0
        
        # Standard clinical definition: a beat differing by >20% from the mean 
        # of the previous beats, or using a fixed threshold.
        # The study focused on 'noise-free qualified sinus beats'[cite: 24, 49].
        
        mean_rr = np.mean(rr_window)
        # Common thresholding for ectopic identification:
        # Any beat < 80% or > 120% of the mean RR interval
        lower_bound = 0.8 * mean_rr
        upper_bound = 1.2 * mean_rr
        
        ectopic_count = np.sum((rr_window < lower_bound) | (rr_window > upper_bound))
        percentage = (ectopic_count / len(rr_window))
        
        return percentage