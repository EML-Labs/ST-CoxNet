import numpy as np
from scipy.signal import welch
from Utils.FeatureExtractor.Base import BaseExtractor

class TraditionalHRVFeatures(BaseExtractor):
    # Here time mean in s or ms. If you need in ms, just multiply the result by 1000.
    def __init__(self, rr_window, time, fs):
        # The HRV features are calculated in 
        self.rr_window = rr_window*time
        self.time = time
        self.len_window = len(rr_window)
        self.diff = np.diff(self.rr_window)
        self.rr1 = self.rr_window[:-1]
        self.rr2 = self.rr_window[1:]
        self.fs = fs
        self.bin_width = 1/self.fs
        self.bins = int((np.max(self.rr_window) - np.min(self.rr_window)) / self.bin_width)
        self.hist, self.bin_edges = np.histogram(self.rr_window, bins=self.bins)
    
    def mean_rr(self) -> float:
        return np.mean(self.rr_window) if self.len_window > 0 else 0.0
    
    def rmssd(self) -> float:
        return np.sqrt(np.mean(self.diff ** 2)) if len(self.diff) > 0 else 0.0
    
    def sdnn(self) -> float:
        return np.std(self.rr_window, ddof=1) if self.len_window > 1 else 0.0
    
    def sdann(self) -> float:
        if np.sum(self.rr_window) >= 300*self.time:  # At least 5 minutes of data
            return np.std(self.rr_window, ddof=1)
        else:
            return 0.0
    
    def sdsd(self) -> float:
        return np.std(self.diff, ddof=1) if len(self.diff) > 1 else 0.0
    
    def sd1(self) -> float:
        return np.sqrt(0.5 * np.var(self.rr2 - self.rr1, ddof=1)) if len(self.rr_window) > 1 else 0.0
    
    def sd2(self) -> float:
        return np.sqrt(0.5 * np.var(self.rr2 + self.rr1, ddof=1)) if len(self.rr_window) > 1 else 0.0
    
    def sd1_sd2_ratio(self) -> float:
        sd1 = self.sd1()
        sd2 = self.sd2()
        return sd1 / sd2 if sd2 > 0 else 0.0
    
    def cv(self) -> float:
        mean = self.mean_rr()
        std = self.sdnn()
        return std / mean if mean > 0 else 0.0
    
    def nn50(self) -> int:
        return np.sum(np.abs(self.diff)/ self.time * 100 > 50) if self.len_window > 1 else 0
    
    def pnn50(self) -> float:
        return self.nn50() / (self.len_window - 1) if self.len_window > 1 else 0.0

    def baseline_width(self) -> float:
        peak_bin = np.argmax(self.hist)
        left_bin = peak_bin
        bin_centres = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        while left_bin > 0 and self.hist[left_bin] > 0:
            left_bin -= 1
        N = bin_centres[left_bin] if left_bin >= 0 else bin_centres[0]
        right_bin = peak_bin
        while right_bin < len(self.hist) - 1 and self.hist[right_bin] > 0:
            right_bin += 1
        M = bin_centres[right_bin] if right_bin < len(self.hist) else bin_centres[-1]
        return M - N
    
    def triangular_index(self) -> float:
        return self.len_window / np.max(self.hist) if np.max(self.hist) > 0 else 0.0

    
    def compute(self) -> float:
        features = {
            "mean_rr": self.mean_rr(),
            "rmssd": self.rmssd(),
            "sdnn": self.sdnn(),
            "sdann": self.sdann(),
            "sdsd": self.sdsd(),
            "sd1": self.sd1(),
            "sd2": self.sd2(),
            "sd1_sd2_ratio": self.sd1_sd2_ratio(),
            "cv": self.cv(),
            "nn50": self.nn50(),
            "pnn50": self.pnn50(),
            "baseline_width": self.baseline_width(),
            "triangular_index": self.triangular_index()
        }        
        return features
