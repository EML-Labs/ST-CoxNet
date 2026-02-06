import numpy as np

class ApproximateEntropy:
    def __init__(self, m:int=2, r:float=0.0):
        self.m = m
        self.r = r

    def _phi(self, data: np.ndarray, m: int) -> np.ndarray:
        N = len(data)
        count = np.zeros(N - m + 1)
        for i in range(N - m + 1):
            template = data[i:i + m]
            for j in range(N - m + 1):
                if np.linalg.norm(template - data[j:j + m]) < self.r:
                    count[i] += 1
        return count / (N - m + 1)

    def compute(self, data: np.ndarray) -> float:
        phi_m = self._phi(data, self.m)
        phi_m_plus_1 = self._phi(data, self.m + 1)
        return np.log(np.sum(phi_m) / np.sum(phi_m_plus_1))
    

class SampleEntropy:
    def __init__(self, m: int = 2,r: float = 0.0):
        self.m = m
        self.r = r

    def _count_matches(self, data: np.ndarray, m: int, r: float) -> int:
        N = len(data)
        count = 0
        # Create templates of length m
        templates = np.array([data[i:i + m] for i in range(N - m + 1)])
        
        for i in range(len(templates) - 1):
            # Use Chebyshev distance (max absolute difference) [cite: 65]
            # Comparison excludes self-matches to remain unbiased [cite: 185]
            distances = np.max(np.abs(templates[i+1:] - templates[i]), axis=1)
            count += np.sum(distances <= r)
        return count

    def compute(self, data: np.ndarray) -> float:
        """
        Calculates Sample Entropy. 
        A linear reduction is a hallmark of altered HR dynamics preceding AF[cite: 232, 293].
        """
        N = len(data)
        if N <= self.m:
            return 0.0

        # The study uses r = 20% of the SDNN 
        r = self.r * np.std(data)
        if r == 0:
            return 0.0

        A = self._count_matches(data, self.m + 1, r)
        B = self._count_matches(data, self.m, r)

        # Handling A=0 to avoid log(0) [RuntimeWarning]
        if B == 0:
            return 0.0
        if A == 0:
            # Return maximum complexity for the given data length if no matches found
            return -np.log(1 / (B + 1e-6))
            
        return float(-np.log(A / B))
    


# import numpy as np

# def calculate_apen(self, rr_window, m=2, r=0.0):
#     """
#     Calculates Approximate Entropy (ApEn).
#     A lower value reflects higher degree of regularity[cite: 186].
#     """
#     N = len(rr_window)
#     if N <= m + 1:
#         return 0.0

#     def _phi(m):
#         x = np.array([rr_window[i:i + m] for i in range(N - m + 1)])
#         # Count similar patterns within distance r
#         C = np.zeros(len(x))
#         for i in range(len(x)):
#             # Use Chebyshev distance as per standard ApEn [cite: 183]
#             distances = np.max(np.abs(x - x[i]), axis=1)
#             C[i] = np.sum(distances <= r) / (N - m + 1)
#         return np.sum(np.log(C)) / (N - m + 1)

#     return abs(_phi(m) - _phi(m + 1))

# def calculate_sampen(self, rr_window, m=2, r=0.0):
#     """
#     Calculates Sample Entropy (SampEn).
#     Found to have a better probability value (p=0.003) for PAF prediction.
#     """
#     N = len(rr_window)
#     if N <= m + 1:
#         return 0.0

#     def _count_matches(m):
#         x = np.array([rr_window[i:i + m] for i in range(N - m)])
#         matches = 0
#         # SampEn does not count self-matches [cite: 185]
#         for i in range(len(x)):
#             distances = np.max(np.abs(x[i+1:] - x[i]), axis=1)
#             matches += np.sum(distances <= r)
#         return matches

#     A = _count_matches(m + 1)
#     B = _count_matches(m)
    
#     if A == 0 or B == 0:
#         return 0.0
#     return -np.log(A / B)

# def calculate_lfhf_ratio(self, rr_window, fs=4.0):
#     """
#     Calculates LF/HF Ratio using Welch's method.
#     Significant decrease observed before onset of AF[cite: 206].
#     """
#     from scipy.signal import welch
    
#     # Study uses LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz) [cite: 58]
#     freqs, psd = welch(rr_window, fs=fs, nperseg=len(rr_window))
    
#     lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
#     hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
    
#     lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
#     hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
    
#     return lf_power / hf_power if hf_power > 0 else 0.0