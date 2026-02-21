import numpy as np
from Utils.FeatureExtractor.Base import BaseExtractor

class ApproximateEntropy(BaseExtractor):
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
    

class SampleEntropy(BaseExtractor):
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
    