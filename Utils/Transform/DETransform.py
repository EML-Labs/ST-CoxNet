import torch

import torch

class DelayEmbeddingTransform:
    def __init__(self, dim: int = 2, delay: int = 3):
        self.dim = dim
        self.delay = delay

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        return: (B, C, dim, T)
        """
        B, C, L = x.shape
        T = L - (self.dim - 1) * self.delay

        if T <= 0:
            raise ValueError("Time series too short for given dim and delay")

        de = torch.zeros((B, C, self.dim, T), device=x.device, dtype=x.dtype)

        for i in range(self.dim):
            de[:, :, i, :] = x[:, :, i * self.delay : i * self.delay + T]

        return de
