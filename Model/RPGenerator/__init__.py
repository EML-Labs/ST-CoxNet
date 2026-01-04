import torch
import torch.nn as nn

class GaussianRP(nn.Module):
    def __init__(self, init_sigma=1.0):
        super().__init__()
        self.log_sigma = nn.Parameter(
            torch.log(torch.tensor(init_sigma))
        )

    def forward(self, x):
        """
        x: (B, C, dim, T)
        return: (B, C, T, T)
        """
        sigma = torch.exp(self.log_sigma)
        
        # 1. Reshape to merge Batch and Channel for calculation
        # x becomes (B*C, dim, T)
        b, c, dim, t = x.shape
        x_reshaped = x.view(b * c, dim, t)
        
        # 2. Permute because cdist expects (Batch, Vector_Count, Vector_Dim)
        # We want distance between Time steps (Vector_Count=T), using features (Vector_Dim=dim)
        x_perm = x_reshaped.transpose(1, 2) # Shape: (B*C, T, dim)

        # 3. Compute Euclidean Distance (L2 norm) efficiently
        # Result shape: (B*C, T, T)
        dist_matrix = torch.cdist(x_perm, x_perm, p=2)
        
        # 4. We need Squared Euclidean for Gaussian formula: dist^2
        dist_sq = dist_matrix.pow(2)
        
        # 5. Apply Gaussian Kernel
        rp = torch.exp(-dist_sq / (2 * sigma ** 2))
        
        # 6. Reshape back to separate Batch and Channel
        rp = rp.view(b, c, t, t)
        
        return rp