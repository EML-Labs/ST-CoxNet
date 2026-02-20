import matplotlib.pyplot as plt
import numpy as np
import torch

class RegressionPlotter:
    def __init__(self):
        self.predictions = []
        self.targets = []

    def update(self, preds:list, targets:list):
        if len(preds) != len(targets):
            raise ValueError("Length of predictions and targets must be the same")
        for i in range(len(preds)):
            self.predictions.append(preds[i])
            self.targets.append(targets[i])

    def plot(self):
        if not self.predictions:
            return None
        
        num_predictors = len(self.predictions[0])
        
        # Reorganize data by predictor
        all_preds = [[] for _ in range(num_predictors)]
        all_targets = [[] for _ in range(num_predictors)]
        
        for sample_idx in range(len(self.predictions)):
            for pred_idx in range(num_predictors):
                pred = self.predictions[sample_idx][pred_idx]
                target = self.targets[sample_idx][pred_idx]
                
                # Convert tensors to numpy/scalar
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().cpu().item() if pred.dim() == 0 else pred.detach().cpu().numpy().flatten()[0]
                if isinstance(target, torch.Tensor):
                    target = target.detach().cpu().item() if target.dim() == 0 else target.detach().cpu().numpy().flatten()[0]
                
                all_preds[pred_idx].append(pred)
                all_targets[pred_idx].append(target)
        
        fig, axes = plt.subplots(1, num_predictors, figsize=(5 * num_predictors, 5))
        if num_predictors == 1:
            axes = [axes]
        
        for i in range(num_predictors):
            preds_array = np.array(all_preds[i])
            targets_array = np.array(all_targets[i])
            
            axes[i].scatter(targets_array, preds_array, alpha=0.5)
            axes[i].set_xlabel("True Values")
            axes[i].set_ylabel("Predicted Values")
            axes[i].set_title(f"Predictor {i+1}")
            
            min_val = np.min(targets_array)
            max_val = np.max(targets_array)
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.tight_layout()
        plt.savefig("regression_plots.png")
        return fig