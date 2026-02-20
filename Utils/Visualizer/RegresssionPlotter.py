import matplotlib.pyplot as plt
import numpy as np
import torch

class RegressionPlotter:
    def __init__(self):
        self.predictions = []
        self.targets = []

    def update(self, preds:list[torch.Tensor], targets:list[torch.Tensor]):
        if len(preds) != len(targets):
            raise ValueError("Length of predictions and targets must be the same")
        for i in range(len(preds)):
            self.predictions.append(preds[i].cpu().numpy())
            self.targets.append(targets[i].cpu().numpy())

    def plot(self):
        num_predictors = len(self.predictions[0])
        fig, axes = plt.subplots(1, num_predictors, figsize=(5 * num_predictors, 5))
        if num_predictors == 1:
            axes = [axes]
        for i in range(num_predictors):
            axes[i].scatter(self.targets[i], self.predictions[i], alpha=0.5)
            axes[i].set_xlabel("True Values")
            axes[i].set_ylabel("Predicted Values")
            axes[i].set_title(f"Predictor {i+1}")
            axes[i].plot([min(self.targets[i]), max(self.targets[i])], [min(self.targets[i]), max(self.targets[i])], 'r--')
        plt.tight_layout()
        plt.savefig("regression_plots.png")
        return fig