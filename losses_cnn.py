import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # Added import statement

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class classification.

        Parameters:
        - alpha: Tensor or None
            Weights for each class. If None, all classes are treated equally.
        - gamma: float
            Focusing parameter gamma >= 0.
        - reduction: str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            elif not isinstance(alpha, torch.Tensor):
                raise TypeError('alpha must be a list, numpy array, or torch Tensor')
            self.alpha = alpha
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the loss function.

        Parameters:
        - inputs: Tensor
            Predicted logits with shape (batch_size, num_classes).
        - targets: Tensor
            Ground truth labels with shape (batch_size).
        """
        # Compute softmax over the classes
        probs = F.softmax(inputs, dim=1)
        # Get the probabilities corresponding to the target classes
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Compute the logarithm of pt
        log_pt = torch.log(pt + 1e-8)  # Add epsilon to avoid log(0)
        # Compute the focal loss term
        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * log_pt

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            loss = loss * at

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
