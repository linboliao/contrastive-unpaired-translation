"""Dice + focal loss for CUTSegModel's nucleus segmentation head.

Both take raw logits (not sigmoid-ed probabilities) for numerical stability --
Dice sigmoids internally, focal uses binary_cross_entropy_with_logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits).flatten(1)
        target = target.flatten(1)
        intersection = (prob * target).sum(dim=1)
        union = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return loss.mean()
