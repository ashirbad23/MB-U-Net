import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: [B, 1, H, W]
        # targets: [B, 1, H, W]

        preds = preds.float()
        targets = targets.float()

        # ---- BCE with logits ----
        bce = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')

        # ---- Focal weight ----
        probs = torch.sigmoid(preds)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal = self.alpha * (1 - pt) ** self.gamma * bce
        focal_loss = focal.mean()

        # ---- Dice ----
        preds = torch.sigmoid(preds)

        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice_loss = 1 - ((2 * intersection + self.smooth) / (union + self.smooth)).mean()

        # ---- Final ----
        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss