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
        preds = preds.float()
        targets = targets.float()

        # BCE
        bce = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')

        probs = torch.sigmoid(preds)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal = self.alpha * (1 - pt) ** self.gamma * bce
        focal_loss = focal.mean()

        # Dice per sample
        preds_sig = torch.sigmoid(preds)

        intersection = (preds_sig * targets).sum(dim=(1, 2, 3))
        union = preds_sig.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        # 🔥 apply dice ONLY where foreground exists
        has_fg = (targets.sum(dim=(1, 2, 3)) > 0).float()

        dice_loss = (dice_loss * has_fg).mean()

        return focal_loss + self.dice_weight * dice_loss


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, smooth=1e-5, pos_weight=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.pos_weight = pos_weight  # optional imbalance control

    def forward(self, preds, targets):
        preds = preds.float()
        targets = targets.float()

        # ---- BCE ----
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=preds.device)
            bce = F.binary_cross_entropy_with_logits(
                preds, targets, pos_weight=pos_weight
            )
        else:
            bce = F.binary_cross_entropy_with_logits(preds, targets)

        # ---- Dice ----
        probs = torch.sigmoid(preds)

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        # keep your good idea: ignore pure background for Dice
        has_fg = (targets.sum(dim=(1, 2, 3)) > 0).float()
        dice_loss = (dice_loss * has_fg).mean()

        return bce + self.dice_weight * dice_loss
