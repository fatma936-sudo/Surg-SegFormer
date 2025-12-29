from __future__ import annotations
import torch

@torch.no_grad()
def dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-7) -> float:
    """
    Macro Dice over classes (including background by default).
    preds/targets: (B,H,W) integer class ids.
    """
    dice = 0.0
    classes = range(num_classes)
    for c in classes:
        pred_i = (preds == c).float()
        target_i = (targets == c).float()
        inter = (pred_i * target_i).sum(dim=(1, 2))
        denom = pred_i.sum(dim=(1, 2)) + target_i.sum(dim=(1, 2))
        dice += (2.0 * inter + eps) / (denom + eps)
    return (dice / len(classes)).mean().item()
