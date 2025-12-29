from __future__ import annotations
import torch

@torch.no_grad()
def miou_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-7) -> float:
    """
    Mean IoU over classes (including background by default).
    preds/targets: (B,H,W) integer class ids.
    """
    miou = 0.0
    classes = range(num_classes)
    for c in classes:
        pred_i = (preds == c).float()
        target_i = (targets == c).float()
        inter = (pred_i * target_i).sum(dim=(1, 2))
        union = pred_i.sum(dim=(1, 2)) + target_i.sum(dim=(1, 2)) - inter
        miou += (inter + eps) / (union + eps)
    return (miou / len(classes)).mean().item()
