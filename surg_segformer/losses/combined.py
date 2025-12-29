from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    """
    Multi-class Tversky loss. Paper uses alpha=0.7, beta=0.3 to penalize false negatives.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,C,H,W)
        targets: (B,H,W) int class ids
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        tgt_1h = F.one_hot(targets.long(), num_classes=num_classes).permute(0,3,1,2).float()

        dims = (0,2,3)
        tp = (probs * tgt_1h).sum(dims)
        fp = (probs * (1 - tgt_1h)).sum(dims)
        fn = ((1 - probs) * tgt_1h).sum(dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - tversky.mean()

class CombinedLoss(nn.Module):
    """
    Combined loss from the paper: alpha_weight * Tversky + (1-alpha_weight) * CE.
    Note: alpha_weight is *not* the same as Tversky alpha. (Yep, notation collision in literature.)
    """
    def __init__(
        self,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        alpha_weight: float = 0.5,
        ce_weight: float | None = None,
    ):
        super().__init__()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.alpha_weight = alpha_weight
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha_weight * self.tversky(logits, targets) + (1.0 - self.alpha_weight) * self.ce(logits, targets)
