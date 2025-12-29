from __future__ import annotations
import torch

def logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W)->(B,H,W)"""
    return torch.argmax(logits, dim=1)

@torch.no_grad()
def probs_max(logits: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W)->(B,H,W) max prob per pixel"""
    return torch.softmax(logits, dim=1).max(dim=1).values
