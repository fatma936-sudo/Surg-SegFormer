from __future__ import annotations
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class AnatomySegFormer(nn.Module):
    """
    SegFormer MiT-B2 fine-tuned for anatomical structures segmentation.
    """
    def __init__(self, num_classes: int, backbone: str = "nvidia/mit-b2"):
        super().__init__()
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.net(pixel_values=pixel_values)
        return out.logits  # (B,C,H,W)
