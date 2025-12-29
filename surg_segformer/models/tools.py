from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel

class ToolSegFormerWithSkipDecoder(nn.Module):
    """
    SegFormer MiT-B5 encoder + lightweight skip-connection decoder for tool segmentation.
    Paper mentions replacing the original decoder with a lightweight design incorporating skip connections.
    This implementation:
      - pulls 4 hidden states from MiT encoder
      - projects each to a common channel dim
      - upsamples to highest resolution (stage1)
      - concatenates + fuses by convs
    """
    def __init__(
        self,
        num_classes: int,
        backbone: str = "nvidia/mit-b5",
        proj_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained(backbone, output_hidden_states=True)

        # MiT hidden sizes for B5: [64, 128, 320, 512] (stage1..4)
        self.proj1 = nn.Conv2d(64,  proj_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(128, proj_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(320, proj_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(512, proj_dim, kernel_size=1)

        in_ch = proj_dim * 4
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(pixel_values=pixel_values)
        # hidden_states: tuple of 4 tensors (B,HW,C) each at different scales
        hs = enc.hidden_states  # len=4
        # Convert each to (B,C,H,W). Shapes depend on input size.
        feats = []
        for h in hs:
            b, n, c = h.shape
            # infer H,W as square-ish; MiT uses (H/4, W/4) etc, but we can get from encoder output
            # SegformerModel also returns feature_maps sometimes, but hidden_states are token seq.
            # Use encoder's `reshape_last_stage` approach: n = h*w
            # We assume input is divisible by 32 and shapes are consistent.
            hw = int(n**0.5)
            feat = h.transpose(1,2).contiguous().view(b, c, hw, hw)
            feats.append(feat)

        x1, x2, x3, x4 = feats
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)
        x3 = self.proj3(x3)
        x4 = self.proj4(x4)

        target_hw = x1.shape[-2:]
        x2 = F.interpolate(x2, size=target_hw, mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=target_hw, mode="bilinear", align_corners=False)
        x4 = F.interpolate(x4, size=target_hw, mode="bilinear", align_corners=False)

        fused = torch.cat([x1, x2, x3, x4], dim=1)
        fused = self.fuse(fused)
        logits = self.classifier(fused)
        # upsample to input resolution
        logits = F.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        return logits
