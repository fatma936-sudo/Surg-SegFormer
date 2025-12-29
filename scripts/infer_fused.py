from __future__ import annotations
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np

from surg_segformer.models import AnatomySegFormer, ToolSegFormerWithSkipDecoder, PriorityWeightedFusion
from surg_segformer.utils import load_checkpoint, ensure_dir

from scripts._cli import load_cfg, add_common_args

def read_rgb(path: str, size: int):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return x

def main():
    ap = argparse.ArgumentParser("Run Surg-SegFormer fused inference (tools + anatomy + fusion)")
    add_common_args(ap)
    ap.add_argument("--anatomy_ckpt", required=True)
    ap.add_argument("--tools_ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/inference")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = cfg["data"]["size"]

    anatomy = AnatomySegFormer(num_classes=cfg["anatomy"]["num_classes"], backbone=cfg["anatomy"]["backbone"]).to(device)
    tools = ToolSegFormerWithSkipDecoder(
        num_classes=cfg["tools"]["num_classes"],
        backbone=cfg["tools"]["backbone"],
        proj_dim=cfg["tools"]["proj_dim"],
        dropout=cfg["tools"]["dropout"],
    ).to(device)

    load_checkpoint(anatomy, args.anatomy_ckpt, map_location=device)
    load_checkpoint(tools, args.tools_ckpt, map_location=device)
    anatomy.eval(); tools.eval()

    fusion = PriorityWeightedFusion(
        tool_background_id_global=cfg["fusion"]["tool_background_id_global"],
        conf_threshold=cfg["fusion"]["conf_threshold"],
        tools_local_to_global=cfg["fusion"].get("tools_local_to_global"),
        anatomy_local_to_global=cfg["fusion"].get("anatomy_local_to_global"),
    )

    x = read_rgb(args.image, size=size).to(device)
    with torch.no_grad():
        a_logits = anatomy(x)
        t_logits = tools(x)
        final = fusion(a_logits, t_logits)[0].cpu().numpy().astype(np.uint8)

    out_dir = ensure_dir(args.out)
    out_path = Path(out_dir) / (Path(args.image).stem + "_final.png")
    cv2.imwrite(str(out_path), final)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
