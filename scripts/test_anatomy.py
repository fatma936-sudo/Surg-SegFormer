from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np

from surg_segformer.data import EndoVis2018SegDataset, EndoVis2018Paths
from surg_segformer.metrics import dice_score, miou_score
from surg_segformer.models.common import logits_to_pred
from surg_segformer.utils import ensure_dir, load_checkpoint

from surg_segformer.models import AnatomySegFormer

from scripts._cli import load_cfg, add_common_args

def main():
    ap = argparse.ArgumentParser("Test anatomy model")
    add_common_args(ap)
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (best.pt).")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_json = Path(cfg["labels_json"])
    labels = json.loads(labels_json.read_text())
    labels = sorted(labels, key=lambda x: x["classid"])

    bg_id = cfg["labels"]["background_id"]
    tool_ids = cfg["labels"]["tool_ids"]
    anatomy_ids = cfg["labels"]["anatomy_ids"]

    paths = EndoVis2018Paths(root=cfg["data"]["root"])
    ds = EndoVis2018SegDataset(
        paths, split=args.split, size=cfg["data"]["size"],
        augment=False, normalize=True,
        task="anatomy",
        background_id=bg_id, tool_ids=tool_ids, anatomy_ids=anatomy_ids,
        remap_to_contiguous=cfg["data"].get("remap_to_contiguous", True),
    )
    loader = DataLoader(ds, batch_size=cfg["test"]["batch_size"], shuffle=False,
                        num_workers=cfg["test"]["num_workers"], pin_memory=True)

    model = AnatomySegFormer(num_classes=cfg["model"]["num_classes"], backbone=cfg["model"]["backbone"]).to(device)
    load_checkpoint(model, args.ckpt, map_location=device)
    model.eval()

    out_dir = ensure_dir(cfg["test"]["out_dir"])
    td, ti, n = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)

            logits = model(x)
            pred = logits_to_pred(logits)

            td += dice_score(pred, y, cfg["model"]["num_classes"])
            ti += miou_score(pred, y, cfg["model"]["num_classes"])
            n += 1

            if args.save_preds:
                for p, img_path in zip(pred.cpu().numpy(), batch["image_path"]):
                    name = Path(img_path).stem + ".png"
                    cv2.imwrite(str(Path(out_dir) / name), p.astype(np.uint8))

    print({"dice": td/n, "miou": ti/n, "n_batches": n})

if __name__ == "__main__":
    main()
