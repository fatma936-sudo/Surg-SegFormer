from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from surg_segformer.data import EndoVis2018SegDataset, EndoVis2018Paths
from surg_segformer.losses import CombinedLoss
from surg_segformer.engine import train_loop

from surg_segformer.models import ToolSegFormerWithSkipDecoder

from scripts._cli import load_cfg, add_common_args

def main():
    ap = argparse.ArgumentParser("Train tools model")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels.json used for consistent class ids
    labels_json = Path(cfg["labels_json"])
    labels = json.loads(labels_json.read_text())
    labels = sorted(labels, key=lambda x: x["classid"])

    bg_id = cfg["labels"]["background_id"]
    tool_ids = cfg["labels"]["tool_ids"]
    anatomy_ids = cfg["labels"]["anatomy_ids"]

    paths = EndoVis2018Paths(root=cfg["data"]["root"])

    train_ds = EndoVis2018SegDataset(
        paths, split="train", size=cfg["data"]["size"],
        augment=cfg["data"]["augment"], normalize=True,
        task="tools",
        background_id=bg_id, tool_ids=tool_ids, anatomy_ids=anatomy_ids,
        remap_to_contiguous=cfg["data"].get("remap_to_contiguous", True),
    )
    val_ds = EndoVis2018SegDataset(
        paths, split="val", size=cfg["data"]["size"],
        augment=False, normalize=True,
        task="tools",
        background_id=bg_id, tool_ids=tool_ids, anatomy_ids=anatomy_ids,
        remap_to_contiguous=cfg["data"].get("remap_to_contiguous", True),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["train"]["num_workers"], pin_memory=True)

    model = ToolSegFormerWithSkipDecoder(num_classes=cfg["model"]["num_classes"], backbone=cfg["model"]["backbone"], proj_dim=cfg["model"]["proj_dim"], dropout=cfg["model"]["dropout"]).to(device)

    criterion = CombinedLoss(
        tversky_alpha=cfg["loss"]["tversky_alpha"],
        tversky_beta=cfg["loss"]["tversky_beta"],
        alpha_weight=cfg["loss"]["alpha_weight"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        out_dir=cfg["train"]["out_dir"],
        num_epochs=cfg["train"]["epochs"],
        num_classes=cfg["model"]["num_classes"],
        use_amp=cfg["train"].get("amp", True),
    )

if __name__ == "__main__":
    main()
