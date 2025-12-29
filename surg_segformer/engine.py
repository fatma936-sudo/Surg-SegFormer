from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from surg_segformer.metrics import dice_score, miou_score
from surg_segformer.models.common import logits_to_pred
from surg_segformer.utils import save_checkpoint, ensure_dir

@dataclass
class EpochStats:
    loss: float
    dice: float
    miou: float

def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    num_classes: int = 2,
    use_amp: bool = True,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    total_miou = 0.0
    n = 0

    pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)
    for batch in pbar:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if is_train:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            pred = logits_to_pred(logits)
            total_dice += dice_score(pred, y, num_classes=num_classes)
            total_miou += miou_score(pred, y, num_classes=num_classes)
            total_loss += loss.item()
            n += 1
            pbar.set_postfix({"loss": total_loss/n, "miou": total_miou/n})

    return EpochStats(loss=total_loss/n, dice=total_dice/n, miou=total_miou/n)

def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    out_dir: str,
    num_epochs: int,
    num_classes: int,
    use_amp: bool = True,
) -> None:
    out = ensure_dir(out_dir)
    best = -1.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, num_epochs+1):
        tr = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, num_classes=num_classes, use_amp=use_amp)
        va = run_one_epoch(model, val_loader, criterion, device, optimizer=None, scaler=None, num_classes=num_classes, use_amp=use_amp)

        print(f"Epoch {epoch}/{num_epochs} | train: loss {tr.loss:.4f} dice {tr.dice:.4f} miou {tr.miou:.4f} | val: loss {va.loss:.4f} dice {va.dice:.4f} miou {va.miou:.4f}")

        save_checkpoint(str(out/"last.pt"), model, epoch=epoch, val_miou=va.miou)
        if va.miou > best:
            best = va.miou
            save_checkpoint(str(out/"best.pt"), model, epoch=epoch, val_miou=va.miou)
            print(f"  ✓ best updated: {best:.4f}")
