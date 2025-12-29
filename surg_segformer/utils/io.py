from __future__ import annotations
import os
import json
from pathlib import Path
import torch

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))

def load_checkpoint(model: torch.nn.Module, ckpt_path: str, map_location="cpu") -> dict:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        return ckpt
    model.load_state_dict(ckpt)
    return {"state_dict": ckpt}

def save_checkpoint(path: str, model: torch.nn.Module, **extra) -> None:
    payload = {"state_dict": model.state_dict(), **extra}
    torch.save(payload, path)
