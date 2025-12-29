from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except Exception:
    A = None


@dataclass
class EndoVis2018Paths:
    """
    EndoVis2018 expected layout (sequence-based):

    root/
      train/
        seq_1/
          left_frames/*.png
          labels/*.png
        ...
      val/
        seq_x/...
      test/
        seq_y/...

    You can adapt this easily for other datasets by modifying the directory traversal in this file.
    """
    root: str
    frames_dirname: str = "left_frames"
    labels_dirname: str = "labels"

    def split_dir(self, split: str) -> Path:
        return Path(self.root) / split


class EndoVis2018SegDataset(Dataset):
    def __init__(
        self,
        paths: EndoVis2018Paths,
        split: str,
        size: int = 512,
        augment: bool = False,
        normalize: bool = True,
        task: str = "all",  # "tools" | "anatomy" | "all"
        # global label ids from labels.json
        background_id: int = 0,
        tool_ids: Optional[List[int]] = None,
        anatomy_ids: Optional[List[int]] = None,
        remap_to_contiguous: bool = True,
        image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        self.paths = paths
        self.split = split
        self.size = int(size)
        self.augment = augment
        self.normalize = normalize
        self.task = task
        self.background_id = int(background_id)
        self.tool_ids = set(tool_ids or [])
        self.anatomy_ids = set(anatomy_ids or [])
        self.remap_to_contiguous = remap_to_contiguous

        split_dir = self.paths.split_dir(split)
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        samples = []
        for seq_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            img_dir = seq_dir / self.paths.frames_dirname
            msk_dir = seq_dir / self.paths.labels_dirname
            if not img_dir.exists() or not msk_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() in image_exts:
                    msk_path = msk_dir / (img_path.stem + ".png")
                    if msk_path.exists():
                        samples.append((img_path, msk_path))
        if not samples:
            raise RuntimeError(f"No image/mask pairs found in {split_dir} with expected EndoVis2018 layout.")

        self.samples = samples

        # Task-specific contiguous remapping
        # - tools model: background + tool classes only
        # - anatomy model: background + anatomy classes only
        # - all: keep global ids (no remap)
        if self.task not in {"tools", "anatomy", "all"}:
            raise ValueError("task must be one of: tools | anatomy | all")

        if self.task == "tools" and self.remap_to_contiguous:
            keep = [self.background_id] + sorted(self.tool_ids)
            self.id_map = {gid: i for i, gid in enumerate(keep)}  # global -> local
            self.inv_id_map = {i: gid for gid, i in self.id_map.items()}  # local -> global
        elif self.task == "anatomy" and self.remap_to_contiguous:
            keep = [self.background_id] + sorted(self.anatomy_ids)
            self.id_map = {gid: i for i, gid in enumerate(keep)}
            self.inv_id_map = {i: gid for gid, i in self.id_map.items()}
        else:
            self.id_map = None
            self.inv_id_map = None

        # Augmentations (geometric family; can be tuned)
        if A is None:
            self.aug = None
        else:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.10, rotate_limit=20,
                    border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=self.background_id, p=0.7
                ),
                A.RandomResizedCrop(height=self.size, width=self.size, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.5),
            ]) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, path: Path) -> np.ndarray:
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if m.ndim == 3:
            m = m[..., 0]
        return m.astype(np.int64)

    def _apply_task_filter(self, mask: np.ndarray) -> np.ndarray:
        if self.task == "all":
            return mask

        if self.task == "tools":
            # Set all non-tool, non-background pixels to background
            keep = self.tool_ids | {self.background_id}
            m = mask.copy()
            m[~np.isin(m, list(keep))] = self.background_id
        else:  # anatomy
            keep = self.anatomy_ids | {self.background_id}
            m = mask.copy()
            m[~np.isin(m, list(keep))] = self.background_id

        # Optional remap to contiguous ids for stable num_classes
        if self.id_map is not None:
            out = np.full_like(m, fill_value=self.id_map[self.background_id])
            for gid, lid in self.id_map.items():
                out[m == gid] = lid
            return out
        return m

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, msk_path = self.samples[idx]
        img = self._read_image(img_path)
        mask = self._read_mask(msk_path)

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.aug is not None:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        mask = self._apply_task_filter(mask)

        img = img.astype(np.float32) / 255.0
        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std

        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask).long()
        return {"pixel_values": img_t, "labels": mask_t, "image_path": str(img_path)}

    @property
    def num_classes(self) -> int:
        if self.task == "all" or not self.remap_to_contiguous:
            # unknown here; user sets via config
            return -1
        return len(self.inv_id_map)
