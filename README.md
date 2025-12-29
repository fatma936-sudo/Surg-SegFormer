# Surg-SegFormer (clean GitHub-ready code)

This repo is a **structured codebase** for a dual-model surgical scene segmentation pipeline:

- **SegAnatomy**: `SegFormer MiT-B2` fine-tuned for anatomical objects
- **SegTool**: `SegFormer MiT-B5` encoder with a **lightweight skip-connection decoder** for surgical tools
- **Fusion**: a priority-weighted conditional fusion operator that combines both outputs into one final mask

The structure follows the paper’s high-level design:
- dual SegFormer instances (B2 anatomy + B5 tools with custom decoder) citeturn1view0
- hybrid **Tversky + Cross Entropy** combined loss with **Tversky α=0.7, β=0.3** citeturn1view0
- geometric augmentations (flips / crops / rotations) citeturn1view0

> Note: the paper names the fusion as “priority-weighted conditional fusion” but does not fully specify the exact equation in the public PDF, so this repo provides a sane, configurable implementation you can adapt. citeturn1view0

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2) Dataset layout

Default expected folder structure:

```
DATA_ROOT/
  images/
    train/*.png
    val/*.png
    test/*.png
  masks/
    train/*.png
    val/*.png
    test/*.png
```

Masks must be **single-channel** images with integer class ids `0..C-1`.

If your EndoVis layout differs, edit the `configs/*.yaml` paths.

## 3) Train

### Anatomy (MiT-B2)

```bash
python scripts/train_anatomy.py --config configs/anatomy.yaml
```

### Tools (MiT-B5 + skip decoder)

```bash
python scripts/train_tools.py --config configs/tools.yaml
```

Checkpoints:
- `outputs/*/best.pt`
- `outputs/*/last.pt`

## 4) Test

```bash
python scripts/test_anatomy.py --config configs/anatomy.yaml --ckpt outputs/anatomy/best.pt
python scripts/test_tools.py   --config configs/tools.yaml   --ckpt outputs/tools/best.pt
```

Add `--save_preds` to dump per-image predicted masks.

## 5) Fused inference (final mask)

```bash
python scripts/infer_fused.py   --config configs/fused_infer.yaml   --anatomy_ckpt outputs/anatomy/best.pt   --tools_ckpt outputs/tools/best.pt   --image path/to/frame.png   --out outputs/inference
```

This produces `*_final.png` masks.

## Where to plug your existing notebook code

You uploaded:
- custom decoder draft
- training/testing scripts
- metrics

I kept **the repo structure** clean and separated, so it’s easy to drop your exact logic in:
- `surg_segformer/models/tools.py`  (your custom decoder)
- `surg_segformer/losses/combined.py`
- `surg_segformer/data/endovis.py`
- `scripts/*.py`

If you share your **full notebooks / full decoder code (not truncated with `...`)**, I can refactor it into this repo directly with a 1:1 match.

## Citation

If you use this repo, please cite the original paper:

**Surg-SegFormer: A Dual Transformer-Based Model for Holistic Surgical Scene Segmentation**. citeturn1view0

## EndoVis2018 Dataset Structure

This codebase follows the **EndoVis2018** sequence-based directory layout by default:

```
DATA_ROOT/
├── train/
│   ├── seq_1/
│   │   ├── left_frames/
│   │   │   ├── frame000.png
│   │   │   └── ...
│   │   └── labels/
│   │       ├── frame000.png
│   │       └── ...
│   ├── seq_2/
│   └── ...
├── val/
│   └── seq_*/
└── test/
    └── seq_*/
```

Users can adapt the loader for other datasets by editing:
`./surg_segformer/data/endovis.py`

## Classes (from labels.json)

The repo reads class IDs from `labels.json` (included in this repo). fileciteturn0file0L1-L64

**Global class ids:**
- 0: background-tissue
- Tools: [1, 2, 3, 6, 7, 8, 9, 11]
- Anatomy: [4, 5, 10]

During training:
- **Tools model** is trained **only** on tool classes (all anatomy pixels are mapped to background).
- **Anatomy model** is trained **only** on anatomy classes (all tool pixels are mapped to background).

At inference:
- Both predictions are mapped back to the **global ids** and fused into a single final mask.
