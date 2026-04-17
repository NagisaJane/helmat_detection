# YOLO26 + GAN + Hybrid Attention for Adverse-Weather PPE Detection

## 1. Overview
This repo provides reproducible scripts for validating:
- B0: YOLO26 baseline
- B1: GAN-only
- B2: Attention-only
- B3: GAN + Hybrid Attention (Full)

Evaluation is performed on an adverse-weather validation split (`val_adverse`).

## 2. Environment
- OS: Windows
- Python: 3.10 (recommended)
- CUDA: optional
- Main deps:
  - torch
  - torchvision
  - ultralytics
  - albumentations
  - opencv-python-headless
  - pyyaml
  - pillow
## dataset link:https://drive.google.com/drive/folders/1igqO6dHc5E0-HXPvKNnuSk-uKldLtxoR?usp=drive_link
Install example:
```bash
pip install torch torchvision ultralytics albumentations opencv-python-headless pyyaml pillow
