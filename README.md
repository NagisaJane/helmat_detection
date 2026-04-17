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
### Create environment (Conda)
conda env create -f environment.yml
conda activate torch_env_py310

### Or with pip
pip install -r requirements.txt
### dataset link:https://drive.google.com/drive/folders/1igqO6dHc5E0-HXPvKNnuSk-uKldLtxoR?usp=drive_link
Install example:
```bash
pip install torch torchvision ultralytics albumentations opencv-python-headless pyyaml pillow
```
## Reproducibility

### 1) Clone
```bash
git clone <https://github.com/NagisaJane/helmat_detection>
cd X-AnyLabeling-main
```
### 2) Create environment
```
conda env create -f environment.yml
conda activate torch_env_py310
```
or
```
pip install -r requirements.txt
```
### 3) Prepare datasets
Main PPE dataset: yolo_dataset01  
Adverse validation set:val_adverse  
GAN pretrain images (VOC sub1k):JPEGImages_sub1k  
Then edit:  

safety.yaml (path -> yolo_dataset01)  
safety_adverse.yaml (path -> yolo_dataset, val -> val_adverse/images)
### 4) Run all pipelines
CMD
```
experiment_pack\00_run_all.bat
```
or step by step
```
experiment_pack\01_train_generator.bat
experiment_pack\02_build_gan_dataset.bat
experiment_pack\03_train_b0_b1.bat
experiment_pack\04_train_b2_b3.bat
experiment_pack\05_eval_val_adverse.bat
```
### 5) Check results
run/eval_val_adverse_summary.txt
