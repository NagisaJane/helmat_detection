@echo off
setlocal

set "PY=C:\Users\Yang\anaconda31\envs\torch_env_py310\python.exe"
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"

pushd "%PROJECT_DIR%"
if errorlevel 1 (
  echo [ERROR] Cannot enter project directory: %PROJECT_DIR%
  exit /b 1
)

echo [RUN] B0 baseline (no GAN, no attention)
"%PY%" train_baseline_nano.py --data "safety.yaml" --model "yolo26n-obb.pt" --epochs 200 --batch 8 --imgsz 640 --name "run\exp_B0_baseline"
if errorlevel 1 (
  echo [ERROR] B0 training failed.
  popd
  exit /b 1
)

echo [RUN] B1 GAN-only
"%PY%" train_baseline_nano.py --data "safety_gan.yaml" --model "yolo26n-obb.pt" --epochs 200 --batch 8 --imgsz 640 --name "run\exp_B1_gan_only"
if errorlevel 1 (
  echo [ERROR] B1 training failed.
  popd
  exit /b 1
)

popd
echo [DONE] 03_train_b0_b1.bat
endlocal
