@echo off
setlocal

set "PY=C:\Users\Yang\anaconda31\envs\torch_env_py310\python.exe"
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"

pushd "%PROJECT_DIR%"
if errorlevel 1 (
  echo [ERROR] Cannot enter project directory: %PROJECT_DIR%
  exit /b 1
)

echo [RUN] B2 attention-only
"%PY%" train_start.py --data "safety.yaml" --model "yolo26-obb-hybrid-only.yaml" --epochs 200 --batch 8 --imgsz 640 --name "run\exp_B2_attn_only"
if errorlevel 1 (
  echo [ERROR] B2 training failed.
  popd
  exit /b 1
)

echo [RUN] B3 full (GAN + attention)
"%PY%" train_start.py --data "safety_gan.yaml" --model "yolo26-obb-hybrid-only.yaml" --epochs 200 --batch 8 --imgsz 640 --name "run\exp_B3_full"
if errorlevel 1 (
  echo [ERROR] B3 training failed.
  popd
  exit /b 1
)

popd
echo [DONE] 04_train_b2_b3.bat
endlocal
