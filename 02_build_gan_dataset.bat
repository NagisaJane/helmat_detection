@echo off
setlocal

set "PY=C:\Users\Yang\anaconda31\envs\torch_env_py310\python.exe"
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"

pushd "%PROJECT_DIR%"
if errorlevel 1 (
  echo [ERROR] Cannot enter project directory: %PROJECT_DIR%
  exit /b 1
)

echo [RUN] Build GAN-augmented detection dataset
"%PY%" build_gan_aug_dataset.py --src-yaml "safety.yaml" --gan-ckpt "gan_generator_voc_sub1k_e10.pt" --out-root "yolo_dataset01_gan" --imgsz 640 --device cuda:0
if errorlevel 1 (
  echo [ERROR] Step 02 failed.
  popd
  exit /b 1
)

popd
echo [DONE] 02_build_gan_dataset.bat
endlocal
