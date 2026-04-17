@echo off
setlocal

set "PY=C:\Users\Yang\anaconda31\envs\torch_env_py310\python.exe"
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"
set "VOC1K_DIR=C:\Users\Yang\Downloads\VOC2028\VOC2028\JPEGImages_sub1k"

pushd "%PROJECT_DIR%"
if errorlevel 1 (
  echo [ERROR] Cannot enter project directory: %PROJECT_DIR%
  exit /b 1
)

echo [RUN] Train GAN generator on VOC sub1k
"%PY%" train_gan_synthetic.py --image-dir "%VOC1K_DIR%" --epochs 10 --batch 1 --imgsz 192 --save "gan_generator_voc_sub1k_e10.pt" --device cuda:0 --workers 0
if errorlevel 1 (
  echo [ERROR] Step 01 failed.
  popd
  exit /b 1
)

popd
echo [DONE] 01_train_generator.bat
endlocal
