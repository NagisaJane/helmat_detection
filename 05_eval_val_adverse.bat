@echo off
setlocal

set "PY=C:\Users\Yang\anaconda31\envs\torch_env_py310\python.exe"
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"

pushd "%PROJECT_DIR%"
if errorlevel 1 (
  echo [ERROR] Cannot enter project directory: %PROJECT_DIR%
  exit /b 1
)

echo [RUN] Evaluate B0/B1/B2/B3 on val_adverse
"%PY%" "experiment_pack\05_eval_val_adverse.py"
if errorlevel 1 (
  echo [ERROR] Step 05 evaluation failed.
  popd
  exit /b 1
)

popd
echo [DONE] 05_eval_val_adverse.bat
endlocal
