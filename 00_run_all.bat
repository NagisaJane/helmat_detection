@echo off
setlocal

set "ROOT=%~dp0"

echo [RUN] 01_train_generator.bat
call "%ROOT%01_train_generator.bat"
if errorlevel 1 goto :fail

echo [RUN] 02_build_gan_dataset.bat
call "%ROOT%02_build_gan_dataset.bat"
if errorlevel 1 goto :fail

echo [RUN] 03_train_b0_b1.bat
call "%ROOT%03_train_b0_b1.bat"
if errorlevel 1 goto :fail

echo [RUN] 04_train_b2_b3.bat
call "%ROOT%04_train_b2_b3.bat"
if errorlevel 1 goto :fail

echo [RUN] 05_eval_val_adverse.bat
call "%ROOT%05_eval_val_adverse.bat"
if errorlevel 1 goto :fail

echo [DONE] ALL STEPS FINISHED SUCCESSFULLY
exit /b 0

:fail
echo [ERROR] Pipeline stopped due to failure.
exit /b 1
