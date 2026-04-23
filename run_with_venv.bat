@echo off
REM Uses C:\venv\cv_ca2 (short path — TensorFlow installs reliably there).
REM For PowerShell do NOT use "cd /d" (that is cmd only). Use: .\run_with_venv.ps1 ...
REM Usage: run_with_venv.bat pneumonia_classification.py --baseline
REM        run_with_venv.bat pneumonia_classification.py
cd /d "%~dp0"
if not exist "C:\venv\cv_ca2\Scripts\python.exe" (
  echo Create the venv first:
  echo   py -3.13 -m venv C:\venv\cv_ca2
  echo   C:\venv\cv_ca2\Scripts\python.exe -m pip install -r "%~dp0requirements.txt"
  exit /b 1
)
"C:\venv\cv_ca2\Scripts\python.exe" %*
