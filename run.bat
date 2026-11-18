@echo off
cd /d "%~dp0"
if not exist .venv (
  echo First time? Running setup...
  call setup.bat
)
echo Starting QLoRA Studio - your browser will open at http://localhost:7861
.venv\Scripts\python.exe app.py
pause
