@echo off
REM One-time setup for QLoRA Studio (adapted from gemma-web-agent\setup.bat).
REM Creates a private Python environment in .venv so nothing else on your PC is touched.
cd /d "%~dp0"

where python >nul 2>nul || (
  echo Python was not found. Install Python 3.10-3.12 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during install. Then run this again.
  pause & exit /b 1
)

if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || (pause & exit /b 1)
)

echo.
echo Installing PyTorch with NVIDIA CUDA support (about 2.5 GB, be patient)...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || (pause & exit /b 1)

echo.
echo Installing the rest...
.venv\Scripts\python.exe -m pip install -r requirements.txt || (pause & exit /b 1)

echo.
.venv\Scripts\python.exe -c "import torch; print('GPU found:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - you can explore models but not train')"
echo.
echo Setup complete!  Double-click run.bat to start QLoRA Studio.
pause
