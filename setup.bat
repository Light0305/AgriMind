@echo off
REM ============================================================
REM   AgriMind - One-Click Setup (Windows)
REM   Double-click to run, or: setup.bat
REM ============================================================

echo ============================================
echo    AgriMind - Crop Diagnosis System
echo    One-Click Setup for Windows
echo ============================================
echo.

REM == Step 1: Python ==
echo [1/5] Checking Python...
where python >nul 2>&1
if %errorlevel% neq 0 goto :need_python
python --version
goto :step2

:need_python
echo   Python not found!
where winget >nul 2>&1
if %errorlevel% neq 0 goto :python_fail
echo   Installing Python 3.12 via winget...
winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
if %errorlevel% neq 0 goto :python_fail
echo.
echo   Python installed! Please CLOSE this window and run setup.bat again.
pause
exit /b 0

:python_fail
echo   [ERROR] Please install Python 3.10+ manually:
echo           https://www.python.org/downloads/
echo   IMPORTANT: Check "Add Python to PATH" during installation!
echo   Then re-run setup.bat.
pause
exit /b 1

REM == Step 2: Node.js ==
:step2
echo [2/5] Checking Node.js...
where node >nul 2>&1
if %errorlevel% neq 0 goto :need_node
node --version
goto :step3

:need_node
echo   Node.js not found!
where winget >nul 2>&1
if %errorlevel% neq 0 goto :node_skip
echo   Installing Node.js LTS via winget...
winget install OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
if %errorlevel% neq 0 goto :node_skip
echo   Node.js installed! Will be available after restarting this script.
goto :step3

:node_skip
echo   [INFO] Node.js 18+ not found. Frontend will not be installed.
echo   Download: https://nodejs.org/
echo   You can re-run setup.bat after installing Node.js.

REM == Step 3: Virtual environment ==
:step3
echo.
echo [3/5] Setting up Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo   Virtual environment created.
) else (
    echo   Virtual environment already exists.
)
call venv\Scripts\activate.bat

REM == Step 4: Backend dependencies ==
echo [4/5] Installing Python dependencies (this may take a few minutes)...
python -m pip install --upgrade pip

echo   Installing PyTorch...
python -m pip install torch torchvision 2>nul
if %errorlevel% neq 0 (
    echo   GPU version failed, installing CPU version...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo   Installing backend packages...
python -m pip install -r backend\requirements.txt

REM GPU-only packages (needed for local 4-bit model loading)
python -c "import torch; assert torch.cuda.is_available()" 2>nul
if %errorlevel% equ 0 (
    echo   GPU detected, installing quantization packages...
    python -m pip install bitsandbytes peft 2>nul
    if %errorlevel% neq 0 echo   Warning: bitsandbytes install failed. Local 4-bit mode may not work.
)
echo   Backend dependencies installed.

REM == Step 5: Frontend ==
echo [5/5] Installing frontend dependencies...
where node >nul 2>&1
if %errorlevel% neq 0 goto :no_frontend
pushd frontend
call npm install 2>nul
popd
echo   Frontend dependencies installed.
goto :post_install

:no_frontend
echo   Skipped - Node.js not available.

REM == Post-install ==
:post_install
echo.
echo Initializing knowledge base...
pushd backend
set PYTHONPATH=.
python -c "from app.rag.indexer import *; print('  Ready')" 2>nul
if %errorlevel% neq 0 echo   Skipped (auto-creates on first use).
popd

if exist "models\agrimind-v2\" (
    echo   Local model found (GPU + API modes available).
) else (
    echo.
    echo   [INFO] No local model detected.
    echo   Will use API mode - requires DashScope API Key.
    echo   For GPU mode, download the full package with model weights.
)

echo.
echo ============================================
echo    Setup Complete!
echo ============================================
echo.
echo Quick Start:
echo.
echo   1. Start backend:
echo      venv\Scripts\activate
echo      cd backend
echo      uvicorn app.main:app --host 0.0.0.0 --port 8000
echo.
echo   2. Start frontend (new terminal):
echo      cd frontend
echo      npm run dev
echo.
echo   3. Open http://localhost:5173 in your browser
echo.
echo   API mode: set AGRIMIND_API_KEY=sk-xxx (or enable in web UI)
echo   GPU mode: ensure models\agrimind-v2\ exists
echo.
pause
