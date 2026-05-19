@echo off
REM ============================================================
REM   AgriMind 一键安装脚本 (Windows)
REM   双击运行或命令行执行: setup.bat
REM ============================================================
setlocal enabledelayedexpansion

echo ============================================
echo    AgriMind - 作物智能会诊系统 一键安装
echo ============================================

REM ── 1. 检测 Python ──────────────────────────
set PYTHON=
for %%p in (python python3) do (
    where %%p >nul 2>&1
    if !errorlevel!==0 (
        %%p --version >nul 2>&1
        if !errorlevel!==0 set PYTHON=%%p
    )
)
if "%PYTHON%"=="" (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [1/5] Python:
%PYTHON% --version

REM ── 2. 创建虚拟环境 ────────────────────────
if not exist "venv" (
    echo [2/5] 创建虚拟环境...
    %PYTHON% -m venv venv
)
call venv\Scripts\activate.bat

REM ── 3. 安装依赖 ────────────────────────────
echo [3/5] 安装 Python 依赖...
pip install -q --upgrade pip

REM 检测 GPU
%PYTHON% -c "import torch; print(torch.cuda.is_available())" 2>nul | findstr "True" >nul
if %errorlevel%==0 (
    echo   检测到 GPU，安装 PyTorch CUDA 版...
    pip install -q torch torchvision
) else (
    echo   未检测到 GPU，安装 PyTorch CPU 版（API 模式可用）...
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

pip install -q transformers accelerate bitsandbytes peft
pip install -q fastapi uvicorn websockets chromadb Pillow sentence-transformers openai
pip install -q qwen-vl-utils

REM ── 4. 初始化知识库 ────────────────────────
echo [4/5] 初始化 RAG 知识库...
cd backend
set PYTHONPATH=.
%PYTHON% app\rag\indexer.py 2>nul || echo   知识库已存在，跳过
cd ..

REM ── 5. 检查模型 ────────────────────────────
echo [5/5] 检查模型文件...
if exist "models\agrimind-v2\" (
    echo   模型已存在，可切换本地 GPU 模式。
) else if exist "models\qwen2.5-vl-7b\" (
    echo   基础模型已存在。
) else (
    echo   [信息] 未检测到本地模型（16GB）。
    echo   系统默认使用 API 模式，无需模型即可运行。
    echo.
    echo   如需本地 GPU 模式，下载完整包：
    echo     Hugging Face / 百度网盘
)

REM ── 完成 ────────────────────────────────────
echo.
echo ============================================
echo    安装完成！
echo ============================================
echo.
echo 启动方式:
echo   # 后端（API 模式，默认）
echo   venv\Scripts\activate
echo   cd backend ^&^& uvicorn app.main:app --host 0.0.0.0 --port 8000
echo.
echo   # 前端
echo   cd frontend ^&^& npm install ^&^& npm run dev
echo.
echo   访问 http://localhost:5173
pause
