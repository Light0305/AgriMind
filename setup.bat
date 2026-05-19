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
) else (
    echo [2/5] 虚拟环境已存在
)
call venv\Scripts\activate.bat

REM ── 3. 安装后端依赖 ────────────────────────
echo [3/5] 安装 Python 依赖...
pip install -q --upgrade pip

REM 检测 GPU
python -c "import torch; print(torch.cuda.is_available())" 2>nul | findstr "True" >nul
if %errorlevel%==0 (
    echo   检测到 GPU，安装 PyTorch CUDA 版...
    pip install -q torch torchvision
) else (
    echo   未检测到 GPU，安装 PyTorch CPU 版...
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

pip install -q -r backend\requirements.txt

REM ── 4. 安装前端依赖 ────────────────────────
echo [4/5] 安装前端依赖...
where node >nul 2>&1
if %errorlevel%==0 (
    cd frontend
    call npm install --silent 2>nul
    cd ..
    echo   前端依赖安装完成
) else (
    echo   [提示] 未检测到 Node.js，跳过前端安装
    echo   如需 Web 界面，请安装 Node.js 18+: https://nodejs.org/
)

REM ── 5. 初始化知识库 ────────────────────────
echo [5/5] 初始化 RAG 知识库...
cd backend
set PYTHONPATH=.
python -c "from app.rag.indexer import *; print('  知识库就绪')" 2>nul || echo   知识库初始化跳过
cd ..

REM ── 检查模型 ────────────────────────────────
if exist "models\agrimind-v2\" (
    echo   模型已存在，支持 GPU 模式和 API 模式
) else (
    echo.
    echo   [提示] 未检测到本地模型
    echo   系统将使用 API 模式运行（需要 DashScope API Key）
    echo   如需本地 GPU 推理，请下载完整包并解压模型到 models\agrimind-v2\
)

REM ── 完成 ────────────────────────────────────
echo.
echo ============================================
echo    安装完成！
echo ============================================
echo.
echo 使用方式:
echo.
echo   1. 启动后端:
echo      venv\Scripts\activate
echo      cd backend ^&^& uvicorn app.main:app --host 0.0.0.0 --port 8000
echo.
echo   2. 启动前端 (新开终端):
echo      cd frontend ^&^& npm run dev
echo.
echo   3. 打开浏览器访问: http://localhost:5173
echo.
echo   API 模式: set AGRIMIND_API_KEY=sk-xxx 或在界面勾选
echo   GPU 模式: 确保 models\agrimind-v2\ 存在
echo.
pause
