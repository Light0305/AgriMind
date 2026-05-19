#!/bin/bash
# ============================================================
# AgriMind 一键安装脚本
# 支持 Linux / WSL2 / macOS
# ============================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   AgriMind — 作物智能会诊系统 一键安装${NC}"
echo -e "${GREEN}============================================${NC}"

# ── 1. 检测 Python ─────────────────────────────────────
PYTHON=""
for py in python3.11 python3.10 python3; do
  if command -v $py &>/dev/null; then
    PYTHON=$py; break
  fi
done
if [ -z "$PYTHON" ]; then
  echo -e "${RED}错误: 未找到 Python 3.10+，请先安装 Python${NC}"
  exit 1
fi
echo -e "${GREEN}[1/5] Python: $($PYTHON --version)${NC}"

# ── 2. 创建虚拟环境 ───────────────────────────────────
if [ ! -d "venv" ]; then
  echo -e "${GREEN}[2/5] 创建虚拟环境...${NC}"
  $PYTHON -m venv venv
else
  echo -e "${GREEN}[2/5] 虚拟环境已存在${NC}"
fi
source venv/bin/activate

# ── 3. 安装后端依赖 ────────────────────────────────────
echo -e "${GREEN}[3/5] 安装 Python 依赖...${NC}"
pip install -q --upgrade pip

# 检测是否有 NVIDIA GPU
HAS_GPU=false
if command -v nvidia-smi &>/dev/null; then
  HAS_GPU=true
fi

if $HAS_GPU; then
  echo "  检测到 NVIDIA GPU，安装 PyTorch CUDA 版..."
  pip install -q torch torchvision
else
  echo "  未检测到 GPU，安装 PyTorch CPU 版（仅 API 模式可用）..."
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -q -r backend/requirements.txt

# ── 4. 安装前端依赖 ────────────────────────────────────
echo -e "${GREEN}[4/5] 安装前端依赖...${NC}"
if command -v node &>/dev/null; then
  NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
  if [ "$NODE_VER" -ge 18 ]; then
    cd frontend && npm install --silent 2>/dev/null && cd ..
    echo "  前端依赖安装完成"
  else
    echo -e "${YELLOW}  Node.js 版本过低 (需要 18+)，跳过前端安装${NC}"
    echo "  请升级 Node.js: https://nodejs.org/"
  fi
else
  echo -e "${YELLOW}  未检测到 Node.js，跳过前端安装${NC}"
  echo "  如需使用 Web 界面，请安装 Node.js 18+: https://nodejs.org/"
fi

# ── 5. 初始化知识库 ────────────────────────────────────
echo -e "${GREEN}[5/5] 初始化 RAG 知识库...${NC}"
cd backend
PYTHONPATH=. python -c "from app.rag.indexer import *; print('  知识库就绪')" 2>/dev/null || echo "  知识库初始化跳过（首次启动时自动创建）"
cd ..

# ── 检查模型 ───────────────────────────────────────────
MODEL_DIR="models/agrimind-v2"
if [ -d "$MODEL_DIR" ]; then
  echo -e "${GREEN}  ✓ 检测到本地模型，支持 GPU 模式和 API 模式${NC}"
else
  echo ""
  echo -e "${YELLOW}  未检测到本地模型 (models/agrimind-v2/)${NC}"
  echo "  系统将使用 API 模式运行（需要 DashScope API Key）"
  echo "  如需本地 GPU 推理，请下载完整包并解压模型到 models/agrimind-v2/"
fi

# ── 完成 ────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   安装完成！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "使用方式："
echo ""
echo "  1. 启动后端:"
echo "     source venv/bin/activate"
echo "     cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  2. 启动前端 (新开终端):"
echo "     cd frontend && npm run dev"
echo ""
echo "  3. 打开浏览器访问: http://localhost:5173"
echo ""
echo "  • API 模式: 设置环境变量 AGRIMIND_API_KEY=sk-xxx 或在界面勾选"
echo "  • GPU 模式: 确保 models/agrimind-v2/ 存在，无需设置 API Key"
echo ""
