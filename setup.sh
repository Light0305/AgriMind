#!/bin/bash
# ============================================================
# AgriMind 一键安装 & 启动脚本
# 支持 Linux / WSL2 / macOS
# ============================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

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
fi
source venv/bin/activate

# ── 3. 安装依赖 ────────────────────────────────────────
echo -e "${GREEN}[3/5] 安装 Python 依赖...${NC}"
pip install -q --upgrade pip

# 检测是否有 GPU
HAS_GPU=false
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
  HAS_GPU=true
fi

if $HAS_GPU; then
  echo "  检测到 GPU，安装 PyTorch CUDA 版..."
  pip install -q torch torchvision
else
  echo "  未检测到 GPU，安装 PyTorch CPU 版（仅 API 模式可用）..."
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -q transformers accelerate bitsandbytes peft
pip install -q fastapi uvicorn websockets chromadb Pillow sentence-transformers openai
pip install -q qwen-vl-utils

# ── 4. 初始化知识库 ────────────────────────────────────
echo -e "${GREEN}[4/5] 初始化 RAG 知识库...${NC}"
cd backend
PYTHONPATH=. python app/rag/indexer.py 2>/dev/null || echo "  知识库已存在，跳过"
cd ..

# ── 5. 检查模型（可选，仅本地 GPU 模式需要） ──────────
echo -e "${GREEN}[5/5] 检查模型文件...${NC}"
MODEL_DIR="models/agrimind-v2"
BASE_MODEL_DIR="models/qwen2.5-vl-7b"

if [ -d "$MODEL_DIR" ] || [ -d "$BASE_MODEL_DIR" ]; then
  echo "  模型已存在，可切换本地 GPU 模式。"
else
  echo "  ℹ️  未检测到本地模型（16GB）。"
  echo "  系统默认使用 API 模式，无需模型即可运行。"
  echo ""
  echo "  如需本地 GPU 模式，下载完整包："
  echo "    Hugging Face: https://huggingface.co/<your-org>/agrimind-v2"
  echo "    百度网盘: [链接]"
  echo ""
  echo "  或下载基础模型："
  echo "    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir $BASE_MODEL_DIR"
fi

# ── 完成 ────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   安装完成！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "启动方式："
echo ""
echo "  # 本地 GPU 模式"
echo "  source venv/bin/activate"
echo "  cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  # API 模式（无需 GPU）"
echo "  export AGRIMIND_API_KEY=your_key_here"
echo "  cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  # 前端"
echo "  cd frontend && npm install && npm run dev"
echo ""
echo "  然后访问 http://localhost:5173"
