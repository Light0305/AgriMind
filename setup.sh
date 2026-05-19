#!/bin/bash
# ============================================================
#   AgriMind - One-Click Setup
#   Supports Linux / WSL2 / macOS
# ============================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   AgriMind - Crop Diagnosis System Setup${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# ── Helper: detect package manager and install ────────────
install_pkg() {
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y "$@"
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y "$@"
  elif command -v yum &>/dev/null; then
    sudo yum install -y "$@"
  elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm "$@"
  elif command -v brew &>/dev/null; then
    brew install "$@"
  else
    return 1
  fi
}

# ── 1. Python ─────────────────────────────────────────────
echo -e "${GREEN}[1/5] Checking Python...${NC}"
PYTHON=""
for py in python3.12 python3.11 python3.10 python3; do
  if command -v "$py" &>/dev/null; then
    PY_VER=$("$py" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
      PYTHON="$py"; break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo -e "${YELLOW}  Python 3.10+ not found. Attempting to install...${NC}"
  if command -v apt-get &>/dev/null; then
    # Debian/Ubuntu: try deadsnakes PPA for newer Python
    sudo apt-get update -qq
    if apt-cache show python3.11 &>/dev/null 2>&1; then
      sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
      PYTHON=python3.11
    elif apt-cache show python3.10 &>/dev/null 2>&1; then
      sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
      PYTHON=python3.10
    else
      # Try adding deadsnakes PPA
      sudo apt-get install -y software-properties-common
      sudo add-apt-repository -y ppa:deadsnakes/ppa
      sudo apt-get update -qq
      sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
      PYTHON=python3.11
    fi
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y python3.11 python3.11-devel || sudo dnf install -y python3
    PYTHON=$(command -v python3.11 || command -v python3)
  elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm python
    PYTHON=python3
  elif command -v brew &>/dev/null; then
    brew install python@3.12
    PYTHON=$(brew --prefix python@3.12)/bin/python3.12
  else
    echo -e "${RED}  [ERROR] Cannot auto-install Python. No supported package manager found.${NC}"
    echo "  Please install Python 3.10+ manually: https://www.python.org/downloads/"
    exit 1
  fi

  # Verify installation
  if [ -z "$PYTHON" ] || ! command -v "$PYTHON" &>/dev/null; then
    echo -e "${RED}  [ERROR] Python installation failed.${NC}"
    echo "  Please install Python 3.10+ manually: https://www.python.org/downloads/"
    exit 1
  fi
fi
echo "  $($PYTHON --version)"

# ── 2. Node.js ────────────────────────────────────────────
echo -e "${GREEN}[2/5] Checking Node.js...${NC}"
NODE_OK=false
if command -v node &>/dev/null; then
  NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
  if [ "$NODE_VER" -ge 18 ] 2>/dev/null; then
    NODE_OK=true
    echo "  Node.js $(node -v)"
  else
    echo -e "${YELLOW}  Node.js version too old ($(node -v), need 18+). Attempting upgrade...${NC}"
  fi
fi

if [ "$NODE_OK" = false ]; then
  echo "  Node.js 18+ not found. Attempting to install..."
  if command -v apt-get &>/dev/null; then
    # NodeSource for Debian/Ubuntu
    if ! command -v curl &>/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y curl
    fi
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
  elif command -v dnf &>/dev/null; then
    if ! command -v curl &>/dev/null; then
      sudo dnf install -y curl
    fi
    curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
    sudo dnf install -y nodejs
  elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm nodejs npm
  elif command -v brew &>/dev/null; then
    brew install node@20
    export PATH="$(brew --prefix node@20)/bin:$PATH"
  else
    echo -e "${YELLOW}  [INFO] Cannot auto-install Node.js. Frontend will not be set up.${NC}"
    echo "  Please install Node.js 18+ manually: https://nodejs.org/"
  fi

  # Re-check
  if command -v node &>/dev/null; then
    NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VER" -ge 18 ] 2>/dev/null; then
      NODE_OK=true
      echo "  Node.js $(node -v) installed successfully."
    fi
  fi
fi

# ── 3. Virtual environment ────────────────────────────────
echo -e "${GREEN}[3/5] Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
  $PYTHON -m venv venv
  echo "  Virtual environment created."
else
  echo "  Virtual environment already exists."
fi
source venv/bin/activate

# ── 4. Backend dependencies ───────────────────────────────
echo -e "${GREEN}[4/5] Installing Python dependencies...${NC}"
pip install -q --upgrade pip

# Detect NVIDIA GPU
HAS_GPU=false
if command -v nvidia-smi &>/dev/null; then
  HAS_GPU=true
fi

if $HAS_GPU; then
  echo "  NVIDIA GPU detected. Installing PyTorch with CUDA..."
  pip install -q torch torchvision
else
  echo "  No GPU detected. Installing PyTorch CPU version (API mode only)..."
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "  Installing backend packages..."
pip install -q -r backend/requirements.txt
echo "  Backend dependencies installed."

# ── 5. Frontend dependencies ──────────────────────────────
echo -e "${GREEN}[5/5] Installing frontend dependencies...${NC}"
if [ "$NODE_OK" = true ]; then
  pushd frontend > /dev/null
  npm install --silent 2>/dev/null
  popd > /dev/null
  echo "  Frontend dependencies installed."
else
  echo -e "${YELLOW}  Skipped - Node.js 18+ not available.${NC}"
fi

# ── Post-install: Knowledge base ──────────────────────────
echo ""
echo "Initializing knowledge base..."
pushd backend > /dev/null
PYTHONPATH=. python -c "from app.rag.indexer import *; print('  Ready')" 2>/dev/null || echo "  Skipped (auto-creates on first use)."
popd > /dev/null

# ── Model detection ───────────────────────────────────────
MODEL_DIR="models/agrimind-v2"
if [ -d "$MODEL_DIR" ]; then
  echo -e "${GREEN}  Local model found (GPU + API modes available).${NC}"
else
  echo ""
  echo -e "${YELLOW}  [INFO] No local model detected (models/agrimind-v2/).${NC}"
  echo "  Will use API mode - requires DashScope API Key."
  echo "  For GPU mode, download the full package with model weights."
fi

# ── Done ──────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Quick Start:"
echo ""
echo "  1. Start backend:"
echo "     source venv/bin/activate"
echo "     cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  2. Start frontend (new terminal):"
echo "     cd frontend && npm run dev"
echo ""
echo "  3. Open http://localhost:5173 in your browser"
echo ""
echo "  API mode: export AGRIMIND_API_KEY=sk-xxx (or enable in web UI)"
echo "  GPU mode: ensure models/agrimind-v2/ exists"
echo ""
