#!/bin/bash
# ============================================================
# AgriMind 打包脚本 — 生成三个发布包
# 在服务器上运行: bash pack.sh
# ============================================================
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$PROJECT_DIR/release"
PACK_NAME="AgriMind"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   AgriMind 打包工具${NC}"
echo -e "${GREEN}============================================${NC}"

# ── 排除列表（适用于所有包）──────────────────────────
EXCLUDE_ALL=(
    --exclude='.git'
    --exclude='__pycache__'
    --exclude='.pytest_cache'
    --exclude='node_modules'
    --exclude='frontend/dist'
    --exclude='.agents'
    --exclude='.env'
    --exclude='.env.local'
    --exclude='*.pyc'
    --exclude='*.log'
    --exclude='venv'
    --exclude='.venv'
    --exclude='release'
    --exclude='pack.sh'
    --exclude='start_backend.sh'
    --exclude='test'
    --exclude='.tmp.driveupload'
    --exclude='tmp'
    --exclude='*.tmp'
    --exclude='.DS_Store'
    --exclude='Thumbs.db'
    --exclude='wandb'
    --exclude='runs'
    --exclude='.claude'
    # 服务器上的杂项文件
    --exclude='run_pipeline.sh'
    --exclude='auto_train.sh'
    --exclude='qlora_qwen_vl.yaml'
    --exclude='*.pth'
    --exclude='*.pt'
    --exclude='*.bin'
    --exclude='*.safetensors'
)

# ============================================================
# 1. 代码包 (AgriMind-code.tar.gz)
#    只包含源代码 + test_images + docs
# ============================================================
echo -e "${GREEN}[1/3] 打包代码包...${NC}"

# 创建临时目录，只复制需要的文件
TMPDIR=$(mktemp -d)
STAGING="$TMPDIR/$PACK_NAME"
mkdir -p "$STAGING"

# 复制核心文件
cp "$PROJECT_DIR/README.md" "$STAGING/"
cp "$PROJECT_DIR/setup.sh" "$STAGING/"
cp "$PROJECT_DIR/setup.bat" "$STAGING/"
cp "$PROJECT_DIR/docker-compose.yml" "$STAGING/"
cp "$PROJECT_DIR/.gitignore" "$STAGING/"

# 后端
cp -r "$PROJECT_DIR/backend" "$STAGING/backend"
rm -rf "$STAGING/backend/__pycache__" "$STAGING/backend/.pytest_cache"
rm -rf "$STAGING/backend/data"  # chromadb runtime data
find "$STAGING/backend" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# 前端（不含 node_modules 和 dist）
cp -r "$PROJECT_DIR/frontend" "$STAGING/frontend"
rm -rf "$STAGING/frontend/node_modules" "$STAGING/frontend/dist"

# 测试图片
cp -r "$PROJECT_DIR/test_images" "$STAGING/test_images"

# 文档
cp -r "$PROJECT_DIR/docs" "$STAGING/docs"

# benchmark（代码 + 数据，不含大文件）
mkdir -p "$STAGING/benchmark/baselines" "$STAGING/benchmark/data"
cp "$PROJECT_DIR/benchmark/"*.py "$STAGING/benchmark/" 2>/dev/null || true
cp "$PROJECT_DIR/benchmark/baselines/"*.py "$STAGING/benchmark/baselines/" 2>/dev/null || true
cp "$PROJECT_DIR/benchmark/data/agrireason_v1.json" "$STAGING/benchmark/data/" 2>/dev/null || true

# training（代码，不含数据）
cp -r "$PROJECT_DIR/training" "$STAGING/training"

# demo
[ -d "$PROJECT_DIR/demo" ] && cp -r "$PROJECT_DIR/demo" "$STAGING/demo"

# 创建空 models 目录占位
mkdir -p "$STAGING/models"
echo "# 模型文件放在这里" > "$STAGING/models/.gitkeep"

tar -czf "$OUTPUT_DIR/AgriMind-code.tar.gz" -C "$TMPDIR" "$PACK_NAME"
rm -rf "$TMPDIR"

CODE_SIZE=$(du -sh "$OUTPUT_DIR/AgriMind-code.tar.gz" | cut -f1)
echo -e "  ✓ 代码包: ${GREEN}$CODE_SIZE${NC}"

# ============================================================
# 2. 完整包 (AgriMind-full.tar.gz)
#    = 代码包 + models/agrimind-v2
# ============================================================
echo -e "${GREEN}[2/3] 打包完整包 (含模型, 耐心等待)...${NC}"

MODEL_DIR="$PROJECT_DIR/models/agrimind-v2"
if [ -d "$MODEL_DIR" ]; then
    TMPDIR=$(mktemp -d)
    STAGING="$TMPDIR/$PACK_NAME"

    # 解压代码包作为基础
    tar -xzf "$OUTPUT_DIR/AgriMind-code.tar.gz" -C "$TMPDIR"

    # 复制模型
    cp -r "$MODEL_DIR" "$STAGING/models/agrimind-v2"

    tar -czf "$OUTPUT_DIR/AgriMind-full.tar.gz" -C "$TMPDIR" "$PACK_NAME"
    rm -rf "$TMPDIR"

    FULL_SIZE=$(du -sh "$OUTPUT_DIR/AgriMind-full.tar.gz" | cut -f1)
    echo -e "  ✓ 完整包: ${GREEN}$FULL_SIZE${NC}"
else
    echo -e "  ${YELLOW}⚠ 未找到 models/agrimind-v2/，跳过完整包${NC}"
fi

# ============================================================
# 3. 附加数据包 (AgriMind-data.tar.gz)
#    data/raw + data/processed + benchmark/results* + checkpoints
# ============================================================
echo -e "${GREEN}[3/3] 打包附加数据包...${NC}"

TMPDIR=$(mktemp -d)
STAGING="$TMPDIR/AgriMind-data"
mkdir -p "$STAGING"

# 复制数据
[ -d "$PROJECT_DIR/data/raw" ] && cp -r "$PROJECT_DIR/data/raw" "$STAGING/data_raw"
[ -d "$PROJECT_DIR/data/processed" ] && cp -r "$PROJECT_DIR/data/processed" "$STAGING/data_processed"
for d in "$PROJECT_DIR/benchmark/results"*; do
    [ -d "$d" ] && cp -r "$d" "$STAGING/$(basename $d)"
done

if [ "$(ls -A "$STAGING" 2>/dev/null)" ]; then
    tar -czf "$OUTPUT_DIR/AgriMind-data.tar.gz" -C "$TMPDIR" "AgriMind-data"
    DATA_SIZE=$(du -sh "$OUTPUT_DIR/AgriMind-data.tar.gz" | cut -f1)
    echo -e "  ✓ 数据包: ${GREEN}$DATA_SIZE${NC}"
else
    echo -e "  ${YELLOW}⚠ 无数据可打包${NC}"
fi
rm -rf "$TMPDIR"

# ── 汇总 ────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   打包完成！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "输出目录: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"*.tar.gz 2>/dev/null
