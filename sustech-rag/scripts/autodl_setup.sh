#!/bin/bash
# ============================================================================
# AutoDL RTX 5090 — 环境一键初始化脚本
# ============================================================================
# 在 AutoDL 实例上运行此脚本以完成环境配置。
#
# 使用方法：
#   1. 登录 AutoDL 实例 (ssh -p XXXXX root@connect.xxxx.gpuhub.com)
#   2. bash scripts/autodl_setup.sh
#   3. 等待安装完成（约 5-10 分钟）
#
# 前置条件：
#   - AutoDL RTX 5090 实例已启动
#   - 实例有 conda 环境
# ============================================================================

set -e

echo "=============================================="
echo " AutoDL RTX 5090 — Environment Setup"
echo "=============================================="

# ── GPU 确认 ──
echo ""
echo "Checking GPU..."
nvidia-smi
echo ""

# ── Conda 环境 ──
echo "Creating conda environment 'rag' (Python 3.11)..."
conda create -n rag python=3.11 -y
source activate rag || conda activate rag

# ── PyTorch (CUDA 12.8) ──
echo ""
echo "Installing PyTorch with CUDA 12.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ── RAG 核心依赖 ──
echo ""
echo "Installing RAG dependencies..."
pip install sentence-transformers chromadb rank-bm25 jieba langchain \
    gradio openai httpx scrapy beautifulsoup4 lxml tqdm rich \
    FlagEmbedding transformers accelerate bitsandbytes \
    langchain-text-splitters langdetect

# ── 验证安装 ──
echo ""
echo "=============================================="
echo " Verifying installation..."
echo "=============================================="

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

python -c "
import sentence_transformers
import chromadb
print(f'sentence-transformers: OK')
print(f'chromadb: OK')
"

echo ""
echo "=============================================="
echo " Setup complete! Next steps:"
echo "  1. export DEEPSEEK_API_KEY='your-key'"
echo "  2. bash scripts/run_all.sh --gpu"
echo "=============================================="
