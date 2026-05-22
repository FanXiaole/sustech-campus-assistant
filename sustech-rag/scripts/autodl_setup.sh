#!/bin/bash
# =============================================================================
# AutoDL GPU Setup — 一键配置 GPU 环境
# Usage: bash scripts/autodl_setup.sh
# =============================================================================
set -e
echo "=== SUSTech RAG — AutoDL Environment Setup ==="

# Conda environment
if ! conda env list 2>/dev/null | grep -q "rag"; then
    echo "[1/4] Creating conda environment..."
    conda create -n rag python=3.11 -y
fi
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate rag

# PyTorch with CUDA
echo "[2/4] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || \
pip install torch torchvision torchaudio

# Core packages
echo "[3/4] Installing Python dependencies..."
pip install sentence-transformers chromadb rank-bm25 jieba langchain \
    gradio openai httpx scrapy beautifulsoup4 lxml tqdm rich \
    transformers accelerate bitsandbytes

# HuggingFace mirror for China
export HF_ENDPOINT=https://hf-mirror.com
grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null || echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

# Verify
echo "[4/4] Verifying..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
python -c "import chromadb; print('ChromaDB OK')"

echo "=== Setup complete ==="
echo "Next steps:"
echo "  export DEEPSEEK_API_KEY='sk-...'"
echo "  python evaluation/run_experiments.py --all"
