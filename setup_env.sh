#!/bin/bash
set -e

ENV_NAME="unsloth_med_qwen"

echo "=== Creating Conda environment: $ENV_NAME ==="

# Create conda environment with Python 3.10
conda create -y -n $ENV_NAME python=3.10

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "=== Installing PyTorch with CUDA 12.1 support ==="
# Install PyTorch for CUDA 12.1 (compatible with RTX 3060 Ampere architecture)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing Unsloth (optimized for consumer GPUs) ==="
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install xformers matching CUDA 12.1
pip install --no-deps xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing training dependencies ==="
# Use HF mirror for users in China
export HF_ENDPOINT="https://hf-mirror.com"
pip install transformers datasets trl peft accelerate bitsandbytes

echo ""
echo "=== Environment setup complete! ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')
"
