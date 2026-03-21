#!/bin/bash
set -euo pipefail

# boontft Parameter Golf — 4090 Setup Script
# Run this on the machine with the 4090

echo "=== boontft Parameter Golf Setup ==="

# 1. Install Tailscale for remote access
if ! command -v tailscale &> /dev/null; then
    echo "Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
    echo "Run 'sudo tailscale up' and sign in to connect."
    echo "Then share the Tailscale IP with Samuel."
else
    echo "Tailscale already installed."
    tailscale ip -4 2>/dev/null || echo "Run 'sudo tailscale up' to connect."
fi

# 2. Clone repo if not already done
REPO_DIR="$HOME/parameter-golf"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning parameter-golf repo..."
    git clone https://github.com/openai/parameter-golf.git "$REPO_DIR"
else
    echo "Repo already exists at $REPO_DIR"
    cd "$REPO_DIR" && git pull
fi
cd "$REPO_DIR"

# 3. Python environment
if [ ! -d ".venv" ]; then
    echo "Creating Python venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install numpy tqdm torch huggingface-hub setuptools \
    "typing-extensions>=4.0" datasets sentencepiece zstandard -q

# 4. Verify CUDA
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
gpu = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_mem // (1024**3)
print(f'GPU: {gpu} ({mem}GB)')
print(f'CUDA: {torch.version.cuda}')
print(f'PyTorch: {torch.__version__}')
"

# 5. Download training data (small subset for iteration)
echo "Downloading FineWeb data (1 shard for fast iteration)..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# 6. Copy our submission code
echo "Setting up boontft submission..."
mkdir -p boontft
# Our code is in the boontft/ dir — sync from Samuel's repo

# 7. Smoke test
echo "Running smoke test (50 iterations)..."
RUN_ID=smoke \
ITERATIONS=50 \
MAX_WALLCLOCK_SECONDS=120 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=131072 \
torchrun --standalone --nproc_per_node=1 boontft/train_gpt.py

echo ""
echo "=== Setup Complete ==="
echo "Quick run:  RUN_ID=test ITERATIONS=500 MAX_WALLCLOCK_SECONDS=300 TRAIN_BATCH_TOKENS=131072 torchrun --standalone --nproc_per_node=1 boontft/train_gpt.py"
echo "Full run:   RUN_ID=full torchrun --standalone --nproc_per_node=1 boontft/train_gpt.py"
