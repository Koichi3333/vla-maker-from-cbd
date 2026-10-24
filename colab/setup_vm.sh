#!/usr/bin/env bash
# Build the GR00T N1.7 training environment on a Colab VM. The steps measured and verified in B-0 (2026-09-12, A100).
#
#   colab/start.sh sends it and starts it in the background (/content/vla/colab/setup_vm.sh).
#   To run it by hand: colab exec -s <name> <<< 'import subprocess; subprocess.Popen("nohup bash /content/vla/colab/setup_vm.sh > /content/groot_setup.log 2>&1 &", shell=True)'
#   -> poll /content/groot_setup.log until SETUP_DONE appears
#
# Measured: uv sync 71 s / N1.7 weights 6.5 GB in 25 s. Under 2 minutes together.
# Cosmos-Reason2-2B is gated, so it is not downloaded here (it is fetched with HF_TOKEN on the first training run).
set -x
REV=51d4c89f72fda44cbf77285c6a8114b52676b8a1   # Commit verified in B-0
T0=$(date +%s)

# The weight download does not depend on the environment, so start it first in the background (uses the kernel's huggingface_hub)
python3 -c 'from huggingface_hub import snapshot_download; print("model at", snapshot_download("nvidia/GR00T-N1.7-3B"))' \
    > /content/groot_dl.log 2>&1 &
DL=$!

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

# Fetching LFS adds close to 1 GB (demo_data etc.), so skip it. But uv resolves dependencies for every platform,
# and if the aarch64 torchcodec wheel (stored in LFS) is still a pointer,
# uv sync fails with "Invalid zip file structure". Fetch the real file for that one wheel only.
mkdir -p /content/Isaac-GR00T && cd /content/Isaac-GR00T && git init -q
git remote add origin https://github.com/NVIDIA/Isaac-GR00T
export GIT_LFS_SKIP_SMUDGE=1   # smudge runs at checkout, so apply it to the whole tree, not just the fetch
git fetch -q --depth 1 origin $REV && git checkout -q FETCH_HEAD
W=scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl
curl -fsSL -o $W https://media.githubusercontent.com/media/NVIDIA/Isaac-GR00T/$REV/$W

uv sync --python 3.12 2>&1
echo "UV_SYNC_EXIT=$?"
# Check in the log above that flash-attn did not fall back to a source build (it should say "from https://...whl")

wait $DL
tail -1 /content/groot_dl.log
echo "TIMING total=$(( $(date +%s) - T0 ))s"
echo SETUP_DONE
