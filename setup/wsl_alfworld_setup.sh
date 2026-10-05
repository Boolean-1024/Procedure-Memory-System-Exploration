#!/bin/bash
# One-time setup of the ALFWorld env server inside WSL (TextWorld does not support Windows).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
SRC=/mnt/d/Project/Thesis_Claude/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/agentenv-alfworld
VENV=$HOME/alfworld-server/.venv
mkdir -p $HOME/alfworld-server
[ -d "$VENV" ] || uv venv --python 3.9 "$VENV"
source "$VENV/bin/activate"
uv pip install "alfworld==0.3.3" "setuptools<70"
uv pip uninstall opencv-python || true
uv pip install -e "$SRC"
export ALFWORLD_DATA=$HOME/.cache/alfworld
[ -d "$ALFWORLD_DATA/json_2.1.1/valid_train" ] || alfworld-download
ls $ALFWORLD_DATA $ALFWORLD_DATA/json_2.1.1
echo SETUP_DONE
