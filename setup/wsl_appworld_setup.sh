#!/bin/bash
# AppWorld (ACE fork) setup inside WSL. Repo lives on D:, venv in WSL home.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export APPWORLD_PROJECT_PATH=/mnt/d/Project/Thesis_Claude/ace-appworld
cd "$APPWORLD_PROJECT_PATH"
VENV=$HOME/appworld-venv
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
source "$VENV/bin/activate"
uv pip install pip
uv pip install -e .
uv pip install -e "experiments[simplified]"
uv pip install "click<8.2"  # typer pinned by appworld breaks on click>=8.2
appworld install --repo
appworld download data
echo "--- splits ---"
for f in data/datasets/*.txt; do echo "$(basename $f): $(grep -c . $f)"; done
echo APPWORLD_SETUP_DONE
