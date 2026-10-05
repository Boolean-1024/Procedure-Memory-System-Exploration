#!/bin/bash
# Start the ALFWorld env server in WSL (reachable from Windows at http://localhost:${1:-36005}).
source "$HOME/alfworld-server/.venv/bin/activate"
export ALFWORLD_DATA=$HOME/.cache/alfworld
exec alfworld --host 0.0.0.0 --port "${1:-36005}"
