#!/bin/bash
# Run the AppWorld memory runner inside WSL. The OpenAI key is read from the Windows
# machine environment at launch time (never written to disk).
source $HOME/appworld-venv/bin/activate
export APPWORLD_PROJECT_PATH=/mnt/d/Project/Thesis_Claude/ace-appworld
export OPENAI_API_KEY="$(powershell.exe -NoProfile -Command "[Environment]::GetEnvironmentVariable('OPENAI_API_KEY','Machine')" | tr -d '\r')"
# OpenAI-compatible endpoint (machine var OPENAI_API_BASE); empty -> official OpenAI
_base="$(powershell.exe -NoProfile -Command "[Environment]::GetEnvironmentVariable('OPENAI_API_BASE','Machine')" | tr -d '\r')"
[ -n "$_base" ] && export OPENAI_BASE_URL="$_base" OPENAI_API_BASE="$_base"
export PYTHONUNBUFFERED=1 MEM0_TELEMETRY=False
cd $APPWORLD_PROJECT_PATH
exec python experiments/code/memory_eval/run_memory.py "$@"
