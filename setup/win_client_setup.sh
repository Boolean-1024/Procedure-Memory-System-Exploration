#!/bin/bash
# Eval-client env (Windows, Git Bash). ALFWorld server itself lives in WSL.
set -euo pipefail
ROOT=/d/Project/Thesis_Claude
EMB=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB
MS=$ROOT/EvoMemBench/EvoMemBench-Memory-Systems
cd $ROOT
[ -d .venv ] || uv venv --python 3.11 .venv
PY=$ROOT/.venv/Scripts/python.exe
uv pip install --python $PY openai tiktoken tqdm requests numpy python-dotenv rank_bm25 nltk "chromadb>=1.0" faiss-cpu scikit-learn jsonlines
uv pip install --python $PY torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $PY sentence-transformers
uv pip install --python $PY --no-deps -e $EMB/agentenv
uv pip install --python $PY -e $MS/mem0
uv pip install --python $PY -e $MS/A-mem
echo CLIENT_SETUP_DONE
