#!/bin/bash
# Install the memory-system dependencies into the AppWorld venv (WSL), so the same
# EvoMemBench adapters (CROSSEP-EMB/scripts/memory) can be used inside AppWorld.
set -e
export PATH="$HOME/.local/bin:$PATH"
source $HOME/appworld-venv/bin/activate
MS=/mnt/d/Project/Thesis_Claude/EvoMemBench/EvoMemBench-Memory-Systems
uv pip install tiktoken python-dotenv rank_bm25 nltk "chromadb>=1.0" faiss-cpu scikit-learn jsonlines tqdm
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install sentence-transformers
uv pip install -e $MS/mem0
uv pip install -e $MS/A-mem
uv pip install --no-deps -e $MS/MemoryOS
uv pip install "httpx[socks]" regex
uv pip install "click<8.2"
python -c "import mem0, agentic_memory.memory_system, memoryos, chromadb, faiss; print('MEMORY_DEPS_OK')"
