#!/bin/bash
# After the queue's BFCL test finishes: rebuild BFCL A-mem with max_tokens=16384 and re-test it.
ROOT=/d/Project/Thesis_Claude
TOOL=$ROOT/EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL
R=$TOOL/cross_episode_results/split_main_gpt-4.1-mini
LOG=$ROOT/night_queue.log
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
until grep -q "BFCL test (3/env) done" "$LOG"; do sleep 60; done
mv "$R/memory/amem" "$R/memory_amem_truncated_4096" 2>/dev/null
mv "$R/test/amem" "$R/test_amem_truncated_4096" 2>/dev/null
cd "$TOOL"
"$ROOT/.venv/Scripts/python.exe" -m bfcl_eval.scripts.cross_episode.run_split_experiment build --systems amem > "$R/logs/build_amem.redo.log" 2>&1
log "BFCL amem rebuild (max_tokens 16384) done (rc=$?)"
"$ROOT/.venv/Scripts/python.exe" -m bfcl_eval.scripts.cross_episode.run_split_experiment test --systems amem --test_per_env 3 --workers 2 > "$R/logs/test12_amem.redo.log" 2>&1
log "BFCL amem re-test done (rc=$?)"
"$ROOT/.venv/Scripts/python.exe" -m bfcl_eval.scripts.cross_episode.run_split_experiment report > /dev/null 2>&1
