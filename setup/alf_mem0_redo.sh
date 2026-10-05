#!/bin/bash
# After the queue's ALFWorld test10: rebuild ALFWorld mem0 from scratch (embedding-input
# truncation fix; keeps strict index order) and re-test it on the same 10 games.
ROOT=/d/Project/Thesis_Claude
EMB=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB
D=$EMB/output/main_gpt-4.1-mini
LOG=$ROOT/night_queue.log
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
until grep -q "ALFWorld test10 (7 systems + no_memory) done" "$LOG"; do sleep 60; done
mv "$D/memory/mem0" "$D/memory_mem0_6_failed" 2>/dev/null
mv "$D/test/mem0" "$D/test_mem0_6_failed" 2>/dev/null
cd "$EMB"
for t in 1 2; do "$ROOT/.venv/Scripts/python.exe" scripts/run_alfworld_main.py build --systems mem0 > "$D/logs/build_mem0.redo.log" 2>&1 && break; done
log "ALFWorld mem0 rebuild (embed truncation) done (rc=$?)"
until curl -s --max-time 5 http://localhost:36005/ >/dev/null; do sleep 30; done
"$ROOT/.venv/Scripts/python.exe" scripts/run_alfworld_main.py test --systems mem0 --test_subset 10 --parallel 4 > "$D/logs/test10_mem0.redo.log" 2>&1
log "ALFWorld mem0 re-test done (rc=$?)"
