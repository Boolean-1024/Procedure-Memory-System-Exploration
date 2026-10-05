#!/bin/bash
# Strict-frozen 10-case test on all three benchmarks (2026-10-04).
#  1) move previous smoke-test outputs aside  2) hash every memory-store file
#  3) run the same 10 test cases per benchmark, memory read-only  4) re-hash and compare.
ROOT=/d/Project/Thesis_Claude
PY=$ROOT/.venv/Scripts/python.exe
EMB=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB
TOOL=$ROOT/EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL
ALF=$EMB/output/main_gpt-4.1-mini
BF=$TOOL/cross_episode_results/split_main_gpt-4.1-mini
APP=$ROOT/ace-appworld/experiments/outputs/thesis_memory
LOG=$ROOT/frozen_test10.log
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "frozen test10 start"
[ -d "$ALF/test" ] && mv "$ALF/test" "$ALF/test_smoke1"
[ -d "$BF/test" ] && mv "$BF/test" "$BF/test_smoke1"
[ -d "$APP/test/test_normal" ] && mv "$APP/test/test_normal" "$APP/test/test_normal_smoke1"
# ALFWorld MemoryOS is still being built -> excluded here, tested separately once done.
"$PY" $ROOT/setup/store_hashes.py snap $ROOT/frozen_hashes_before.json \
    --exclude "CROSSEP-EMB/output/main_gpt-4.1-mini/memory/memoryos" >> "$LOG" 2>&1

until curl -s --max-time 5 http://localhost:36005/ >/dev/null; do
  powershell.exe -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','--','bash','/mnt/d/Project/Thesis_Claude/setup/wsl_alfworld_server.sh','36005' -WindowStyle Hidden" >/dev/null
  sleep 30
done

( cd "$EMB" && "$PY" scripts/run_alfworld_main.py test --test_subset 10 --parallel 4 \
    --systems no_memory awm reasoning_bank ace mem0 amem bm25 vector > "$ALF/logs/frozen_test10.log" 2>&1
  log "ALFWorld test done (rc=$?)" ) &
( cd "$TOOL" && "$PY" -m bfcl_eval.scripts.cross_episode.run_split_experiment test --test_per_env 3,3,2,2 \
    --workers 2 > "$BF/logs/frozen_test10.log" 2>&1
  log "BFCL test done (rc=$?)" ) &
( powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh test --test_n 10 --procs 4" > "$APP/logs/frozen_test10.log" 2>&1
  log "AppWorld test done (rc=$?)" ) &
wait

"$PY" $ROOT/setup/store_hashes.py check $ROOT/frozen_hashes_before.json >> "$LOG" 2>&1
log "hash check rc=$? (0 = every memory file unchanged)"
( cd "$EMB" && "$PY" scripts/run_alfworld_main.py report > /dev/null 2>&1 )
( cd "$TOOL" && "$PY" -m bfcl_eval.scripts.cross_episode.run_split_experiment report > /dev/null 2>&1 )
powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh report" > /dev/null 2>&1
log "frozen test10 finished"
