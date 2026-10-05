#!/bin/bash
# Strict-frozen test on the first 40 test cases of every benchmark (2026-10-04).
#   ALFWorld : test games 2420..2459 (index order)
#   BFCL     : first 10 test samples of each of the 4 environments, dataset-id order (= 40)
#   AppWorld : first 40 tasks of test_normal (official order)
# One process per system (memory is read from temporary copies, so parallel reads are safe).
# ALFWorld MemoryOS is still being built -> run separately afterwards (alf_memoryos_first40.sh).
ROOT=/d/Project/Thesis_Claude
PY=$ROOT/.venv/Scripts/python.exe
EMB=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB
TOOL=$ROOT/EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL
ALF=$EMB/output/main_gpt-4.1-mini
BF=$TOOL/cross_episode_results/split_main_gpt-4.1-mini
APP=$ROOT/ace-appworld/experiments/outputs/thesis_memory
LOG=$ROOT/first40_test.log
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "first40 start"
# keep the previous 10-case frozen results; ALFWorld memoryos (not yet tested) stays in test/
mkdir -p "$ALF/test_frozen10"
for d in "$ALF"/test/*; do [ "$(basename "$d")" != "memoryos" ] && mv "$d" "$ALF/test_frozen10/"; done
[ -d "$BF/test" ] && mv "$BF/test" "$BF/test_frozen10"
[ -d "$APP/test/test_normal" ] && mv "$APP/test/test_normal" "$APP/test/test_normal_frozen10"

"$PY" $ROOT/setup/store_hashes.py snap $ROOT/first40_hashes_before.json \
    --exclude "CROSSEP-EMB/output/main_gpt-4.1-mini/memory/memoryos" >> "$LOG" 2>&1

until curl -s --max-time 5 http://localhost:36005/ >/dev/null; do
  powershell.exe -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','--','bash','/mnt/d/Project/Thesis_Claude/setup/wsl_alfworld_server.sh','36005' -WindowStyle Hidden" >/dev/null
  sleep 30
done

for s in no_memory awm reasoning_bank ace mem0 amem bm25 vector; do
  ( cd "$EMB" && "$PY" scripts/run_alfworld_main.py test --test_first 40 --parallel 3 --force_parallel \
      --systems $s > "$ALF/logs/first40_$s.log" 2>&1; log "ALFWorld $s done (rc=$?)" ) &
  sleep 5   # stagger: each run rewrites run_meta.json at startup
done
for s in no_memory awm reasoning_bank ace mem0 amem memoryos bm25 vector; do
  ( cd "$TOOL" && "$PY" -m bfcl_eval.scripts.cross_episode.run_split_experiment test --systems $s \
      --test_per_env 10 --test_order id --workers 2 --force_parallel > "$BF/logs/first40_$s.log" 2>&1
    log "BFCL $s done (rc=$?)" ) &
done
for s in no_memory ace awm reasoning_bank mem0 amem memoryos bm25 vector; do
  ( powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh test --systems $s --test_first 40 --procs 3 --force_parallel" > "$APP/logs/first40_$s.log" 2>&1
    log "AppWorld $s done (rc=$?)" ) &
done
wait

"$PY" $ROOT/setup/store_hashes.py check $ROOT/first40_hashes_before.json >> "$LOG" 2>&1
log "hash check rc=$? (0 = every memory file unchanged)"
( cd "$EMB" && "$PY" scripts/run_alfworld_main.py report > /dev/null 2>&1 )
( cd "$TOOL" && "$PY" -m bfcl_eval.scripts.cross_episode.run_split_experiment report > /dev/null 2>&1 )
powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh report" > /dev/null 2>&1
log "first40 finished"
