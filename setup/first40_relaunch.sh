#!/bin/bash
# Relaunch of the first-40 test for ALFWorld + AppWorld after the WSL OOM at 23:34
# (27 AppWorld workers exhausted WSL memory; the OOM killer also took down the ALFWorld server).
# AppWorld now runs as 2 sequential lanes x 3 workers. BFCL keeps running from first40_test.sh.
ROOT=/d/Project/Thesis_Claude
PY=$ROOT/.venv/Scripts/python.exe
EMB=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB
TOOL=$ROOT/EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL
ALF=$EMB/output/main_gpt-4.1-mini
APP=$ROOT/ace-appworld/experiments/outputs/thesis_memory
LOG=$ROOT/first40_test.log
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ensure_server() {
  until curl -s --max-time 5 http://localhost:36005/ >/dev/null; do
    log "ALFWorld server down -> restarting"
    powershell.exe -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','--','bash','/mnt/d/Project/Thesis_Claude/setup/wsl_alfworld_server.sh','36005' -WindowStyle Hidden" >/dev/null
    sleep 40
  done
}
log "relaunch ALFWorld + AppWorld (after OOM), attempt 2"
ensure_server

# AppWorld: 2 lanes, systems run one after another inside each lane
( powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh test --systems no_memory ace awm reasoning_bank mem0 --test_first 40 --procs 3 --force_parallel" > "$APP/logs/first40_laneA.log" 2>&1
  log "AppWorld lane A (no_memory ace awm reasoning_bank mem0) done (rc=$?)" ) &
sleep 20
( powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh test --systems amem memoryos bm25 vector --test_first 40 --procs 3 --force_parallel" > "$APP/logs/first40_laneB.log" 2>&1
  log "AppWorld lane B (amem memoryos bm25 vector) done (rc=$?)" ) &

# ALFWorld: one process per system; up to 3 resume rounds in case the server drops again
for s in no_memory awm reasoning_bank ace mem0 amem bm25 vector; do
  ( cd "$EMB"
    for t in 1 2 3; do
      ensure_server
      "$PY" scripts/run_alfworld_main.py test --test_first 40 --parallel 3 --force_parallel --systems $s >> "$ALF/logs/first40_$s.log" 2>&1
      n=$(ls "$ALF/test/$s" 2>/dev/null | grep -c '^alfworld_')
      [ "$n" -ge 40 ] && break
      log "ALFWorld $s: $n/40 after round $t, retrying"
      sleep 60
    done
    log "ALFWorld $s done ($(ls "$ALF/test/$s" | grep -c '^alfworld_')/40)" ) &
  sleep 5
done
wait

"$PY" $ROOT/setup/store_hashes.py check $ROOT/first40_hashes_before.json >> "$LOG" 2>&1
log "FINAL hash check rc=$? (0 = every memory file unchanged)"
( cd "$EMB" && "$PY" scripts/run_alfworld_main.py report > /dev/null 2>&1 )
( cd "$TOOL" && "$PY" -m bfcl_eval.scripts.cross_episode.run_split_experiment report > /dev/null 2>&1 )
powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh report" > /dev/null 2>&1
log "relaunch finished"
