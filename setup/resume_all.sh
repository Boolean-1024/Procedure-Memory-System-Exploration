#!/bin/bash
# Resume everything after the 2026-10-04 23:57 credit exhaustion, now via the OPENAI_API_BASE proxy.
#   lane M: ALFWorld MemoryOS build (164/200 -> 200) -> hash snapshot -> first-40 test (frozen) -> hash check
#   lane A: ALFWorld first-40 for ace / amem / bm25 / vector (finished games are skipped)
#   lane W: AppWorld first-40, 2 sub-lanes x 3 workers (WSL memory limit)
#   then  : final hash check of every other memory store + reports
ROOT=/d/Project/Thesis_Claude
PY=$ROOT/.venv/Scripts/python.exe
EMB=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB
TOOL=$ROOT/EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL
ALF=$EMB/output/main_gpt-4.1-mini
APP=$ROOT/ace-appworld/experiments/outputs/thesis_memory
LOG=$ROOT/resume_all.log
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ensure_server() {
  until curl -s --max-time 5 http://localhost:36005/ >/dev/null; do
    log "ALFWorld server down -> restarting"
    powershell.exe -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','--','bash','/mnt/d/Project/Thesis_Claude/setup/wsl_alfworld_server.sh','36005' -WindowStyle Hidden" >/dev/null
    sleep 40
  done
}
alf_test() {  # $1 = system; resume until 40/40 (max 3 rounds)
  for t in 1 2 3; do
    ensure_server
    ( cd "$EMB" && "$PY" scripts/run_alfworld_main.py test --test_first 40 --parallel 3 --force_parallel \
        --systems $1 >> "$ALF/logs/first40_$1.log" 2>&1 )
    rc=$?
    n=$(ls "$ALF/test/$1" 2>/dev/null | grep -c '^alfworld_')
    [ "$n" -ge 40 ] && break
    [ $rc -eq 3 ] && { log "ALFWorld $1: FATAL API error (rc=3), stopping"; break; }
    log "ALFWorld $1: $n/40 after round $t, retrying"; sleep 60
  done
  log "ALFWorld $1 first40: $(ls "$ALF/test/$1" 2>/dev/null | grep -c '^alfworld_')/40"
}
mos_hash() {  # $1 = output json
  "$PY" - "$1" <<'PYEOF'
import hashlib, json, os, sys
base = r"D:/Project/Thesis_Claude/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/output/main_gpt-4.1-mini/memory/memoryos"
h = {os.path.relpath(os.path.join(r, f), base): hashlib.md5(open(os.path.join(r, f), "rb").read()).hexdigest()
     for r, _, fs in os.walk(base) for f in fs}
json.dump(h, open(sys.argv[1], "w"), indent=1)
PYEOF
}

log "resume_all start (proxy: $(powershell.exe -NoProfile -Command "[Environment]::GetEnvironmentVariable('OPENAI_API_BASE','Machine')" | tr -d '\r'))"
ensure_server

# lane M: ALFWorld MemoryOS
(
  ( cd "$EMB" && "$PY" scripts/run_alfworld_main.py build --systems memoryos > "$ALF/logs/build_memoryos.resume2.log" 2>&1 )
  rc=$?
  n=$(wc -l < "$ALF/memory/memoryos/build_log.jsonl.done")
  log "ALFWorld memoryos build done (rc=$rc, $n/200)"
  if [ "$n" -ge 200 ]; then
    mos_hash $ROOT/first40_memoryos_hashes_before.json
    log "ALFWorld memoryos store hash snapshot taken"
    alf_test memoryos
    mos_hash $ROOT/first40_memoryos_hashes_after.json
    if cmp -s $ROOT/first40_memoryos_hashes_before.json $ROOT/first40_memoryos_hashes_after.json; then
      log "ALFWorld memoryos hash check: unchanged (OK)"
    else
      log "ALFWorld memoryos hash check: CHANGED"
    fi
  fi
) &
M=$!

# lane A: unfinished ALFWorld systems
for s in ace amem bm25 vector; do alf_test $s & sleep 5; done

# lane W: AppWorld, 2 sub-lanes
( powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh test --systems no_memory ace awm reasoning_bank mem0 --test_first 40 --procs 3 --force_parallel" > "$APP/logs/first40_resume_laneA.log" 2>&1
  log "AppWorld lane A (no_memory ace awm reasoning_bank mem0) done (rc=$?)" ) &
sleep 20
( powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh test --systems amem memoryos bm25 vector --test_first 40 --procs 3 --force_parallel" > "$APP/logs/first40_resume_laneB.log" 2>&1
  log "AppWorld lane B (amem memoryos bm25 vector) done (rc=$?)" ) &

wait
"$PY" $ROOT/setup/store_hashes.py check $ROOT/first40_hashes_before.json >> "$LOG" 2>&1
log "FINAL hash check (all stores except ALFWorld memoryos) rc=$? (0 = unchanged)"
( cd "$EMB" && "$PY" scripts/run_alfworld_main.py report > /dev/null 2>&1 )
( cd "$TOOL" && "$PY" -m bfcl_eval.scripts.cross_episode.run_split_experiment report > /dev/null 2>&1 )
powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh report" > /dev/null 2>&1
log "resume_all finished"
