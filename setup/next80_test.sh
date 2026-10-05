#!/bin/bash
# Extend the strict-frozen test from the first 40 to the first 120 cases (2026-10-05).
#   ALFWorld : test games 2420..2539 (index order)          -> +80
#   BFCL     : all 25 test samples of each of the 4 envs    -> +60 (the whole test set)
#   AppWorld : first 120 tasks of test_normal               -> +80
# Finished cases are skipped by every runner, so only the new ones are run.
# ALFWorld MemoryOS waits for resume_all.sh (build + first-40 test) and then extends to 120.
ROOT=/d/Project/Thesis_Claude
PY=$ROOT/.venv/Scripts/python.exe
EMB=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB
TOOL=$ROOT/EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL
ALF=$EMB/output/main_gpt-4.1-mini
BF=$TOOL/cross_episode_results/split_main_gpt-4.1-mini
APP=$ROOT/ace-appworld/experiments/outputs/thesis_memory
LOG=$ROOT/next80_test.log
N=120
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ensure_server() {
  until curl -s --max-time 5 http://localhost:36005/ >/dev/null; do
    log "ALFWorld server down -> restarting"
    powershell.exe -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','--','bash','/mnt/d/Project/Thesis_Claude/setup/wsl_alfworld_server.sh','36005' -WindowStyle Hidden" >/dev/null
    sleep 40
  done
}
alf_test() {  # $1 = system; resume until N/N (max 3 rounds)
  for t in 1 2 3; do
    ensure_server
    ( cd "$EMB" && "$PY" scripts/run_alfworld_main.py test --test_first $N --parallel 3 --force_parallel \
        --systems $1 >> "$ALF/logs/first${N}_$1.log" 2>&1 )
    rc=$?
    n=$(ls "$ALF/test/$1" 2>/dev/null | grep -c '^alfworld_')
    [ "$n" -ge $N ] && break
    [ $rc -eq 3 ] && { log "ALFWorld $1: FATAL API error (rc=3), stopping"; break; }
    log "ALFWorld $1: $n/$N after round $t, retrying"; sleep 60
  done
  log "ALFWorld $1 first$N: $(ls "$ALF/test/$1" 2>/dev/null | grep -c '^alfworld_')/$N"
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

log "next80 start (first $N per benchmark)"
"$PY" $ROOT/setup/store_hashes.py snap $ROOT/first120_hashes_before.json \
    --exclude "CROSSEP-EMB/output/main_gpt-4.1-mini/memory/memoryos" >> "$LOG" 2>&1
log "store hash snapshot taken (all stores except ALFWorld memoryos)"
ensure_server

# ALFWorld: 8 ready systems
for s in no_memory awm reasoning_bank ace mem0 amem bm25 vector; do alf_test $s & sleep 5; done

# ALFWorld MemoryOS: after resume_all.sh has finished its build + first-40 test
(
  # resume the build (finished trajectories are skipped via build_log.jsonl.done)
  ( cd "$EMB" && "$PY" scripts/run_alfworld_main.py build --systems memoryos >> "$ALF/logs/build_memoryos.resume3.log" 2>&1 )
  nb=$(wc -l < "$ALF/memory/memoryos/build_log.jsonl.done")
  log "ALFWorld memoryos build: $nb/200"
  if [ "$nb" -ge 200 ]; then
    mos_hash $ROOT/first120_memoryos_hashes_before.json
    alf_test memoryos
    mos_hash $ROOT/first120_memoryos_hashes_after.json
    if cmp -s $ROOT/first120_memoryos_hashes_before.json $ROOT/first120_memoryos_hashes_after.json; then
      log "ALFWorld memoryos first$N hash check: unchanged (OK)"
    else
      log "ALFWorld memoryos first$N hash check: CHANGED"
    fi
  else
    log "ALFWorld memoryos: build incomplete, test not run"
  fi
) &

# BFCL: whole test set (25 per env)
for s in no_memory awm reasoning_bank ace mem0 amem memoryos bm25 vector; do
  ( cd "$TOOL" && "$PY" -m bfcl_eval.scripts.cross_episode.run_split_experiment test --systems $s \
      --test_per_env 25 --test_order id --workers 2 --force_parallel >> "$BF/logs/first100_$s.log" 2>&1
    log "BFCL $s done (rc=$?)" ) &
done

# AppWorld: 2 lanes x 3 workers (WSL memory limit)
( powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh test --systems no_memory ace awm reasoning_bank mem0 --test_first $N --procs 3 --force_parallel" > "$APP/logs/first${N}_laneA.log" 2>&1
  log "AppWorld lane A (no_memory ace awm reasoning_bank mem0) done (rc=$?)" ) &
sleep 20
( powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh test --systems amem memoryos bm25 vector --test_first $N --procs 3 --force_parallel" > "$APP/logs/first${N}_laneB.log" 2>&1
  log "AppWorld lane B (amem memoryos bm25 vector) done (rc=$?)" ) &

wait
"$PY" $ROOT/setup/store_hashes.py check $ROOT/first120_hashes_before.json >> "$LOG" 2>&1
log "FINAL hash check (all stores except ALFWorld memoryos) rc=$? (0 = unchanged)"
( cd "$EMB" && "$PY" scripts/run_alfworld_main.py report > /dev/null 2>&1 )
( cd "$TOOL" && "$PY" -m bfcl_eval.scripts.cross_episode.run_split_experiment report > /dev/null 2>&1 )
powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh report" > /dev/null 2>&1
log "next80 finished"
