#!/bin/bash
# Overnight queue (2026-10-04): finish ALFWorld rollout, build all memories for ALFWorld + BFCL,
# then a 10-sample read-only test per system. Runs detached; progress -> night_queue.log.
# Concurrency is kept low because the OpenAI org limit is 200K TPM for gpt-4.1-mini.
ROOT=/d/Project/Thesis_Claude
PY=$ROOT/.venv/Scripts/python.exe
EMB=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB
TOOL=$ROOT/EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL
ALF_RUN=$EMB/output/main_gpt-4.1-mini
BF_RUN=$TOOL/cross_episode_results/split_main_gpt-4.1-mini
LOG=$ROOT/night_queue.log
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
# Count python.exe processes whose command line matches $1 (excludes this shell/powershell itself).
nproc_match() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count" | tr -d '\r'; }
wait_for_none() { while [ "$(nproc_match "$1")" != "0" ]; do sleep 60; done; }
alf() { (cd "$EMB" && "$PY" scripts/run_alfworld_main.py "$@"); }
bf()  { (cd "$TOOL" && "$PY" -m bfcl_eval.scripts.cross_episode.run_split_experiment "$@"); }
ensure_server() {
  if ! curl -s --max-time 5 http://localhost:36005/ >/dev/null; then
    log "ALFWorld server down -> restarting"
    powershell.exe -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','--','bash','/mnt/d/Project/Thesis_Claude/setup/wsl_alfworld_server.sh','36005' -WindowStyle Hidden" >/dev/null
    for i in $(seq 1 40); do curl -s --max-time 3 http://localhost:36005/ >/dev/null && break; sleep 5; done
  fi
}

log "queue start"

# 1. ALFWorld rollout: wait for the running one, then resume until all 200 exist (max 4 tries).
wait_for_none 'run_alfworld_main.py rollout'
for t in 1 2 3 4; do
  n=$(ls "$ALF_RUN/train_rollouts" | grep -c '^alfworld_')
  log "ALFWorld rollout: $n/200 trajectories"
  [ "$n" -ge 200 ] && break
  ensure_server
  alf rollout --parallel 4 >> "$ALF_RUN/logs/rollout_queue.log" 2>&1
done
log "ALFWorld rollout final: $(ls "$ALF_RUN/train_rollouts" | grep -c '^alfworld_')/200"

build_alf() {  # resumable; retry once on error exit
  for t in 1 2; do alf build --systems "$1" > "$ALF_RUN/logs/build_$1.queue.log" 2>&1 && break; done
  log "ALFWorld build $1 done (rc=$?)"
}

# 2a. ALFWorld MemoryOS (slowest, ~6 h) on its own lane.
( build_alf memoryos ) &
LANE_A=$!

# 2b. ALFWorld other systems, one at a time, then their 10-game test.
(
  for s in awm reasoning_bank ace mem0 amem bm25 vector; do build_alf "$s"; done
  ensure_server
  alf test --test_subset 10 --parallel 4 --systems no_memory awm reasoning_bank ace mem0 amem bm25 vector \
      > "$ALF_RUN/logs/test10.queue.log" 2>&1
  log "ALFWorld test10 (7 systems + no_memory) done (rc=$?)"
) &
LANE_B=$!

# 2c. BFCL: wait for the ACE / MemoryOS builds already running, rebuild mem0 + A-mem, then test.
(
  wait_for_none 'run_split_experiment build'
  log "BFCL: ace/memoryos builds finished"
  for s in mem0 amem; do
    bf build --systems "$s" > "$BF_RUN/logs/build_$s.queue.log" 2>&1
    log "BFCL build $s done (rc=$?)"
  done
  bf test --test_per_env 3 --workers 2 > "$BF_RUN/logs/test12.queue.log" 2>&1
  log "BFCL test (3/env) done (rc=$?)"
  bf report > /dev/null 2>&1
) &
LANE_C=$!

wait $LANE_B
wait $LANE_A
ensure_server
alf test --test_subset 10 --parallel 4 --systems memoryos >> "$ALF_RUN/logs/test10.queue.log" 2>&1
log "ALFWorld test10 memoryos done (rc=$?)"
alf report > /dev/null 2>&1
wait $LANE_C
log "queue finished"
