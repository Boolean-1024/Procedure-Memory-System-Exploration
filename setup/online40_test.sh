#!/bin/bash
# Online (non-frozen) test, first 40 cases per benchmark, in order (2026-10-05).
# Every memory system starts from checkpoint frozen_v1_2026-10-05 (working copies made by
# setup/online_init.py) and updates its own store after every test case: inject -> solve ->
# update, strictly sequential per system. Different systems run side by side (separate stores).
#   ALFWorld : games 2420..2459              -> test_online/<sys>/
#   BFCL     : first 10 ids of each env      -> test_online/<sys>/<env>/
#   AppWorld : first 40 tasks of test_normal -> test_online/test_normal/<sys>/  (ACE: official online no-GT)
# no_memory is not rerun (nothing to update); compare with the frozen no_memory results.
ROOT=/d/Project/Thesis_Claude
PY=$ROOT/.venv/Scripts/python.exe
EMB=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB
TOOL=$ROOT/EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL
ALF=$EMB/output/main_gpt-4.1-mini
BF=$TOOL/cross_episode_results/split_main_gpt-4.1-mini
APP=$ROOT/ace-appworld/experiments/outputs/thesis_memory
LOG=$ROOT/online40_test.log
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ensure_server() {
  until curl -s --max-time 5 http://localhost:36005/ >/dev/null; do
    log "ALFWorld server down -> restarting"
    powershell.exe -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','--','bash','/mnt/d/Project/Thesis_Claude/setup/wsl_alfworld_server.sh','36005' -WindowStyle Hidden" >/dev/null
    sleep 40
  done
}
alf_online() {  # $1 = system; resume (finished games are skipped) until 40/40, max 3 rounds
  for t in 1 2 3; do
    ensure_server
    ( cd "$EMB" && "$PY" scripts/run_alfworld_main.py test --online --test_first 40 --systems $1 \
        >> "$ALF/logs/online40_$1.log" 2>&1 )
    rc=$?
    n=$(ls "$ALF/test_online/$1" 2>/dev/null | grep -c '^alfworld_')
    [ "$n" -ge 40 ] && break
    [ $rc -eq 3 ] && { log "ALFWorld $1: FATAL API error (rc=3), stopping"; break; }
    log "ALFWorld $1: $n/40 after round $t, retrying"; sleep 60
  done
  log "ALFWorld $1 online40: $(ls "$ALF/test_online/$1" 2>/dev/null | grep -c '^alfworld_')/40"
}
app_lane() {  # sequential systems in one lane (WSL memory limit)
  for s in "$@"; do
    powershell.exe -NoProfile -Command "wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_appworld_run.sh test --online --systems $s --test_first 40" >> "$APP/logs/online40_$s.log" 2>&1
    log "AppWorld $s online40 done (rc=$?): $(ls $APP/test_online/test_normal/$s 2>/dev/null | grep -c '_[0-9]*\.json$')/40"
  done
}

log "online40 start"
"$PY" $ROOT/setup/memory_checkpoint.py diff frozen_v1_2026-10-05 >> "$LOG" 2>&1
log "frozen stores vs checkpoint before the online run: rc=$? (0 = identical)"
ensure_server

for s in awm reasoning_bank ace mem0 amem memoryos bm25 vector; do alf_online $s & sleep 5; done

for s in awm reasoning_bank ace mem0 amem memoryos bm25 vector; do
  ( cd "$TOOL" && "$PY" -m bfcl_eval.scripts.cross_episode.run_split_experiment test --online --systems $s \
      --test_per_env 10 --test_order id >> "$BF/logs/online40_$s.log" 2>&1
    log "BFCL $s online40 done (rc=$?)" ) &
done

app_lane ace awm reasoning_bank &
sleep 20
app_lane mem0 amem memoryos &
sleep 20
app_lane bm25 vector &

wait
"$PY" $ROOT/setup/memory_checkpoint.py diff frozen_v1_2026-10-05 >> "$LOG" 2>&1
log "frozen stores vs checkpoint after the online run: rc=$? (0 = identical)"
log "online40 finished"
