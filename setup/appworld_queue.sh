#!/bin/bash
# AppWorld memory queue (runs inside WSL, detached). Progress -> /mnt/d/Project/Thesis_Claude/appworld_queue.log
#   lane A: official ACE offline (no-GT) adaptation on train (90 tasks)
#   lane B: no-memory train rollout -> build 7 adapter-based systems (MemoryOS in its own process)
#   then  : 10-task smoke test on test_normal for every system + report
ROOT=/mnt/d/Project/Thesis_Claude
LOG=$ROOT/appworld_queue.log
RUN=$ROOT/ace-appworld/experiments/outputs/thesis_memory
mkdir -p $RUN/logs
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
R() { bash $ROOT/setup/wsl_appworld_run.sh "$@"; }

source $HOME/appworld-venv/bin/activate
export APPWORLD_PROJECT_PATH=$ROOT/ace-appworld
export OPENAI_API_KEY="$(powershell.exe -NoProfile -Command "[Environment]::GetEnvironmentVariable('OPENAI_API_KEY','Machine')" | tr -d '\r')"
export PYTHONUNBUFFERED=1

log "appworld queue start"

(
  cd $APPWORLD_PROJECT_PATH
  appworld run THESIS_ACE_offline_no_GT_adaptation > $RUN/logs/ace_adaptation.log 2>&1
  log "ACE offline adaptation done (rc=$?); playbook: $(wc -c < experiments/playbooks/thesis_ace_offline_no_gt_gpt-4.1-mini.txt 2>/dev/null) chars"
) &
LANE_A=$!

(
  for t in 1 2 3; do
    R rollout --procs 4 >> $RUN/logs/rollout.log 2>&1 && break
    log "rollout attempt $t had errors; retrying missing tasks"
  done
  log "train rollout: $(ls $RUN/train_rollouts | grep -c json)/90 tasks"
  R build --systems memoryos > $RUN/logs/build_memoryos.log 2>&1 &
  MOS=$!
  for s in awm reasoning_bank mem0 amem bm25 vector; do
    R build --systems $s > $RUN/logs/build_$s.log 2>&1
    log "build $s done (rc=$?)"
  done
  wait $MOS
  log "build memoryos done (rc=$?)"
) &
LANE_B=$!

wait $LANE_A
wait $LANE_B
R test --test_n 10 --procs 4 > $RUN/logs/test10.log 2>&1
log "test10 done (rc=$?)"
R report > /dev/null 2>&1
log "appworld queue finished"
