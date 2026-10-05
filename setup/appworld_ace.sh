#!/bin/bash
# Official ACE offline (no-GT) adaptation on AppWorld train, then the 10-task test for ACE.
ROOT=/mnt/d/Project/Thesis_Claude
LOG=$ROOT/appworld_queue.log
RUN=$ROOT/ace-appworld/experiments/outputs/thesis_memory
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
source $HOME/appworld-venv/bin/activate
export APPWORLD_PROJECT_PATH=$ROOT/ace-appworld
export OPENAI_API_KEY="$(powershell.exe -NoProfile -Command "[Environment]::GetEnvironmentVariable('OPENAI_API_KEY','Machine')" | tr -d '\r')"
export PYTHONUNBUFFERED=1
cd $APPWORLD_PROJECT_PATH
log "ACE offline adaptation (re)start"
appworld run THESIS_ACE_offline_no_GT_adaptation > $RUN/logs/ace_adaptation.log 2>&1
log "ACE offline adaptation done (rc=$?); playbook: $(wc -c < experiments/playbooks/thesis_ace_offline_no_gt_gpt-4.1-mini.txt 2>/dev/null) chars"
# wait for the main queue's test step so the two don't overlap on the same split
until grep -q "test10 done" $LOG; do sleep 120; done
bash $ROOT/setup/wsl_appworld_run.sh test --systems ace --test_n 10 --procs 4 > $RUN/logs/test10_ace.log 2>&1
log "ACE test10 done (rc=$?)"
