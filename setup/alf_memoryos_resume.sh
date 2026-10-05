#!/bin/bash
# Resume ALFWorld MemoryOS build from trajectory #97 (96/200 done when credits ran out), then test10.
ROOT=/d/Project/Thesis_Claude
EMB=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB
D=$EMB/output/main_gpt-4.1-mini
LOG=$ROOT/night_queue.log
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
cd "$EMB"
log "ALFWorld memoryos resume start ($(wc -l < $D/memory/memoryos/build_log.jsonl.done)/200 done)"
"$ROOT/.venv/Scripts/python.exe" scripts/run_alfworld_main.py build --systems memoryos > "$D/logs/build_memoryos.resume.log" 2>&1
rc=$?
log "ALFWorld memoryos build done (rc=$rc, $(wc -l < $D/memory/memoryos/build_log.jsonl.done)/200)"
[ $rc -eq 0 ] || exit $rc
if ! curl -s --max-time 5 http://localhost:36005/ >/dev/null; then
  powershell.exe -NoProfile -Command "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','--','bash','/mnt/d/Project/Thesis_Claude/setup/wsl_alfworld_server.sh','36005' -WindowStyle Hidden" >/dev/null
  for i in $(seq 1 40); do curl -s --max-time 3 http://localhost:36005/ >/dev/null && break; sleep 5; done
fi
"$ROOT/.venv/Scripts/python.exe" scripts/run_alfworld_main.py test --systems memoryos --test_subset 10 --parallel 4 > "$D/logs/test10_memoryos.log" 2>&1
log "ALFWorld test10 memoryos done (rc=$?)"
"$ROOT/.venv/Scripts/python.exe" scripts/run_alfworld_main.py report > /dev/null 2>&1
log "memoryos resume finished"
