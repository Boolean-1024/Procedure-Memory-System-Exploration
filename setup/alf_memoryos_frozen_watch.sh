#!/bin/bash
# Hash-guard the ALFWorld MemoryOS test: snapshot the store the moment the build finishes
# (build_summary.json written), then compare after the queued test10 finishes.
ROOT=/d/Project/Thesis_Claude
PY=$ROOT/.venv/Scripts/python.exe
M=$ROOT/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/output/main_gpt-4.1-mini/memory/memoryos
LOG=$ROOT/frozen_test10.log
export PYTHONUTF8=1
until [ -f "$M/build_summary.json" ]; do sleep 2; done
"$PY" - <<PYEOF > $ROOT/frozen_hashes_alf_memoryos.json
import hashlib, os, json
base=r"$M".replace("/d/","D:/",1)
print(json.dumps({os.path.relpath(os.path.join(r,f),base): hashlib.md5(open(os.path.join(r,f),'rb').read()).hexdigest()
                  for r,_,fs in os.walk(base) for f in fs}))
PYEOF
echo "[$(date '+%m-%d %H:%M:%S')] ALFWorld memoryos store snapshot taken ($(grep -o '": "' $ROOT/frozen_hashes_alf_memoryos.json | wc -l) files)" >> $LOG
until grep -q "ALFWorld test10 memoryos done" $ROOT/night_queue.log; do sleep 30; done
"$PY" - <<PYEOF >> $LOG
import hashlib, os, json
base=r"$M".replace("/d/","D:/",1)
before=json.load(open(r"$ROOT/frozen_hashes_alf_memoryos.json".replace("/d/","D:/",1)))
now={os.path.relpath(os.path.join(r,f),base): hashlib.md5(open(os.path.join(r,f),'rb').read()).hexdigest() for r,_,fs in os.walk(base) for f in fs}
ch=[k for k in before if before[k]!=now.get(k)]; add=[k for k in now if k not in before]
print(f"ALFWorld memoryos hash check: {len(before)} files, changed={len(ch)} added={len(add)}", ch[:10], add[:10])
PYEOF
