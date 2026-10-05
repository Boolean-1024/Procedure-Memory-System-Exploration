"""Snapshot / compare MD5 hashes of every memory-store file (strict-frozen check).

  python setup/store_hashes.py snap  out.json   # before testing
  python setup/store_hashes.py check out.json   # after testing: lists any changed/added/removed file
Excludes stores that are still being built (pass --exclude <substring> ...).
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORES = {
    "ALFWorld": "EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/output/main_gpt-4.1-mini/memory",
    "BFCL": "EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL/cross_episode_results/split_main_gpt-4.1-mini/memory",
    "AppWorld": "ace-appworld/experiments/outputs/thesis_memory/memory",
    "AppWorld-ACE-playbook": "ace-appworld/experiments/playbooks/thesis_ace_offline_no_gt_gpt-4.1-mini.txt",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(excludes):
    out = {}
    for name, rel in STORES.items():
        base = os.path.join(ROOT, rel)
        files = [base] if os.path.isfile(base) else [
            os.path.join(r, f) for r, _, fs in os.walk(base) for f in fs]
        for p in files:
            key = os.path.relpath(p, ROOT).replace("\\", "/")
            if any(x in key for x in excludes):
                continue
            out[key] = md5(p)
    return out


def main():
    mode, path = sys.argv[1], sys.argv[2]
    excludes = sys.argv[sys.argv.index("--exclude") + 1:] if "--exclude" in sys.argv else []
    now = snapshot(excludes)
    if mode == "snap":
        json.dump({"excludes": excludes, "hashes": now}, open(path, "w"), indent=1)
        print(f"snapshot: {len(now)} files -> {path}")
        return
    old = json.load(open(path))
    now = snapshot(old["excludes"])
    before = old["hashes"]
    changed = [k for k in before if k in now and before[k] != now[k]]
    removed = [k for k in before if k not in now]
    added = [k for k in now if k not in before]
    print(f"checked {len(before)} files: changed={len(changed)} added={len(added)} removed={len(removed)}")
    for label, lst in (("CHANGED", changed), ("ADDED", added), ("REMOVED", removed)):
        for k in lst[:30]:
            print(f"  {label}: {k}")
    sys.exit(1 if (changed or added or removed) else 0)


if __name__ == "__main__":
    main()
