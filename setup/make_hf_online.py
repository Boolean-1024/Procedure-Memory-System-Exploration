"""Assemble hf_online/ with the online (non-frozen) test run for the Hugging Face dataset.

Adds, per benchmark, under <bench>/online/:
    test/<system>/...      traces of the online run (first 40 cases, memory updated after each case)
    memory/<system>/...    the stores AFTER the online run (start state = the frozen stores in <bench>/memory/)
    evaluation/<system>/   AppWorld unit-test reports of the online run
ALFWorld and AppWorld MemoryOS were not run online and are not included.
Local paths are replaced with <ROOT> and every text file is scanned for API keys.

    python setup/make_hf_online.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_hf_dataset as base  # noqa: E402  (copy / scrub / scan helpers)

OUT = base.ROOT / "hf_online"
ign = shutil.ignore_patterns("__pycache__", "*.lock", "*.pyc")


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    print("ALFWorld")
    a = OUT / "alfworld/online"
    for s in ["awm", "reasoning_bank", "ace", "mem0", "amem", "bm25", "vector"]:
        base.copy(base.ALF / "test_online" / s, a / "test" / s, ign)
        base.copy(base.ALF / "memory_online" / s, a / "memory" / s, ign)

    print("BFCL")
    b = OUT / "bfcl/online"
    for s in ["awm", "reasoning_bank", "ace", "mem0", "amem", "memoryos", "bm25", "vector"]:
        base.copy(base.BFCL / "test_online" / s, b / "test" / s, ign)
        base.copy(base.BFCL / "memory_online" / s, b / "memory" / s, ign)

    print("AppWorld")
    p = OUT / "appworld/online"
    tm = base.APP / "thesis_memory"
    base.copy(tm / "test_order_normal.json", p / "test_order_normal.json")
    for s in ["ace", "awm", "reasoning_bank", "mem0", "amem", "bm25", "vector"]:
        base.copy(tm / "test_online/test_normal" / s, p / "test" / s, ign)  # ace: per-task playbook snapshots
        if s != "ace":
            base.copy(tm / "memory_online" / s, p / "memory" / s, ign)
        tested = {f.stem for f in (p / "test" / s).glob("*_*.json")}
        for task in sorted((base.APP / f"thesis_online_test_normal_{s}" / "tasks").glob("*")):
            if task.name in tested:
                base.copy(task / "evaluation", p / "evaluation" / s / task.name, ign)
    pb = base.ROOT / "ace-appworld/experiments/playbooks"
    base.copy(pb / "thesis_ace_online_start_gpt-4.1-mini.txt", p / "memory/ace/playbook_start.txt")
    base.copy(pb / "thesis_ace_online_gpt-4.1-mini.txt", p / "memory/ace/playbook_after_online.txt")


def main():
    build()
    base.OUT = OUT  # scrub_and_scan works on base.OUT
    n_text, scrubbed, hits = base.scrub_and_scan()
    files = [f for f in OUT.rglob("*") if f.is_file()]
    print(f"\nfiles: {len(files)}, size: {sum(f.stat().st_size for f in files) / 2**20:.1f} MiB, "
          f"text files: {n_text}, path-scrubbed: {scrubbed}")
    for d in sorted(OUT.iterdir()):
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        print(f"  {d.name:10s} {sz / 2**20:8.1f} MiB")
    if hits:
        print(f"\n!! {len(hits)} possible secrets:")
        for h in hits[:30]:
            print("  ", h)
        sys.exit(2)
    print("secret scan: clean")


if __name__ == "__main__":
    main()
