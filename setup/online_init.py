"""Create the working copies for the online (non-frozen) test from a memory checkpoint.

Online runs read AND write these copies; the frozen stores and the checkpoint stay untouched:
    ALFWorld : .../main_gpt-4.1-mini/memory_online/<system>/
    BFCL     : .../split_main_gpt-4.1-mini/memory_online/<system>/<env>/
    AppWorld : .../thesis_memory/memory_online/<system>/
               experiments/playbooks/thesis_ace_online_start_gpt-4.1-mini.txt (ACE start playbook)

    python setup/online_init.py [checkpoint_name] [--force]
--force replaces existing working copies (and the ACE online playbook) with fresh ones.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_checkpoint as mc  # noqa: E402

ACE_PB_DIR = mc.ROOT / "ace-appworld/experiments/playbooks"
TARGETS = {
    "alfworld": mc.STORES["alfworld"].parent / "memory_online",
    "bfcl": mc.STORES["bfcl"].parent / "memory_online",
    "appworld": mc.STORES["appworld"].parent / "memory_online",
    "appworld_ace_playbook": ACE_PB_DIR / "thesis_ace_online_start_gpt-4.1-mini.txt",
}
ACE_ONLINE_OUT = ACE_PB_DIR / "thesis_ace_online_gpt-4.1-mini.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", default="frozen_v1_2026-10-05")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not mc.verify(a.checkpoint):
        sys.exit("checkpoint damaged; not creating working copies")
    m = mc.load_manifest(a.checkpoint)
    existing = [str(p) for p in TARGETS.values() if p.exists()] + ([str(ACE_ONLINE_OUT)] if ACE_ONLINE_OUT.exists() else [])
    if existing and not a.force:
        sys.exit("working copies already exist (use --force to reset them):\n  " + "\n  ".join(existing))
    for key, dst in TARGETS.items():
        mc._rm(dst)
        mc._copy(mc.CKPT_DIR / a.checkpoint / key, dst)
        mc.set_readonly(dst, False)
        ok = mc.report("copy", key, m["stores"][key]["files"], mc.hash_tree(dst))
        if not ok:
            sys.exit(f"copy of {key} does not match the checkpoint")
    mc._rm(ACE_ONLINE_OUT)  # ACE online continues from this file once it exists
    print(f"online working copies created from {a.checkpoint!r}")


if __name__ == "__main__":
    main()
