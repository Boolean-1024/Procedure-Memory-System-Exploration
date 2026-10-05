"""Assemble the Hugging Face dataset folder (hf_dataset/) from the experiment outputs.

Copies the shared no-memory training rollouts, the built memory stores, and the first-40
strict-frozen test results of ALFWorld / BFCL / AppWorld. Local absolute paths in text files
are replaced with <ROOT>, and every text file is scanned for API keys before anything is
written. Re-run after the ALFWorld MemoryOS build to add it (--with_alf_memoryos).

    python setup/make_hf_dataset.py [--with_alf_memoryos]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf_dataset"
ALF = ROOT / "EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/output/main_gpt-4.1-mini"
BFCL = ROOT / "EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL/cross_episode_results/split_main_gpt-4.1-mini"
APP = ROOT / "ace-appworld/experiments/outputs"
APP_PLAYBOOK = ROOT / "ace-appworld/experiments/playbooks/thesis_ace_offline_no_gt_gpt-4.1-mini.txt"

TEXT_EXT = {".json", ".jsonl", ".md", ".csv", ".txt", ".log", ".yaml", ".yml", ".jsonnet", ".py", ".out"}
KEY_PATTERNS = [re.compile(p) for p in (r"sk-[A-Za-z0-9_-]{20,}", r"ghp_[A-Za-z0-9]{30,}", r"hf_[A-Za-z0-9]{30,}")]
# local path prefixes as they appear in raw text and in JSON-escaped text
PATH_FORMS = [
    str(ROOT), str(ROOT).replace("\\", "/"), str(ROOT).replace("\\", "\\\\"),
    "/mnt/d/Project/Thesis_Claude", "/d/Project/Thesis_Claude",
]


def machine_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as k:
                val = winreg.QueryValueEx(k, name)[0]
        except OSError:
            pass
    return val


def copy(src: Path, dst: Path, ignore=None):
    if not src.exists():
        print(f"  [skip, missing] {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)
    else:
        shutil.copy2(src, dst)


def build(with_alf_memoryos: bool):
    card = (OUT / "README.md").read_text(encoding="utf-8") if (OUT / "README.md").exists() else None
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    if card:  # keep the hand-written dataset card across rebuilds
        (OUT / "README.md").write_text(card, encoding="utf-8")
    ign = shutil.ignore_patterns("__pycache__", "*.lock", "*.pyc")

    print("ALFWorld")
    a = OUT / "alfworld"
    copy(ALF / "train_indices.json", a / "split/train_indices.json")
    copy(ALF / "run_meta.json", a / "split/run_meta.json")
    copy(ALF / "train_rollouts", a / "train_rollouts", ign)
    for sysdir in sorted((ALF / "memory").iterdir()):
        if sysdir.name == "memoryos" and not with_alf_memoryos:
            continue
        copy(sysdir, a / "memory" / sysdir.name, ign)
    copy(ALF / "test", a / "test", ign)
    copy(ALF / "report.md", a / "report.md")
    copy(ALF / "report.csv", a / "report.csv")

    print("BFCL")
    b = OUT / "bfcl"
    copy(BFCL / "split.json", b / "split/split.json")
    copy(BFCL / "train_rollouts", b / "train_rollouts", ign)
    copy(BFCL / "memory", b / "memory", ign)
    copy(BFCL / "test", b / "test", ign)
    copy(BFCL / "report.md", b / "report.md")
    copy(BFCL / "report.csv", b / "report.csv")

    print("AppWorld")
    p = OUT / "appworld"
    tm = APP / "thesis_memory"
    copy(tm / "train_ids.json", p / "split/train_ids.json")
    copy(tm / "train_rollouts", p / "train_rollouts", ign)
    copy(tm / "memory", p / "memory", ign)
    copy(APP_PLAYBOOK, p / "memory/ace/playbook.txt")
    copy(tm / "test/test_normal", p / "test", ign)
    copy(tm / "report_test_normal.md", p / "report.md")
    # official AppWorld per-task evaluation reports (unit-test results) for the tested tasks
    for sysout in sorted(APP.glob("thesis_test_test_normal_*")):
        system = sysout.name.removeprefix("thesis_test_test_normal_")
        tested = {f.stem for f in (p / "test" / system).glob("*.json")}
        for task in sorted((sysout / "tasks").glob("*")):
            if task.name in tested:
                copy(task / "evaluation", p / "evaluation" / system / task.name, ign)


def scrub_and_scan():
    secrets = [s for s in (machine_env("OPENAI_API_KEY"), os.environ.get("HF_TOKEN", "")) if len(s) > 10]
    hits, scrubbed, n_text = [], 0, 0
    for f in OUT.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in TEXT_EXT:
            continue
        n_text += 1
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = text
        for form in PATH_FORMS:
            new = new.replace(form, "<ROOT>")
        if new != text:
            f.write_text(new, encoding="utf-8")
            scrubbed += 1
        if any(s in new for s in secrets):
            hits.append(f"{f.relative_to(OUT)}: exact machine key")
        for pat in KEY_PATTERNS:
            for m in pat.finditer(new):
                hits.append(f"{f.relative_to(OUT)}: {m.group(0)[:12]}...")
    return n_text, scrubbed, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_alf_memoryos", action="store_true")
    args = ap.parse_args()
    build(args.with_alf_memoryos)
    n_text, scrubbed, hits = scrub_and_scan()
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    n = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"\nfiles: {n}, size: {total / 2**20:.1f} MiB, text files: {n_text}, path-scrubbed: {scrubbed}")
    for top in sorted(OUT.iterdir()):
        if top.is_dir():
            sz = sum(f.stat().st_size for f in top.rglob("*") if f.is_file())
            print(f"  {top.name:10s} {sz / 2**20:8.1f} MiB")
    if hits:
        print(f"\n!! {len(hits)} possible secrets found:")
        for h in hits[:30]:
            print("  ", h)
        sys.exit(2)
    print("secret scan: clean")


if __name__ == "__main__":
    main()
