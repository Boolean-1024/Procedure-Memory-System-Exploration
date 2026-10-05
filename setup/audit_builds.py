"""Audit memory builds for silent failures (rate limits swallowed by vendored libraries).

For every system's build_log.jsonl: number of update() calls, how many made zero
memory-side LLM calls, explicit errors; plus failure strings found in the build stdout.
Usage: python setup/audit_builds.py
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {
    "ALFWorld": os.path.join(ROOT, "EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/output/main_gpt-4.1-mini"),
    "BFCL": os.path.join(ROOT, "EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL/cross_episode_results/split_main_gpt-4.1-mini"),
    "AppWorld": os.path.join(ROOT, "ace-appworld/experiments/outputs/thesis_memory"),
}
# Systems that never call an LLM on write, or only for some trajectories.
NO_LLM = {"bm25", "vector"}
FAIL_PAT = re.compile(r"LLM extraction failed|Error in memory evolution|update failed|rate_limit_exceeded|Traceback")


def audit(bench, run):
    print(f"\n=== {bench}: {run}")
    print(f"{'system':15s} {'updates':>7s} {'0-LLM':>6s} {'errors':>6s} {'fail-lines':>10s}  note")
    for sysdir in sorted(glob.glob(os.path.join(run, "memory", "*"))):
        name = os.path.basename(sysdir)
        logs = glob.glob(os.path.join(sysdir, "**", "build_log.jsonl"), recursive=True)
        n = zero = err = 0
        for lf in logs:
            for line in open(lf, encoding="utf-8"):
                rec = json.loads(line)
                n += 1
                zero += len(rec.get("llm_calls", [])) == 0
                err += bool(rec.get("error"))
        fails = 0
        for out in glob.glob(os.path.join(run, "logs", f"build_{name}*")):
            if "ratelimited" in out:
                continue
            fails += sum(1 for l in open(out, encoding="utf-8", errors="replace") if FAIL_PAT.search(l))
        note = ""
        if name in NO_LLM:
            note = "no LLM on write (expected)"
        elif name == "awm":
            note = "0-LLM expected for failed trajectories"
        elif name in ("amem",):
            note = "0-LLM expected only for the first note per store"
        elif zero:
            note = "CHECK: updates without LLM output"
        print(f"{name:15s} {n:7d} {zero:6d} {err:6d} {fails:10d}  {note}")


for bench, run in RUNS.items():
    if os.path.isdir(run):
        audit(bench, run)
