"""BFCL multi-turn main experiment with an explicit train/test split.

BFCL has no official train split. We split each of the 4 environments
(gorilla_fs, vehicle_control, trading_bot, travel_api; 50 samples each) 1:1 with a
fixed seed -> 25 train / 25 test per environment (100 / 100 overall).

Protocol (identical for every memory system):
  split   : write <run_dir>/split.json (reused if it already exists)
  rollout : a no-memory agent runs all train samples once (FC mode);
            trajectories -> <run_dir>/train_rollouts/result.jsonl
  build   : for each system and environment, a fresh memory store ingests that
            environment's train trajectories via the backend's own update()
            (in split order, serially). Write log: memory/<sys>/<env>/build_log.jsonl
  test    : each system runs the environment's test samples with its store loaded
            read-only (--memory-readonly). Retrieval log: test/<sys>/<env>/memory_log.jsonl
            The no-memory baseline runs on the same test samples.
  report  : success rate / progress score overall and per environment -> report.md / report.csv

Usage (from CROSSEP-TOOL/):
  python -m bfcl_eval.scripts.cross_episode.run_split_experiment all
  python -m bfcl_eval.scripts.cross_episode.run_split_experiment test --systems no_memory ace
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from bfcl_eval.memory.factory import build_memory
from bfcl_eval.memory.llm_env import embed_model, llm_api_key, llm_model
from bfcl_eval.memory.logging_wrapper import LoggedMemory
from bfcl_eval.scripts.cross_episode.run_batch_generate import memory_kwargs

IDS_DIR = Path(__file__).resolve().parent / "ids"
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # CROSSEP-TOOL/
ENVS = ["gorilla_fs", "vehicle_control", "trading_bot", "travel_api"]

# system name -> (bfcl --memory-type, top_k). top_k follows the EvoMemBench launch scripts.
SYSTEMS = {
    "awm": ("agent_workflow", 3),
    "reasoning_bank": ("reasoning_bank", 3),
    "ace": ("ace", 3),
    "mem0": ("mem0", 3),
    "amem": ("amem", 3),
    "memoryos": ("memoryos", 3),
    "bm25": ("bm25", 10),
    "vector": ("qwen3_embedding", 10),
}
ALL_SYSTEMS = list(SYSTEMS)
SERIAL_TEST = {"mem0", "amem", "memoryos"}  # read-time thread safety unverified


def make_split(run_dir: Path, seed: int, train_frac: float) -> dict:
    path = run_dir / "split.json"
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    split = {"seed": seed, "train_frac": train_frac, "envs": {}}
    for env in ENVS:
        ids = json.load(open(IDS_DIR / f"ids_{env}.json", encoding="utf-8"))["multi_turn_ours"]
        ids = list(ids)
        random.Random(f"{seed}-{env}").shuffle(ids)
        k = round(len(ids) * train_frac)
        split["envs"][env] = {"train": ids[:k], "test": ids[k:]}
    json.dump(split, open(path, "w", encoding="utf-8"), indent=2)
    return split


def run(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(c if len(c) < 120 else c[:117] + "..." for c in cmd)
    print(f"\n$ {printable}", flush=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    with open(log_file, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        return proc.wait()


def generate_cmd(a, out_dir: Path, run_name: str, ids: list[str], workers: int) -> list[str]:
    # run_batch_generate treats an empty --ids as "all 200 samples"; never allow that here.
    assert ids, f"empty id list for {run_name}"
    cmd = [sys.executable, "-m", "bfcl_eval.scripts.cross_episode.run_batch_generate",
           "--num-workers", str(workers), "--output-dir", str(out_dir), "--run-name", run_name,
           "--mode", a.mode, "--ids", ",".join(ids), "--model", a.model, "--resume"]
    return cmd


def evaluate_cmd(result_jsonl: Path) -> list[str]:
    return [sys.executable, "-m", "bfcl_eval.scripts.cross_episode.run_batch_evaluate",
            "--input", str(result_jsonl), "--num-workers", "4"]


# ── steps ────────────────────────────────────────────────────────────────────

def step_rollout(a, split) -> int:
    ids = [i for env in ENVS for i in split["envs"][env]["train"]]
    out = a.run_dir
    rc = run(generate_cmd(a, out, "train_rollouts", ids, a.workers) + ["--memory-type", "none"],
             a.run_dir / "logs" / "rollout.log")
    rc |= run(evaluate_cmd(out / "train_rollouts" / "result.jsonl"), a.run_dir / "logs" / "rollout_eval.log")
    return rc


def load_rollouts(a) -> dict[str, dict]:
    path = a.run_dir / "train_rollouts" / "result.jsonl"
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("error") is None:
                    recs[r["id"]] = r  # last successful record wins (resume-safe)
    return recs


def load_train_success(a) -> dict[str, bool]:
    """id -> success of each train trajectory (from run_batch_evaluate's per_sample.csv)."""
    path = a.run_dir / "train_rollouts" / "per_sample.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: int(float(r["success"])) == 1 for r in csv.DictReader(f)}


def step_build(a, split) -> int:
    recs = load_rollouts(a)
    success = load_train_success(a)
    mem_model = os.environ.get("BFCL_MEMORY_MODEL") or a.model
    rc = 0
    for name in a.systems:
        if name == "no_memory":
            continue
        mtype, top_k = SYSTEMS[name]
        for env in ENVS:
            mdir = a.run_dir / "memory" / name / env
            log_path = mdir / "build_log.jsonl"
            done_path = mdir / "build_done.txt"
            done = set(done_path.read_text().split()) if done_path.exists() else set()
            kwargs, _ = memory_kwargs(mtype, mdir, api_key=llm_api_key(), mem_model=mem_model,
                                      clear=not done, readonly=False, top_k=top_k)
            memory = LoggedMemory(build_memory(mtype, **kwargs), log_path=str(log_path), phase="build")
            train_ids = split["envs"][env]["train"]
            missing = [i for i in train_ids if i not in recs]
            if missing:
                print(f"[warn] {name}/{env}: {len(missing)} train trajectories missing/errored: {missing[:5]}")
            totals = {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0,
                      "latency_s": 0.0, "n_llm_calls": 0, "n_embed_calls": 0}
            errors = []
            t0 = time.time()
            pending = [i for i in train_ids if i in recs and i not in done]
            print(f"[build] {name}/{env}: {len(pending)} pending, {len(done)} done")
            for sid in pending:
                memory.set_context(sid)
                memory.begin_sample()
                try:
                    if mtype == "agent_workflow":
                        # Thesis fix: the backend defaults to is_correct=True, i.e. it would
                        # induce workflows from failed trajectories too. Pass the evaluated
                        # outcome so AWM learns from successes only, as on ALFWorld.
                        if sid not in success:
                            raise RuntimeError("train success unknown; run `rollout` (evaluate) first")
                        memory.update(recs[sid]["full_message_history"],
                                      metadata={"is_correct": success[sid]})
                    else:
                        memory.update(recs[sid]["full_message_history"])
                    with open(done_path, "a") as f:
                        f.write(sid + "\n")
                except Exception as e:  # logged by LoggedMemory as well
                    errors.append({"id": sid, "error": f"{type(e).__name__}: {e}"})
                    print(f"  {sid} ERROR: {e}")
                u = memory.drain_usage()
                for k in totals:
                    totals[k] += getattr(u, k, 0) or 0
            summ_path = mdir / "build_summary.json"
            prev = json.load(open(summ_path, encoding="utf-8")) if summ_path.exists() else {}
            summary = {"system": name, "memory_type": mtype, "env": env, "top_k": top_k,
                       "n_train": len(train_ids), "errors": errors,
                       "wall_clock_s": time.time() - t0 + prev.get("wall_clock_s", 0)}
            for k, v in totals.items():
                summary[k] = v + prev.get(k, 0)
            json.dump(summary, open(summ_path, "w", encoding="utf-8"), indent=2)
            rc |= 1 if errors else 0
    return rc


def step_test(a, split) -> int:
    rc = 0
    for name in a.systems:
        for env in ENVS:
            ids = split["envs"][env]["test"]
            if a.test_order == "id":  # dataset order (multi_turn_ours_<n> ascending)
                ids = sorted(ids, key=lambda x: int(x.rsplit("_", 1)[1]))
            if a.test_per_env:
                # "3" = 3 per env; "3,3,2,2" = per env in ENVS order. Split order is a seeded shuffle.
                counts = [int(x) for x in str(a.test_per_env).split(",")]
                k = counts[ENVS.index(env)] if len(counts) == len(ENVS) else counts[0]
                ids = ids[:k]
            if not ids:
                continue
            out_dir = a.run_dir / ("test_online" if a.online else "test") / name
            workers = 1 if (a.online or (name in SERIAL_TEST and not a.force_parallel)) else a.workers
            cmd = generate_cmd(a, out_dir, env, ids, workers)
            if name == "no_memory":
                cmd += ["--memory-type", "none"]
            elif a.online:
                # online test: memory_online/ working copy, updated after every sample in id order
                mtype, top_k = SYSTEMS[name]
                mdir = a.run_dir / "memory_online" / name / env
                if not (mdir / "build_summary.json").exists():
                    print(f"[skip] {name}/{env}: no online working copy (run online_init first)")
                    rc |= 1
                    continue
                cmd += ["--memory-type", mtype, "--memory-load-from", str(mdir), "--memory-online",
                        "--memory-top-k", str(top_k),
                        "--memory-log", str(out_dir / env / "memory_log.jsonl"), "--log-phase", "online"]
            else:
                mtype, top_k = SYSTEMS[name]
                mdir = a.run_dir / "memory" / name / env
                if not (mdir / "build_summary.json").exists():
                    print(f"[skip] {name}/{env}: memory not built yet")
                    rc |= 1
                    continue
                cmd += ["--memory-type", mtype, "--memory-load-from", str(mdir), "--memory-readonly",
                        "--memory-top-k", str(top_k),
                        "--memory-log", str(out_dir / env / "memory_log.jsonl"), "--log-phase", "test"]
            log = a.run_dir / "logs" / (f"test_online_{name}_{env}.log" if a.online else f"test_{name}_{env}.log")
            rc |= run(cmd, log)
            rc |= run(evaluate_cmd(out_dir / env / "result.jsonl"), log)
    return rc


def step_report(a, split) -> int:
    rows = []
    for name in ["no_memory"] + ALL_SYSTEMS:
        per_env, all_s = {}, []
        for env in ENVS:
            p = a.run_dir / "test" / name / env / "per_sample.csv"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                samples = list(csv.DictReader(f))
            per_env[env] = samples
            all_s += samples
        if not all_s:
            continue

        def sr(ss):
            return round(sum(int(float(s["success"])) for s in ss) / len(ss), 4) if ss else None

        def pg(ss):
            return round(sum(float(s["progress"]) for s in ss) / len(ss), 4) if ss else None

        build_tokens = 0
        for env in ENVS:
            bp = a.run_dir / "memory" / name / env / "build_summary.json"
            if bp.exists():
                b = json.load(open(bp, encoding="utf-8"))
                build_tokens += b.get("input_tokens", 0) + b.get("output_tokens", 0)
        row = {
            "system": name, "n_test": len(all_s),
            "success_rate": sr(all_s), "progress": pg(all_s),
            **{f"sr_{e}": sr(per_env.get(e, [])) for e in ENVS},
            "test_inference_tokens": sum(int(float(s.get("total_tokens_inference") or 0)) for s in all_s),
            "test_memory_tokens": sum(int(float(s.get("total_tokens_memory") or 0)) for s in all_s),
            "build_memory_tokens": build_tokens,
            "n_errors": sum(1 for s in all_s if s.get("error")),
        }
        rows.append(row)
    if not rows:
        print("No test results found.")
        return 1
    with open(a.run_dir / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    head = list(rows[0])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join("-" if r[k] is None else str(r[k]) for k in head) + " |")
    md = ["# BFCL multi-turn main experiment (train/test split)", "",
          f"Run dir: `{a.run_dir}`  ",
          f"Model: `{a.model}`, embeddings: `{embed_model()}`, mode: {a.mode}  ",
          f"Split: seed={split['seed']}, train_frac={split['train_frac']} per environment "
          f"({sum(len(v['train']) for v in split['envs'].values())} train / "
          f"{sum(len(v['test']) for v in split['envs'].values())} test)", "", *lines, ""]
    (a.run_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["split", "rollout", "build", "test", "report", "all"])
    p.add_argument("--run_dir", type=Path, default=None)
    p.add_argument("--systems", nargs="+", default=None, help=f"subset of {['no_memory'] + ALL_SYSTEMS}")
    p.add_argument("--model", default="", help="agent + memory LLM (default $LLM_MODEL or gpt-4.1-mini)")
    p.add_argument("--mode", choices=["FC", "prompting"], default="FC")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force_parallel", action="store_true")
    p.add_argument("--online", action="store_true",
                   help="test: update memory after every sample (sequential) on memory_online/ -> test_online/")
    p.add_argument("--test_per_env", default="",
                   help="test only the first N test samples of each environment: '3', or per env "
                        "in ENVS order '3,3,2,2' (default: all 25)")
    p.add_argument("--test_order", choices=["split", "id"], default="split",
                   help="order within each environment before --test_per_env: seeded split order or dataset id order")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_frac", type=float, default=0.5)
    a = p.parse_args()

    if not llm_api_key():
        p.error("No API key: set OPENAI_API_KEY (or LLM_API_KEY).")
    a.model = a.model or llm_model()
    os.environ["LLM_MODEL"] = a.model
    a.run_dir = (a.run_dir or PROJECT_ROOT / "cross_episode_results" / f"split_main_{a.model}").resolve()
    a.run_dir.mkdir(parents=True, exist_ok=True)
    if a.systems is None:
        a.systems = (["no_memory"] if a.step in ("test", "all") else []) + ALL_SYSTEMS
    bad = [s for s in a.systems if s != "no_memory" and s not in SYSTEMS]
    if bad:
        p.error(f"unknown systems: {bad}")

    split = make_split(a.run_dir, a.seed, a.train_frac)
    rc = 0
    if a.step in ("rollout", "all"):
        rc |= step_rollout(a, split)
    if a.step in ("build", "all"):
        rc |= step_build(a, split)
    if a.step in ("test", "all"):
        rc |= step_test(a, split)
    if a.step in ("report", "all"):
        rc |= step_report(a, split)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
