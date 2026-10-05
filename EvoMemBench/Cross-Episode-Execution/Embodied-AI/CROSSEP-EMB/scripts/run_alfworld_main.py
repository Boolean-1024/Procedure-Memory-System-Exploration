#!/usr/bin/env python
"""ALFWorld main experiment: build memory on the train split, evaluate read-only on the test split.

Protocol (identical for every system):
  1. rollout : a no-memory agent plays 200 train games (stratified from idx 0..2419,
               json_2.1.1/train; --train_subset 0 = all) once;
               trajectories + rewards are saved to <run_dir>/train_rollouts/.
  2. build   : each memory system ingests exactly those trajectories through its own
               update() (scripts/build_memory_from_trajectories.py). Write log per system.
  3. test    : each system is evaluated on the 200 test games (idx 2420..2619,
               json_2.1.1/valid_train) with --readonly_memory. Retrieval log per system.
               The no-memory baseline runs on the same 200 games.
  4. report  : success rate (overall + per task type) and token cost -> report.md / report.csv

Usage (ALFWorld server must be running, see THESIS_README.md):
  python scripts/run_alfworld_main.py all
  python scripts/run_alfworld_main.py rollout --train_subset 0            # all 2420 train games
  python scripts/run_alfworld_main.py build --systems ace awm
  python scripts/run_alfworld_main.py test  --systems no_memory ace awm
  python scripts/run_alfworld_main.py report
"""
import argparse
import collections
import csv
import json
import os
import random
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from memory._llm import embed_api_key, embed_model, llm_api_key, llm_model  # noqa: E402

PROMPTS = os.path.join(SCRIPTS_DIR, "prompts")
MAPPINGS = os.path.join(BASE_DIR, "agentenv-alfworld", "configs")
TRAIN_RANGE = (0, 2420)
TEST_RANGE = (2420, 2620)
TASK_TYPES = [
    "pick_and_place_simple", "pick_two_obj_and_place", "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_heat_then_place_in_recep", "look_at_obj_in_light",
]
AGENT_ID = "alfworld_train"  # single shared store per system (main experiment, not per-category)

# Systems whose read-time thread safety has not been verified: tested with parallel=1.
SERIAL_TEST = {"mem0", "amem", "memoryos"}


def system_specs(mem_dir: str, model: str, emb: str) -> dict:
    """name -> (memory_type, config dict, store paths). Hyper-params follow the
    EvoMemBench all-pairs launchers (scripts/eval_alfworld_all_*_batch_all_pairs.sh).
    API keys are never written to config files; adapters resolve them from env."""
    d = lambda *p: os.path.join(mem_dir, *p)  # noqa: E731
    llm_cfg = {"model": model}
    return {
        "awm": ("agent_workflow_memory", {
            "store_path": d("awm", "awm_store.json"),
            "top_k": 3,
            "enable_induction": True,
            "ark_batch_config": llm_cfg,
        }, [d("awm")]),
        "reasoning_bank": ("reasoning_bank", {
            "bank_path": d("reasoning_bank", "bank.jsonl"),
            "embeddings_path": d("reasoning_bank", "embeddings.jsonl"),
            "topk": 1,
            "ark_batch_config": llm_cfg,
            "embedder_config": {"model": emb},
            "successful_prompt_path": os.path.join(PROMPTS, "alfworld_reasoning_bank_successful_prompt.txt"),
            "failed_prompt_path": os.path.join(PROMPTS, "alfworld_reasoning_bank_failed_prompt.txt"),
        }, [d("reasoning_bank")]),
        "ace": ("ace", {
            "playbook_path": d("ace", "playbook.txt"),
            "initial_playbook_path": os.path.join(PROMPTS, "alfworld_initial_playbook.txt"),
            "ark_batch_config": llm_cfg,
            "reflector_prompt_path": os.path.join(PROMPTS, "alfworld_ace_reflector_prompt.txt"),
            "curator_prompt_path": os.path.join(PROMPTS, "alfworld_ace_curator_prompt.txt"),
        }, [d("ace")]),
        "mem0": ("mem0", {
            "agent_id": AGENT_ID,
            "search_limit": 3,
            "history_db_path": d("mem0", "history.db"),
            "ark_batch_config": llm_cfg,
            "embedder_config": {"provider": "openai", "config": {"model": emb}},
            "llm_config": {"provider": "openai", "config": {"model": model}},
            "vector_store_config": {"provider": "chroma", "config": {
                "collection_name": AGENT_ID, "path": d("mem0", "chroma")}},
        }, [d("mem0")]),
        "amem": ("amem", {
            "agent_id": AGENT_ID,
            "storage_path": d("amem"),
            "llm_model": model,
            "search_limit": 3,
            "evo_threshold": 99999,
            "llm": {},
            "embed": {"model": emb},
        }, [d("amem")]),
        "memoryos": ("memoryos", {
            "user_id": AGENT_ID,
            "storage_path": d("memoryos"),
            "short_term_capacity": 10,
            "llm_model": model,
            "search_limit": 3,
            "llm": {},
            "embed": {"model": emb},
        }, [d("memoryos")]),
        "bm25": ("bm25", {
            "agent_id": AGENT_ID,
            "top_k": 10,
            "chunk_size": 1024,
            "memory_dir": d("bm25"),
        }, [d("bm25")]),
        "vector": ("qwen3_embedding", {
            "agent_id": AGENT_ID,
            "top_k": 10,
            "chunk_size": 1024,
            "memory_dir": d("vector"),
            "embed_model": emb,
        }, [d("vector")]),
    }


ALL_SYSTEMS = ["awm", "reasoning_bank", "ace", "mem0", "amem", "memoryos", "bm25", "vector"]


def _type_counts(mapping_file: str) -> collections.Counter:
    rows = json.load(open(mapping_file, encoding="utf-8"))
    return collections.Counter(r["task_type"].split("-")[0] for r in rows)


def stratified(mapping_file: str, offset: int, n: int | None, seed: int = 0,
               target_mapping: str | None = None) -> list[int]:
    """All indices, or exactly n indices sampled per task type (deterministic).

    Per-type quotas are proportional to ``target_mapping``'s type distribution
    (default: the sampled file's own), allocated by largest remainder so they sum to n.
    """
    rows = json.load(open(mapping_file, encoding="utf-8"))
    idx = list(range(offset, offset + len(rows)))
    if not n or n >= len(rows):
        return idx
    by_type = collections.defaultdict(list)
    for i, r in zip(idx, rows):
        by_type[r["task_type"].split("-")[0]].append(i)
    dist = _type_counts(target_mapping or mapping_file)
    total = sum(dist.values())
    exact = {t: n * dist.get(t, 0) / total for t in by_type}
    quota = {t: int(v) for t, v in exact.items()}
    for t in sorted(exact, key=lambda t: exact[t] - quota[t], reverse=True)[: n - sum(quota.values())]:
        quota[t] += 1
    rng = random.Random(seed)
    out = []
    for t, lst in sorted(by_type.items()):
        out += rng.sample(lst, min(quota[t], len(lst)))
    return sorted(out)


def task_type_of(idx: int) -> str:
    if idx >= TEST_RANGE[0]:
        rows = json.load(open(os.path.join(MAPPINGS, "mappings_test.json"), encoding="utf-8"))
        return rows[idx - TEST_RANGE[0]]["task_type"].split("-")[0]
    rows = json.load(open(os.path.join(MAPPINGS, "mappings_train.json"), encoding="utf-8"))
    return rows[idx]["task_type"].split("-")[0]


def child_env(model: str, emb: str) -> dict:
    # Resolving the keys first copies machine-level Windows variables into os.environ.
    llm_api_key(), embed_api_key()
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["LLM_MODEL"] = model
    env["EMBED_MODEL"] = emb
    return env


def run(cmd: list[str], env: dict, log_file: str | None = None) -> int:
    print("\n$ " + " ".join(cmd if len(" ".join(cmd)) < 600 else cmd[:6] + ["..."]), flush=True)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
            return proc.wait()
    return subprocess.call(cmd, env=env)


def indices_arg(idx: list[int], full: tuple[int, int]) -> list[str]:
    if idx == list(range(*full)):
        return ["--start_idx", str(full[0]), "--num_samples", str(full[1] - full[0])]
    return ["--indices", json.dumps(idx)]


# ── steps ────────────────────────────────────────────────────────────────────

def step_rollout(a, env):
    dist = os.path.join(MAPPINGS, "mappings_test.json") if a.train_dist == "test" else None
    idx = stratified(os.path.join(MAPPINGS, "mappings_train.json"), 0, a.train_subset, a.seed, dist)
    out = os.path.join(a.run_dir, "train_rollouts")
    json.dump(idx, open(os.path.join(a.run_dir, "train_indices.json"), "w"))
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "eval_alfworld_with_memory.py"),
           "--port", str(a.port), "--max_rounds", str(a.max_rounds), "--parallel", str(a.parallel),
           "--no_memory", "--output_dir", out, *indices_arg(idx, TRAIN_RANGE)]
    if a.temperature is not None:
        cmd += ["--temperature", str(a.temperature)]
    return run(cmd, env, os.path.join(a.run_dir, "logs", "rollout.log"))


def step_build(a, env, specs):
    train_idx_file = os.path.join(a.run_dir, "train_indices.json")
    idx = json.load(open(train_idx_file)) if os.path.exists(train_idx_file) else None
    rc = 0
    for name in a.systems:
        if name == "no_memory":
            continue
        mtype, cfg, stores = specs[name]
        cfg_path = os.path.join(a.run_dir, "configs", f"{name}.json")
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2)
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "build_memory_from_trajectories.py"),
               "--traj_dir", os.path.join(a.run_dir, "train_rollouts"),
               "--memory_type", mtype, "--memory_config", cfg_path,
               "--memory_log", os.path.join(a.run_dir, "memory", name, "build_log.jsonl"),
               "--store_paths", json.dumps(stores), "--parallel", str(a.build_parallel)]
        if idx is not None:
            cmd += ["--indices", json.dumps(idx)]
        rc |= run(cmd, env, os.path.join(a.run_dir, "logs", f"build_{name}.log"))
    return rc


def step_test(a, env, specs):
    if a.test_first:  # first N test games in index order (2420, 2421, ...)
        idx = list(range(TEST_RANGE[0], TEST_RANGE[0] + a.test_first))
    else:
        idx = stratified(os.path.join(MAPPINGS, "mappings_test.json"), TEST_RANGE[0], a.test_subset, a.seed)
    rc = 0
    for name in a.systems:
        out = os.path.join(a.run_dir, "test", name)
        par = 1 if (name in SERIAL_TEST and not a.force_parallel) else a.parallel
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "eval_alfworld_with_memory.py"),
               "--port", str(a.port), "--max_rounds", str(a.max_rounds), "--parallel", str(par),
               "--output_dir", out, *indices_arg(idx, TEST_RANGE)]
        if a.temperature is not None:
            cmd += ["--temperature", str(a.temperature)]
        if name == "no_memory":
            cmd += ["--no_memory"]
        else:
            mtype, _, _ = specs[name]
            cfg_path = os.path.join(a.run_dir, "configs", f"{name}.json")
            if not os.path.exists(cfg_path):
                print(f"[skip] {name}: no config/memory yet, run `build` first")
                rc |= 1
                continue
            cmd += ["--memory_type", mtype, "--memory_config", cfg_path, "--readonly_memory",
                    "--memory_log", os.path.join(out, "retrieval_log.jsonl"), "--log_phase", "test"]
        rc |= run(cmd, env, os.path.join(a.run_dir, "logs", f"test_{name}.log"))
    return rc


def step_report(a):
    rows = []
    systems = ["no_memory"] + ALL_SYSTEMS
    for name in systems:
        sp = os.path.join(a.run_dir, "test", name, "summary.json")
        if not os.path.exists(sp):
            continue
        s = json.load(open(sp, encoding="utf-8"))
        per = collections.defaultdict(lambda: [0, 0])
        for r in s["results"]:
            t = task_type_of(r["data_idx"])
            per[t][0] += r.get("success", 0)
            per[t][1] += 1
        bp = os.path.join(a.run_dir, "memory", name, "build_summary.json")
        b = json.load(open(bp, encoding="utf-8")) if os.path.exists(bp) else {}
        row = {
            "system": name,
            "n_test": s["num_samples"],
            "success_rate": round(s["success_rate"], 4),
            "avg_score": round(s["avg_score"], 4),
            "avg_rounds": round(s["avg_rounds_per_sample"], 2),
            "test_agent_tokens": s["agent_prompt_tokens"] + s["agent_completion_tokens"],
            "test_memory_tokens": s["memory_input_tokens"] + s["memory_output_tokens"],
            "build_memory_tokens": b.get("update_input_tokens", 0) + b.get("update_output_tokens", 0),
            "build_embedding_tokens": b.get("update_embedding_tokens", 0),
            "n_errors": sum(1 for r in s["results"] if "error" in r),
        }
        for t in TASK_TYPES:
            ok, n = per.get(t, (0, 0))
            row[t] = round(ok / n, 4) if n else None
        rows.append(row)
    if not rows:
        print("No test summaries found.")
        return 1
    cols = list(rows[0].keys())
    with open(os.path.join(a.run_dir, "report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    short = {"pick_and_place_simple": "pick", "pick_two_obj_and_place": "pick2",
             "pick_clean_then_place_in_recep": "clean", "pick_cool_then_place_in_recep": "cool",
             "pick_heat_then_place_in_recep": "heat", "look_at_obj_in_light": "look"}
    head = ["system", "n_test", "success_rate", *[short[t] for t in TASK_TYPES],
            "avg_rounds", "test_agent_tokens", "test_memory_tokens", "build_memory_tokens", "n_errors"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        vals = [r["system"], r["n_test"], r["success_rate"], *[r[t] for t in TASK_TYPES],
                r["avg_rounds"], f"{r['test_agent_tokens']:,}", f"{r['test_memory_tokens']:,}",
                f"{r['build_memory_tokens']:,}", r["n_errors"]]
        lines.append("| " + " | ".join("-" if v is None else str(v) for v in vals) + " |")
    meta = json.load(open(os.path.join(a.run_dir, "run_meta.json"))) if os.path.exists(
        os.path.join(a.run_dir, "run_meta.json")) else {}
    md = ["# ALFWorld main experiment", "", f"Run dir: `{a.run_dir}`",
          f"Meta: `{json.dumps(meta)}`", "", *lines, ""]
    open(os.path.join(a.run_dir, "report.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["rollout", "build", "test", "report", "all"])
    p.add_argument("--run_dir", default="")
    p.add_argument("--systems", nargs="+", default=None,
                   help=f"subset of {['no_memory'] + ALL_SYSTEMS} (default: all)")
    p.add_argument("--model", default="", help="agent + memory LLM (default $LLM_MODEL or gpt-4.1-mini)")
    p.add_argument("--embed_model", default="", help="default $EMBED_MODEL or text-embedding-3-small")
    p.add_argument("--port", type=int, default=36005)
    p.add_argument("--max_rounds", type=int, default=20)
    p.add_argument("--parallel", type=int, default=8, help="episodes in flight (rollout/test)")
    p.add_argument("--build_parallel", type=int, default=1, help="concurrent update() calls")
    p.add_argument("--force_parallel", action="store_true",
                   help=f"also use --parallel for {sorted(SERIAL_TEST)} at test time")
    p.add_argument("--train_subset", type=int, default=200,
                   help="stratified subset of the 2420 train games (default 200, like EvoMemBench's "
                        "200-game scale); 0 = all 2420")
    p.add_argument("--train_dist", choices=["test", "train"], default="test",
                   help="per-task-type quotas of the train subset: 'test' = same type mix as the "
                        "200 test games (46/45/37/28/25/19), 'train' = proportional to the train split")
    p.add_argument("--test_subset", type=int, default=None,
                   help="stratified subset of the 200 test games (pilot runs)")
    p.add_argument("--test_first", type=int, default=0,
                   help="test the first N test games in index order (overrides --test_subset)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    model = a.model or llm_model()
    emb = a.embed_model or embed_model()
    a.run_dir = os.path.abspath(a.run_dir or os.path.join(BASE_DIR, "output", f"main_{model}"))
    os.makedirs(a.run_dir, exist_ok=True)
    mem_dir = os.path.join(a.run_dir, "memory")
    specs = system_specs(mem_dir, model, emb)
    if a.systems is None:
        a.systems = (["no_memory"] if a.step in ("test", "all") else []) + ALL_SYSTEMS
    bad = [s for s in a.systems if s != "no_memory" and s not in specs]
    if bad:
        p.error(f"unknown systems: {bad}")
    if not llm_api_key():
        p.error("No API key: set OPENAI_API_KEY (or LLM_API_KEY).")

    meta_path = os.path.join(a.run_dir, "run_meta.json")
    meta = {"model": model, "embed_model": emb, "max_rounds": a.max_rounds,
            "train_subset": a.train_subset, "train_dist": a.train_dist, "test_subset": a.test_subset,
            "temperature": a.temperature, "seed": a.seed}
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if (old.get("model"), old.get("embed_model")) != (model, emb):
            p.error(f"run_dir was created with {old}; use a different --run_dir")
        # Keep settings recorded by earlier steps (e.g. train_subset from `rollout`).
        meta = {**old, **{k: v for k, v in meta.items() if v is not None}}
    json.dump(meta, open(meta_path, "w"), indent=2)

    env = child_env(model, emb)
    rc = 0
    if a.step in ("rollout", "all"):
        rc |= step_rollout(a, env)
    if a.step in ("build", "all"):
        rc |= step_build(a, env, specs)
    if a.step in ("test", "all"):
        rc |= step_test(a, env, specs)
    if a.step in ("report", "all"):
        rc |= step_report(a)
    sys.exit(rc)


if __name__ == "__main__":
    main()
