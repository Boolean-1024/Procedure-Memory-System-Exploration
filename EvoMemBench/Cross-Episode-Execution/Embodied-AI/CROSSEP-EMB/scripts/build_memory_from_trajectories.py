#!/usr/bin/env python
"""Offline memory construction: replay a fixed set of training trajectories into a memory backend.

Every memory system ingests *the same* trajectories (produced once by a no-memory agent
on the ALFWorld train split), via the backend's own ``update(conversation, data_idx, reward)``.
This removes rollout variance between systems: differences at test time come only from
how each system writes / retrieves memory.

Usage:
  python scripts/build_memory_from_trajectories.py \
      --traj_dir output/main/train_rollouts \
      --memory_type ace --memory_config output/main/configs/ace.json \
      --memory_log output/main/memory/ace/build_log.jsonl

Resumable: indices already ingested are recorded in <memory_log>.done and skipped on rerun.
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from eval_alfworld_with_memory import load_env  # noqa: E402  (also loads .env on import)
from memory import MemoryCallStats, create_memory  # noqa: E402
from memory.logging_wrapper import LoggedMemory  # noqa: E402


def load_trajectories(traj_dir, indices=None):
    files = glob.glob(os.path.join(traj_dir, "alfworld_*.json"))
    by_idx = {}
    for fn in files:
        m = re.search(r"alfworld_(\d+)\.json$", fn)
        if m:
            by_idx[int(m.group(1))] = fn
    order = sorted(by_idx) if indices is None else [i for i in indices if i in by_idx]
    missing = [] if indices is None else [i for i in indices if i not in by_idx]
    return [(i, by_idx[i]) for i in order], missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_dir", required=True)
    p.add_argument("--memory_type", required=True)
    p.add_argument("--memory_config", default="")
    p.add_argument("--memory_log", required=True)
    p.add_argument("--store_paths", default="[]",
                   help="JSON list of storage paths whose size is snapshotted after each update")
    p.add_argument("--indices", default="", help="JSON list; default = every trajectory in traj_dir")
    p.add_argument("--parallel", type=int, default=1,
                   help="Concurrent update() calls. 1 = deterministic order (recommended for "
                        "stateful systems such as ACE / AWM / mem0).")
    p.add_argument("--only_success", action="store_true",
                   help="Ingest only reward>=1 trajectories for every system (default: pass all, "
                        "each backend applies its own reward filter).")
    args = p.parse_args()

    load_env()
    kwargs = {}
    if args.memory_config:
        raw = open(args.memory_config, encoding="utf-8").read()
        raw = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), raw)
        kwargs = json.loads(raw)
    inner = create_memory(args.memory_type, read_only=False, **kwargs)
    memory = LoggedMemory(inner, log_path=args.memory_log, phase="build",
                          store_paths=json.loads(args.store_paths))

    indices = json.loads(args.indices) if args.indices else None
    trajs, missing = load_trajectories(args.traj_dir, indices)
    if missing:
        print(f"[warn] {len(missing)} requested trajectories not found, e.g. {missing[:5]}")

    done_path = args.memory_log + ".done"
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            done = {int(x) for x in f.read().split()}
    pending = [(i, fn) for i, fn in trajs if i not in done]
    print(f"[build] {args.memory_type}: {len(trajs)} trajectories, {len(done)} already ingested, "
          f"{len(pending)} pending, parallel={args.parallel}")

    done_lock = threading.Lock()
    totals = {"stats": MemoryCallStats(), "n": 0, "skipped": 0, "errors": []}

    def ingest(idx, fn):
        rec = json.load(open(fn, encoding="utf-8"))
        conv = rec["conversations"]
        reward = rec.get("reward", 0.0)
        if args.only_success and reward < 1:
            return idx, None, "skipped"
        stats = memory.update(conv, idx, reward=reward)
        return idx, stats, "ok"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futs = {ex.submit(ingest, i, fn): i for i, fn in pending}
        for fut in tqdm(as_completed(futs), total=len(futs), unit="traj", dynamic_ncols=True):
            i = futs[fut]
            try:
                idx, stats, status = fut.result()
                if status == "skipped":
                    totals["skipped"] += 1
                else:
                    totals["stats"] = totals["stats"] + stats
                    totals["n"] += 1
                with done_lock, open(done_path, "a") as f:
                    f.write(f"{idx}\n")
            except Exception as e:
                totals["errors"].append({"data_idx": i, "error": f"{type(e).__name__}: {e}"})
                tqdm.write(f"  idx={i} ERROR: {e}")

    summary = {
        "memory_type": args.memory_type,
        "traj_dir": os.path.abspath(args.traj_dir),
        "num_trajectories": len(trajs),
        "ingested_this_run": totals["n"],
        "skipped": totals["skipped"],
        "previously_ingested": len(done),
        "errors": totals["errors"],
        "wall_clock_seconds": time.time() - t0,
        **totals["stats"].to_dict(prefix="update_"),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(args.memory_log)), "build_summary.json")
    # Accumulate across resumed runs.
    if os.path.exists(out):
        prev = json.load(open(out, encoding="utf-8"))
        for k in ("update_input_tokens", "update_output_tokens", "update_embedding_tokens",
                  "update_cached_tokens", "update_latency", "wall_clock_seconds"):
            summary[k] = summary.get(k, 0) + prev.get(k, 0)
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=2))
    if totals["errors"]:
        print(f"[build] {len(totals['errors'])} errors; rerun the same command to retry them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
