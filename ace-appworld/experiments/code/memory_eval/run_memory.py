#!/usr/bin/env python
"""AppWorld x memory systems (thesis): rollout -> build -> test -> report.

Same protocol as ALFWorld / BFCL:
  rollout : a no-memory ReAct agent (ACE's evaluation agent + generator prompt, empty playbook)
            solves the 90 train tasks once; trajectory + AppWorld unit-test outcome saved.
  build   : every memory system ingests exactly those trajectories through the SAME
            EvoMemBench adapters used for ALFWorld (CROSSEP-EMB/scripts/memory), via
            update(conversation, idx, reward=1 if all unit tests pass else 0).
  test    : read-only. Retrieved memory text fills the generator prompt's {{ playbook }}
            slot (no_memory: empty; ace: ACE's offline-trained playbook).
  report  : task goal completion (TGC, all tests pass) and mean test pass rate.

ACE itself is trained with the official ACE pipeline (`appworld run THESIS_ACE_offline_no_GT_adaptation`),
not by this script; its playbook is only read here at test time.

Run inside WSL with the AppWorld venv:  python experiments/code/memory_eval/run_memory.py all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT = os.environ.setdefault("APPWORLD_PROJECT_PATH", "/mnt/d/Project/Thesis_Claude/ace-appworld")
EVOMEM_SCRIPTS = "/mnt/d/Project/Thesis_Claude/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/scripts"
if EVOMEM_SCRIPTS not in sys.path:
    sys.path.insert(0, EVOMEM_SCRIPTS)

from memory import create_memory  # noqa: E402  (installs OpenAI retry/compat hooks)
from memory._llm import embed_model, llm_model  # noqa: E402
from memory.logging_wrapper import LoggedMemory  # noqa: E402

GEN_PROMPT = f"{PROJECT}/experiments/prompts/appworld_react_generator_prompt.txt"
MEM_PROMPTS = f"{PROJECT}/experiments/memory_prompts"
ACE_PLAYBOOK_DEFAULT = f"{PROJECT}/experiments/playbooks/thesis_ace_offline_no_gt_gpt-4.1-mini.txt"
SYSTEMS = ["awm", "reasoning_bank", "mem0", "amem", "memoryos", "bm25", "vector"]
SERIAL_TEST = {"mem0", "amem", "memoryos"}  # read-time thread/process safety unverified
OUTPUT_CHARS_IN_MEMORY = 2000  # long API outputs (docs, listings) are clipped in the memory copy only

SYS_TEXT = ("You are an AI assistant that completes a user's day-to-day digital tasks in AppWorld by "
            "writing Python code that calls app APIs (apis.<app>.<api>) step by step, reading each "
            "execution output, and finally calling apis.supervisor.complete_task().")


# ── conversation format shared with the EvoMemBench adapters ──────────────────
# conv[0] system text · conv[1] ack · conv[2] "Your task is to: ..." (adapters use it as the
# retrieval query / task line) · then alternating assistant code / user execution output.

def base_conversation(instruction: str, supervisor) -> list[dict]:
    first = getattr(supervisor, "first_name", "") or ""
    last = getattr(supervisor, "last_name", "") or ""
    return [
        {"role": "user", "content": SYS_TEXT},
        {"role": "assistant", "content": "OK."},
        {"role": "user", "content": f"Your task is to: {instruction}\n(Supervisor: {first} {last})".strip()},
    ]


def trajectory_conversation(base: list[dict], agent_messages: list[dict]) -> list[dict]:
    conv = [dict(m) for m in base]
    for m in agent_messages:
        content = m.get("content") or ""
        if m["role"] == "user" and len(content) > OUTPUT_CHARS_IN_MEMORY:
            content = content[:OUTPUT_CHARS_IN_MEMORY] + "\n[... output clipped ...]"
        conv.append({"role": m["role"], "content": content})
    return conv


# ── memory configs (hyper-params as in the ALFWorld main experiment) ──────────

def memory_spec(name: str, mem_dir: str, model: str, emb: str):
    d = lambda *p: os.path.join(mem_dir, *p)  # noqa: E731
    llm_cfg = {"model": model}
    agent_id = "appworld_train"
    specs = {
        "awm": ("agent_workflow_memory", {"store_path": d("awm", "awm_store.json"), "top_k": 3,
                                          "enable_induction": True, "ark_batch_config": llm_cfg}),
        "reasoning_bank": ("reasoning_bank", {
            "bank_path": d("reasoning_bank", "bank.jsonl"),
            "embeddings_path": d("reasoning_bank", "embeddings.jsonl"), "topk": 1,
            "ark_batch_config": llm_cfg, "embedder_config": {"model": emb},
            "successful_prompt_path": f"{MEM_PROMPTS}/appworld_reasoning_bank_successful_prompt.txt",
            "failed_prompt_path": f"{MEM_PROMPTS}/appworld_reasoning_bank_failed_prompt.txt"}),
        "mem0": ("mem0", {
            "agent_id": agent_id, "search_limit": 3, "history_db_path": d("mem0", "history.db"),
            "ark_batch_config": llm_cfg,
            "embedder_config": {"provider": "openai", "config": {"model": emb}},
            "llm_config": {"provider": "openai", "config": {"model": model}},
            "vector_store_config": {"provider": "chroma", "config": {
                "collection_name": agent_id, "path": d("mem0", "chroma")}}}),
        "amem": ("amem", {"agent_id": agent_id, "storage_path": d("amem"), "llm_model": model,
                          "search_limit": 3, "evo_threshold": 99999, "llm": {}, "embed": {"model": emb}}),
        "memoryos": ("memoryos", {"user_id": agent_id, "storage_path": d("memoryos"),
                                  "short_term_capacity": 10, "llm_model": model, "search_limit": 3,
                                  "llm": {}, "embed": {"model": emb}}),
        "bm25": ("bm25", {"agent_id": agent_id, "top_k": 10, "chunk_size": 1024, "memory_dir": d("bm25")}),
        "vector": ("qwen3_embedding", {"agent_id": agent_id, "top_k": 10, "chunk_size": 1024,
                                       "memory_dir": d("vector"), "embed_model": emb}),
    }
    return specs[name]


# ── agent ─────────────────────────────────────────────────────────────────────

def make_agent(run_dir: str, model: str, max_steps: int, memory=None, fixed_playbook: str = ""):
    from appworld_experiments.code.ace.evaluation_react import SimplifiedReActAgent

    empty_pb = os.path.join(run_dir, "empty_playbook.txt")
    if not os.path.exists(empty_pb):
        open(empty_pb, "w").close()

    class MemoryReActAgent(SimplifiedReActAgent):
        """ACE's evaluation ReAct agent; the {{ playbook }} slot is filled per task."""

        def initialize(self, world):
            self.injected = ""
            self.inject_stats = None
            if memory is not None:
                conv = base_conversation(world.task.instruction, world.task.supervisor)
                if hasattr(memory, "set_context"):
                    memory.set_context(world.task_id)
                new_conv, stats = memory.inject(conv)
                before, after = conv[0]["content"], new_conv[0]["content"]
                self.injected = after[len(before):].strip() if after.startswith(before) else after
                self.inject_stats = stats.to_dict()
            self.playbook = fixed_playbook or self.injected
            super().initialize(world)

        def solve_task(self, task_id, experiment_name=None):
            """Same loop as evaluation_agent.Agent.solve_task, but evaluates while the world is
            still open (as ACE's adaptation agent does) and returns the TestTracker."""
            from appworld import AppWorld
            from appworld_experiments.code.ace.evaluation_agent import ExecutionIO
            self.cost_tracker.reset(task_id)
            tracker = None
            with AppWorld(task_id=task_id, experiment_name=experiment_name, **self.appworld_config) as world:
                outputs = []
                self.initialize(world)
                for _ in range(self.max_steps):
                    self.step_number += 1
                    inputs, cost, _ = self.next_execution_inputs_and_cost(outputs, "")
                    if inputs:
                        outputs = [ExecutionIO(content=world.execute(i.content), metadata=i.metadata)
                                   for i in inputs]
                        self.cost_tracker.add(task_id, cost)
                    if world.task_completed() or self.cost_tracker.exceeded():
                        break
                res = world.evaluate()
                tracker = res[0] if isinstance(res, tuple) else res
            self.logger.complete_task()
            return tracker

    gen_cfg = {
        "name": model, "provider": "openai", "temperature": 0, "seed": 100,
        "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
        "n": 1, "response_format": {"type": "text"},
        "retry_after_n_seconds": 10, "use_cache": False, "max_retries": 50,
    }
    return MemoryReActAgent(
        generator_model_config=gen_cfg,
        appworld_config={"random_seed": 123},
        logger_config={"color": False, "verbose": False},
        generator_prompt_file_path=GEN_PROMPT,
        trained_playbook_file_path=empty_pb,
        ignore_multiple_calls=True,
        max_steps=max_steps,
        max_cost_overall=1000,
        max_cost_per_task=10,
        log_lm_calls=True,
    )


def run_tasks_worker(args: dict) -> list[str]:
    """Solve a chunk of tasks in one process; write one JSON per task. Returns error strings."""
    os.environ.setdefault("APPWORLD_PROJECT_PATH", PROJECT)

    memory = None
    if args["system"] not in ("no_memory", "ace"):
        mtype, cfg = memory_spec(args["system"], args["mem_dir"], args["model"], args["emb"])
        memory = LoggedMemory(create_memory(mtype, read_only=True, **cfg),
                              log_path=os.path.join(args["out_dir"], "retrieval_log.jsonl"), phase="test")
    fixed = open(args["ace_playbook"], encoding="utf-8").read() if args["system"] == "ace" else ""
    agent = make_agent(args["run_dir"], args["model"], args["max_steps"], memory, fixed)
    # solve_tasks() normally does this; we call solve_task() per task to evaluate in between.
    agent.logger.initialize(experiment_name=args["experiment"], num_tasks=len(args["task_ids"]),
                            num_processes=1, process_index=0)
    errors = []
    for tid in args["task_ids"]:
        out = os.path.join(args["out_dir"], f"{tid}.json")
        if os.path.exists(out):
            continue
        t0 = time.time()
        try:
            tracker = agent.solve_task(tid, args["experiment"])
            msgs = agent.messages[agent.num_instruction_messages:]
            base = base_conversation(agent.world.task.instruction, agent.world.task.supervisor)
            rec = {
                "task_id": tid,
                "instruction": agent.world.task.instruction,
                "success": bool(tracker.success),
                "pass_count": tracker.pass_count,
                "num_tests": tracker.num_tests,
                "steps": agent.step_number,
                "cost_usd": agent.cost_tracker.task_costs.get(tid, 0) if hasattr(agent.cost_tracker, "task_costs") else None,
                "latency": time.time() - t0,
                "injected": getattr(agent, "injected", ""),
                "inject_stats": getattr(agent, "inject_stats", None),
                "conversation": trajectory_conversation(base, msgs),
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
            print(f"[{args['system']}] {tid} success={rec['success']} tests={rec['pass_count']}/{rec['num_tests']} "
                  f"steps={rec['steps']} {rec['latency']:.0f}s", flush=True)
        except Exception as e:
            errors.append(f"{tid}: {type(e).__name__}: {e}")
            traceback.print_exc()
    return errors


def run_split(system: str, task_ids: list[str], out_dir: str, a, experiment: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    todo = [t for t in task_ids if not os.path.exists(os.path.join(out_dir, f"{t}.json"))]
    procs = 1 if (system in SERIAL_TEST and not a.force_parallel) else max(1, min(a.procs, len(todo)))
    print(f"[{system}] {len(task_ids)} tasks, {len(todo)} to run, {procs} processes -> {out_dir}", flush=True)
    if not todo:
        return 0
    chunks = [todo[i::procs] for i in range(procs)]
    base = {"system": system, "out_dir": out_dir, "run_dir": a.run_dir, "mem_dir": a.mem_dir,
            "model": a.model, "emb": a.emb, "max_steps": a.max_steps, "experiment": experiment,
            "ace_playbook": a.ace_playbook}
    errors = []
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for fut in as_completed([ex.submit(run_tasks_worker, dict(base, task_ids=c)) for c in chunks]):
            errors += fut.result()
    if errors:
        print(f"[{system}] {len(errors)} errors (rerun to retry):\n  " + "\n  ".join(errors[:10]), flush=True)
    return 1 if errors else 0


# ── steps ─────────────────────────────────────────────────────────────────────

def step_rollout(a):
    from appworld import load_task_ids
    ids = load_task_ids("train")
    json.dump(ids, open(os.path.join(a.run_dir, "train_ids.json"), "w"))
    return run_split("no_memory", ids, os.path.join(a.run_dir, "train_rollouts"), a, "thesis_train_rollout")


def step_build(a):
    ids = json.load(open(os.path.join(a.run_dir, "train_ids.json")))
    rc = 0
    for name in a.systems:
        if name in ("no_memory", "ace"):
            continue
        mtype, cfg = memory_spec(name, a.mem_dir, a.model, a.emb)
        sdir = os.path.join(a.mem_dir, name)
        os.makedirs(sdir, exist_ok=True)
        json.dump(cfg, open(os.path.join(sdir, "config.json"), "w"), indent=2)
        log_path = os.path.join(sdir, "build_log.jsonl")
        done_path = log_path + ".done"
        done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
        memory = LoggedMemory(create_memory(mtype, read_only=False, **cfg), log_path=log_path, phase="build",
                              store_paths=[sdir])
        t0, tok_in, tok_out, errs = time.time(), 0, 0, []
        print(f"[build] {name}: {len(ids)} trajectories, {len(done)} done", flush=True)
        for i, tid in enumerate(ids):
            if tid in done:
                continue
            p = os.path.join(a.run_dir, "train_rollouts", f"{tid}.json")
            if not os.path.exists(p):
                errs.append(f"{tid}: missing rollout")
                continue
            rec = json.load(open(p, encoding="utf-8"))
            try:
                memory.set_context(tid)
                st = memory.update(rec["conversation"], i, reward=1.0 if rec["success"] else 0.0)
                tok_in += st.input_tokens
                tok_out += st.output_tokens
                with open(done_path, "a") as f:
                    f.write(tid + "\n")
            except Exception as e:
                errs.append(f"{tid}: {type(e).__name__}: {e}")
                print(f"  {tid} ERROR {e}", flush=True)
        summ_path = os.path.join(sdir, "build_summary.json")
        prev = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
        json.dump({"system": name, "memory_type": mtype, "n_train": len(ids), "errors": errs,
                   "input_tokens": tok_in + prev.get("input_tokens", 0),
                   "output_tokens": tok_out + prev.get("output_tokens", 0),
                   "wall_clock_s": time.time() - t0 + prev.get("wall_clock_s", 0)},
                  open(summ_path, "w"), indent=2)
        print(f"[build] {name} done: {len(errs)} errors", flush=True)
        rc |= 1 if errs else 0
    return rc


def test_ids(a) -> list[str]:
    from appworld import load_task_ids
    ids = load_task_ids(a.test_split)
    if a.test_first:  # first N tasks in the split's official order
        return ids[: a.test_first]
    if a.test_n and a.test_n < len(ids):
        step = len(ids) / a.test_n  # evenly spread over the split (different scenarios)
        ids = [ids[math.floor(i * step)] for i in range(a.test_n)]
    return ids


def step_test(a):
    ids = test_ids(a)
    rc = 0
    for name in a.systems:
        if name == "ace" and not os.path.exists(a.ace_playbook):
            print(f"[skip] ace: trained playbook not found at {a.ace_playbook}")
            rc |= 1
            continue
        if name not in ("no_memory", "ace") and not os.path.exists(os.path.join(a.mem_dir, name, "build_summary.json")):
            print(f"[skip] {name}: memory not built")
            rc |= 1
            continue
        rc |= run_split(name, ids, os.path.join(a.run_dir, "test", a.test_split, name), a,
                        f"thesis_test_{a.test_split}_{name}")
    return rc


def step_report(a):
    rows = []
    for name in ["no_memory", "ace"] + SYSTEMS:
        d = os.path.join(a.run_dir, "test", a.test_split, name)
        if not os.path.isdir(d):
            continue
        recs = [json.load(open(os.path.join(d, f), encoding="utf-8")) for f in os.listdir(d)
                if f.endswith(".json")]
        if not recs:
            continue
        n = len(recs)
        rows.append({
            "system": name, "n": n,
            "TGC": round(sum(r["success"] for r in recs) / n, 4),
            "test_pass_rate": round(sum(r["pass_count"] / max(r["num_tests"], 1) for r in recs) / n, 4),
            "avg_steps": round(sum(r["steps"] for r in recs) / n, 1),
            # ACE's playbook is a fixed prompt insert, not recorded per task as "injected".
            "avg_injected_chars": (len(open(a.ace_playbook, encoding="utf-8").read())
                                   if name == "ace" and os.path.exists(a.ace_playbook)
                                   else round(sum(len(r.get("injected") or "") for r in recs) / n)),
        })
    train = os.path.join(a.run_dir, "train_rollouts")
    tr = [json.load(open(os.path.join(train, f), encoding="utf-8")) for f in os.listdir(train)] if os.path.isdir(train) else []
    head = ["system", "n", "TGC", "test_pass_rate", "avg_steps", "avg_injected_chars"]
    md = ["# AppWorld x memory systems", "",
          f"Model `{a.model}`, embeddings `{a.emb}`, split `{a.test_split}`, max_steps {a.max_steps}.  ",
          f"Train rollouts: {len(tr)} tasks, TGC {sum(r['success'] for r in tr)}/{len(tr)}.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(str(r[k]) for k in head) + " |" for r in rows]
    open(os.path.join(a.run_dir, f"report_{a.test_split}.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["rollout", "build", "test", "report", "all"])
    p.add_argument("--run_dir", default=f"{PROJECT}/experiments/outputs/thesis_memory")
    p.add_argument("--systems", nargs="+", default=None)
    p.add_argument("--model", default="")
    p.add_argument("--embed_model", default="")
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--force_parallel", action="store_true")
    p.add_argument("--test_split", default="test_normal")
    p.add_argument("--test_n", type=int, default=0, help="evenly spaced subset of the test split (0 = all)")
    p.add_argument("--test_first", type=int, default=0, help="first N tasks of the split, in order")
    p.add_argument("--ace_playbook", default=ACE_PLAYBOOK_DEFAULT)
    a = p.parse_args()
    a.model = a.model or llm_model()
    a.emb = a.embed_model or embed_model()
    os.environ["LLM_MODEL"], os.environ["EMBED_MODEL"] = a.model, a.emb
    os.makedirs(a.run_dir, exist_ok=True)
    a.mem_dir = os.path.join(a.run_dir, "memory")
    if a.systems is None:
        a.systems = (["no_memory", "ace"] if a.step in ("test", "all") else []) + SYSTEMS
    rc = 0
    if a.step in ("rollout", "all"):
        rc |= step_rollout(a)
    if a.step in ("build", "all"):
        rc |= step_build(a)
    if a.step in ("test", "all"):
        rc |= step_test(a)
    if a.step in ("report", "all", "test"):
        rc |= step_report(a)
    sys.exit(rc)


if __name__ == "__main__":
    main()
