"""Cross-episode inference driver — generate phase only.

Runs BaseHandler.inference_multi_turn_prompting for every selected sample and
writes each result as a JSON line to ``<output_dir>/<run_name>/result.jsonl``.
The JSONL uses the same field names as the original BFCL ``*_result.json`` files
(id, result, input_token_count, output_token_count, latency, inference_log) plus
fork-specific telemetry fields (category, per_turn_totals, sample_totals,
memory_usage, force_quit, full_message_history, error).

Pass the generated ``result.jsonl`` to ``run_batch_evaluate`` to compute
success_rate, progress_score and per-sample JSON artefacts.

Usage:
    python -m bfcl_eval.scripts.cross_episode.run_batch_generate [options]

Reads LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (falling back to OPENAI_API_KEY /
OPENAI_BASE_URL, model gpt-4.1-mini) from the environment (or PROJECT_ROOT/.env).
Embedding-based memories use EMBED_API_KEY / EMBED_BASE_URL / EMBED_MODEL with the
same fallbacks (default text-embedding-3-small).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
from bfcl_eval.constants.eval_config import (
    DOTENV_PATH,
    POSSIBLE_ANSWER_PATH,
    PROMPT_PATH,
)
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.factory import build_memory
from bfcl_eval.memory.llm_env import embed_api_key, embed_base_url, embed_model, llm_api_key, llm_model
from bfcl_eval.memory.logging_wrapper import LoggedMemory
from bfcl_eval.model_handler.api_inference.ark_batch import ArkBatchFCHandler, ArkBatchPromptingHandler
from bfcl_eval.scripts.cross_episode.aggregator import load_id_to_category
from bfcl_eval.scripts.cross_episode.func_doc_loader import load_function_docs
from bfcl_eval.utils import load_file, make_json_serializable

DATASET_PATH = PROMPT_PATH / "BFCL_v4_multi_turn_ours.json"
IDS_DIR = Path(__file__).resolve().parent / "ids"


TOP_K_TYPES = ("mem0", "bm25", "qwen3_embedding", "graphrag", "agent_kb", "agent_workflow",
               "lightweight", "amem", "memos", "memoryos")
EMBED_TYPES = ("mem0", "reasoning_bank", "qwen3_embedding", "graphrag", "amem", "memos", "memoryos")


def memory_kwargs(memory_type: str, memory_dir: Path, *, api_key: str, mem_model: str,
                  clear: bool, readonly: bool, top_k: int) -> tuple[dict, dict]:
    """Constructor kwargs for ``build_memory(memory_type, **kwargs)`` + config-log fields.

    Embeddings go to an OpenAI-compatible endpoint (EMBED_API_KEY/EMBED_BASE_URL/EMBED_MODEL,
    falling back to OPENAI_*; default text-embedding-3-small). The parameter names
    ``ark_*`` / ``dashscope_*`` are kept for backend compatibility only.
    """
    memory_dir = Path(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict = dict(storage_dir=memory_dir, ark_api_key=api_key, ark_model=mem_model,
                        clear=clear, readonly=readonly)
    log: dict = {}
    if memory_type in EMBED_TYPES:
        e_key = embed_api_key()
        if not e_key:
            raise ValueError(f"Memory ({memory_type}) requires EMBED_API_KEY or OPENAI_API_KEY")
        e_model = os.environ.get("BFCL_EMBED_MODEL") or embed_model()
        kwargs.update(dashscope_api_key=e_key, dashscope_base_url=embed_base_url(),
                      embed_model=e_model)
        log["embed_model"] = e_model
    if memory_type in TOP_K_TYPES:
        kwargs.update(top_k=top_k)
        log["memory_top_k"] = top_k
    return kwargs, log


class DeferredUpdateMemory:
    """Thesis (online test): hold back the handler's end-of-sample ``update()`` so the sample can
    be checked first, then ingest it with the same signal as the build step (AWM: ``is_correct``).

    The handler calls begin_sample -> utilize -> ... -> update -> drain_usage; ``update`` here
    only records the trajectory, and ``commit()`` performs the real (logged) write afterwards.
    """

    def __init__(self, inner, memory_type: str):
        self.inner = inner
        self.memory_type = memory_type
        self.readonly = False
        self.pending: Optional[list] = None
        self.last_snippet: str = ""

    def begin_sample(self) -> None:
        self.pending, self.last_snippet = None, ""
        self.inner.begin_sample()

    def utilize(self, query: str) -> str:
        self.last_snippet = self.inner.utilize(query) or ""
        return self.last_snippet

    def update(self, trajectory: list[dict], **kwargs) -> None:
        self.pending = list(trajectory)

    def drain_usage(self):
        return self.inner.drain_usage()

    def commit(self, success: bool) -> int:
        """Write the held trajectory without the injected memory message. Returns #messages."""
        traj = self.pending or []
        if (traj and traj[0].get("role") == "system" and self.last_snippet
                and (traj[0].get("content") or "").strip() == self.last_snippet.strip()):
            traj = traj[1:]  # the retrieved-memory system message added in FC mode
        if self.memory_type == "agent_workflow":
            self.inner.update(traj, metadata={"is_correct": bool(success)})
        else:
            self.inner.update(traj)
        self.pending = None
        return len(traj)

    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)


_GT_CACHE: dict = {}


def check_sample(handler, entry: dict, raw_result, model: str) -> int:
    """1 if the sample passes the official multi-turn checker (same logic as run_batch_evaluate)."""
    from bfcl_eval.scripts.cross_episode.run_batch_evaluate import GROUND_TRUTH_PATH, _evaluate_entry
    if not _GT_CACHE:
        for gt in load_file(GROUND_TRUTH_PATH, sort_by_id=False, use_lock=False):
            _GT_CACHE[gt["id"]] = gt["ground_truth"]
    slug = model.replace("/", "_").replace(":", "_")
    success, _, _ = _evaluate_entry(handler, entry["id"], raw_result, _GT_CACHE.get(entry["id"], []),
                                    copy.deepcopy(entry), slug)
    return success


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", type=str, default=None,
                   help="Comma-separated list of sample ids to run.")
    p.add_argument("--output-dir", type=Path, default=Path("cross_episode_results"))
    p.add_argument("--run-name", type=str, default="auto")
    p.add_argument("--max-step", type=int, default=MAXIMUM_STEP_LIMIT)
    p.add_argument("--request-timeout", type=int, default=1200)
    p.add_argument("--resume", action="store_true",
                   help="Skip samples already present in result.jsonl.")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not call the API; print the planned ids and exit.")
    # Memory flags
    p.add_argument("--memory-type",
                   choices=["none", "mem0", "ace", "reasoning_bank",
                            "bm25", "qwen3_embedding", "graphrag",
                            "agent_kb", "agent_workflow", "skillweaver", "lightweight",
                            "amem", "memos", "memoryos"],
                   default="none",
                   help="Memory backend to use (default: none).")
    p.add_argument("--memory-clear", action=argparse.BooleanOptionalAction, default=True,
                   help="Clear the memory store before this run (default: True).")
    p.add_argument("--memory-readonly", action="store_true", default=False,
                   help="Only read from memory (utilize), never write (update).")
    p.add_argument("--memory-online", action="store_true", default=False,
                   help="Online test: after each sample, check it and then update memory "
                        "(requires --num-workers 1; samples run in dataset order).")
    p.add_argument("--memory-load-from", type=Path, default=None,
                   help="Load an existing memory store from this directory (implies --no-memory-clear).")
    p.add_argument("--memory-top-k", type=int, default=3,
                   help="Number of memories to retrieve per sample (default: 3).")
    p.add_argument("--model", type=str, default="",
                   help="Agent model (default: $LLM_MODEL or gpt-4.1-mini).")
    p.add_argument("--memory-log", type=Path, default=None,
                   help="JSONL write/retrieval log (default: <output>/memory_log.jsonl).")
    p.add_argument("--log-phase", type=str, default="",
                   help="Phase tag stored in each memory-log line.")
    p.add_argument("--mode", choices=["prompting", "FC"], default="prompting",
                   help="LLM call mode: 'prompting' (default) or 'FC' (native function calling).")
    return p.parse_args()


def setup_logging(error_log_path: Path) -> logging.Logger:
    logger = logging.getLogger("cross_episode_generate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(error_log_path)
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def derive_run_name(model: str) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    return f"{safe_model}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def filter_entries(
    entries: list[dict],
    explicit_ids: list[str] | None,
    limit: int | None,
    skip_ids: set[str],
) -> list[dict]:
    out = entries
    if explicit_ids is not None:
        wanted = set(explicit_ids)
        out = [e for e in out if e["id"] in wanted]
    if skip_ids:
        out = [e for e in out if e["id"] not in skip_ids]
    if limit is not None:
        out = out[:limit]
    return out


def process_one_sample(
    handler: ArkBatchPromptingHandler,
    entry: dict,
    category: str,
    jsonl_path: Path,
    jsonl_lock: threading.Lock,
    logger: logging.Logger,
    memory: Optional[Memory] = None,
    model: str = "",
) -> Optional[str]:
    """Run inference for one sample and append a JSON line to result.jsonl.

    Returns the error string if inference failed, else None.
    """
    test_entry = copy.deepcopy(entry)
    test_entry["function"] = load_function_docs(
        test_entry["involved_classes"], test_entry.get("excluded_function")
    )

    error: Optional[str] = None
    all_responses: list = []
    metadata: dict = {}
    if memory is not None and hasattr(memory, "set_context"):
        memory.set_context(entry["id"])
    try:
        all_responses, metadata = handler.inference(
            test_entry,
            include_input_log=True,
            exclude_state_log=False,
            external_memory=memory,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "inference failed for %s: %r\n%s",
            entry["id"],
            exc,
            traceback.format_exc(),
        )

    online_update = None
    if isinstance(memory, DeferredUpdateMemory) and memory.pending is not None:
        try:
            success = check_sample(handler, entry, all_responses, model) if error is None else 0
            n_msg = memory.commit(success)
            online_update = {"success": success, "num_messages": n_msg}
        except Exception as exc:
            online_update = {"error": f"{type(exc).__name__}: {exc}"}
            logger.warning("online memory update failed for %s: %r\n%s",
                           entry["id"], exc, traceback.format_exc())

    line = make_json_serializable({
        # Original BFCL-compatible fields
        "id": entry["id"],
        "result": all_responses,
        "input_token_count": metadata.get("input_token_count", []),
        "output_token_count": metadata.get("output_token_count", []),
        "latency": metadata.get("latency", []),
        "inference_log": metadata.get("inference_log", []),
        # Fork-specific telemetry consumed by run_batch_evaluate
        "category": category,
        "involved_classes": entry["involved_classes"],
        "num_turns": len(entry["question"]),
        "per_turn_totals": metadata.get("per_turn_totals", []),
        "sample_totals": metadata.get("sample_totals", {}),
        "memory_usage": metadata.get("memory_usage"),
        "online_update": online_update,
        "force_quit": metadata.get("force_quit", False),
        "full_message_history": metadata.get("full_message_history", []),
        "error": error,
    })

    with jsonl_lock:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    return error


def main() -> int:
    args = parse_args()

    if DOTENV_PATH.exists():
        load_dotenv(dotenv_path=DOTENV_PATH, override=False)

    # Agent + memory LLM: LLM_API_KEY/LLM_BASE_URL/LLM_MODEL (fallback OPENAI_*, gpt-4.1-mini).
    api_key = llm_api_key()
    model = args.model or llm_model()
    missing = [name for name, val in [("LLM_API_KEY/OPENAI_API_KEY", api_key), ("LLM_MODEL", model)] if not val]
    if missing and not args.dry_run:
        print(f"Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        return 2
    model = model or "<unset>"

    run_name = args.run_name if args.run_name != "auto" else derive_run_name(model)
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "errors.log")

    # ------------------------------------------------------------------ #
    # Memory setup                                                         #
    # ------------------------------------------------------------------ #
    memory: Optional[Memory] = None
    memory_config_log: dict = {"memory_type": args.memory_type}

    if args.memory_type != "none" and not args.dry_run:
        mem_model = os.environ.get("BFCL_MEMORY_MODEL") or model
        memory_dir: Path = (
            args.memory_load_from if args.memory_load_from else (output_dir / "memory")
        )
        do_clear = args.memory_clear and not args.memory_load_from
        try:
            common_kwargs, extra_log = memory_kwargs(
                args.memory_type, memory_dir, api_key=api_key, mem_model=mem_model,
                clear=do_clear, readonly=args.memory_readonly, top_k=args.memory_top_k,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        memory_config_log.update(extra_log)

        logger.info(
            "Memory: type=%s, dir=%s, clear=%s, readonly=%s%s",
            args.memory_type, memory_dir, do_clear, args.memory_readonly,
            f", top_k={args.memory_top_k}" if "memory_top_k" in extra_log else "",
        )
        memory = build_memory(args.memory_type, **common_kwargs)
        memory = LoggedMemory(
            memory,
            log_path=str(args.memory_log or (output_dir / "memory_log.jsonl")),
            phase=args.log_phase or ("test" if args.memory_readonly else "online"),
        )
        if args.memory_online:
            if args.memory_readonly or args.num_workers != 1:
                print("--memory-online needs a writable store and --num-workers 1", file=sys.stderr)
                return 2
            memory = DeferredUpdateMemory(memory, args.memory_type)
        memory_config_log.update({
            "memory_online": args.memory_online,
            "memory_dir": str(memory_dir),
            "memory_model": mem_model,
            "memory_clear": do_clear,
            "memory_readonly": args.memory_readonly,
        })

    entries = load_file(DATASET_PATH, sort_by_id=False, use_lock=False)
    id_to_category = load_id_to_category(IDS_DIR)

    jsonl_path = output_dir / "result.jsonl"
    if args.ids is not None and not args.ids.strip():
        print("--ids was given but is empty; refusing to run the full dataset.", file=sys.stderr)
        return 2
    explicit_ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else None

    skip_ids: set[str] = set()
    if args.resume and jsonl_path.exists():
        with open(jsonl_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                    if rec.get("id") and rec.get("error") is None:
                        skip_ids.add(rec["id"])
                except Exception:
                    continue

    selected = filter_entries(entries, explicit_ids, args.limit, skip_ids)

    config = {
        "model": model,
        "mode": args.mode,
        "run_name": run_name,
        "num_workers": args.num_workers,
        "limit": args.limit,
        "ids": explicit_ids,
        "max_step": args.max_step,
        "request_timeout": args.request_timeout,
        "resume": args.resume,
        "skipped_via_resume": sorted(skip_ids),
        "selected_count": len(selected),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "memory": memory_config_log,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(
        f"Generate {run_name}: {len(selected)} samples "
        f"({len(skip_ids)} resumed-skip), workers={args.num_workers}, output={output_dir}"
    )

    if args.dry_run:
        for e in selected:
            print(e["id"])
        return 0

    # One shared handler — ArkBatchClient uses per-thread TLS internally.
    if args.mode == "FC":
        handler = ArkBatchFCHandler(
            model_name=model,
            temperature=0,
            registry_name=model,
            is_fc_model=True,
            ark_api_key=api_key,
            request_timeout=args.request_timeout,
        )
    else:
        handler = ArkBatchPromptingHandler(
            model_name=model,
            temperature=0,
            registry_name=model,
            is_fc_model=False,
            ark_api_key=api_key,
            request_timeout=args.request_timeout,
        )

    jsonl_lock = threading.Lock()

    def _task(entry):
        category = id_to_category.get(entry["id"], "uncategorized")
        return process_one_sample(
            handler, entry, category, jsonl_path, jsonl_lock, logger, memory=memory, model=model
        )

    n_error = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(_task, e): e["id"] for e in selected}
        with tqdm(total=len(futures), desc="generate") as pbar:
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    err = fut.result()
                except Exception as exc:
                    logger.warning(f"sample {sid} worker failed: {exc!r}\n{traceback.format_exc()}")
                    err = f"{type(exc).__name__}: {exc}"
                if err is not None:
                    n_error += 1
                pbar.update(1)

    logger.info(
        f"Done. {len(selected) - n_error}/{len(selected)} succeeded, "
        f"{n_error} errored. Results at {jsonl_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
