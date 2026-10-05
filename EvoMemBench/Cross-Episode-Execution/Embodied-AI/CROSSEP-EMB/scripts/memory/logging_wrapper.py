"""Write / retrieval logging for any BaseMemory backend.

Wraps a backend and appends one JSON line per call to ``log_path``:

  {"event": "inject", "data_idx", "phase", "query", "injected", "injected_chars",
   "stats", "llm_calls", "ts"}
  {"event": "update", "data_idx", "phase", "reward", "num_turns", "skipped_read_only",
   "stats", "llm_calls", "store_files", "error", "ts"}

``injected`` is exactly the text the backend added to the system prompt, so the
retrieval log shows what each test episode actually saw. ``llm_calls`` holds the
memory-side LLM outputs captured in the calling thread (what got written).
"""
from __future__ import annotations

import json
import os
import threading
import time

from ._llm import capture_llm_calls
from .base import BaseMemory, MemoryCallStats


def _dir_snapshot(paths: list[str]) -> dict:
    """Total file count / bytes under each storage path (cheap growth indicator)."""
    snap = {}
    for p in paths:
        if not p:
            continue
        if os.path.isfile(p):
            snap[p] = {"files": 1, "bytes": os.path.getsize(p)}
        elif os.path.isdir(p):
            n = b = 0
            for root, _, files in os.walk(p):
                for fn in files:
                    try:
                        b += os.path.getsize(os.path.join(root, fn))
                        n += 1
                    except OSError:
                        pass
            snap[p] = {"files": n, "bytes": b}
    return snap


class LoggedMemory(BaseMemory):
    def __init__(self, inner: BaseMemory, log_path: str, phase: str = "",
                 store_paths: list[str] | None = None, max_chars: int = 20000):
        super().__init__(memory_type=inner.memory_type, read_only=inner.read_only)
        self.inner = inner
        self.log_path = log_path
        self.phase = phase
        self.store_paths = store_paths or []
        self.max_chars = max_chars
        self._lock = threading.Lock()
        self._ctx = threading.local()
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    # Called by the eval loop before inject() so the log knows the episode id.
    def set_context(self, data_idx: int | None) -> None:
        self._ctx.data_idx = data_idx

    def _write(self, rec: dict) -> None:
        rec["ts"] = time.time()
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def inject(self, conversation):
        before = conversation[0]["content"] if conversation else ""
        query = conversation[2]["content"] if len(conversation) > 2 else ""
        with capture_llm_calls(self.max_chars) as cap:
            new_conv, stats = self.inner.inject(conversation)
        after = new_conv[0]["content"] if new_conv else ""
        if after == before:
            injected = ""
        elif after.startswith(before):
            injected = after[len(before):]
        elif after.endswith(before):
            injected = after[: len(after) - len(before)]
        else:
            injected = after
        self._write({
            "event": "inject",
            "phase": self.phase,
            "data_idx": getattr(self._ctx, "data_idx", None),
            "query": query[: self.max_chars],
            "injected": injected[: self.max_chars],
            "injected_chars": len(injected),
            "stats": stats.to_dict(),
            "llm_calls": cap.calls,
        })
        return new_conv, stats

    def update(self, conversation, data_idx=None, reward=None):
        rec = {
            "event": "update",
            "phase": self.phase,
            "data_idx": data_idx,
            "reward": reward,
            "num_turns": len(conversation),
            "skipped_read_only": bool(self.inner.read_only),
        }
        stats = MemoryCallStats()
        with capture_llm_calls(self.max_chars) as cap:
            try:
                stats = self.inner.update(conversation, data_idx, reward=reward)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["stats"] = stats.to_dict()
                rec["llm_calls"] = cap.calls
                self._write(rec)
                raise
        rec["stats"] = stats.to_dict()
        rec["llm_calls"] = cap.calls
        if self.store_paths and not self.inner.read_only:
            rec["store_files"] = _dir_snapshot(self.store_paths)
        self._write(rec)
        return stats

    def load_from_disk(self, **kwargs):
        return self.inner.load_from_disk(**kwargs)

    def __getattr__(self, name):
        # Delegate backend-specific attributes (only reached for missing attrs).
        return getattr(self.__dict__["inner"], name)
