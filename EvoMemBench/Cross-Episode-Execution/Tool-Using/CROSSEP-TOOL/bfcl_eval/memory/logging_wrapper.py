"""Write / retrieval logging for any BFCL ``Memory`` backend.

Appends one JSON line per call to ``log_path``:
  {"event": "utilize", "phase", "sample_id", "query", "snippet", "snippet_chars", "llm_calls", "ts"}
  {"event": "update",  "phase", "sample_id", "num_messages", "skipped_read_only",
   "llm_calls", "error", "ts"}
``llm_calls`` holds the memory-side LLM outputs captured in the calling thread,
i.e. what the backend extracted / wrote.
"""
from __future__ import annotations

import json
import os
import threading
import time

from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.llm_env import capture_llm_calls


class LoggedMemory(Memory):
    def __init__(self, inner: Memory, log_path: str, phase: str = "", max_chars: int = 20000):
        self.inner = inner
        self.readonly = inner.readonly
        self.log_path = log_path
        self.phase = phase
        self.max_chars = max_chars
        self._lock = threading.Lock()
        self._ctx = threading.local()
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def set_context(self, sample_id: str | None) -> None:
        self._ctx.sample_id = sample_id

    def _write(self, rec: dict) -> None:
        rec["ts"] = time.time()
        rec.setdefault("sample_id", getattr(self._ctx, "sample_id", None))
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Memory interface -------------------------------------------------------
    def begin_sample(self) -> None:
        self.inner.begin_sample()

    def drain_usage(self):
        return self.inner.drain_usage()

    def utilize(self, system_prompt: str) -> str:
        with capture_llm_calls(self.max_chars) as cap:
            snippet = self.inner.utilize(system_prompt)
        self._write({
            "event": "utilize", "phase": self.phase,
            "query": (system_prompt or "")[: self.max_chars],
            "snippet": (snippet or "")[: self.max_chars],
            "snippet_chars": len(snippet or ""),
            "llm_calls": cap.calls,
        })
        return snippet

    def update(self, trajectory: list[dict], **kwargs) -> None:
        rec = {"event": "update", "phase": self.phase, "num_messages": len(trajectory),
               "skipped_read_only": bool(self.inner.readonly)}
        if kwargs:
            rec["kwargs"] = kwargs
        with capture_llm_calls(self.max_chars) as cap:
            try:
                self.inner.update(trajectory, **kwargs)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["llm_calls"] = cap.calls
                self._write(rec)
                raise
        rec["llm_calls"] = cap.calls
        self._write(rec)

    @classmethod
    def load_from_disk(cls, **backend_kwargs):
        raise NotImplementedError("wrap an already-constructed backend instead")

    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)
