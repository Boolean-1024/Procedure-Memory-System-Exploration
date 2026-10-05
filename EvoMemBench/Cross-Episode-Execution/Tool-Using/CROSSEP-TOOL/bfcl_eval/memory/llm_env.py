"""OpenAI-compatible LLM / embedding client factory (replaces the Volcengine Ark SDK).

Every place that used to build ``volcenginesdkarkruntime.Ark(api_key=...)`` and call
``client.batch.chat.completions.create(...)`` now builds a client here and calls the
standard ``client.chat.completions.create(...)``.

Resolution order for each setting (first non-empty wins):
  api_key : cfg["api_key"]  -> LLM_API_KEY  -> OPENAI_API_KEY
  base_url: cfg["base_url"] -> LLM_BASE_URL -> OPENAI_BASE_URL -> https://api.openai.com/v1
  model   : cfg["model"]    -> LLM_MODEL    -> "gpt-4.1-mini"
Embeddings use EMBED_API_KEY / EMBED_BASE_URL / EMBED_MODEL with the same fallbacks
(default model "text-embedding-3-small").
"""
from __future__ import annotations

import os
import sys
import threading

# mem0 telemetry opens a process-global Qdrant store (~/.mem0/migrations_qdrant), which breaks
# when several mem0 instances live in one process, and phones home. Disable unless set.
os.environ.setdefault("MEM0_TELEMETRY", "False")

DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _getenv(name: str) -> str:
    val = os.environ.get(name, "")
    if not val and sys.platform == "win32":
        # Machine/user-level variables set after the parent shell started are not
        # inherited; read them from the registry so `setx`-style keys still work.
        try:
            import winreg
            for root, sub in (
                (winreg.HKEY_CURRENT_USER, "Environment"),
                (winreg.HKEY_LOCAL_MACHINE,
                 r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            ):
                try:
                    with winreg.OpenKey(root, sub) as k:
                        val = winreg.QueryValueEx(k, name)[0]
                        if val:
                            os.environ[name] = val
                            break
                except OSError:
                    continue
        except ImportError:
            pass
    return val


def _first(*vals: str | None) -> str | None:
    for v in vals:
        if v:
            return v
    return None


def llm_api_key(cfg: dict | None = None) -> str | None:
    return _first((cfg or {}).get("api_key"), _getenv("LLM_API_KEY"), _getenv("OPENAI_API_KEY"))


def llm_base_url(cfg: dict | None = None) -> str:
    return _first((cfg or {}).get("base_url"), _getenv("LLM_BASE_URL"), _getenv("OPENAI_BASE_URL"),
                  _getenv("OPENAI_API_BASE"), DEFAULT_BASE_URL)


def llm_model(cfg: dict | None = None) -> str:
    return _first((cfg or {}).get("model"), _getenv("LLM_MODEL"), DEFAULT_LLM_MODEL)


def embed_api_key(cfg: dict | None = None) -> str | None:
    return _first((cfg or {}).get("api_key"), _getenv("EMBED_API_KEY"), llm_api_key())


def embed_base_url(cfg: dict | None = None) -> str:
    return _first((cfg or {}).get("base_url"), _getenv("EMBED_BASE_URL"), llm_base_url())


def embed_model(cfg: dict | None = None) -> str:
    return _first((cfg or {}).get("model"), _getenv("EMBED_MODEL"), DEFAULT_EMBED_MODEL)


def make_chat_client(cfg: dict | None = None):
    """Return an ``openai.OpenAI`` client for chat completions."""
    from openai import OpenAI
    return OpenAI(api_key=llm_api_key(cfg), base_url=llm_base_url(cfg), max_retries=3)


def make_embed_client(cfg: dict | None = None):
    """Return an ``openai.OpenAI`` client for embeddings."""
    from openai import OpenAI
    return OpenAI(api_key=embed_api_key(cfg), base_url=embed_base_url(cfg), max_retries=3)


def _is_reasoning_model(model: str | None) -> bool:
    m = (model or "").lower().split("/")[-1]
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


_capture_tls = threading.local()


class capture_llm_calls:
    """Context manager: record every chat completion made *in this thread*.

    Used by the memory logger so each update()/inject() log entry contains the
    memory-side LLM outputs (e.g. induced workflows, curator deltas, mem0 facts).
    """

    def __init__(self, max_chars: int = 20000):
        self.max_chars = max_chars
        self.calls: list[dict] = []

    def __enter__(self):
        self._prev = getattr(_capture_tls, "sink", None)
        _capture_tls.sink = self
        return self

    def __exit__(self, *exc):
        _capture_tls.sink = self._prev
        return False

    def record(self, model, messages, response) -> None:
        try:
            out = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            last_user = ""
            for m in reversed(messages or []):
                if m.get("role") == "user":
                    c = m.get("content")
                    last_user = c if isinstance(c, str) else str(c)
                    break
            self.calls.append({
                "model": model,
                "prompt_tail": last_user[-2000:],
                "output": out[: self.max_chars],
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            })
        except Exception:
            pass


RETRY_ATTEMPTS = int(os.environ.get("LLM_RETRY_ATTEMPTS", "30"))


def _with_retry(fn, *args, **kwargs):
    """Retry rate limits / transient server errors with capped exponential backoff.

    Several vendored libraries (mem0, A-mem) catch exceptions around their LLM calls
    and silently skip the write, so a 429 under the account's TPM limit would quietly
    produce an incomplete memory. Retrying here, below every library, prevents that.
    """
    import random
    import time
    import openai
    transient = (openai.RateLimitError, openai.APITimeoutError,
                 openai.APIConnectionError, openai.InternalServerError)
    fatal = (openai.AuthenticationError, openai.PermissionDeniedError)
    billing_words = ("insufficient_quota", "quota", "balance", "credit", "billing", "余额", "额度", "欠费")
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except fatal as e:
            # Bad key / no access / out of balance on a proxy: never let a library swallow it.
            print(f"[llm] FATAL {type(e).__name__}: {e}", flush=True)
            os._exit(3)
        except openai.APIStatusError as e:
            if not isinstance(e, transient) and any(w in str(e).lower() for w in billing_words):
                print(f"[llm] FATAL billing error: {e}", flush=True)
                os._exit(3)
            if not isinstance(e, transient):
                raise
            if any(w in str(e).lower() for w in billing_words) and "rate" not in str(e).lower():
                print(f"[llm] FATAL billing error (429): {e}", flush=True)
                os._exit(3)
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            delay = min(60.0, 2.0 * 2 ** attempt) + random.uniform(0, 1.5)
            print(f"[llm retry {attempt + 1}/{RETRY_ATTEMPTS}] {type(e).__name__}; sleeping {delay:.1f}s",
                  flush=True)
            time.sleep(delay)
        except transient as e:
            if getattr(e, "code", None) == "insufficient_quota" or "insufficient_quota" in str(e):
                # Out of credits is not transient. Exit hard: a library that swallows the
                # exception would otherwise record an empty write as done (2026-10-04).
                print(f"[llm] FATAL insufficient_quota (API credits exhausted): {e}", flush=True)
                os._exit(3)
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            delay = min(60.0, 2.0 * 2 ** attempt) + random.uniform(0, 1.5)
            print(f"[llm retry {attempt + 1}/{RETRY_ATTEMPTS}] {type(e).__name__}; sleeping {delay:.1f}s",
                  flush=True)
            time.sleep(delay)


def install_openai_hooks() -> None:
    """Patch ``openai`` chat completions once per process.

    1. Reasoning-model compatibility (gpt-5*, o*): those models reject
       ``temperature != 1`` and ``max_tokens``, which A-mem / MemoryOS hard-code.
       We drop ``temperature``/``top_p`` and translate ``max_tokens`` into a generous
       ``max_completion_tokens`` (reasoning tokens count against it).
       Non-reasoning models are passed through untouched.
    2. Thread-local capture of outputs for :class:`capture_llm_calls`.
    """
    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        return
    if getattr(Completions.create, "_thesis_hooked", False):
        return
    orig = Completions.create

    def create(self, *args, **kwargs):
        if _is_reasoning_model(kwargs.get("model")):
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            if "max_tokens" in kwargs:
                mt = kwargs.pop("max_tokens")
                kwargs.setdefault("max_completion_tokens", max(int(mt or 0), 8192))
        resp = _with_retry(orig, self, *args, **kwargs)
        sink = getattr(_capture_tls, "sink", None)
        if sink is not None and not kwargs.get("stream"):
            sink.record(kwargs.get("model"), kwargs.get("messages"), resp)
        return resp

    create._thesis_hooked = True
    Completions.create = create

    # Embeddings get the same rate-limit retry.
    try:
        from openai.resources.embeddings import Embeddings
        orig_emb = Embeddings.create

        def emb_create(self, *args, **kwargs):
            return _with_retry(orig_emb, self, *args, **kwargs)

        Embeddings.create = emb_create
    except ImportError:
        pass


def frozen_copy(path):
    """Read-only (test) mode: return a temporary copy of a store file/dir so the original is
    never opened. Vector DBs (ChromaDB, Qdrant local) rewrite index/SQLite files on open even
    when only queried; the thesis requires memory files to stay byte-identical during testing."""
    import atexit, os, shutil, tempfile
    if not path or not os.path.exists(str(path)):
        return path
    tmp = tempfile.mkdtemp(prefix="frozen_mem_")
    dst = os.path.join(tmp, os.path.basename(str(path).rstrip("/\\")))
    if os.path.isdir(str(path)):
        shutil.copytree(str(path), dst)
    else:
        shutil.copy2(str(path), dst)
    atexit.register(shutil.rmtree, tmp, True)
    return dst



def _sync_base_url_env() -> None:
    """OPENAI_API_BASE (older name, set on this machine) -> OPENAI_BASE_URL, which the OpenAI SDK,
    LiteLLM and mem0 read directly. Also makes sure registry-only keys reach child processes."""
    base = _getenv("OPENAI_BASE_URL") or _getenv("OPENAI_API_BASE")
    if base:
        os.environ["OPENAI_BASE_URL"] = base
    _getenv("OPENAI_API_KEY")


_sync_base_url_env()
install_openai_hooks()
