"""Save / verify / restore checkpoints of every experience-memory store.

A checkpoint is a full copy of the built memory stores of all three benchmarks plus an
MD5 manifest, so a store that was modified during a non-frozen (online-update) test run
can be put back byte-for-byte.

    python setup/memory_checkpoint.py save    <name> [--note "..."]
    python setup/memory_checkpoint.py verify  <name>          # checkpoint copy intact?
    python setup/memory_checkpoint.py diff    <name>          # live stores vs checkpoint
    python setup/memory_checkpoint.py restore <name> [--only alfworld/mem0 ...]
    python setup/memory_checkpoint.py list

`restore` first saves the current live state as an automatic checkpoint
(`auto_before_restore_<time>`), then replaces the live stores and verifies them against
the manifest. Checkpoints live in <ROOT>/checkpoints/<name>/ and are made read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[1]))
CKPT_DIR = ROOT / "checkpoints"

# live store locations: key -> path (directories are copied whole)
STORES = {
    "alfworld": ROOT / "EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/output/main_gpt-4.1-mini/memory",
    "bfcl": ROOT / "EvoMemBench/Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL/cross_episode_results/split_main_gpt-4.1-mini/memory",
    "appworld": ROOT / "ace-appworld/experiments/outputs/thesis_memory/memory",
    "appworld_ace_playbook": ROOT / "ace-appworld/experiments/playbooks/thesis_ace_offline_no_gt_gpt-4.1-mini.txt",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(path: Path) -> dict[str, str]:
    if path.is_file():
        return {"<file>": md5(path)}  # name-independent: the copy is stored under its key
    return {f.relative_to(path).as_posix(): md5(f) for f in sorted(path.rglob("*")) if f.is_file()}


def set_readonly(path: Path, readonly: bool):
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    for f in files:
        mode = f.stat().st_mode
        os.chmod(f, (mode & ~stat.S_IWRITE) if readonly else (mode | stat.S_IWRITE))


def _rm(path: Path):
    if not path.exists():
        return
    if path.is_file():
        set_readonly(path, False)
        path.unlink()
    else:
        set_readonly(path, False)
        shutil.rmtree(path)


def _copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
    else:
        shutil.copytree(src, dst)


def save(name: str, note: str = "", keys=None) -> Path:
    dest = CKPT_DIR / name
    if dest.exists():
        sys.exit(f"checkpoint {name!r} already exists: {dest}")
    keys = keys or list(STORES)
    manifest = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note,
                "root": str(ROOT), "stores": {}}
    for key in keys:
        src = STORES[key]
        if not src.exists():
            print(f"  [skip, missing] {key}: {src}")
            continue
        _copy(src, dest / key)
        hashes = hash_tree(dest / key)
        if hashes != hash_tree(src):  # store changed while copying
            _rm(dest)
            sys.exit(f"{key} changed during copy; stop all runs and retry")
        manifest["stores"][key] = {"source": str(src.relative_to(ROOT)), "is_file": src.is_file(),
                                   "n_files": len(hashes), "files": hashes,
                                   "bytes": sum(f.stat().st_size for f in ([dest / key] if src.is_file()
                                                else (dest / key).rglob("*")) if f.is_file())}
        print(f"  saved {key:22s} {len(hashes):5d} files  {manifest['stores'][key]['bytes'] / 2**20:8.1f} MiB")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    set_readonly(dest, True)
    print(f"checkpoint saved: {dest}")
    return dest


def load_manifest(name: str) -> dict:
    p = CKPT_DIR / name / "manifest.json"
    if not p.exists():
        sys.exit(f"no checkpoint {name!r} ({p})")
    return json.loads(p.read_text(encoding="utf-8"))


def compare(expected: dict[str, str], actual: dict[str, str]) -> tuple[list, list, list]:
    changed = [k for k in expected if k in actual and actual[k] != expected[k]]
    removed = [k for k in expected if k not in actual]
    added = [k for k in actual if k not in expected]
    return changed, added, removed


def report(label: str, key: str, expected: dict, actual: dict) -> bool:
    changed, added, removed = compare(expected, actual)
    ok = not (changed or added or removed)
    print(f"  {label} {key:22s} {'OK' if ok else 'DIFFERS'}  changed={len(changed)} added={len(added)} removed={len(removed)}")
    for k in (changed + added + removed)[:5]:
        print(f"      {k}")
    return ok


def verify(name: str) -> bool:
    m = load_manifest(name)
    ok = True
    for key, info in m["stores"].items():
        ok &= report("ckpt", key, info["files"], hash_tree(CKPT_DIR / name / key))
    print("checkpoint intact" if ok else "!! checkpoint copy is damaged")
    return ok


def diff(name: str) -> bool:
    m = load_manifest(name)
    same = True
    for key, info in m["stores"].items():
        live = STORES[key]
        same &= report("live", key, info["files"], hash_tree(live) if live.exists() else {})
    print("live stores identical to checkpoint" if same else "live stores differ from checkpoint")
    return same


def restore(name: str, only=None):
    m = load_manifest(name)
    if not verify(name):
        sys.exit("refusing to restore from a damaged checkpoint")
    keys = only or list(m["stores"])
    # partial keys like alfworld/mem0 restore one system's store only
    auto = f"auto_before_restore_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"saving current live state as {auto!r} first")
    save(auto, note=f"automatic backup before restoring {name!r}",
         keys=sorted({k.split('/')[0] for k in keys}))
    for key in keys:
        top, _, sub = key.partition("/")
        src = CKPT_DIR / name / top / sub if sub else CKPT_DIR / name / top
        dst = STORES[top] / sub if sub else STORES[top]
        _rm(dst)
        _copy(src, dst)
        set_readonly(dst, False)  # live stores must stay writable
        print(f"  restored {key}")
    ok = True
    for key in keys:
        top, _, sub = key.partition("/")
        exp = m["stores"][top]["files"]
        if sub:
            exp = {k[len(sub) + 1:]: v for k, v in exp.items() if k.startswith(sub + "/")}
        ok &= report("live", key, exp, hash_tree(STORES[top] / sub if sub else STORES[top]))
    print("restore verified" if ok else "!! restore verification failed")


def list_ckpts():
    for d in sorted(CKPT_DIR.glob("*/manifest.json")):
        m = json.loads(d.read_text(encoding="utf-8"))
        n = sum(s["n_files"] for s in m["stores"].values())
        mb = sum(s["bytes"] for s in m["stores"].values()) / 2**20
        print(f"{m['name']:40s} {m['created']}  {n:5d} files {mb:8.1f} MiB  {m.get('note', '')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["save", "verify", "diff", "restore", "list"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--note", default="")
    ap.add_argument("--only", nargs="+", help="restore only these keys, e.g. alfworld bfcl/mem0")
    a = ap.parse_args()
    if a.cmd != "list" and not a.name:
        ap.error("name required")
    if a.cmd == "save":
        save(a.name, a.note)
    elif a.cmd == "verify":
        sys.exit(0 if verify(a.name) else 1)
    elif a.cmd == "diff":
        sys.exit(0 if diff(a.name) else 1)
    elif a.cmd == "restore":
        restore(a.name, a.only)
    else:
        list_ckpts()


if __name__ == "__main__":
    main()
