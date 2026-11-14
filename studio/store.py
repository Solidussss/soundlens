"""Project storage with snapshot history (undo/redo) and a small job queue.

Layout under ``<SOUNDLENS_DATA_DIR>/studio``::

    projects/<owner>/<project_id>/project.json      current document
    projects/<owner>/<project_id>/history/NNNNNN.json  one snapshot per revision
    projects/<owner>/<project_id>/history.json      {"revision", "head", "log": [...]}
    projects/<owner>/<project_id>/assets/           uploaded / provider audio
    projects/<owner>/<project_id>/exports/
    cache/stems/                                    content-addressed stems (shared)

Writes are atomic (temp file + rename) like the rest of SoundLens. Undo moves
``head`` back without deleting anything; a new edit after undo discards the
redo branch (standard editor behaviour). Projects are never deleted by edits.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

MAX_HISTORY = 200
_SAFE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(path)


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not _SAFE.match(value):
        raise ValueError("invalid id")
    return value


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.projects = self.root / "projects"
        self.cache = self.root / "cache" / "stems"
        self.projects.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def prune_cache(self, max_bytes: int | None = None) -> dict:
        """Evict least-recently-used cached stems/mixes above ``max_bytes``.

        Safe because every stem is a deterministic function of the project document:
        an evicted stem is simply re-rendered on next use. Uploaded/provider audio lives
        in project ``assets`` and is never touched here.
        """
        import os as _os
        limit = max_bytes if max_bytes is not None else int(float(_os.getenv("SOUNDLENS_STUDIO_CACHE_GB", "5")) * 1024 ** 3)
        files = [p for p in self.cache.iterdir() if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        removed = 0
        if total > limit:
            for p in sorted(files, key=lambda p: p.stat().st_atime):
                if total <= limit * 0.8:
                    break
                size = p.stat().st_size
                p.unlink(missing_ok=True)
                total -= size
                removed += 1
        return {"bytes": total, "removed": removed}

    def lock(self, project_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(project_id, threading.Lock())

    def dir(self, owner: str, project_id: str) -> Path:
        return self.projects / safe_id(owner) / safe_id(project_id)

    def assets_dir(self, owner: str, project_id: str) -> Path:
        d = self.dir(owner, project_id) / "assets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def exports_dir(self, owner: str, project_id: str) -> Path:
        d = self.dir(owner, project_id) / "exports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- documents --------------------------------------------------------
    def create(self, owner: str, project: dict, note: str = "generated") -> dict:
        d = self.dir(owner, project["id"])
        if d.exists():
            raise ValueError("project already exists")
        (d / "history").mkdir(parents=True)
        _atomic_write(d / "history" / "000000.json", project)
        _atomic_write(d / "project.json", project)
        _atomic_write(d / "history.json", {"revision": 0, "head": 0, "log": [{"rev": 0, "note": note, "at": time.time()}]})
        return project

    def load(self, owner: str, project_id: str) -> dict:
        path = self.dir(owner, project_id) / "project.json"
        if not path.exists():
            raise FileNotFoundError(project_id)
        return json.loads(path.read_text())

    def history(self, owner: str, project_id: str) -> dict:
        return json.loads((self.dir(owner, project_id) / "history.json").read_text())

    def commit(self, owner: str, project_id: str, project: dict, note: str, actions: list | None = None) -> int:
        d = self.dir(owner, project_id)
        h = self.history(owner, project_id)
        # drop the redo branch
        for entry in [e for e in h["log"] if e["rev"] > h["head"]]:
            (d / "history" / f"{entry['rev']:06d}.json").unlink(missing_ok=True)
        h["log"] = [e for e in h["log"] if e["rev"] <= h["head"]]
        rev = h["head"] + 1
        _atomic_write(d / "history" / f"{rev:06d}.json", project)
        _atomic_write(d / "project.json", project)
        h["log"].append({"rev": rev, "note": note[:200], "at": time.time(), "actions": actions or []})
        h["revision"], h["head"] = rev, rev
        # keep the first snapshot (the original generation) forever; trim the middle
        while len(h["log"]) > MAX_HISTORY:
            victim = h["log"].pop(1)
            (d / "history" / f"{victim['rev']:06d}.json").unlink(missing_ok=True)
        _atomic_write(d / "history.json", h)
        return rev

    def move_head(self, owner: str, project_id: str, step: int) -> dict:
        d = self.dir(owner, project_id)
        h = self.history(owner, project_id)
        revs = [e["rev"] for e in h["log"]]
        i = revs.index(h["head"]) + step
        if not 0 <= i < len(revs):
            raise ValueError("nothing to undo" if step < 0 else "nothing to redo")
        h["head"] = revs[i]
        project = json.loads((d / "history" / f"{revs[i]:06d}.json").read_text())
        _atomic_write(d / "project.json", project)
        _atomic_write(d / "history.json", h)
        return project

    def list(self, owner: str) -> list[dict]:
        base = self.projects / safe_id(owner)
        out = []
        if not base.exists():
            return out
        for d in base.iterdir():
            f = d / "project.json"
            if f.exists():
                try:
                    p = json.loads(f.read_text())
                    out.append({"id": p["id"], "title": p.get("title"), "character_id": p.get("character_id"),
                                "updated_at": p.get("updated_at"), "bpm": p["params"]["bpm"], "key": p["params"]["key"],
                                "scale": p["params"]["scale"], "tracks": len(p.get("tracks", []))})
                except (OSError, ValueError, KeyError):
                    continue
        return sorted(out, key=lambda x: x.get("updated_at") or "", reverse=True)

    def delete(self, owner: str, project_id: str) -> None:
        d = self.dir(owner, project_id)
        if d.exists():
            trash = self.root / "trash" / f"{safe_id(owner)}_{safe_id(project_id)}_{int(time.time())}"
            trash.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(trash))  # recoverable, like account deletion backups


class JobQueue:
    """In-process background jobs with persisted state (survive a page reload, not a server restart)."""

    def __init__(self, root: Path, workers: int = 2):
        self.root = Path(root) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="studio-job")
        self.lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return self.root / f"{safe_id(job_id)}.json"

    def get(self, job_id: str) -> dict:
        p = self._path(job_id)
        if not p.exists():
            raise FileNotFoundError(job_id)
        return json.loads(p.read_text())

    def _update(self, job_id: str, **fields) -> None:
        with self.lock:
            job = self.get(job_id)
            job.update(fields)
            job["updated_at"] = time.time()
            _atomic_write(self._path(job_id), job)

    def submit(self, owner: str, kind: str, fn: Callable[[Callable[[float, str], None]], dict], meta: dict | None = None) -> dict:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = {"id": job_id, "owner": owner, "kind": kind, "status": "queued", "progress": 0.0, "message": "queued",
               "result": None, "error": None, "meta": meta or {}, "created_at": time.time(), "updated_at": time.time()}
        _atomic_write(self._path(job_id), job)

        def progress(frac: float, msg: str) -> None:
            self._update(job_id, progress=round(max(0.0, min(1.0, frac)), 3), message=msg)

        def run():
            self._update(job_id, status="running", message="starting")
            try:
                result = fn(progress)
                self._update(job_id, status="done", progress=1.0, message="done", result=result)
            except Exception as e:  # recorded for the UI; never crashes the server
                traceback.print_exc()
                self._update(job_id, status="failed", message="failed", error=f"{type(e).__name__}: {e}")

        self.pool.submit(run)
        return job
