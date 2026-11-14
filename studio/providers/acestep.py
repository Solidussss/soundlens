"""ACE-Step 1.5 client (self-hosted REST server, ``uv run acestep-api``, default port 8001).

Endpoints and field names follow the project's docs/en/API.md as read on
2026-11-13 (see docs/studio/research_generation_models.md):

* ``POST /release_task`` (multipart) → ``task_id``; ``task_type`` ∈ text2music, repaint, cover, lego, extract, complete
* ``POST /query_result`` ``{"task_id_list": [...]}`` → status 0 running / 1 done / 2 failed, result file URLs
* ``GET /v1/audio?path=...`` → audio bytes; ``GET /health``

**Not yet exercised against a live server** (no GPU and no network route to the
weights in the development sandbox). Response parsing is therefore defensive and
every request/response is logged in the job record. ``lego``/``extract``/``complete``
need a *Base* model (``acestep-v15-base``); Turbo only supports text2music/repaint/cover.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .base import Provider, ProviderUnavailable

# SoundLens roles → ACE-Step track classes (TRACK_NAMES in acestep/constants.py). There is no "808" class.
ROLE_TO_TRACK = {
    "808": "bass", "bass": "bass", "kick": "drums", "snare": "drums", "clap": "drums", "hat": "drums", "perc": "percussion",
    "melody": "synth", "lead": "synth", "counter_melody": "synth", "bell": "keyboard", "piano": "keyboard", "chords": "keyboard",
    "pad": "synth", "atmosphere": "fx", "fx": "fx", "lead_vocal": "vocals", "backing_vocal": "backing_vocals", "adlib": "backing_vocals",
}
BASE_ONLY = {"lego", "extract", "complete"}


class AceStepProvider(Provider):
    name = "acestep"
    capabilities = {"text2music", "repaint", "cover", "lego", "extract", "complete", "vocals"}
    cost_note = "self-hosted GPU; vendor-quoted <10 s per song on an RTX 3090 (unverified by us)"

    def __init__(self, base_url: str | None = None, timeout_s: float = 600.0):
        self.base_url = (base_url or os.getenv("ACESTEP_API_URL", "")).rstrip("/")
        self.base_model = os.getenv("ACESTEP_BASE_MODEL", "acestep-v15-base")
        self.turbo_model = os.getenv("ACESTEP_TURBO_MODEL", "acestep-v15-turbo")
        self.timeout_s = timeout_s

    def available(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "set ACESTEP_API_URL to a running ACE-Step 1.5 API server (needs a GPU host)"
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/health", timeout=5)
            return (r.status_code == 200, "ok" if r.status_code == 200 else f"health check returned {r.status_code}")
        except Exception as e:
            return False, f"server unreachable ({type(e).__name__})"

    # -- low level ---------------------------------------------------------
    def _release(self, fields: dict[str, Any], files: dict[str, Path] | None = None) -> str:
        import httpx
        data = {k: (json.dumps(v) if isinstance(v, (dict, list)) else str(v)) for k, v in fields.items() if v is not None}
        handles = {k: (p.name, p.open("rb"), "audio/wav") for k, p in (files or {}).items()}
        try:
            r = httpx.post(f"{self.base_url}/release_task", data=data, files=handles or None, timeout=60)
        finally:
            for _, fh, _ in handles.values():
                fh.close()
        r.raise_for_status()
        body = r.json()
        task_id = body.get("task_id") or (body.get("data") or {}).get("task_id")
        if not task_id:
            raise ProviderUnavailable(f"unexpected release_task response: {str(body)[:300]}")
        return str(task_id)

    def _wait(self, task_id: str, progress=None) -> dict:
        import httpx
        t0 = time.time()
        while time.time() - t0 < self.timeout_s:
            r = httpx.post(f"{self.base_url}/query_result", json={"task_id_list": [task_id]}, timeout=30)
            r.raise_for_status()
            body = r.json()
            items = body.get("data") if isinstance(body.get("data"), list) else body.get("results") or body.get("data") or body
            item = items[0] if isinstance(items, list) and items else (items.get(task_id) if isinstance(items, dict) else None)
            status = (item or {}).get("status")
            if status == 1:
                return item
            if status == 2:
                raise RuntimeError(f"ACE-Step task failed: {str(item)[:300]}")
            if progress:
                progress(min(0.9, (time.time() - t0) / 120.0), "ACE-Step is generating")
            time.sleep(2.0)
        raise TimeoutError("ACE-Step task timed out")

    def _download(self, item: dict, dest: Path) -> Path:
        import httpx
        result = item.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                pass
        candidates = []
        def walk(x):
            if isinstance(x, str) and (x.endswith((".wav", ".mp3", ".flac")) or "/v1/audio" in x):
                candidates.append(x)
            elif isinstance(x, dict):
                for v in x.values(): walk(v)
            elif isinstance(x, list):
                for v in x: walk(v)
        walk(result if result is not None else item)
        if not candidates:
            raise RuntimeError(f"no audio file in ACE-Step result: {str(item)[:300]}")
        url = candidates[0]
        if not url.startswith("http"):
            url = f"{self.base_url}/v1/audio?path={url}" if "/v1/audio" not in url else f"{self.base_url}{url}"
        r = httpx.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest

    # -- tasks -------------------------------------------------------------
    def text2music(self, caption: str, lyrics: str, bpm: float, key_scale: str, duration_s: float, dest: Path, progress=None) -> Path:
        self.require()
        tid = self._release({"task_type": "text2music", "caption": caption, "prompt": caption, "lyrics": lyrics or "[Instrumental]",
                             "bpm": int(bpm), "key_scale": key_scale, "time_signature": 4, "audio_duration": float(duration_s),
                             "model": self.turbo_model})
        return self._download(self._wait(tid, progress), dest)

    def repaint(self, src: Path, start_s: float, end_s: float, caption: str, lyrics: str, dest: Path, progress=None, base: bool = False) -> Path:
        self.require()
        tid = self._release({"task_type": "repaint", "caption": caption, "prompt": caption, "lyrics": lyrics or "[Instrumental]",
                             "repainting_start": start_s, "repainting_end": end_s, "model": self.base_model if base else self.turbo_model},
                            files={"src_audio": src})
        return self._download(self._wait(tid, progress), dest)

    def lego(self, context_mix: Path, role: str, caption: str, lyrics: str, bpm: float, key_scale: str, dest: Path, progress=None) -> Path:
        """Generate one track (by ACE-Step class) that fits the existing audio context."""
        self.require()
        track = ROLE_TO_TRACK.get(role)
        if not track:
            raise ProviderUnavailable(f"ACE-Step has no track class for {role}")
        tid = self._release({"task_type": "lego", "instruction": f"Generate the {track} track based on the audio context:",
                             "track_name": track, "caption": caption, "prompt": caption, "lyrics": lyrics or "[Instrumental]",
                             "bpm": int(bpm), "key_scale": key_scale, "model": self.base_model}, files={"src_audio": context_mix})
        return self._download(self._wait(tid, progress), dest)

    def extract(self, mix: Path, role: str, dest: Path, progress=None) -> Path:
        self.require()
        track = ROLE_TO_TRACK.get(role, role)
        tid = self._release({"task_type": "extract", "instruction": f"Extract the {track} track from the audio:", "track_name": track,
                             "model": self.base_model}, files={"src_audio": mix})
        return self._download(self._wait(tid, progress), dest)
