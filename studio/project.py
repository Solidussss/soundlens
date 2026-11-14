"""The editable project document.

A project is plain JSON so it can be saved, diffed, versioned (undo/redo) and
sent to the browser. Audio is never stored in the document: MIDI tracks are
re-rendered deterministically from their notes + instrument + seed, and audio
tracks reference files in the project's ``assets`` folder.

Time is measured in **beats** (quarter notes) everywhere in the document. A
MIDI clip holds notes whose ``t`` is relative to the clip's own timeline;
``offset`` says where in that timeline the clip window starts (so trimming
the left edge of a clip does not move its notes).
"""
from __future__ import annotations

import copy
import json
import secrets
from datetime import datetime, timezone
from typing import Any, Iterable

from .theory import SCALES, pitch_class

SCHEMA = "soundlens.studio.project/1"

ROLES = [
    "kick", "snare", "clap", "hat", "perc", "808", "bass", "melody", "counter_melody",
    "chords", "pad", "bell", "piano", "lead", "atmosphere", "fx",
    "lead_vocal", "backing_vocal", "adlib", "mix", "other",
]
DRUM_ROLES = {"kick", "snare", "clap", "hat", "perc"}
MIXER_FIELDS = {"gain_db", "pan", "mute", "solo", "name", "color"}


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(5)}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def beats_per_bar(project: dict) -> int:
    return int(project["params"].get("time_signature", 4))


def total_beats(project: dict) -> float:
    if project.get("sections"):
        last = max(project["sections"], key=lambda s: s["start_bar"] + s["bars"])
        return float((last["start_bar"] + last["bars"]) * beats_per_bar(project))
    end = 0.0
    for tr in project.get("tracks", []):
        for c in tr.get("clips", []):
            end = max(end, c["start"] + c["length"])
    return end


def beats_to_seconds(project: dict, beats: float) -> float:
    return float(beats) * 60.0 / float(project["params"]["bpm"])


def section_by_ref(project: dict, ref: str | int) -> dict:
    """Find a section by id, name ("verse", "verse 2", "second verse") or index."""
    secs = project.get("sections", [])
    if isinstance(ref, int):
        return secs[ref]
    r = str(ref).strip().lower()
    for s in secs:
        if s["id"] == ref:
            return s
    ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4, "last": -1, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
    words = r.replace("-", " ").split()
    n = None
    name_words = []
    for w in words:
        if w in ordinals:
            n = ordinals[w]
        elif w.isdigit():
            n = int(w)
        else:
            name_words.append(w)
    name = " ".join(name_words) or r
    if name in ("hook", "chorus"):
        matches = [s for s in secs if s["name"] in ("hook", "chorus")]
    else:
        matches = [s for s in secs if s["name"] == name]
    if not matches:
        raise ValueError(f"no section called {ref!r}; sections: {[s['label'] for s in secs]}")
    if n is None:
        return matches[0]
    if n == -1:
        return matches[-1]
    if not (1 <= n <= len(matches)):
        raise ValueError(f"there are only {len(matches)} {name} section(s)")
    return matches[n - 1]


def section_beats(project: dict, section: dict) -> tuple[float, float]:
    bpb = beats_per_bar(project)
    return float(section["start_bar"] * bpb), float((section["start_bar"] + section["bars"]) * bpb)


def label_sections(sections: list[dict]) -> None:
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for s in sections:
        totals[s["name"]] = totals.get(s["name"], 0) + 1
    for s in sections:
        counts[s["name"]] = counts.get(s["name"], 0) + 1
        s["label"] = s["name"].title() + (f" {counts[s['name']]}" if totals[s["name"]] > 1 else "")


def track_by_ref(project: dict, ref: str) -> dict:
    """Find a track by id, exact name, or role (first match)."""
    r = str(ref).strip().lower()
    tracks = project.get("tracks", [])
    for t in tracks:
        if t["id"] == ref:
            return t
    for t in tracks:
        if t["name"].lower() == r:
            return t
    for t in tracks:
        if t["role"] == r:
            return t
    raise ValueError(f"no track {ref!r}; tracks: {[t['name'] for t in tracks]}")


def clip_notes_absolute(clip: dict) -> list[dict]:
    """Notes of a MIDI clip on the song timeline, cropped to the clip window."""
    out = []
    lo, hi = clip.get("offset", 0.0), clip.get("offset", 0.0) + clip["length"]
    for n in clip.get("notes", []):
        if lo <= n["t"] < hi:
            m = dict(n)
            m["t"] = clip["start"] + (n["t"] - lo)
            m["d"] = min(n["d"], hi - n["t"])
            out.append(m)
    return out


def track_notes_absolute(track: dict) -> list[dict]:
    notes: list[dict] = []
    for c in track.get("clips", []):
        if not c.get("muted"):
            notes.extend(clip_notes_absolute(c))
    return sorted(notes, key=lambda n: (n["t"], n["p"]))


def make_track(name: str, role: str, instrument: dict | None, notes: Iterable[dict] | None = None, length: float = 0.0,
               kind: str = "midi", seed: int = 0, gain_db: float = 0.0, pan: float = 0.0, fx: list | None = None,
               color: str | None = None, generator: dict | None = None) -> dict:
    clip = {"id": new_id("clip"), "start": 0.0, "length": float(length), "offset": 0.0}
    if kind == "midi":
        clip["notes"] = [dict(n) for n in (notes or [])]
    return {
        "id": new_id("trk"), "name": name, "role": role, "kind": kind,
        "instrument": instrument, "clips": [clip] if length > 0 else [],
        "fx": fx or [], "gain_db": gain_db, "pan": pan, "mute": False, "solo": False,
        "color": color, "seed": int(seed), "generator": generator or {},
    }


def validate_project(project: dict) -> None:
    """Raise ValueError if the document is inconsistent. Cheap enough to run after every edit."""
    if project.get("schema") != SCHEMA:
        raise ValueError("not a SoundLens Studio project")
    p = project["params"]
    if not (40 <= float(p["bpm"]) <= 240):
        raise ValueError("bpm must be 40–240")
    pitch_class(p["key"])
    if p["scale"] not in SCALES:
        raise ValueError(f"unknown scale {p['scale']!r}")
    ids = set()
    for t in project["tracks"]:
        if t["id"] in ids:
            raise ValueError(f"duplicate track id {t['id']}")
        ids.add(t["id"])
        if t["role"] not in ROLES:
            raise ValueError(f"unknown role {t['role']!r}")
        if t["kind"] not in ("midi", "audio"):
            raise ValueError("track kind must be midi or audio")
        if not (-60 <= float(t["gain_db"]) <= 24):
            raise ValueError("gain_db out of range")
        if not (-1 <= float(t["pan"]) <= 1):
            raise ValueError("pan out of range")
        for c in t["clips"]:
            if c["length"] <= 0 or c["start"] < 0 or c.get("offset", 0) < 0:
                raise ValueError(f"bad clip geometry in {t['name']}")
            if t["kind"] == "midi":
                for n in c.get("notes", []):
                    if not (0 <= int(n["p"]) <= 127) or n["d"] <= 0 or n["t"] < 0 or not (0 <= n.get("v", 0.8) <= 1):
                        raise ValueError(f"bad note in {t['name']}: {n}")
            elif not c.get("source"):
                raise ValueError(f"audio clip without source in {t['name']}")


def clone(project: dict) -> dict:
    return copy.deepcopy(project)


def dumps(project: dict) -> str:
    return json.dumps(project, sort_keys=True, separators=(",", ":"))


def summary(project: dict) -> dict[str, Any]:
    """Compact description used by the orchestrator / LLM prompt."""
    return {
        "title": project.get("title"),
        "bpm": project["params"]["bpm"], "key": project["params"]["key"], "scale": project["params"]["scale"],
        "sections": [{"id": s["id"], "label": s["label"], "bars": s["bars"]} for s in project.get("sections", [])],
        "tracks": [{"id": t["id"], "name": t["name"], "role": t["role"], "kind": t["kind"],
                    "instrument": (t.get("instrument") or {}).get("preset"), "gain_db": t["gain_db"],
                    "fx": [f["type"] for f in t.get("fx", [])], "mute": t["mute"]} for t in project["tracks"]],
    }
