"""Validated, reversible edit operations.

Every change to a project — from a mixer knob, the piano roll, the timeline
or a natural-language command — is expressed as one of these actions:

    {"op": "add_effect", "track": "808", "type": "distortion", "params": {...}}

``apply_actions`` works on a copy and validates the whole document afterwards,
so a failing batch changes nothing. Undo/redo is snapshot based
(:mod:`studio.store`), which makes every action reversible without writing an
inverse for each one.

``diff_tracks`` compares stem cache keys before/after, so each edit reports
exactly which tracks will sound different — and the PoC verifies the others
are bit-identical.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

from . import composer, dsp, synth
from .project import (DRUM_ROLES, ROLES, beats_per_bar, clone, label_sections, make_track, new_id, section_beats,
                      section_by_ref, total_beats, track_by_ref, validate_project)
from .render import stem_key


class ActionError(ValueError):
    pass


TONAL_EXCLUDE = DRUM_ROLES | {"atmosphere"}
GROUPS = {
    "drums": lambda p: [t for t in p["tracks"] if t["role"] in DRUM_ROLES],
    "all": lambda p: list(p["tracks"]),
    "melodic": lambda p: [t for t in p["tracks"] if t["role"] not in DRUM_ROLES and t["role"] != "808"],
}


def _tracks(project: dict, ref) -> list[dict]:
    refs = ref if isinstance(ref, list) else [ref]
    out: list[dict] = []
    for r in refs:
        if isinstance(r, str) and r.lower() in GROUPS:
            out += GROUPS[r.lower()](project)
        else:
            try:
                out.append(track_by_ref(project, r))
            except ValueError as e:
                raise ActionError(str(e))
    seen, uniq = set(), []
    for t in out:
        if t["id"] not in seen:
            seen.add(t["id"]); uniq.append(t)
    return uniq


def _one(project: dict, ref) -> dict:
    ts = _tracks(project, ref)
    if len(ts) != 1:
        raise ActionError(f"expected one track for {ref!r}, got {len(ts)}")
    return ts[0]


def _num(a: dict, key: str, lo: float, hi: float, default=None, required=False) -> float | None:
    if key not in a or a[key] is None:
        if required:
            raise ActionError(f"missing {key!r}")
        return default
    try:
        v = float(a[key])
    except (TypeError, ValueError):
        raise ActionError(f"{key} must be a number")
    if not math.isfinite(v) or not (lo <= v <= hi):
        raise ActionError(f"{key}={v} must be between {lo} and {hi}")
    return v


def _character(ctx: dict) -> dict:
    c = ctx.get("character")
    if not c:
        raise ActionError("this edit needs the project's character profile")
    return c


# --- notes helpers ------------------------------------------------------------

def _flatten_midi(track: dict) -> list[dict]:
    from .project import track_notes_absolute
    return track_notes_absolute(track)


def _set_single_clip(track: dict, notes: list[dict], length: float) -> None:
    """Replace a MIDI track's clips with one clip holding absolute-time notes."""
    track["clips"] = [{"id": new_id("clip"), "start": 0.0, "length": float(length), "offset": 0.0,
                       "notes": sorted([dict(n) for n in notes], key=lambda n: (n["t"], n["p"]))}] if length > 0 else []


def _clip(track: dict, clip_id: str) -> dict:
    for c in track["clips"]:
        if c["id"] == clip_id:
            return c
    raise ActionError(f"no clip {clip_id!r} on {track['name']}")


def _section_ranges(project: dict, refs) -> list[tuple[float, float, dict]]:
    if refs is None:
        return []
    refs = refs if isinstance(refs, list) else [refs]
    out = []
    for r in refs:
        try:
            s = section_by_ref(project, r)
        except (ValueError, IndexError) as e:
            raise ActionError(str(e))
        lo, hi = section_beats(project, s)
        out.append((lo, hi, s))
    return out


# --- operations -----------------------------------------------------------------

def op_set_mixer(p, a, ctx):
    for t in _tracks(p, a.get("track")):
        if "gain_db" in a:
            t["gain_db"] = _num(a, "gain_db", -60, 24)
        if "delta_db" in a:
            t["gain_db"] = float(min(24.0, max(-60.0, t["gain_db"] + _num(a, "delta_db", -60, 60))))
        if "pan" in a:
            t["pan"] = _num(a, "pan", -1, 1)
        for k in ("mute", "solo"):
            if k in a:
                t[k] = bool(a[k])
        if "name" in a:
            name = str(a["name"]).strip()[:40]
            if not name:
                raise ActionError("track name cannot be empty")
            t["name"] = name
    return "mixer"


def _fx_target(p, a) -> list[dict]:
    if a.get("track") == "master":
        return [p["master"]]
    return _tracks(p, a.get("track"))


def op_add_effect(p, a, ctx):
    try:
        fx = dsp.validate_effect({"type": a.get("type"), "params": a.get("params") or {}})
    except ValueError as e:
        raise ActionError(str(e))
    for tgt in _fx_target(p, a):
        f = copy.deepcopy(fx)
        f["id"] = new_id("fx")
        pos = a.get("position")
        if pos is None:
            tgt.setdefault("fx", []).append(f)
        else:
            tgt.setdefault("fx", []).insert(int(pos), f)
    return "sound"


def _find_fx(tgt: dict, a: dict) -> int:
    chain = tgt.get("fx", [])
    if "effect_id" in a:
        for i, f in enumerate(chain):
            if f.get("id") == a["effect_id"]:
                return i
    if "index" in a:
        i = int(a["index"])
        if 0 <= i < len(chain):
            return i
    if "type" in a:
        for i in range(len(chain) - 1, -1, -1):
            if chain[i]["type"] == a["type"]:
                return i
    raise ActionError("effect not found")


def op_update_effect(p, a, ctx):
    for tgt in _fx_target(p, a):
        i = _find_fx(tgt, a)
        f = tgt["fx"][i]
        params = dict(f.get("params") or {})
        for k, v in (a.get("params") or {}).items():
            params[k] = v
        for k, dv in (a.get("delta") or {}).items():
            spec = dsp.EFFECTS[f["type"]]["params"].get(k)
            if spec is None:
                raise ActionError(f"{f['type']} has no numeric parameter {k}")
            lo, hi, default = spec
            params[k] = min(hi, max(lo, float(params.get(k, default)) + float(dv)))
        try:
            nf = dsp.validate_effect({"type": f["type"], "params": params, "enabled": a.get("enabled", f.get("enabled", True)), "id": f.get("id")})
        except ValueError as e:
            raise ActionError(str(e))
        tgt["fx"][i] = nf
    return "sound"


def op_remove_effect(p, a, ctx):
    for tgt in _fx_target(p, a):
        tgt["fx"].pop(_find_fx(tgt, a))
    return "sound"


def op_set_instrument(p, a, ctx):
    name = a.get("preset")
    if name not in synth.PRESETS:
        raise ActionError(f"unknown instrument {name!r}")
    for t in _tracks(p, a.get("track")):
        if t["kind"] != "midi":
            raise ActionError(f"{t['name']} is an audio track; changing its instrument needs note data (transcribe it first) or a generative provider")
        old_engine = (t.get("instrument") or {}).get("engine")
        new = synth.preset(name)
        drum_engines = {"kick", "snare", "clap", "hat", "perc"}
        if (old_engine in drum_engines) != (new["engine"] in drum_engines):
            raise ActionError(f"{synth.PRESETS[name]['label']} can't play the {t['name']} part (drum vs pitched instrument)")
        t["instrument"] = new
    return "sound"


def op_set_instrument_params(p, a, ctx):
    for t in _tracks(p, a.get("track")):
        inst = t.get("instrument")
        if not inst:
            raise ActionError(f"{t['name']} has no instrument")
        base = synth.PRESETS.get(inst.get("preset"), {}).get("params", {})
        allowed = set(base) | set(inst.get("params", {}))
        for k, v in (a.get("params") or {}).items():
            if k not in allowed:
                raise ActionError(f"{inst.get('preset')} has no parameter {k!r} (has {sorted(allowed)})")
            if isinstance(v, (int, float)):
                if not math.isfinite(float(v)) or abs(float(v)) > 1e5:
                    raise ActionError(f"bad value for {k}")
                inst["params"][k] = float(v)
            else:
                inst["params"][k] = v
        for k, dv in (a.get("delta") or {}).items():
            if k not in allowed:
                raise ActionError(f"{inst.get('preset')} has no parameter {k!r}")
            lim = a.get("limits", {}).get(k, [0.0, 1.0] if k == "drive" else [-1e5, 1e5])
            inst["params"][k] = float(min(lim[1], max(lim[0], float(inst["params"].get(k, base.get(k, 0.0))) + float(dv))))
    return "sound"


def op_regenerate_part(p, a, ctx):
    character = _character(ctx)
    ranges = _section_ranges(p, a.get("sections"))
    only = {s["id"] for _, _, s in ranges} if ranges else None
    for t in _tracks(p, a.get("track")):
        if t["kind"] != "midi" or (t.get("generator") or {}).get("engine") != "procedural":
            raise ActionError(f"{t['name']} wasn't made by the composer; regenerate it with a generative provider instead")
        seed = int(a["seed"]) if "seed" in a else composer._seed(t["generator"].get("role_seed", 0), "regen", ctx.get("revision", 0), t["id"])
        opts = {k: v for k, v in (a.get("options") or {}).items() if k in ("rate", "roll_density", "glide_prob")}
        if "rate" in opts and opts["rate"] not in (1.0, 0.5, 1 / 3, 0.25, 1 / 6, 0.125):
            raise ActionError("hat rate must be 1, 1/2, 1/3, 1/4, 1/6 or 1/8 of a beat")
        new_notes = composer.generate_role_notes(p, character, t["role"], seed, only_sections=only, **opts)
        if only:
            keep = [n for n in _flatten_midi(t) if not any(lo <= n["t"] < hi for lo, hi, _ in ranges)]
            new_notes = keep + new_notes
        _set_single_clip(t, new_notes, total_beats(p))
        if not only:
            t["generator"]["role_seed"] = seed
        if opts:
            t["generator"].setdefault("options", {}).update(opts)
    return "notes"


def op_add_part(p, a, ctx):
    character = _character(ctx)
    role = a.get("role")
    if role not in ROLES:
        raise ActionError(f"unknown role {role!r}")
    if any(t["role"] == role for t in p["tracks"]) and not a.get("allow_duplicate"):
        raise ActionError(f"there is already a {role} track")
    preset_name = a.get("preset")
    if preset_name is not None and preset_name not in synth.PRESETS:
        raise ActionError(f"unknown instrument {preset_name!r}")
    seed = int(a.get("seed", composer._seed(p["seed"], role, "added", ctx.get("revision", 0))))
    try:
        t = composer.build_track(p, character, role, seed, preset_name=preset_name)
    except ValueError as e:
        raise ActionError(str(e))
    if not t["clips"] or not any(c.get("notes") for c in t["clips"]):
        raise ActionError(f"the composer produced no {role} notes for this arrangement")
    p["tracks"].append(t)
    return "notes"


def op_remove_track(p, a, ctx):
    ids = {t["id"] for t in _tracks(p, a.get("track"))}
    p["tracks"] = [t for t in p["tracks"] if t["id"] not in ids]
    return "structure"


def _cut_audio_clips(track: dict, lo: float, hi: float) -> None:
    out = []
    for c in track["clips"]:
        cs, ce = c["start"], c["start"] + c["length"]
        if ce <= lo or cs >= hi:
            out.append(c); continue
        if cs < lo:
            left = dict(c, id=new_id("clip"), length=lo - cs)
            out.append(left)
        if ce > hi:
            right = dict(c, id=new_id("clip"), start=hi, offset=c.get("offset", 0.0) + (hi - cs), length=ce - hi)
            out.append(right)
    track["clips"] = out


def op_clear_section(p, a, ctx):
    ranges = _section_ranges(p, a.get("sections") or a.get("section"))
    if not ranges:
        raise ActionError("which section?")
    for t in _tracks(p, a.get("track")):
        if t["kind"] == "midi":
            notes = [n for n in _flatten_midi(t) if not any(lo <= n["t"] < hi for lo, hi, _ in ranges)]
            _set_single_clip(t, notes, total_beats(p))
        else:
            for lo, hi, _ in ranges:
                _cut_audio_clips(t, lo, hi)
    return "notes"


def op_transpose(p, a, ctx):
    st = int(_num(a, "semitones", -24, 24, required=True))
    targets = _tracks(p, a.get("track", "all"))
    whole_song = a.get("track", "all") == "all"
    for t in targets:
        if t["role"] in TONAL_EXCLUDE or t["kind"] != "midi":
            continue
        for c in t["clips"]:
            for n in c.get("notes", []):
                n["p"] = int(n["p"] + st)
                if not 0 <= n["p"] <= 127:
                    raise ActionError("transposition moves notes out of MIDI range")
    if whole_song:
        from .theory import NOTE_NAMES, pitch_class
        p["params"]["key"] = NOTE_NAMES[(pitch_class(p["params"]["key"]) + st) % 12]
        for h in p.get("harmony", []):
            h["root_pc"] = (h["root_pc"] + st) % 12
            h["pcs"] = [(x + st) % 12 for x in h["pcs"]]
        for t in targets:
            if t["role"] == "atmosphere":
                for c in t["clips"]:
                    for n in c.get("notes", []):
                        n["p"] = int(n["p"] + st)
    return "notes"


def op_set_tempo(p, a, ctx):
    bpm = _num(a, "bpm", 40, 240, required=True)
    for t in p["tracks"]:
        if t["kind"] == "audio":
            for c in t["clips"]:
                c["source"].setdefault("origin_bpm", p["params"]["bpm"])
    p["params"]["bpm"] = bpm
    return "tempo"


def op_quantize(p, a, ctx):
    grid = _num(a, "grid", 1 / 32, 4, default=0.25)
    strength = _num(a, "strength", 0, 1, default=1.0)
    for t in _tracks(p, a.get("track")):
        for c in t["clips"]:
            for n in c.get("notes", []):
                q = round(n["t"] / grid) * grid
                n["t"] = round(n["t"] + (q - n["t"]) * strength, 5)
    return "notes"


def op_set_notes(p, a, ctx):
    t = _one(p, a.get("track"))
    if t["kind"] != "midi":
        raise ActionError("only MIDI tracks have notes")
    notes = a.get("notes")
    if not isinstance(notes, list) or len(notes) > 20000:
        raise ActionError("notes must be a list (max 20000)")
    clean = []
    for n in notes:
        try:
            m = {"t": float(n["t"]), "d": float(n["d"]), "p": int(n["p"]), "v": float(n.get("v", 0.8))}
        except (KeyError, TypeError, ValueError):
            raise ActionError(f"bad note {n}")
        if n.get("slide"):
            m["slide"] = True
        clean.append(m)
    if a.get("clip"):
        _clip(t, a["clip"])["notes"] = clean
    else:
        _set_single_clip(t, clean, max(total_beats(p), max((n["t"] + n["d"] for n in clean), default=0)))
    return "notes"


def op_move_clip(p, a, ctx):
    t = _one(p, a.get("track"))
    c = _clip(t, a.get("clip"))
    c["start"] = _num(a, "start", 0, 1e5, required=True)
    return "arrangement"


def op_trim_clip(p, a, ctx):
    t = _one(p, a.get("track"))
    c = _clip(t, a.get("clip"))
    if "left" in a:  # move the left edge without moving the content
        left = _num(a, "left", 0, 1e5)
        delta = left - c["start"]
        if delta >= c["length"] or c.get("offset", 0) + delta < 0:
            raise ActionError("trim would empty the clip or go before its start")
        c["start"], c["offset"], c["length"] = left, c.get("offset", 0) + delta, c["length"] - delta
    if "length" in a:
        c["length"] = _num(a, "length", 1 / 64, 1e5)
    return "arrangement"


def op_split_clip(p, a, ctx):
    t = _one(p, a.get("track"))
    c = _clip(t, a.get("clip"))
    at = _num(a, "at", 0, 1e5, required=True)
    if not (c["start"] < at < c["start"] + c["length"]):
        raise ActionError("split point must be inside the clip")
    left_len = at - c["start"]
    right = copy.deepcopy(c)
    right.update(id=new_id("clip"), start=at, offset=c.get("offset", 0) + left_len, length=c["length"] - left_len)
    c["length"] = left_len
    t["clips"].insert(t["clips"].index(c) + 1, right)
    return "arrangement"


def op_duplicate_clip(p, a, ctx):
    t = _one(p, a.get("track"))
    c = _clip(t, a.get("clip"))
    dup = copy.deepcopy(c)
    dup["id"] = new_id("clip")
    dup["start"] = _num(a, "start", 0, 1e5, default=c["start"] + c["length"])
    t["clips"].append(dup)
    return "arrangement"


def op_delete_clip(p, a, ctx):
    t = _one(p, a.get("track"))
    c = _clip(t, a.get("clip"))
    t["clips"].remove(c)
    return "arrangement"


def _shift_timeline(p: dict, at: float, delta: float) -> None:
    """Shift everything at/after ``at`` beats by ``delta`` beats (MIDI notes and audio clips)."""
    for t in p["tracks"]:
        if t["kind"] == "midi":
            notes = []
            for n in _flatten_midi(t):
                if n["t"] >= at:
                    n = dict(n, t=n["t"] + delta)
                notes.append(n)
            _set_single_clip(t, [n for n in notes if n["t"] >= 0], total_beats(p))
        else:
            if delta > 0:
                _split_audio_at(t, at)
            for c in t["clips"]:
                if c["start"] >= at:
                    c["start"] += delta


def _split_audio_at(t: dict, at: float) -> None:
    for c in list(t["clips"]):
        if c["start"] < at < c["start"] + c["length"]:
            left_len = at - c["start"]
            right = dict(c, id=new_id("clip"), start=at, offset=c.get("offset", 0) + left_len, length=c["length"] - left_len)
            c["length"] = left_len
            t["clips"].insert(t["clips"].index(c) + 1, right)


def op_duplicate_section(p, a, ctx):
    (lo, hi, s), = _section_ranges(p, a.get("section"))
    length = hi - lo
    bpb = beats_per_bar(p)
    copies = {}
    for t in p["tracks"]:
        if t["kind"] == "midi":
            copies[t["id"]] = [dict(n, t=n["t"] + length) for n in _flatten_midi(t) if lo <= n["t"] < hi]
        else:
            _split_audio_at(t, lo); _split_audio_at(t, hi)
            copies[t["id"]] = [dict(c, id=new_id("clip"), start=c["start"] + length) for c in t["clips"] if lo <= c["start"] < hi]
    # grow sections/harmony first so total_beats includes the new bars
    idx = p["sections"].index(s)
    for later in p["sections"][idx + 1:]:
        later["start_bar"] += s["bars"]
    new_sec = dict(s, id=new_id("sec"), start_bar=s["start_bar"] + s["bars"])
    p["sections"].insert(idx + 1, new_sec)
    label_sections(p["sections"])
    harm = {h["bar"]: h for h in p.get("harmony", [])}
    new_h = []
    for h in p.get("harmony", []):
        if h["bar"] >= s["start_bar"] + s["bars"]:
            new_h.append(dict(h, bar=h["bar"] + s["bars"]))
        else:
            new_h.append(h)
    for b in range(s["bars"]):
        if s["start_bar"] + b in harm:
            new_h.append(dict(harm[s["start_bar"] + b], bar=s["start_bar"] + s["bars"] + b))
    p["harmony"] = sorted(new_h, key=lambda h: h["bar"])
    _shift_timeline(p, hi, length)
    for t in p["tracks"]:
        if t["kind"] == "midi":
            _set_single_clip(t, _flatten_midi(t) + copies[t["id"]], total_beats(p))
        else:
            t["clips"].extend(copies[t["id"]])
    _ = bpb
    return "arrangement"


def op_delete_section(p, a, ctx):
    (lo, hi, s), = _section_ranges(p, a.get("section"))
    if len(p["sections"]) <= 1:
        raise ActionError("can't delete the only section")
    length = hi - lo
    for t in p["tracks"]:
        if t["kind"] == "midi":
            _set_single_clip(t, [n for n in _flatten_midi(t) if not (lo <= n["t"] < hi)], total_beats(p))
        else:
            _cut_audio_clips(t, lo, hi)
    idx = p["sections"].index(s)
    p["sections"].pop(idx)
    for later in p["sections"][idx:]:
        later["start_bar"] -= s["bars"]
    label_sections(p["sections"])
    end_bar = s["start_bar"] + s["bars"]
    p["harmony"] = [dict(h, bar=h["bar"] - s["bars"]) if h["bar"] >= end_bar else h
                    for h in p.get("harmony", []) if not (s["start_bar"] <= h["bar"] < end_bar)]
    _shift_timeline(p, hi, -length)
    return "arrangement"


def op_add_audio_track(p, a, ctx):
    role = a.get("role", "other")
    if role not in ROLES:
        raise ActionError(f"unknown role {role!r}")
    uri = str(a.get("uri", ""))
    if not uri or uri.startswith(("/", "\\")) or ".." in uri.replace("\\", "/").split("/"):
        raise ActionError("bad audio reference (must be a file in the project's assets)")
    length = _num(a, "length", 1 / 64, 1e5, required=True)
    t = make_track(a.get("name") or role.title(), role, None, kind="audio", length=length,
                   gain_db=float(a.get("gain_db", 0.0)), color=a.get("color") or "#9fa6bb",
                   generator=a.get("generator") or {"engine": "upload"})
    t["clips"][0]["start"] = _num(a, "start", 0, 1e5, default=0.0)
    t["clips"][0]["source"] = {"uri": uri, "origin_bpm": float(a.get("origin_bpm") or p["params"]["bpm"])}
    p["tracks"].append(t)
    return "structure"


def op_set_master(p, a, ctx):
    if "gain_db" in a:
        p["master"]["gain_db"] = _num(a, "gain_db", -24, 24)
    return "mixer"


OPS: dict[str, Callable[[dict, dict, dict], str]] = {
    "set_mixer": op_set_mixer, "add_effect": op_add_effect, "update_effect": op_update_effect, "remove_effect": op_remove_effect,
    "set_instrument": op_set_instrument, "set_instrument_params": op_set_instrument_params,
    "regenerate_part": op_regenerate_part, "add_part": op_add_part, "remove_track": op_remove_track,
    "clear_section": op_clear_section, "transpose": op_transpose, "set_tempo": op_set_tempo, "quantize": op_quantize,
    "set_notes": op_set_notes, "move_clip": op_move_clip, "trim_clip": op_trim_clip, "split_clip": op_split_clip,
    "duplicate_clip": op_duplicate_clip, "delete_clip": op_delete_clip, "duplicate_section": op_duplicate_section,
    "delete_section": op_delete_section, "add_audio_track": op_add_audio_track, "set_master": op_set_master,
}


def diff_tracks(before: dict, after: dict) -> dict[str, Any]:
    """Which tracks will sound different, which only changed mixer settings, which are untouched."""
    b = {t["id"]: t for t in before["tracks"]}
    a = {t["id"]: t for t in after["tracks"]}
    changed, mixer, same = [], [], []
    for tid, t in a.items():
        if tid not in b:
            changed.append({"id": tid, "name": t["name"], "change": "added"})
            continue
        if stem_key(before, b[tid]) != stem_key(after, t):
            changed.append({"id": tid, "name": t["name"], "change": "sound"})
        elif any(b[tid].get(k) != t.get(k) for k in ("gain_db", "pan", "mute", "solo", "name")):
            mixer.append({"id": tid, "name": t["name"]})
        else:
            same.append({"id": tid, "name": t["name"]})
    removed = [{"id": tid, "name": t["name"]} for tid, t in b.items() if tid not in a]
    master = before.get("master") != after.get("master")
    return {"sound_changed": changed, "mixer_changed": mixer, "untouched": same, "removed": removed, "master_changed": master}


def apply_actions(project: dict, actions: list[dict], character: dict | None = None, revision: int = 0) -> tuple[dict, dict]:
    if not isinstance(actions, list) or not actions:
        raise ActionError("no actions")
    if len(actions) > 50:
        raise ActionError("too many actions in one edit")
    p = clone(project)
    ctx = {"character": character, "revision": revision}
    kinds = []
    for i, a in enumerate(actions):
        op = OPS.get(a.get("op"))
        if op is None:
            raise ActionError(f"unknown operation {a.get('op')!r}")
        try:
            kinds.append(op(p, a, ctx))
        except ActionError as e:
            raise ActionError(f"step {i + 1} ({a.get('op')}): {e}")
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise ActionError(f"step {i + 1} ({a.get('op')}): invalid request ({e})")
    try:
        validate_project(p)
    except ValueError as e:
        raise ActionError(f"edit rejected, project would be invalid: {e}")
    from .project import now_iso
    p["updated_at"] = now_iso()
    return p, {"kinds": kinds, "diff": diff_tracks(project, p)}
