"""Structured song generation (Architecture B).

The song is generated as coordinated musical components, in order:

1. arrangement (sections) scaled to the requested duration
2. harmony (one diatonic chord per bar, per-section progressions)
3. parts, each as note data: kick → snare/clap → hats → perc → 808 (follows
   the kick) → chords → melody (motif-based, follows the chords) → extras
4. each part becomes its own track with its own instrument, fx and seed

Every part is reproducible from ``(project harmony, role, seed)``, so a single
part can be regenerated later without touching any other part.

This engine is procedural (rules + seeded randomness), not a neural model.
It is the *structure* layer; neural providers (ACE-Step, Stable Audio) plug in
through :mod:`studio.providers` to render or replace layers with generated audio.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from . import synth
from .project import (SCHEMA, label_sections, make_track, new_id, now_iso, total_beats,
                      track_by_ref, track_notes_absolute)
from .theory import NOTE_NAMES, diatonic_chord, key_label, pitch_class, scale_pitches

COMPOSER_VERSION = 1
STEPS = 16  # 16th-note grid in 4/4
STEP = 0.25  # beats

ROLE_NAMES = {"kick": "Kick", "snare": "Snare", "clap": "Clap", "hat": "Hi-Hats", "perc": "Percussion", "808": "808",
              "melody": "Melody", "chords": "Chords", "counter_melody": "Counter Melody", "bell": "Bells",
              "atmosphere": "Atmosphere", "pad": "Pad", "lead": "Lead", "piano": "Piano"}
ROLE_COLORS = {"kick": "#ff5d73", "snare": "#ff9f43", "clap": "#ffc048", "hat": "#feca57", "perc": "#e0a96d", "808": "#7275ff",
               "melody": "#3ee0c5", "chords": "#4aa3ff", "counter_melody": "#7cf29a", "bell": "#c98bff", "atmosphere": "#8a9099",
               "pad": "#4aa3ff", "lead": "#3ee0c5", "piano": "#f3d9a4"}

# which roles play in which section type (others are silent there)
SECTION_ROLES = {
    "intro": {"melody", "chords", "atmosphere", "bell"},
    "verse": {"kick", "snare", "clap", "hat", "perc", "808", "melody", "chords", "atmosphere", "counter_melody"},
    "hook": {"kick", "snare", "clap", "hat", "perc", "808", "melody", "chords", "atmosphere", "bell", "counter_melody"},
    "bridge": {"hat", "melody", "chords", "atmosphere", "bell"},
    "outro": {"melody", "chords", "atmosphere"},
}

KICK_PATTERNS = {
    "rage": ["x.....x...x.....", "x..x......x..x..", "x.....x.x.......", "x...x.....x.x..."],
    "trap": ["x.......x.x.....", "x..x......x.....", "x.....x...x....x"],
    "plugg": ["x.......x.x.....", "x..x......x.....", "x.....x...x....x", "x..x....x......."],
    "sparse": ["x...........x...", "x.......x.......", "x..........x...."],
    "boombap": ["x.........x.....", "x.x.......x..x..", "x......x..x....."],
    "jersey": ["x..x..x.x...x...", "x..x..x.x.x.x.x."],
}

UNSUPPORTED_WORDS = ["guitar", "violin", "strings", "cello", "flute", "sax", "saxophone", "trumpet", "brass", "choir", "orchestra", "harp", "organ"]
PROMPT_INSTRUMENTS = [
    (("music box",), "melody", "music_box"), (("electric piano", "rhodes", "keys"), "melody", "rhodes"),
    (("felt piano",), "melody", "piano_felt"), (("piano",), "melody", "piano"), (("bell",), "melody", "bell_bright"),
    (("pluck",), "melody", "pluck_soft"), (("dark synth",), "melody", "dark_synth"), (("supersaw", "synth lead", "rage synth"), "melody", "rage_lead"),
    (("blown", "distorted 808", "hard 808", "clipped 808"), "808", "808_blown"), (("clean 808", "smooth 808", "soft 808"), "808", "808_clean"),
]


def _seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def interpret_prompt(prompt: str) -> dict:
    """Deterministic keyword reading of the user's prompt. Everything applied is reported back."""
    p = (prompt or "").lower()
    hints: dict[str, Any] = {"applied": [], "unsupported": [], "instruments": {}}
    if any(w in p for w in ("dark", "evil", "sinister", "haunt", "sad", "eerie", "menac", "cold")):
        hints["mood"] = "dark"; hints["applied"].append("dark mood → minor-family scale")
    elif any(w in p for w in ("happy", "bright", "uplift", "sweet", "dream", "warm", "love")):
        hints["mood"] = "bright"; hints["applied"].append("bright mood → major-family scale")
    if any(w in p for w in ("fast", "hype", "energetic", "aggressive", "crazy", "rage")):
        hints["tempo_bias"] = 1.0; hints["energy_bias"] = 0.15; hints["applied"].append("high energy → upper tempo range")
    elif any(w in p for w in ("slow", "chill", "calm", "relax", "mellow", "lofi", "lo-fi")):
        hints["tempo_bias"] = -1.0; hints["energy_bias"] = -0.15; hints["applied"].append("calm → lower tempo range")
    for words, role, preset_name in PROMPT_INSTRUMENTS:
        if role not in hints["instruments"] and any(w in p for w in words):
            hints["instruments"][role] = preset_name
            hints["applied"].append(f"'{words[0]}' → {role} uses {synth.PRESETS[preset_name]['label']}")
    for w in UNSUPPORTED_WORDS:
        if w in p:
            hints["unsupported"].append(f"'{w}' has no instrument in the procedural engine yet (needs a generative provider)")
    if any(w in p for w in ("no drums", "without drums", "drumless")):
        hints["no_drums"] = True; hints["applied"].append("no drums")
    return hints


def _choose_key(character: dict, params: dict, hints: dict, rng) -> tuple[str, str]:
    scales = character["keys"]["preferred_scales"]
    if params.get("scale"):
        scale = params["scale"]
    elif hints.get("mood") == "dark":
        dark = [s for s in scales if s in ("minor", "phrygian", "harmonic_minor", "dorian", "locrian")] or ["minor"]
        scale = dark[int(rng.integers(len(dark)))]
    elif hints.get("mood") == "bright":
        bright = [s for s in scales if s in ("major", "lydian", "mixolydian")] or ["major"]
        scale = bright[int(rng.integers(len(bright)))]
    else:
        scale = scales[0]
    if params.get("key"):
        key = NOTE_NAMES[pitch_class(params["key"])]
    else:
        roots = character["keys"].get("preferred_roots") or NOTE_NAMES
        key = roots[int(rng.integers(len(roots)))]
    return key, scale


def _choose_bpm(character: dict, params: dict, hints: dict) -> float:
    if params.get("bpm"):
        return float(params["bpm"])
    t = character["tempo"]
    bias = hints.get("tempo_bias", 0.0)
    if bias > 0:
        return float(round(t["default"] + (t["max"] - t["default"]) * 0.6))
    if bias < 0:
        return float(round(t["default"] - (t["default"] - t["min"]) * 0.6))
    return float(t["default"])


def plan_sections(character: dict, bpm: float, duration_s: float, template: list[str] | None = None) -> list[dict]:
    items = [(s.split(":")[0], int(s.split(":")[1])) for s in (template or character["arrangement"]["template"])]
    target = max(8, round(duration_s * bpm / 240.0))
    total = sum(b for _, b in items)
    if target > total * 1.25:
        # longer songs: repeat the middle (verse, hook) instead of stretching sections
        middle = [it for it in items if it[0] in ("verse", "hook")][-2:] or items[1:3]
        out = items[:-1]
        while sum(b for _, b in out) + items[-1][1] + sum(b for _, b in middle) <= target:
            out += middle
        items = out + [items[-1]]
    elif target < total:
        f = target / total
        items = [(n, max(2, int(round(b * f / 2.0)) * 2)) for n, b in items]
    sections, bar = [], 0
    for name, bars in items:
        sections.append({"id": new_id("sec"), "name": name, "start_bar": bar, "bars": bars,
                         "energy": {"intro": 0.35, "verse": 0.7, "hook": 1.0, "bridge": 0.5, "outro": 0.3}.get(name, 0.6)})
        bar += bars
    label_sections(sections)
    return sections


def plan_harmony(character: dict, sections: list[dict], key: str, scale: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(_seed(seed, "harmony"))
    progs = character["harmony"]["progressions"]
    size = int(character["harmony"].get("chord_size", 3))
    by_type: dict[str, list[int]] = {}
    order = list(rng.permutation(len(progs)))
    for i, name in enumerate(["hook", "verse", "intro", "bridge", "outro"]):
        by_type[name] = progs[order[i % len(progs)]] if name not in ("intro", "outro") else None
    by_type["intro"] = by_type["hook"]
    by_type["outro"] = by_type["hook"]
    root = pitch_class(key)
    harmony = []
    for s in sections:
        prog = by_type.get(s["name"]) or by_type["hook"]
        for b in range(s["bars"]):
            deg = int(prog[b % len(prog)])
            pcs = diatonic_chord(root, scale, deg, size)
            harmony.append({"bar": s["start_bar"] + b, "degree": deg, "root_pc": pcs[0] % 12, "pcs": [p % 12 for p in pcs],
                            "chord_offsets": [p - pcs[0] for p in pcs]})
    return harmony


# ---------------------------------------------------------------------------
# helpers shared by part generators
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self, project: dict, character: dict):
        self.p = project
        self.c = character
        self.sections = project["sections"]
        self.harmony = {h["bar"]: h for h in project["harmony"]}
        self.key_pc = pitch_class(project["params"]["key"])
        self.scale = project["params"]["scale"]
        self.energy = float(project["params"].get("energy", 0.7))
        self.swing = float(character.get("swing", 0.0))
        self.feel = character["tempo"].get("feel", "half")
        self.hum = character.get("humanize", {})
        self.bpm = float(project["params"]["bpm"])
        self.total_bars = int(total_beats(project) // 4)

    def section_of_bar(self, bar: int) -> dict:
        for s in self.sections:
            if s["start_bar"] <= bar < s["start_bar"] + s["bars"]:
                return s
        return self.sections[-1]

    def plays(self, role: str, bar: int, only: set[str] | None) -> bool:
        s = self.section_of_bar(bar)
        if only is not None and s["id"] not in only:
            return False
        return role in SECTION_ROLES.get(s["name"], SECTION_ROLES["verse"])

    def step_time(self, bar: int, step: float) -> float:
        t = bar * 4 + step * STEP
        if self.swing and int(step) == step and int(step) % 2 == 1:
            t += self.swing * STEP * 2  # delay odd 16ths
        return t

    def humanize(self, notes: list[dict], rng) -> list[dict]:
        tm = self.hum.get("timing_ms", 0) / 1000.0 * self.bpm / 60.0
        vel = self.hum.get("velocity", 0)
        for n in notes:
            if tm:
                n["t"] = max(0.0, n["t"] + float(rng.normal(0, tm / 2)))
            if vel:
                n["v"] = float(np.clip(n["v"] + rng.normal(0, vel / 2), 0.05, 1.0))
        return notes


def _r(x: float) -> float:
    return round(float(x), 5)


def _finish(notes: list[dict]) -> list[dict]:
    for n in notes:
        n["t"], n["d"], n["v"] = _r(n["t"]), _r(max(n["d"], 1 / 64)), _r(n["v"])
    return sorted(notes, key=lambda n: (n["t"], n["p"]))


def _last_bar_before_hook(ctx: _Ctx, bar: int) -> bool:
    s = ctx.section_of_bar(bar)
    if bar != s["start_bar"] + s["bars"] - 1:
        return False
    idx = ctx.sections.index(s)
    return idx + 1 < len(ctx.sections) and ctx.sections[idx + 1]["name"] == "hook"


# ---------------------------------------------------------------------------
# drums
# ---------------------------------------------------------------------------

def gen_kick(ctx: _Ctx, seed: int, only=None) -> list[dict]:
    spec = ctx.c["drums"].get("kick") or {}
    pats = KICK_PATTERNS.get(spec.get("pattern", "trap"), KICK_PATTERNS["trap"])
    rng = np.random.default_rng(seed)
    base = [pats[int(rng.integers(len(pats)))] for _ in range(2)]  # an A/B pair keeps the groove recognisable
    notes = []
    for bar in range(ctx.total_bars):
        if not ctx.plays("kick", bar, only):
            continue
        pat = base[bar % 2] if (bar % 4 != 3 or rng.random() > 0.5) else pats[int(rng.integers(len(pats)))]
        for step, ch in enumerate(pat):
            if ch != "x":
                continue
            if _last_bar_before_hook(ctx, bar) and step >= 8:
                continue  # classic drop-out before the hook
            notes.append({"t": ctx.step_time(bar, step), "d": 0.25, "p": 36, "v": 0.95 if step == 0 else 0.85})
    return _finish(ctx.humanize(notes, rng))


def _backbeat_steps(ctx: _Ctx) -> list[int]:
    return [8] if ctx.feel == "half" else [4, 12]


def gen_snare(ctx: _Ctx, seed: int, only=None, role: str = "snare") -> list[dict]:
    rng = np.random.default_rng(seed)
    notes = []
    for bar in range(ctx.total_bars):
        if not ctx.plays(role, bar, only):
            continue
        s = ctx.section_of_bar(bar)
        for st in _backbeat_steps(ctx):
            notes.append({"t": ctx.step_time(bar, st), "d": 0.25, "p": 38, "v": 0.9})
        is_phrase_end = (bar - s["start_bar"]) % 4 == 3
        if is_phrase_end and s["name"] == "hook" and rng.random() < 0.6 * ctx.energy:
            for st in (14, 15) if rng.random() < 0.5 else (13, 15):
                notes.append({"t": ctx.step_time(bar, st), "d": 0.25, "p": 38, "v": 0.55})
    return _finish(ctx.humanize(notes, rng))


def gen_hat(ctx: _Ctx, seed: int, only=None, rate: float | None = None, roll_density: float | None = None) -> list[dict]:
    spec = ctx.c["drums"].get("hat") or {}
    rng = np.random.default_rng(seed)
    rate = float(rate if rate is not None else spec.get("rate", 0.5))
    rd = float(roll_density if roll_density is not None else spec.get("roll_density", 0.2))
    triplets = bool(spec.get("triplet_rolls", False))
    open_prob = float(spec.get("open_prob", 0.0))
    notes = []
    for bar in range(ctx.total_bars):
        if not ctx.plays("hat", bar, only):
            continue
        s = ctx.section_of_bar(bar)
        dens = rd * (0.6 if s["name"] == "verse" else 1.0) * (0.6 + 0.8 * ctx.energy)
        for beat in range(4):
            t0 = bar * 4 + beat
            roll = rng.random() < dens and (beat in (1, 3) or rng.random() < 0.3)
            if roll:
                kind = rng.choice(["32", "16t", "8t"] if triplets else ["32", "16"])
                div = {"32": 0.125, "16t": 1 / 6, "8t": 1 / 3, "16": 0.25}[kind]
                k = int(round(1 / div))
                for j in range(k):
                    notes.append({"t": t0 + j * div, "d": div, "p": 42, "v": 0.45 + 0.4 * (j / max(k - 1, 1))})
                continue
            k = max(1, int(round(1 / rate)))
            for j in range(k):
                t = t0 + j * rate
                step = int(round((t - bar * 4) / STEP))
                if ctx.swing and step % 2 == 1:
                    t += ctx.swing * STEP * 2
                accent = 0.82 if j == 0 else 0.62
                is_open = open_prob > 0 and j == k - 1 and beat in (1, 3) and rng.random() < open_prob * 4
                notes.append({"t": t, "d": rate, "p": 46 if is_open else 42, "v": accent})
    return _finish(ctx.humanize(notes, rng))


def gen_perc(ctx: _Ctx, seed: int, only=None) -> list[dict]:
    spec = ctx.c["drums"].get("perc") or {}
    rng = np.random.default_rng(seed)
    dens = float(spec.get("density", 0.3))
    shaker = (spec.get("preset") == "perc_shaker")
    motif = [st for st in range(STEPS) if (rng.random() < dens and st % 4 != 0)] or [6]
    notes = []
    for bar in range(ctx.total_bars):
        if not ctx.plays("perc", bar, only):
            continue
        steps = range(0, STEPS, 2) if shaker else motif
        for st in steps:
            v = (0.7 if st % 4 == 2 else 0.45) if shaker else 0.7
            notes.append({"t": ctx.step_time(bar, st), "d": 0.25, "p": 60, "v": v})
    return _finish(ctx.humanize(notes, rng))


# ---------------------------------------------------------------------------
# 808 / bass
# ---------------------------------------------------------------------------

def _bass_pitch(root_pc: int) -> int:
    return 28 + ((root_pc - 4) % 12)  # E1..D#2: 41–78 Hz fundamentals


def gen_808(ctx: _Ctx, seed: int, only=None, kick_notes: list[dict] | None = None, glide_prob: float | None = None) -> list[dict]:
    spec = ctx.c["bass"]
    rng = np.random.default_rng(seed)
    gp = float(glide_prob if glide_prob is not None else spec.get("glide_prob", 0.3))
    notes = []
    if spec.get("style", "follow_kick") == "follow_kick" and kick_notes:
        hits = sorted(n["t"] for n in kick_notes)
        for i, t in enumerate(hits):
            bar = int(t // 4)
            if not ctx.plays("808", bar, only):
                continue
            h = ctx.harmony.get(bar)
            if h is None:
                continue
            nxt = hits[i + 1] if i + 1 < len(hits) else t + 2.0
            length = max(0.25, min(nxt - t, 3.0))
            p = _bass_pitch(h["root_pc"])
            notes.append({"t": t, "d": length * 0.97, "p": p, "v": 0.95})
            if length >= 1.0 and rng.random() < gp:
                # 808 slide: jump up an octave (or a fifth) and glide back down
                up = p + (12 if rng.random() < 0.6 else 7)
                ts = t + length * 0.55
                notes[-1]["d"] = (ts - t) * 0.99
                notes.append({"t": ts, "d": length * 0.2, "p": up, "v": 0.9, "slide": True})
                notes.append({"t": ts + length * 0.2, "d": length * 0.25 * 0.97, "p": p, "v": 0.85, "slide": True})
    else:
        for bar in range(ctx.total_bars):
            if not ctx.plays("808", bar, only):
                continue
            h = ctx.harmony[bar]
            p = _bass_pitch(h["root_pc"])
            if _last_bar_before_hook(ctx, bar):
                notes.append({"t": bar * 4, "d": 1.9, "p": p, "v": 0.95})
                continue
            notes.append({"t": bar * 4, "d": 2.9, "p": p, "v": 0.95})
            if rng.random() < 0.5 + gp:
                tgt = p + (12 if rng.random() < gp else 0)
                notes.append({"t": bar * 4 + 3, "d": 0.95, "p": tgt, "v": 0.85, "slide": tgt != p})
    return _finish(notes)


# ---------------------------------------------------------------------------
# harmony + melody
# ---------------------------------------------------------------------------

def _voice(pcs: list[int], lo: int, hi: int, prev: list[int] | None) -> list[int]:
    cands = []
    for inv in range(len(pcs)):
        order = pcs[inv:] + pcs[:inv]
        base = lo + ((order[0] - lo) % 12)
        voicing, last = [], None
        for pc in order:
            m = base if last is None else last + ((pc - last) % 12 or 12)
            voicing.append(m)
            last = m
        if voicing[-1] <= hi + 3:
            cands.append(voicing)
    if not cands:
        cands = [[lo + ((pc - lo) % 12) for pc in pcs]]
    if prev:
        return min(cands, key=lambda v: sum(abs(a - b) for a, b in zip(sorted(v), sorted(prev))))
    return cands[0]


def gen_chords(ctx: _Ctx, seed: int, only=None, role: str = "chords") -> list[dict]:
    spec = ctx.c["harmony"]
    lo, hi = spec.get("register", [52, 68])
    rng = np.random.default_rng(seed)
    rhythm = spec.get("rhythm", "sustain")
    stab_pat = [[0, 6, 10], [0, 3, 8, 14], [0, 8, 11]][int(rng.integers(3))]
    notes, prev = [], None
    for bar in range(ctx.total_bars):
        if not ctx.plays(role, bar, only):
            continue
        h = ctx.harmony[bar]
        voicing = _voice(h["pcs"], lo, hi, prev)
        prev = voicing
        if rhythm == "stabs":
            for st in stab_pat:
                for m in voicing:
                    notes.append({"t": ctx.step_time(bar, st), "d": 0.9 if st == stab_pat[-1] else min(0.75, (stab_pat[(stab_pat.index(st) + 1) % len(stab_pat)] - st) * STEP * 0.9), "p": m, "v": 0.7})
        else:
            for m in voicing:
                notes.append({"t": bar * 4, "d": 3.96, "p": m, "v": 0.6})
    return _finish(ctx.humanize(notes, rng))


def _motif(rng, bars: int, density: float, step_bias: float, n_scale: int) -> list[tuple[int, int, int]]:
    """Return [(step, length_steps, scale_index_offset)] for ``bars`` bars."""
    total = bars * STEPS
    weights = np.array([1.0 if s % 8 == 0 else 0.75 if s % 4 == 0 else 0.5 if s % 2 == 0 else 0.25 for s in range(total)])
    on = rng.random(total) < np.clip(weights * density * 1.25, 0, 0.95)
    on[0] = on[0] or rng.random() < 0.7
    if not on.any():
        on[0] = True
    steps = [s for s in range(total) if on[s]]
    out, idx = [], 0
    for i, s in enumerate(steps):
        nxt = steps[i + 1] if i + 1 < len(steps) else total
        length = int(min(nxt - s, 6 if rng.random() < 0.5 else 3))
        out.append((s, max(1, length), idx))
        move = int(rng.choice([-1, 1])) if rng.random() < step_bias else int(rng.choice([-4, -3, -2, 2, 3, 4]))
        idx = int(np.clip(idx + move, -5, 6))
    return out


def gen_melody(ctx: _Ctx, seed: int, only=None, role: str = "melody", register: list[int] | None = None,
               density_scale: float = 1.0, avoid: list[dict] | None = None) -> list[dict]:
    spec = ctx.c["melody"]
    rng = np.random.default_rng(seed)
    lo, hi = register or spec.get("register", [70, 86])
    pool = scale_pitches(ctx.key_pc, ctx.scale, lo, hi)
    centre = len(pool) // 2
    mb = int(spec.get("motif_bars", 1))
    dens = float(spec.get("density", 0.6)) * (0.75 + 0.5 * ctx.energy) * density_scale
    rep = float(spec.get("repetition", 0.8))
    motifs = {name: _motif(np.random.default_rng(_seed(seed, name)), mb, dens * (0.7 if name == "verse" else 1.0),
                           float(spec.get("step_bias", 0.6)), len(pool))
              for name in ("hook", "verse", "bridge")}
    motifs["intro"] = motifs["outro"] = motifs["hook"]
    avoid_steps = set()
    for n in avoid or []:
        avoid_steps.add(int(round(n["t"] / STEP)))
    notes = []
    for bar in range(ctx.total_bars):
        if not ctx.plays(role, bar, only):
            continue
        s = ctx.section_of_bar(bar)
        rel = bar - s["start_bar"]
        motif = motifs.get(s["name"], motifs["hook"])
        chord_pcs = set(ctx.harmony[bar]["pcs"])
        phrase_pos = rel % 4
        vary = phrase_pos == 3 and rng.random() > rep * 0.5
        for st, ln, off in motif:
            if st // STEPS != rel % mb:
                continue
            step = st % STEPS
            if s["name"] in ("intro", "outro") and step % 4 != 0 and rng.random() < 0.5:
                continue  # thinner statement of the hook in intro/outro
            if vary and rng.random() < 0.5:
                off += int(rng.choice([-2, -1, 1, 2]))
            gi = int(np.clip(centre + off, 0, len(pool) - 1))
            m = pool[gi]
            if step % 4 == 0 and m % 12 not in chord_pcs:  # strong beats land on chord tones
                for d in (1, -1, 2, -2):
                    j = gi + d
                    if 0 <= j < len(pool) and pool[j] % 12 in chord_pcs:
                        m = pool[j]
                        break
            abs_step = bar * STEPS + step
            if abs_step in avoid_steps:
                continue
            notes.append({"t": ctx.step_time(bar, step), "d": ln * STEP * 0.92, "p": int(m), "v": 0.8 if step % 4 == 0 else 0.68})
    return _finish(ctx.humanize(notes, rng))


def gen_counter_melody(ctx: _Ctx, seed: int, only=None, melody_notes: list[dict] | None = None) -> list[dict]:
    lo, hi = ctx.c["melody"].get("register", [70, 86])
    rng = np.random.default_rng(seed)
    notes = []
    melody_steps = {int(round(n["t"] / STEP)) for n in (melody_notes or [])}
    for bar in range(ctx.total_bars):
        if not ctx.plays("counter_melody", bar, only):
            continue
        h = ctx.harmony[bar]
        tones = sorted({(lo - 12) + ((pc - (lo - 12)) % 12) for pc in h["pcs"]})
        for st in (2, 6, 10, 14):
            if bar * STEPS + st in melody_steps or rng.random() < 0.35:
                continue
            notes.append({"t": ctx.step_time(bar, st), "d": STEP * 3.5, "p": int(tones[int(rng.integers(len(tones)))]), "v": 0.6})
    return _finish(notes)


def gen_bell(ctx: _Ctx, seed: int, only=None) -> list[dict]:
    lo, hi = 76, 96
    rng = np.random.default_rng(seed)
    order = [0, 1, 2, 1] if rng.random() < 0.5 else [0, 2, 1, 2]
    notes = []
    for bar in range(ctx.total_bars):
        if not ctx.plays("bell", bar, only):
            continue
        s = ctx.section_of_bar(bar)
        if s["name"] != "hook" and rng.random() < 0.5:
            continue
        h = ctx.harmony[bar]
        tones = sorted({lo + ((pc - lo) % 12) for pc in h["pcs"]})
        for k in range(8):
            notes.append({"t": bar * 4 + k * 0.5, "d": 0.45, "p": int(tones[order[k % 4] % len(tones)]), "v": 0.55})
    return _finish(notes)


def gen_atmosphere(ctx: _Ctx, seed: int, only=None) -> list[dict]:
    notes = []
    for s in ctx.sections:
        if only is not None and s["id"] not in only:
            continue
        if "atmosphere" not in SECTION_ROLES.get(s["name"], set()):
            continue
        root = ctx.harmony[s["start_bar"]]["root_pc"]
        notes.append({"t": s["start_bar"] * 4, "d": s["bars"] * 4 - 0.5, "p": 72 + root, "v": 0.6})
    return _finish(notes)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def role_preset(character: dict, role: str) -> str | None:
    if role in ("kick", "snare", "clap", "hat", "perc"):
        spec = character["drums"].get(role)
        return spec["preset"] if spec else None
    if role == "808":
        return character["bass"]["preset"]
    if role == "melody":
        return character["melody"]["preset"]
    if role == "counter_melody":
        return "pluck_soft" if character["melody"]["preset"] != "pluck_soft" else "bell_dark"
    if role == "chords":
        return character["harmony"].get("preset")
    for e in character.get("extra", []):
        if e["role"] == role:
            return e["preset"]
    return {"bell": "glass_keys", "atmosphere": "atmos_air"}.get(role)


def role_fx(character: dict, role: str) -> list[dict]:
    if role == "808":
        return [dict(f) for f in character["bass"].get("fx", [])]
    if role in ("melody", "counter_melody"):
        return [dict(f) for f in character["melody"].get("fx", [])]
    if role in ("chords", "bell", "atmosphere"):
        return [{"type": "reverb", "params": {"mix": 0.25, "decay_s": 2.2}}]
    if role == "hat":
        return [{"type": "stereo", "params": {"width": 1.3}}]
    return []


def generate_role_notes(project: dict, character: dict, role: str, seed: int, only_sections: set[str] | None = None,
                        **overrides) -> list[dict]:
    """Notes for one role over the whole song (or only the given section ids)."""
    ctx = _Ctx(project, character)
    if role == "kick":
        return gen_kick(ctx, seed, only_sections)
    if role in ("snare", "clap"):
        return gen_snare(ctx, seed, only_sections, role=role)
    if role == "hat":
        return gen_hat(ctx, seed, only_sections, rate=overrides.get("rate"), roll_density=overrides.get("roll_density"))
    if role == "perc":
        return gen_perc(ctx, seed, only_sections)
    if role == "808":
        try:
            kick = track_notes_absolute(track_by_ref(project, "kick"))
        except ValueError:
            kick = None
        return gen_808(ctx, seed, only_sections, kick_notes=kick, glide_prob=overrides.get("glide_prob"))
    if role == "chords":
        return gen_chords(ctx, seed, only_sections)
    if role == "melody":
        return gen_melody(ctx, seed, only_sections)
    if role == "counter_melody":
        try:
            mel = track_notes_absolute(track_by_ref(project, "melody"))
        except ValueError:
            mel = []
        return gen_counter_melody(ctx, seed, only_sections, melody_notes=mel)
    if role == "bell":
        return gen_bell(ctx, seed, only_sections)
    if role == "atmosphere":
        return gen_atmosphere(ctx, seed, only_sections)
    raise ValueError(f"the composer cannot generate role {role!r}")


def build_track(project: dict, character: dict, role: str, seed: int, preset_name: str | None = None, **overrides) -> dict:
    preset_name = preset_name or role_preset(character, role)
    if preset_name is None:
        raise ValueError(f"{character['name']} has no instrument for {role}")
    instrument = synth.preset(preset_name)
    if role == "808" and character["bass"].get("decay_s"):
        instrument["params"]["decay_s"] = character["bass"]["decay_s"]
    notes = generate_role_notes(project, character, role, seed, **overrides)
    gains = character["mix"].get("track_gains", {})
    return make_track(
        name=ROLE_NAMES.get(role, role.title()), role=role, instrument=instrument, notes=notes,
        length=total_beats(project), seed=_seed(seed, "synth"), gain_db=float(gains.get(role, -10)),
        pan={"perc": 0.25, "counter_melody": -0.2, "bell": 0.2}.get(role, 0.0),
        fx=role_fx(character, role), color=ROLE_COLORS.get(role),
        generator={"engine": "procedural", "version": COMPOSER_VERSION, "role_seed": seed, "character": character["id"]},
    )


def generate_project(character: dict, params: dict, seed: int | None = None) -> dict:
    seed = int(seed if seed is not None else _seed(now_iso(), params.get("prompt", "")) % 1_000_000)
    hints = interpret_prompt(params.get("prompt", ""))
    rng = np.random.default_rng(_seed(seed, "global"))
    key, scale = _choose_key(character, params, hints, rng)
    bpm = _choose_bpm(character, params, hints)
    energy = float(np.clip(float(params.get("energy", 0.7)) + hints.get("energy_bias", 0.0), 0.0, 1.0))
    duration = float(params.get("duration_s", 60))
    sections = plan_sections(character, bpm, duration, params.get("arrangement"))
    project = {
        "schema": SCHEMA, "id": new_id("prj"), "title": params.get("title") or f"{character['name']} sketch",
        "created_at": now_iso(), "updated_at": now_iso(), "character_id": character["id"], "seed": seed,
        "params": {"bpm": bpm, "key": key, "scale": scale, "time_signature": 4, "energy": energy,
                   "duration_s": duration, "prompt": params.get("prompt", ""), "mood": params.get("mood") or hints.get("mood"),
                   "vocals": params.get("vocals") or {"mode": "instrumental"}},
        "sections": sections, "harmony": [], "tracks": [],
        "master": {"gain_db": 0.0, "fx": [dict(f) for f in character["mix"].get("master_fx", [])],
                   "target_lufs": character["mix"].get("target_lufs")},
        "generation_notes": {"prompt_hints": hints, "key_label": key_label(pitch_class(key), scale),
                             "limitations": ["Only 4/4 is supported by the procedural composer.",
                                             "Vocals require a generative provider (not available in this engine)."]},
    }
    project["harmony"] = plan_harmony(character, sections, key, scale, seed)
    overrides = dict(hints.get("instruments", {}))
    overrides.update(params.get("instruments") or {})
    roles = []
    if not hints.get("no_drums") and not params.get("no_drums"):
        roles += [r for r in ("kick", "snare", "clap", "hat", "perc") if character["drums"].get(r)]
    roles += ["808", "chords", "melody"] + [e["role"] for e in character.get("extra", [])]
    for role in roles:
        if role == "chords" and not character["harmony"].get("preset"):
            continue
        tr = build_track(project, character, role, _seed(seed, role), preset_name=overrides.get(role))
        for e in character.get("extra", []):
            if e["role"] == role and "gain_db" in e:
                tr["gain_db"] = float(e["gain_db"])
        project["tracks"].append(tr)
    vocals = project["params"]["vocals"]
    if vocals.get("mode") == "vocals":
        project["generation_notes"]["vocals"] = ("Vocals were requested, but the procedural engine cannot sing or rap. "
                                                 "Connect the ACE-Step provider (lego task, 'vocals' track) or upload a vocal.")
    return project
