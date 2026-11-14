"""Natural language → validated edit actions.

The orchestrator never produces audio. It produces a list of actions from
:mod:`studio.actions`; the action system validates them and the renderer makes
the sound. Requests are interpreted in this order of preference:

1. deterministic DSP (gain, EQ, distortion, reverb, ...)
2. structured note edits (change instrument, transpose, clear a section)
3. targeted regeneration of a part or section (composer, new seed)

``RuleInterpreter`` is deterministic and works offline. ``LLMInterpreter`` is
optional (needs ``OPENAI_API_KEY`` and ``SOUNDLENS_STUDIO_LLM=1``) and its
output goes through exactly the same validation. If neither understands a
request, the response says so instead of guessing.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from . import synth
from .actions import ActionError, apply_actions
from .project import DRUM_ROLES, section_by_ref, summary

TRACK_WORDS = [
    (r"\b808s?\b|\bsub\b|\bbass\b", "808"),
    (r"\bkicks?\b", "kick"),
    (r"\bsnares?\b", "snare"),
    (r"\bclaps?\b", "clap"),
    (r"\bhi[- ]?hats?\b|\bhats?\b|\bhihats?\b", "hat"),
    (r"\bperc(ussion)?\b|\bshakers?\b|\brims?\b", "perc"),
    (r"\bcounter[- ]?melod(y|ies)\b", "counter_melody"),
    (r"\b(main )?melod(y|ies)\b|\blead\b|\btop line\b", "melody"),
    (r"\bchords?\b|\bpads?\b|\bharmony\b", "chords"),
    (r"\bbells?\b", "bell"),
    (r"\batmos(phere)?\b|\bambien(ce|t)\b|\btexture\b", "atmosphere"),
    (r"\bdrums?\b|\bbeat\b(?! battle)", "drums"),
    (r"\bvocals?\b|\bvox\b|\bsinging\b|\brap\b|\bad[- ]?libs?\b", "vocals"),
    (r"\b(entire|whole) (song|track|beat)\b|\beverything\b|\bthe song\b|\bmaster\b|\bmix\b", "master"),
]

INSTRUMENT_WORDS = [
    ("dark synth", "dark_synth"), ("rage synth", "rage_lead"), ("supersaw", "supersaw_lead"), ("synth lead", "supersaw_lead"),
    ("dark bell", "bell_dark"), ("music box", "music_box"), ("glass keys", "glass_keys"), ("glass pad", "pad_glass"),
    ("electric piano", "rhodes"), ("rhodes", "rhodes"), ("felt piano", "piano_felt"), ("piano", "piano"), ("keys", "rhodes"),
    ("bells", "bell_bright"), ("bell", "bell_bright"), ("pluck", "pluck_soft"), ("synth", "supersaw_lead"),
    ("warm pad", "pad_warm"), ("pad", "pad_warm"),
    ("hard kick", "kick_hard"), ("soft kick", "kick_soft"), ("punchy kick", "kick_punch"),
    ("clean 808", "808_clean"), ("warm 808", "808_warm"), ("blown 808", "808_blown"), ("distorted 808", "808_distorted"),
    ("dark hats", "hat_dark"), ("tick hats", "hat_tick"), ("crisp hats", "hat_crisp"),
    ("fat snare", "snare_fat"), ("tight snare", "snare_tight"), ("dusty snare", "snare_lofi"),
]
UNAVAILABLE_INSTRUMENTS = ["guitar", "violin", "strings", "cello", "flute", "sax", "trumpet", "brass", "choir", "organ", "harp"]
ADDABLE = [(r"counter[- ]?melod", "counter_melody"), (r"\bbells?\b", "bell"), (r"atmos|ambien|texture", "atmosphere"),
           (r"perc|shaker|rim", "perc"), (r"\bclaps?\b", "clap"), (r"\bsnares?\b", "snare"), (r"\bchords?\b|\bpads?\b", "chords")]
SECTION_RE = re.compile(r"\b(?:(first|second|third|fourth|last|1st|2nd|3rd|4th)\s+)?(intro|verse|hook|chorus|bridge|outro)(?:\s*(\d))?\b")


def _amount(text: str, base: float) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:db|decibels?)", text)
    if m:
        return float(m.group(1))
    if re.search(r"\b(a lot|much|way|really|super|very)\b", text):
        return base * 2
    if re.search(r"\b(a (little|bit|touch)|slightly|tad)\b", text):
        return base / 2
    return base


class Interpretation(dict):
    pass


class RuleInterpreter:
    name = "rules"

    def __init__(self, project: dict):
        self.p = project
        self.roles = {t["role"] for t in project["tracks"]}

    # -- targets ----------------------------------------------------------------
    INSTRUMENT_REFS = ("piano", "bell", "pluck", "synth", "pad", "keys", "guitar", "strings")

    def _targets(self, text: str) -> list[str]:
        # in "replace X with Y" / "change X to Y", only X names the track
        if re.search(r"\b(replace|swap|change|switch|turn)\b", text):
            text = re.split(r"\b(with|to|into|for)\b", text)[0]
        found = []
        for pat, role in TRACK_WORDS:
            if re.search(pat, text):
                found.append(role)
        # "the piano" etc. as a *track* reference: the track currently playing that instrument
        for word in self.INSTRUMENT_REFS:
            if not re.search(rf"\b{word}s?\b", text) or any(re.search(pat, word) for pat, _ in TRACK_WORDS):
                continue
            plural = bool(re.search(rf"\b{word}s\b", text))
            matches = [t for t in self.p["tracks"]
                       if word in synth.PRESETS.get((t.get("instrument") or {}).get("preset", ""), {}).get("label", "").lower()]
            # prefer the plainest match ("Felt piano" over "Electric piano") unless the user said "pianos"
            matches.sort(key=lambda t: ("electric" in synth.PRESETS[t["instrument"]["preset"]]["label"].lower(), t["role"] != "melody"))
            if not matches:
                raise ActionError(f"there's no {word} in this song (tracks: {', '.join(t['name'] for t in self.p['tracks'])})")
            for t in (matches if plural else matches[:1]):
                if t["role"] not in found:
                    found.insert(0, t["role"])
        if "drums" in found:
            found = [r for r in found if r not in DRUM_ROLES or r == "drums"]
        if "counter_melody" in found and "melody" in found:
            found.remove("melody")
        return found

    def _section(self, text: str, ctx: dict) -> str | None:
        m = SECTION_RE.search(text)
        if m:
            ordinal, name, num = m.groups()
            ref = " ".join(x for x in (ordinal, name, num) if x)
            section_by_ref(self.p, ref)  # raises if missing
            if not ordinal and not num:
                # "the chorus" means every chorus, so repeated sections stay identical
                names = ("hook", "chorus") if name in ("hook", "chorus") else (name,)
                return [s["id"] for s in self.p["sections"] if s["name"] in names]
            return [ref]
        if re.search(r"\bthis (section|part)\b|\bhere\b", text):
            if not ctx.get("section"):
                raise ActionError("select a section on the timeline first (or name it, e.g. “verse 2”)")
            return [ctx["section"]]
        return None

    def _sec_label(self, refs: list[str]) -> str:
        return ", ".join(section_by_ref(self.p, r)["label"] for r in refs)

    def _track_ref(self, role: str) -> str:
        if role in ("drums", "master"):
            return role
        if role not in self.roles:
            raise ActionError(f"there is no {role.replace('_', ' ')} track in this song")
        return role

    # -- main -------------------------------------------------------------------
    def interpret(self, text: str, ctx: dict | None = None) -> Interpretation:
        ctx = ctx or {}
        raw = text.strip()
        clauses = [c.strip() for c in re.split(r"\s*(?:,|;|\bthen\b|\band also\b|\band\b(?= (?:make|add|remove|turn|change|replace|mute|solo|pan|regenerate|give|put|take|more|less|a bit|a little|louder|quieter|brighter|darker|wider|punchier|dirtier|cleaner|longer|faster|slower)\b))\s*", raw.lower()) if c.strip()]
        actions, notes, problems = [], [], []
        last_targets: list[str] = []
        for clause in clauses:
            try:
                acts, expl, last_targets = self._clause(clause, ctx, last_targets)
                actions += acts
                notes += expl
            except ActionError as e:
                problems.append(str(e))
        return Interpretation(actions=actions, explanation=notes, problems=problems, interpreter=self.name)

    def _clause(self, c: str, ctx: dict, carry: list[str]):
        targets = self._targets(c) or carry or ([ctx["track_role"]] if ctx.get("track_role") else [])
        section = self._section(c, ctx)
        expl: list[str] = []

        if "vocals" in targets:
            vocal_tracks = [t for t in self.p["tracks"] if t["role"] in ("lead_vocal", "backing_vocal", "adlib")]
            if not vocal_tracks:
                raise ActionError("this song has no vocal track yet. Vocals need a generative vocal provider (ACE-Step) "
                                  "or an uploaded/recorded vocal; I won't fake one")
            targets = [t["id"] for t in vocal_tracks]

        # tempo / key for the whole song
        if re.search(r"\b(bpm|tempo)\b", c) or (("master" in targets or not targets) and re.search(r"\b(faster|slower|speed (it )?up|slow (it )?down)\b", c) and "hat" not in targets):
            bpm = float(self.p["params"]["bpm"])
            m = re.search(r"(\d{2,3})\s*bpm", c)
            new = float(m.group(1)) if m else round(bpm * (1.1 if re.search(r"faster|speed", c) else 0.9))
            return [{"op": "set_tempo", "bpm": new}], [f"tempo {bpm:g} → {new:g} BPM (MIDI parts re-render exactly; audio parts would be time-stretched)"], targets
        m = re.search(r"\b(up|down|raise|lower)\b.*?\b(\d+)?\s*(semitones?|half[- ]steps?|steps?|octaves?|key)\b", c) or re.search(r"\btranspose\b.*?(-?\d+)", c)
        if m and re.search(r"transpose|semitone|half[- ]step|octave|\bkey\b|pitch", c):
            num = re.search(r"(-?\d+)", c)
            n = int(num.group(1)) if num else 1
            if "octave" in c:
                n *= 12
            if re.search(r"\bdown|lower\b", c):
                n = -abs(n)
            whole = not targets or targets == ["master"]
            tr = "all" if whole else [self._track_ref(t) for t in targets if t not in ("master",)]
            return [{"op": "transpose", "track": tr, "semitones": n}], [f"transpose {'song' if whole else ', '.join(targets)} by {n:+d} semitones"], targets

        # add a part
        if re.search(r"\b(add|give( it)?|put in|layer)\b", c):
            for pat, role in ADDABLE:
                if re.search(pat, c):
                    if role in self.roles and role != "counter_melody":
                        raise ActionError(f"there's already a {role} track; ask to regenerate it or change its sound")
                    preset = next((p for w, p in INSTRUMENT_WORDS if w in c and synth.PRESETS[p]["engine"] not in ("kick", "snare", "clap", "hat", "perc", "808")), None)
                    a = {"op": "add_part", "role": role}
                    if preset and role in ("counter_melody", "bell", "chords"):
                        a["preset"] = preset
                    return [a], [f"add a new {role.replace('_', ' ')} part written against the existing chords" + (" and melody" if role == "counter_melody" else "")], [role]

        regen = re.search(r"\b(regenerate|rewrite|redo|re-?do|new|different|another|reroll|re-roll|vary|variation)\b", c)
        if regen and (not self._targets(c) or self._targets(c) == ["master"]) and (section or "master" in targets or not targets):
            if re.search(r"\brewrite\b", c):
                refs = [r for r in ("melody", "counter_melody") if r in self.roles]
                what = "the melody"
            else:
                refs = [t["id"] for t in self.p["tracks"] if (t.get("generator") or {}).get("engine") == "procedural"]
                what = "every composed part"
            a = {"op": "regenerate_part", "track": refs}
            if section:
                a["sections"] = section
            return [a], [f"regenerate {what}" + (f" in {self._sec_label(section)} only" if section else "") + "; everything outside that range is kept"], targets
        if not targets:
            hint = next((w for w in ("piano", "bell", "pluck", "synth", "pad", "keys", "guitar") if w in c), None)
            if hint:
                raise ActionError(f"there's no {hint} in this song; pick a track or name one ({', '.join(t['name'] for t in self.p['tracks'])})")
            raise ActionError(f"which track should I change? (I couldn't find one in “{c}”)")
        refs = [self._track_ref(t) for t in targets]
        fx_target = ["master"] if refs == ["master"] else refs

        # remove drums/parts from a section, or remove a track
        if re.search(r"\b(remove|take (\w+ )*out|pull (\w+ )*out|drop|cut|kill|no|mute|delete|strip|without)\b", c) and section:
            return [{"op": "clear_section", "track": refs, "sections": section}], [f"remove {', '.join(targets)} from {self._sec_label(section)} (other sections untouched)"], targets
        if re.search(r"\b(unmute|bring back)\b", c):
            return [{"op": "set_mixer", "track": refs, "mute": False}], [f"unmute {', '.join(targets)}"], targets
        if re.search(r"\bmute\b", c):
            return [{"op": "set_mixer", "track": refs, "mute": True}], [f"mute {', '.join(targets)}"], targets
        if re.search(r"\bsolo\b", c):
            return [{"op": "set_mixer", "track": refs, "solo": not re.search(r"unsolo|un-solo", c)}], [f"solo {', '.join(targets)}"], targets
        if re.search(r"\b(remove|delete|get rid of)\b", c):
            return [{"op": "remove_track", "track": refs}], [f"remove the {', '.join(targets)} track (undo brings it back)"], targets

        # instrument change
        m = re.search(r"\b(replace|swap|change|switch|turn|make|use)\b.*?\b(with|to|into|for|a|an)\b\s+(.*)$", c)
        if m and re.search(r"replace|swap|switch|change .* (to|into|with)|turn .* into|make (it|the \w+) (a|an) |use (a|an) ", c):
            tail = m.group(3)
            for w in UNAVAILABLE_INSTRUMENTS:
                if w in tail:
                    raise ActionError(f"there's no {w} instrument in the built-in synth engine yet; that needs a generative provider (ACE-Step lego) or a sample instrument")
            preset = next((p for w, p in INSTRUMENT_WORDS if w in tail), None)
            if preset:
                return ([{"op": "set_instrument", "track": refs, "preset": preset}],
                        [f"{', '.join(targets)} now plays the same notes on {synth.PRESETS[preset]['label']}"], targets)

        acts: list[dict] = []
        # hats faster / slower / rolls
        if "hat" in targets and re.search(r"faster|quicker|double|more rolls|busier|slower|half|fewer rolls|less rolls|simpler", c):
            opt = dict((self.p and next((t for t in self.p["tracks"] if t["role"] == "hat"), {}).get("generator", {}).get("options", {})) or {})
            rate = float(opt.get("rate", 0.5))
            rd = float(opt.get("roll_density", 0.3))
            if re.search(r"faster|quicker|double|busier", c):
                rate = max(0.125, rate / 2)
            if re.search(r"slower|half|simpler", c):
                rate = min(1.0, rate * 2)
            if re.search(r"more rolls|busier|faster", c):
                rd = min(1.0, rd + 0.25)
            if re.search(r"fewer rolls|less rolls|simpler|slower", c):
                rd = max(0.0, rd - 0.25)
            a = {"op": "regenerate_part", "track": "hat", "options": {"rate": rate, "roll_density": rd}}
            if section:
                a["sections"] = section
            return [a], [f"rewrite the hi-hat pattern at 1/{int(round(4 / rate))} notes, roll density {rd:.2f}"], targets

        # regeneration
        if regen:
            a = {"op": "regenerate_part", "track": refs}
            if section:
                a["sections"] = section
            return [a], [f"regenerate {', '.join(targets) if targets != ['master'] else 'all composed parts'}" + (f" in {self._sec_label(section)} only" if section else "") + "; every other part is kept"], targets

        # mixer
        if re.search(r"\b(louder|turn (it |\w+ )?up|boost|more volume|up the volume|raise)\b", c) and not re.search(r"distort|reverb|bright", c):
            d = _amount(c, 3.0)
            if refs == ["master"]:
                return [{"op": "set_master", "gain_db": min(24.0, self.p["master"].get("gain_db", 0) + d)}], [f"master +{d:g} dB"], targets
            return [{"op": "set_mixer", "track": refs, "delta_db": d}], [f"{', '.join(targets)} +{d:g} dB"], targets
        if re.search(r"\b(quieter|softer|turn (it |\w+ )?down|lower the volume|less loud|too loud)\b", c):
            d = _amount(c, 3.0)
            if refs == ["master"]:
                return [{"op": "set_master", "gain_db": max(-24.0, self.p["master"].get("gain_db", 0) - d)}], [f"master −{d:g} dB"], targets
            return [{"op": "set_mixer", "track": refs, "delta_db": -d}], [f"{', '.join(targets)} −{d:g} dB"], targets
        if re.search(r"\bpan\b", c):
            v = -0.6 if "left" in c else 0.6 if "right" in c else 0.0
            return [{"op": "set_mixer", "track": refs, "pan": v}], [f"pan {', '.join(targets)} {'left' if v < 0 else 'right' if v > 0 else 'center'}"], targets

        def has_fx(kind):
            if refs == ["master"]:
                return any(f["type"] == kind for f in self.p["master"].get("fx", []))
            return all(any(f["type"] == kind for f in next(t for t in self.p["tracks"] if t["role"] == r or t["id"] == r).get("fx", [])) for r in refs if r not in ("drums",)) and "drums" not in refs

        # distortion
        if re.search(r"less distort|cleaner|clean (it|the)|less dirty|less grit|smoother", c):
            if has_fx("distortion"):
                acts.append({"op": "update_effect", "track": fx_target, "type": "distortion", "delta": {"drive_db": -8}})
            if "808" in refs:
                acts.append({"op": "set_instrument_params", "track": "808", "delta": {"drive": -0.35}})
            if not acts:
                raise ActionError(f"{', '.join(targets)} has no distortion to reduce")
            return acts, ["reduce distortion (DSP)"], targets
        if re.search(r"distort|dirt|grit|crunch|blown|clip|nastier|harder|saturat|fuzz|aggressive|grimy", c):
            if re.search(r"saturat|warm", c) and not re.search(r"distort|blown|clip", c):
                acts.append({"op": "add_effect", "track": fx_target, "type": "saturation", "params": {"amount": 0.6 if re.search('more|heav', c) else 0.4}})
                return acts, ["add saturation (DSP)"], targets
            step = _amount(c, 6.0) if "db" in c else (10.0 if re.search(r"a lot|much|way|really|very|blown", c) else 6.0)
            if has_fx("distortion"):
                acts.append({"op": "update_effect", "track": fx_target, "type": "distortion", "delta": {"drive_db": step}})
            else:
                params = {"drive_db": 12.0 + step, "mode": "hard" if re.search("clip|blown|hard", c) else "soft"}
                if "808" in refs:
                    params.update({"keep_sub_hz": 70.0, "tone_hz": 6500.0})
                acts.append({"op": "add_effect", "track": fx_target, "type": "distortion", "params": params})
            if "808" in refs:
                acts.append({"op": "set_instrument_params", "track": "808", "delta": {"drive": 0.25}})
            if re.search(r"aggressive", c) and refs != ["master"]:
                acts.append({"op": "add_effect", "track": fx_target, "type": "compressor", "params": {"threshold_db": -20, "ratio": 6, "attack_ms": 3, "release_ms": 80, "makeup_db": 4}})
                acts.append({"op": "add_effect", "track": fx_target, "type": "eq", "params": {"bands": [{"type": "peak", "freq": 3000, "gain_db": 4, "q": 0.9}]}})
            return acts, [f"more distortion on {', '.join(targets)} (DSP waveshaper{', sub kept clean below 70 Hz' if '808' in refs else ''})"], targets

        # tone
        if re.search(r"brighter|more air|crisper|sharper|more top|more highs|shine", c):
            g = _amount(c, 4.0)
            return [{"op": "add_effect", "track": fx_target, "type": "eq", "params": {"bands": [{"type": "highshelf", "freq": 6000, "gain_db": min(g, 12), "q": 0.7}]}}], [f"high shelf +{g:g} dB at 6 kHz"], targets
        if re.search(r"darker|duller|muffle|less bright|warmer|softer tone|underwater|filter", c):
            cutoff = 700.0 if "underwater" in c else 2500.0
            return [{"op": "add_effect", "track": fx_target, "type": "filter", "params": {"mode": "lowpass", "cutoff": cutoff, "slope": 24}}], [f"low-pass filter at {cutoff:g} Hz"], targets
        if re.search(r"more (low end|bass|bottom|weight)|heavier|fatter|thicker", c) and "808" not in refs:
            return [{"op": "add_effect", "track": fx_target, "type": "eq", "params": {"bands": [{"type": "lowshelf", "freq": 150, "gain_db": 4, "q": 0.7}]}}], ["low shelf +4 dB at 150 Hz"], targets
        if re.search(r"more (low end|bass|bottom|weight)|heavier|fatter|thicker|longer", c) and "808" in refs:
            return [{"op": "set_instrument_params", "track": "808", "delta": {"decay_s": 0.4}, "limits": {"decay_s": [0.1, 4.0]}},
                    {"op": "set_mixer", "track": "808", "delta_db": 1.5}], ["longer 808 tail and +1.5 dB"], targets

        # space
        if re.search(r"less reverb|drier|dry|less wet|closer", c):
            if not has_fx("reverb"):
                raise ActionError(f"{', '.join(targets)} has no reverb")
            return [{"op": "update_effect", "track": fx_target, "type": "reverb", "delta": {"mix": -0.15}}], ["less reverb"], targets
        if re.search(r"reverb|wetter|spac(e|ey|ious)|bigger|room|hall|ethereal|dreamy", c):
            if has_fx("reverb"):
                return [{"op": "update_effect", "track": fx_target, "type": "reverb", "delta": {"mix": 0.15, "decay_s": 0.8}}], ["more reverb"], targets
            return [{"op": "add_effect", "track": fx_target, "type": "reverb", "params": {"mix": 0.3, "decay_s": 2.5}}], ["add reverb"], targets
        if re.search(r"delay|echo", c):
            return [{"op": "add_effect", "track": fx_target, "type": "delay", "params": {"beats": 0.75, "feedback": 0.35, "mix": 0.25}}], ["add a dotted-eighth delay"], targets
        if re.search(r"punch|compress|tighter|glue", c):
            return [{"op": "add_effect", "track": fx_target, "type": "compressor", "params": {"threshold_db": -18, "ratio": 4, "attack_ms": 15, "release_ms": 100, "makeup_db": 3}}], ["add compression"], targets
        if re.search(r"wider|stereo|spread", c):
            return [{"op": "add_effect", "track": fx_target, "type": "stereo", "params": {"width": 1.6}}], ["widen stereo image"], targets
        if re.search(r"\bmono\b|narrow", c):
            return [{"op": "add_effect", "track": fx_target, "type": "stereo", "params": {"width": 0.0}}], ["collapse to mono"], targets
        if re.search(r"lo-?fi|bitcrush|crushed|8[- ]?bit|retro", c):
            return [{"op": "add_effect", "track": fx_target, "type": "bitcrush", "params": {"bits": 8, "downsample": 2, "mix": 0.6}}], ["bit-crush"], targets
        if re.search(r"quantiz|on beat|on the grid|tighten", c):
            return [{"op": "quantize", "track": refs, "grid": 0.25, "strength": 1.0}], ["quantize to 1/16"], targets
        raise ActionError(f"I don't know how to “{c}” yet")


class LLMInterpreter:
    """Optional: asks an LLM for actions as JSON. Output is validated exactly like rule output."""
    name = "llm"

    def __init__(self, project: dict):
        self.p = project

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("SOUNDLENS_STUDIO_LLM") == "1"

    def interpret(self, text: str, ctx: dict | None = None) -> Interpretation:
        from openai import OpenAI
        from .dsp import EFFECTS
        from .actions import OPS
        system = (
            "You translate a music producer's request into JSON edit actions for SoundLens Studio. "
            "Never invent audio; only use the listed operations. Prefer DSP effects for tone changes, set_instrument for "
            "instrument swaps, regenerate_part only when new notes are needed. If the request is impossible (e.g. vocals that "
            "do not exist), return no actions and explain. Reply with JSON: {\"actions\": [...], \"explanation\": [..], \"problems\": [..]}.\n"
            f"Operations: {sorted(OPS)}\nEffects and numeric ranges: { {k: v['params'] for k, v in EFFECTS.items()} }\n"
            f"Instrument presets: { {k: v['label'] for k, v in synth.PRESETS.items()} }\n"
            "Track refs may be a track id, a role, or the groups 'drums', 'all', 'master'. Section refs look like 'verse 2', 'hook', 'last hook'."
        )
        user = json.dumps({"request": text, "context": ctx or {}, "project": summary(self.p)})
        client = OpenAI()
        resp = client.chat.completions.create(model=os.getenv("SOUNDLENS_STUDIO_LLM_MODEL", "gpt-4o-mini"),
                                              response_format={"type": "json_object"},
                                              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        data = json.loads(resp.choices[0].message.content or "{}")
        return Interpretation(actions=list(data.get("actions") or []), explanation=list(data.get("explanation") or []),
                              problems=list(data.get("problems") or []), interpreter=self.name)


def interpret(project: dict, text: str, ctx: dict | None = None, character: dict | None = None, allow_llm: bool = True) -> Interpretation:
    """Interpret, then dry-run the actions so the caller only ever sees valid edits."""
    if not text or len(text) > 500:
        return Interpretation(actions=[], explanation=[], problems=["say what to change (max 500 characters)"], interpreter="none")
    result = RuleInterpreter(project).interpret(text, ctx)
    if not result["actions"] and allow_llm and LLMInterpreter.available():
        try:
            llm = LLMInterpreter(project).interpret(text, ctx)
            if llm["actions"]:
                result = llm
            else:
                result["problems"] += llm["problems"]
        except Exception as e:  # network/quota errors must not break editing
            result["problems"].append(f"language model unavailable ({type(e).__name__})")
    if result["actions"]:
        try:
            apply_actions(project, result["actions"], character)
        except ActionError as e:
            result["problems"].append(str(e))
            result["actions"] = []
    return result
