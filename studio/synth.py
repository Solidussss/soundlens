"""DSP instruments that turn note data into audio.

Every instrument here is synthesized from first principles (oscillators,
noise, envelopes, filters). There are no samples, so there are no sample
licensing questions, and every sound is a set of editable parameters.

Notes are dicts: ``{"t": start_beats, "d": length_beats, "p": midi, "v": 0..1, "slide": bool}``.
``render_notes`` returns stereo ``float64`` audio of exactly ``n`` samples.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
from scipy import signal

from . import dsp
from .theory import midi_to_hz

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# oscillators
# ---------------------------------------------------------------------------

def _poly_blep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    out[m] = x + x - x * x - 1.0
    m2 = t > 1.0 - dt
    x2 = (t[m2] - 1.0) / dt[m2]
    out[m2] = x2 * x2 + x2 + x2 + 1.0
    return out


def saw_from_freq(freq: np.ndarray, sr: int, phase0: float = 0.0) -> np.ndarray:
    dt = np.clip(freq / sr, 1e-9, 0.49)
    phase = (phase0 + np.cumsum(dt)) % 1.0
    return (2.0 * phase - 1.0) - _poly_blep(phase, dt)


def square_from_freq(freq: np.ndarray, sr: int, phase0: float = 0.0, width: float = 0.5) -> np.ndarray:
    dt = np.clip(freq / sr, 1e-9, 0.49)
    phase = (phase0 + np.cumsum(dt)) % 1.0
    sq = np.where(phase < width, 1.0, -1.0)
    sq += _poly_blep(phase, dt)
    sq -= _poly_blep((phase + (1.0 - width)) % 1.0, dt)
    return sq


def sine_from_freq(freq: np.ndarray, sr: int, phase0: float = 0.0) -> np.ndarray:
    return np.sin(phase0 + TWO_PI * np.cumsum(freq) / sr)


def _adsr(n: int, sr: int, a: float, d: float, s: float, r: float, gate_n: int) -> np.ndarray:
    """ADSR envelope (seconds). ``gate_n`` = samples the note is held."""
    env = np.zeros(n)
    t = np.arange(n) / sr
    a = max(a, 1e-4)
    att = np.clip(t / a, 0.0, 1.0)
    dec = s + (1.0 - s) * np.exp(-np.maximum(t - a, 0.0) / max(d, 1e-4))
    env = np.where(t < a, att, dec)
    if gate_n < n:
        level_at_release = env[max(gate_n - 1, 0)]
        tr = (np.arange(n - gate_n)) / sr
        env[gate_n:] = level_at_release * np.exp(-tr / max(r, 1e-4))
    return env


def _beats_to_samples(beats: float, bpm: float, sr: int) -> int:
    return int(round(beats * 60.0 / bpm * sr))


def _vel_gain(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0)) ** 1.5


def _rng(seed: int, note: dict) -> np.random.Generator:
    """Per-note randomness keyed by the note's *content*, not its position in the list.

    Adding or removing other notes therefore never changes how an existing note sounds,
    which is what keeps section-level regeneration from touching the rest of the track.
    """
    key = (int(seed) * 1_000_003 + int(round(note["t"] * 9600)) * 7919 + int(note.get("p", 0)) * 104729) & 0xFFFFFFFF
    return np.random.default_rng(key)


def _place(buf: np.ndarray, start: int, sig: np.ndarray) -> None:
    n = buf.shape[-1]
    if start >= n or start + sig.shape[-1] <= 0:
        return
    s0 = max(0, -start)
    end = min(n, start + sig.shape[-1])
    buf[..., max(start, 0):end] += sig[..., s0:s0 + end - max(start, 0)]


# ---------------------------------------------------------------------------
# drum engines (one-shot per hit)
# ---------------------------------------------------------------------------

def kick_hit(sr: int, p: dict, v: float, rng) -> np.ndarray:
    dur = p.get("decay_ms", 380) / 1000.0 * 2.2
    n = int(sr * dur)
    t = np.arange(n) / sr
    f0, f1 = p.get("pitch_start", 170.0), p.get("pitch_end", 50.0)
    freq = f1 + (f0 - f1) * np.exp(-t / (p.get("pitch_decay_ms", 35) / 1000.0))
    body = sine_from_freq(freq, sr) * np.exp(-t / (p.get("decay_ms", 380) / 1000.0))
    click = rng.standard_normal(n) * np.exp(-t / 0.0025) * p.get("click", 0.35)
    click = signal.sosfilt(dsp._biquad("highpass", 1800, sr), click)
    y = body + click
    drive = p.get("drive", 0.2)
    if drive > 0:
        k = 1 + drive * 8
        y = np.tanh(y * k) / math.tanh(k)
    return y * _vel_gain(v)


def snare_hit(sr: int, p: dict, v: float, rng) -> np.ndarray:
    nd = p.get("noise_decay_ms", 170) / 1000.0
    n = int(sr * nd * 4)
    t = np.arange(n) / sr
    body_f = p.get("body_hz", 190.0)
    freq = body_f * (1 + 0.5 * np.exp(-t / 0.01))
    body = sine_from_freq(freq, sr) * np.exp(-t / (p.get("body_decay_ms", 90) / 1000.0)) * p.get("body", 0.55)
    noise = rng.standard_normal(n)
    noise = signal.sosfilt(np.vstack([dsp._biquad("highpass", p.get("noise_hp", 1500), sr), dsp._biquad("lowpass", p.get("tone_hz", 9000), sr)]), noise)
    noise *= np.exp(-t / nd) * p.get("snap", 0.8)
    y = body + noise
    return np.tanh(y * 1.5) * _vel_gain(v)


def clap_hit(sr: int, p: dict, v: float, rng) -> np.ndarray:
    decay = p.get("decay_ms", 220) / 1000.0
    n = int(sr * decay * 4)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    noise = signal.sosfilt(dsp._biquad("bandpass", p.get("tone_hz", 1400), sr, 0.9), noise)
    env = np.zeros(n)
    spread = p.get("spread_ms", 11) / 1000.0
    for k in range(int(p.get("bursts", 4))):
        tk = np.maximum(t - k * spread, 0.0)
        on = t >= k * spread
        env += on * np.exp(-tk / (0.006 if k < p.get("bursts", 4) - 1 else decay))
    return np.tanh(noise * env * 2.2) * _vel_gain(v)


_HAT_RATIOS = np.array([2.0, 3.0, 4.16, 5.43, 6.79, 8.21])


def hat_hit(sr: int, p: dict, v: float, rng, open_: bool) -> np.ndarray:
    decay = (p.get("open_decay_ms", 320) if open_ else p.get("decay_ms", 45)) / 1000.0
    n = int(sr * decay * 5)
    t = np.arange(n) / sr
    base = p.get("metal_hz", 205.0)
    metal = sum(square_from_freq(np.full(n, base * r), sr, phase0=rng.uniform()) for r in _HAT_RATIOS)
    noise = rng.standard_normal(n) * p.get("noise", 0.6)
    y = metal / len(_HAT_RATIOS) + noise
    y = signal.sosfilt(np.vstack([dsp._biquad("highpass", p.get("hp_hz", 7000), sr), dsp._biquad("highpass", p.get("hp_hz", 7000), sr),
                                  dsp._biquad("peak", p.get("tone_hz", 10500), sr, 1.2, 4.0)]), y)
    y *= np.exp(-t / decay)
    return y * 1.4 * _vel_gain(v)


def perc_hit(sr: int, p: dict, v: float, rng, pitch: int) -> np.ndarray:
    mode = p.get("mode", "rim")
    if mode == "shaker":
        n = int(sr * 0.12)
        t = np.arange(n) / sr
        y = rng.standard_normal(n) * (t / 0.012).clip(0, 1) * np.exp(-t / 0.04)
        y = signal.sosfilt(dsp._biquad("highpass", 5000, sr), y)
        return y * 0.8 * _vel_gain(v)
    if mode == "tom":
        n = int(sr * 0.5)
        t = np.arange(n) / sr
        f = midi_to_hz(pitch) * (1 + 0.6 * np.exp(-t / 0.03))
        return sine_from_freq(f, sr) * np.exp(-t / 0.18) * _vel_gain(v)
    # rim / wood click
    n = int(sr * 0.08)
    t = np.arange(n) / sr
    y = sine_from_freq(np.full(n, p.get("tone_hz", 1700.0)), sr) * np.exp(-t / 0.012)
    y += rng.standard_normal(n) * np.exp(-t / 0.003) * 0.5
    return y * _vel_gain(v)


def _render_drums(engine: str, notes: list[dict], p: dict, bpm: float, n: int, sr: int, seed: int) -> np.ndarray:
    out = np.zeros(n)
    ordered = sorted(notes, key=lambda x: x["t"])
    starts = [_beats_to_samples(nt["t"], bpm, sr) for nt in ordered]
    for i, note in enumerate(ordered):
        rng = _rng(seed, note)
        start = starts[i]
        v = note.get("v", 0.8)
        if engine == "kick":
            hit = kick_hit(sr, p, v, rng)
        elif engine == "snare":
            hit = snare_hit(sr, p, v, rng)
        elif engine == "clap":
            hit = clap_hit(sr, p, v, rng)
        elif engine == "hat":
            is_open = note.get("p", 42) == 46
            hit = hat_hit(sr, p, v, rng, is_open)
            # the next hit on the same kit piece chokes a ringing open hat, like a real kit
            if is_open and i + 1 < len(ordered):
                cut = starts[i + 1] - start
                if 0 < cut < hit.size:
                    fade = min(int(0.004 * sr), cut)
                    hit = hit.copy()
                    hit[cut - fade:cut] *= np.linspace(1.0, 0.0, fade)
                    hit[cut:] = 0.0
        else:
            hit = perc_hit(sr, p, v, rng, note.get("p", 60))
        _place(out, start, hit)
    return out


# ---------------------------------------------------------------------------
# 808 (monophonic, glides, punch, built-in drive)
# ---------------------------------------------------------------------------

def _render_808(notes: list[dict], p: dict, bpm: float, n: int, sr: int, seed: int) -> np.ndarray:
    notes = sorted(notes, key=lambda x: x["t"])
    if not notes:
        return np.zeros(n)
    freq = np.zeros(n)
    amp = np.zeros(n)
    glide_s = p.get("glide_ms", 70) / 1000.0
    decay_s = p.get("decay_s", 1.6)
    punch_st = p.get("punch_semitones", 7.0)
    punch_s = p.get("punch_ms", 25) / 1000.0
    release_s = p.get("release_ms", 40) / 1000.0
    prev_hz = None
    last_amp = 0.0
    for i, note in enumerate(notes):
        start = max(0, _beats_to_samples(note["t"], bpm, sr))
        if start >= n:
            break
        end_gate = min(n, start + max(1, _beats_to_samples(note["d"], bpm, sr)))
        nxt = _beats_to_samples(notes[i + 1]["t"], bpm, sr) if i + 1 < len(notes) else n
        seg_end = min(n, max(nxt, end_gate) if i + 1 >= len(notes) else nxt)
        seg_end = max(seg_end, start + 1)
        seg_n = seg_end - start
        t = np.arange(seg_n) / sr
        hz = midi_to_hz(note["p"])
        slide = bool(note.get("slide")) and prev_hz is not None
        if slide:
            f = hz + (prev_hz - hz) * np.exp(-t / max(glide_s / 3.0, 1e-4))
            a_env = last_amp + (1.0 - last_amp) * np.clip(t / 0.004, 0, 1)  # legato: no hard retrigger
            a_env = a_env * np.exp(-t / decay_s)
        else:
            f = hz * 2.0 ** (punch_st * np.exp(-t / max(punch_s, 1e-4)) / 12.0)
            a_env = np.clip(t / 0.002, 0, 1) * np.exp(-t / decay_s)
        gate_len = end_gate - start
        if gate_len < seg_n:
            rel_t = np.arange(seg_n - gate_len) / sr
            a_env[gate_len:] = a_env[gate_len - 1] * np.exp(-rel_t / max(release_s, 1e-4))
        freq[start:seg_end] = f
        amp[start:seg_end] = a_env * _vel_gain(note.get("v", 0.9))
        prev_hz = hz
        last_amp = float(a_env[-1])
    freq[freq <= 0] = 40.0
    phase = TWO_PI * np.cumsum(freq) / sr
    tri_mix = p.get("tri", 0.0)
    wave = np.sin(phase) * (1 - tri_mix) + tri_mix * (2 / math.pi) * np.arcsin(np.sin(phase))
    # 2nd harmonic helps small speakers hear the 808 (like real 808 "tops")
    wave += p.get("harmonic2", 0.08) * np.sin(2 * phase)
    y = wave * amp
    drive = p.get("drive", 0.0)
    if drive > 0:
        k = 1.0 + drive * 14.0
        y = np.tanh(y * k) / math.tanh(k)
    # tiny anti-click fade where amplitude jumps (retriggers)
    return y


# ---------------------------------------------------------------------------
# tonal engines (polyphonic)
# ---------------------------------------------------------------------------

def _note_audio_saw(note, p, bpm, sr, rng) -> np.ndarray:
    gate = max(1, _beats_to_samples(note["d"], bpm, sr))
    rel = p.get("release", 0.25)
    n = gate + int(rel * sr * 5)
    hz = midi_to_hz(note["p"] + p.get("octave", 0) * 12)
    voices = int(p.get("voices", 5))
    detune = p.get("detune_cents", 18.0)
    t = np.arange(n) / sr
    vib = 1.0 + (2 ** (p.get("vibrato_cents", 0.0) / 1200.0) - 1.0) * np.sin(TWO_PI * p.get("vibrato_hz", 5.0) * t) * np.clip((t - 0.25) / 0.3, 0, 1)
    out = np.zeros((2, n))
    wave = p.get("wave", "saw")
    for k in range(voices):
        spread = 0.0 if voices == 1 else (k / (voices - 1)) * 2 - 1
        f = hz * 2 ** (spread * detune / 1200.0) * vib
        osc = square_from_freq(f, sr, rng.uniform()) if wave == "square" else saw_from_freq(f, sr, rng.uniform())
        pan = spread * p.get("width", 0.8)
        out[0] += osc * math.cos((pan + 1) * math.pi / 4)
        out[1] += osc * math.sin((pan + 1) * math.pi / 4)
    out /= math.sqrt(voices)
    env = _adsr(n, sr, p.get("attack", 0.005), p.get("decay", 0.3), p.get("sustain", 0.7), rel, gate)
    cutoff = p.get("cutoff", 4000.0)
    dark = dsp.sos_filter(out, np.vstack([dsp._biquad("lowpass", cutoff, sr, p.get("resonance", 0.9)), dsp._biquad("lowpass", cutoff, sr)]))
    env_amt = p.get("filter_env", 0.0)
    if env_amt > 0:
        bright_cut = min(cutoff * (1 + 6 * env_amt), sr * 0.45)
        bright = dsp.sos_filter(out, np.vstack([dsp._biquad("lowpass", bright_cut, sr, p.get("resonance", 0.9)), dsp._biquad("lowpass", bright_cut, sr)]))
        fenv = np.exp(-t / max(p.get("filter_decay", 0.2), 1e-3))
        y = bright * fenv + dark * (1 - fenv)
    else:
        y = dark
    return y * env * _vel_gain(note.get("v", 0.8))


def _note_audio_fm(note, p, bpm, sr, rng) -> np.ndarray:
    gate = max(1, _beats_to_samples(note["d"], bpm, sr))
    decay = p.get("decay", 1.2)
    n = gate + int(min(decay * 3, 6.0) * sr) if p.get("ring", True) else gate + int(p.get("release", 0.1) * sr * 5)
    hz = midi_to_hz(note["p"] + p.get("octave", 0) * 12)
    t = np.arange(n) / sr
    mod_ratio = p.get("mod_ratio", 3.5)
    index = p.get("index", 3.0) * np.exp(-t / max(p.get("index_decay", 0.25), 1e-3)) + p.get("index_sustain", 0.2)
    mod = np.sin(TWO_PI * hz * mod_ratio * t + rng.uniform() * TWO_PI)
    car = np.sin(TWO_PI * hz * p.get("car_ratio", 1.0) * t + index * mod)
    if p.get("ring", True):
        env = np.clip(t / max(p.get("attack", 0.002), 1e-4), 0, 1) * np.exp(-t / decay)
        if gate < n and p.get("damp_on_release", False):
            env[gate:] *= np.exp(-(np.arange(n - gate) / sr) / 0.08)
    else:
        env = _adsr(n, sr, p.get("attack", 0.002), decay, p.get("sustain", 0.0), p.get("release", 0.1), gate)
    y = car * env * _vel_gain(note.get("v", 0.8))
    return np.vstack([y, y])


def _note_audio_keys(note, p, bpm, sr, rng) -> np.ndarray:
    """Additive piano-like tone with mild inharmonicity and partial-dependent decay."""
    gate = max(1, _beats_to_samples(note["d"], bpm, sr))
    hz = midi_to_hz(note["p"] + p.get("octave", 0) * 12)
    decay = p.get("decay", 2.5) * (440.0 / max(hz, 60.0)) ** 0.35
    n = gate + int(min(decay * 2.0, 5.0) * sr)
    t = np.arange(n) / sr
    B = p.get("inharmonicity", 0.0004)
    y = np.zeros(n)
    bright = p.get("brightness", 0.6)
    for k in range(1, 14):
        fk = hz * k * math.sqrt(1 + B * k * k)
        if fk > sr * 0.45:
            break
        ak = (bright ** (k - 1)) / k ** 0.6
        y += ak * np.sin(TWO_PI * fk * t + rng.uniform() * TWO_PI) * np.exp(-t * k ** 0.7 / decay)
    hammer = rng.standard_normal(n) * np.exp(-t / 0.004) * 0.05
    y = (y + hammer) * np.clip(t / 0.002, 0, 1)
    # dampers come down at note-off
    if gate < n:
        y[gate:] *= np.exp(-(np.arange(n - gate) / sr) / p.get("release", 0.18))
    y *= _vel_gain(note.get("v", 0.8)) * 0.5
    return np.vstack([y, y])


def _render_poly(engine: str, notes: list[dict], p: dict, bpm: float, n: int, sr: int, seed: int) -> np.ndarray:
    out = np.zeros((2, n))
    fn = {"saw": _note_audio_saw, "fm": _note_audio_fm, "keys": _note_audio_keys}[engine]
    for i, note in enumerate(sorted(notes, key=lambda x: (x["t"], x["p"]))):
        start = _beats_to_samples(note["t"], bpm, sr)
        if start >= n:
            continue
        _place(out, start, fn(note, p, bpm, sr, _rng(seed, note)))
    return out


def _render_texture(notes: list[dict], p: dict, bpm: float, n: int, sr: int, seed: int) -> np.ndarray:
    """Atmosphere: band-passed stereo noise swells gated by notes (pitch sets the centre)."""
    out = np.zeros((2, n))
    for i, note in enumerate(notes):
        rng = _rng(seed, note)
        start = _beats_to_samples(note["t"], bpm, sr)
        length = max(1, _beats_to_samples(note["d"], bpm, sr))
        m = length + int(sr * 1.0)
        t = np.arange(m) / sr
        noise = rng.standard_normal((2, m))
        fc = midi_to_hz(note["p"])
        noise = dsp.sos_filter(noise, np.vstack([dsp._biquad("bandpass", fc, sr, p.get("q", 2.5)), dsp._biquad("lowpass", p.get("cutoff", 6000), sr)]))
        lfo = 0.6 + 0.4 * np.sin(TWO_PI * p.get("lfo_hz", 0.2) * t + rng.uniform() * TWO_PI)
        env = _adsr(m, sr, p.get("attack", 1.0), 1.0, 1.0, p.get("release", 0.8), length)
        _place(out, start, noise * env * lfo * _vel_gain(note.get("v", 0.6)) * 2.0)
    return out


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    # drums
    "kick_punch": {"engine": "kick", "label": "Punchy kick", "params": {"pitch_start": 180, "pitch_end": 52, "pitch_decay_ms": 30, "decay_ms": 300, "click": 0.4, "drive": 0.25}},
    "kick_hard": {"engine": "kick", "label": "Hard distorted kick", "params": {"pitch_start": 220, "pitch_end": 55, "pitch_decay_ms": 22, "decay_ms": 260, "click": 0.6, "drive": 0.8}},
    "kick_soft": {"engine": "kick", "label": "Soft round kick", "params": {"pitch_start": 120, "pitch_end": 48, "pitch_decay_ms": 45, "decay_ms": 420, "click": 0.1, "drive": 0.05}},
    "snare_tight": {"engine": "snare", "label": "Tight snare", "params": {"body_hz": 200, "body_decay_ms": 70, "noise_decay_ms": 140, "snap": 0.9, "tone_hz": 9000}},
    "snare_fat": {"engine": "snare", "label": "Fat snare", "params": {"body_hz": 170, "body_decay_ms": 120, "noise_decay_ms": 210, "snap": 0.75, "tone_hz": 7000, "body": 0.8}},
    "snare_lofi": {"engine": "snare", "label": "Dusty snare", "params": {"body_hz": 180, "body_decay_ms": 80, "noise_decay_ms": 160, "snap": 0.6, "tone_hz": 4500, "noise_hp": 900}},
    "clap_wide": {"engine": "clap", "label": "Wide clap", "params": {"decay_ms": 230, "tone_hz": 1300, "bursts": 4, "spread_ms": 11}},
    "clap_snappy": {"engine": "clap", "label": "Snappy clap", "params": {"decay_ms": 140, "tone_hz": 1800, "bursts": 3, "spread_ms": 8}},
    "hat_crisp": {"engine": "hat", "label": "Crisp hats", "params": {"decay_ms": 40, "open_decay_ms": 300, "hp_hz": 7500, "tone_hz": 11000}},
    "hat_dark": {"engine": "hat", "label": "Dark hats", "params": {"decay_ms": 55, "open_decay_ms": 380, "hp_hz": 5200, "tone_hz": 8000, "noise": 0.4}},
    "hat_tick": {"engine": "hat", "label": "Tick hats", "params": {"decay_ms": 22, "open_decay_ms": 200, "hp_hz": 9000, "tone_hz": 12500}},
    "perc_rim": {"engine": "perc", "label": "Rim", "params": {"mode": "rim", "tone_hz": 1700}},
    "perc_shaker": {"engine": "perc", "label": "Shaker", "params": {"mode": "shaker"}},
    "perc_tom": {"engine": "perc", "label": "Tom", "params": {"mode": "tom"}},
    # 808s
    "808_clean": {"engine": "808", "label": "Clean sub 808", "params": {"decay_s": 1.8, "glide_ms": 70, "punch_semitones": 5, "punch_ms": 20, "drive": 0.0, "harmonic2": 0.06}},
    "808_warm": {"engine": "808", "label": "Warm 808", "params": {"decay_s": 1.6, "glide_ms": 80, "punch_semitones": 7, "punch_ms": 25, "drive": 0.25, "harmonic2": 0.1}},
    "808_distorted": {"engine": "808", "label": "Distorted 808", "params": {"decay_s": 1.5, "glide_ms": 70, "punch_semitones": 9, "punch_ms": 25, "drive": 0.65, "tri": 0.2, "harmonic2": 0.12}},
    "808_blown": {"engine": "808", "label": "Blown-out 808", "params": {"decay_s": 1.3, "glide_ms": 60, "punch_semitones": 12, "punch_ms": 30, "drive": 1.0, "tri": 0.4, "harmonic2": 0.2}},
    # melodic
    "bell_bright": {"engine": "fm", "label": "Bright bell", "params": {"mod_ratio": 3.5, "index": 3.2, "index_decay": 0.3, "decay": 1.4, "index_sustain": 0.15}},
    "bell_dark": {"engine": "fm", "label": "Dark bell", "params": {"mod_ratio": 2.0, "index": 2.0, "index_decay": 0.4, "decay": 1.8, "index_sustain": 0.05}},
    "music_box": {"engine": "fm", "label": "Music box", "params": {"mod_ratio": 7.0, "index": 1.6, "index_decay": 0.08, "decay": 0.9, "octave": 1}},
    "pluck_soft": {"engine": "fm", "label": "Soft pluck", "params": {"mod_ratio": 1.0, "index": 2.4, "index_decay": 0.06, "decay": 0.35, "index_sustain": 0.0}},
    "glass_keys": {"engine": "fm", "label": "Glass keys", "params": {"mod_ratio": 14.0, "index": 0.9, "index_decay": 0.2, "decay": 1.6, "index_sustain": 0.1}},
    "rhodes": {"engine": "fm", "label": "Electric piano", "params": {"mod_ratio": 1.0, "index": 1.8, "index_decay": 0.5, "decay": 1.6, "index_sustain": 0.3, "damp_on_release": True}},
    "piano": {"engine": "keys", "label": "Piano", "params": {"decay": 2.4, "brightness": 0.6, "inharmonicity": 0.0004}},
    "piano_felt": {"engine": "keys", "label": "Felt piano", "params": {"decay": 2.0, "brightness": 0.35, "inharmonicity": 0.0002, "release": 0.3}},
    "supersaw_lead": {"engine": "saw", "label": "Supersaw lead", "params": {"voices": 7, "detune_cents": 22, "cutoff": 5200, "attack": 0.004, "decay": 0.25, "sustain": 0.8, "release": 0.18, "filter_env": 0.4, "width": 0.9}},
    "rage_lead": {"engine": "saw", "label": "Rage synth", "params": {"voices": 7, "detune_cents": 32, "cutoff": 3200, "resonance": 1.4, "attack": 0.003, "decay": 0.2, "sustain": 0.75, "release": 0.12, "filter_env": 0.8, "filter_decay": 0.12, "width": 1.0, "vibrato_cents": 12}},
    "dark_synth": {"engine": "saw", "label": "Dark synth", "params": {"voices": 4, "detune_cents": 12, "wave": "square", "cutoff": 1100, "resonance": 1.2, "attack": 0.01, "decay": 0.4, "sustain": 0.6, "release": 0.3, "filter_env": 0.5, "filter_decay": 0.25, "width": 0.6}},
    "pad_warm": {"engine": "saw", "label": "Warm pad", "params": {"voices": 5, "detune_cents": 14, "cutoff": 1500, "attack": 0.6, "decay": 1.0, "sustain": 0.85, "release": 1.2, "width": 1.0}},
    "pad_glass": {"engine": "saw", "label": "Glass pad", "params": {"voices": 6, "detune_cents": 9, "cutoff": 3800, "attack": 0.9, "decay": 1.5, "sustain": 0.8, "release": 1.5, "width": 1.0, "vibrato_cents": 6, "vibrato_hz": 4.2}},
    "atmos_air": {"engine": "texture", "label": "Air texture", "params": {"q": 1.2, "attack": 1.5, "release": 1.5, "lfo_hz": 0.15, "cutoff": 9000}},
    "atmos_dark": {"engine": "texture", "label": "Dark drone", "params": {"q": 3.0, "attack": 1.0, "release": 1.2, "lfo_hz": 0.3, "cutoff": 2500}},
}

ENGINES = {"kick", "snare", "clap", "hat", "perc", "808", "saw", "fm", "keys", "texture"}


def preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"unknown instrument preset {name!r}")
    p = copy.deepcopy(PRESETS[name])
    p["preset"] = name
    return p


def render_notes(instrument: dict, notes: list[dict], bpm: float, n: int, sr: int, seed: int = 0) -> np.ndarray:
    engine = instrument.get("engine")
    params = instrument.get("params", {})
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}")
    if engine in ("kick", "snare", "clap", "hat", "perc"):
        return dsp.as_stereo(_render_drums(engine, notes, params, bpm, n, sr, seed))
    if engine == "808":
        return dsp.as_stereo(_render_808(notes, params, bpm, n, sr, seed))
    if engine == "texture":
        return _render_texture(notes, params, bpm, n, sr, seed)
    return _render_poly(engine, notes, params, bpm, n, sr, seed)
