"""Real audio effects. Every effect is a pure function ``(audio, sr, **params) -> audio``.

Audio is ``float64`` with shape ``(2, n)`` (stereo). Effects are deterministic:
the same input and parameters always produce identical output, which is what
lets the studio prove an edit left other tracks untouched.

Non-linear effects (distortion, clipping) are oversampled to keep aliasing
down. Dynamics use block-rate envelope followers (64 samples ≈ 1.5 ms at
44.1 kHz) so they run in pure NumPy without a per-sample Python loop.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

Audio = np.ndarray


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float, floor: float = -120.0) -> float:
    return float(20.0 * math.log10(x)) if x > 0 else floor


def as_stereo(x: np.ndarray) -> Audio:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.vstack([x, x])
    if x.shape[0] == 1:
        return np.vstack([x[0], x[0]])
    if x.shape[0] != 2 and x.shape[-1] == 2:
        x = x.T
    return x[:2].copy() if x.shape[0] >= 2 else x


def _biquad(kind: str, freq: float, sr: int, q: float = 0.707, gain_db: float = 0.0) -> np.ndarray:
    """RBJ audio-EQ-cookbook biquad as a single SOS row."""
    freq = float(np.clip(freq, 10.0, sr * 0.49))
    a_gain = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * max(q, 1e-3))
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * a_gain, -2 * cw, 1 - alpha * a_gain]
        a = [1 + alpha / a_gain, -2 * cw, 1 - alpha / a_gain]
    elif kind in ("lowshelf", "highshelf"):
        sq = 2.0 * math.sqrt(a_gain) * alpha
        if kind == "lowshelf":
            b = [a_gain * ((a_gain + 1) - (a_gain - 1) * cw + sq),
                 2 * a_gain * ((a_gain - 1) - (a_gain + 1) * cw),
                 a_gain * ((a_gain + 1) - (a_gain - 1) * cw - sq)]
            a = [(a_gain + 1) + (a_gain - 1) * cw + sq,
                 -2 * ((a_gain - 1) + (a_gain + 1) * cw),
                 (a_gain + 1) + (a_gain - 1) * cw - sq]
        else:
            b = [a_gain * ((a_gain + 1) + (a_gain - 1) * cw + sq),
                 -2 * a_gain * ((a_gain - 1) + (a_gain + 1) * cw),
                 a_gain * ((a_gain + 1) + (a_gain - 1) * cw - sq)]
            a = [(a_gain + 1) - (a_gain - 1) * cw + sq,
                 2 * ((a_gain - 1) - (a_gain + 1) * cw),
                 (a_gain + 1) - (a_gain - 1) * cw - sq]
    else:
        raise ValueError(f"unknown filter type {kind!r}")
    b = np.asarray(b) / a[0]
    a = np.asarray(a) / a[0]
    return np.concatenate([b, a])[None, :]


def sos_filter(x: Audio, sos: np.ndarray) -> Audio:
    return signal.sosfilt(sos, x, axis=-1)


def _block_envelope(level: np.ndarray, sr: int, attack_ms: float, release_ms: float, block: int = 64) -> np.ndarray:
    """Attack/release smoothing of a non-negative control signal at block rate.

    ``level`` is per-sample; returns a per-sample smoothed envelope.
    Uses the block maximum so fast transients are never missed.
    """
    n = level.shape[-1]
    nb = int(math.ceil(n / block))
    pad = nb * block - n
    lv = np.pad(level, (0, pad)).reshape(nb, block).max(axis=1)
    block_sr = sr / block
    a_att = math.exp(-1.0 / max(attack_ms * 1e-3 * block_sr, 1e-6))
    a_rel = math.exp(-1.0 / max(release_ms * 1e-3 * block_sr, 1e-6))
    env = np.empty(nb)
    e = 0.0
    for i in range(nb):  # nb ≈ 700 per second of audio: cheap
        v = lv[i]
        coeff = a_att if v > e else a_rel
        e = coeff * e + (1.0 - coeff) * v
        env[i] = e
    # linear interpolation back to sample rate (block centres)
    centres = (np.arange(nb) + 0.5) * block
    return np.interp(np.arange(n), centres, env)


def _oversampled(x: Audio, factor: int, fn: Callable[[Audio], Audio]) -> Audio:
    if factor <= 1:
        return fn(x)
    up = signal.resample_poly(x, factor, 1, axis=-1)
    y = fn(up)
    return signal.resample_poly(y, 1, factor, axis=-1)[..., : x.shape[-1]]


# ---------------------------------------------------------------------------
# effects
# ---------------------------------------------------------------------------

def gain(x: Audio, sr: int, db: float = 0.0) -> Audio:
    return x * db_to_lin(db)


def eq(x: Audio, sr: int, bands: list[dict] | None = None) -> Audio:
    """Parametric EQ. Each band: {type, freq, gain_db, q}."""
    if not bands:
        return x
    sos = np.vstack([
        _biquad(b.get("type", "peak"), float(b.get("freq", 1000.0)), sr,
                float(b.get("q", 0.707)), float(b.get("gain_db", 0.0)))
        for b in bands
    ])
    return sos_filter(x, sos)


def filter_(x: Audio, sr: int, mode: str = "lowpass", cutoff: float = 1000.0, resonance: float = 0.707, slope: int = 12) -> Audio:
    stages = max(1, int(slope) // 12)
    sos = np.vstack([_biquad(mode, cutoff, sr, resonance if i == 0 else 0.707) for i in range(stages)])
    return sos_filter(x, sos)


def _shape(x: np.ndarray, mode: str, bias: float) -> np.ndarray:
    if mode == "soft":
        return np.tanh(x + bias) - math.tanh(bias)
    if mode == "hard":
        return np.clip(x + bias, -1.0, 1.0) - float(np.clip(bias, -1, 1))
    if mode == "foldback":
        # triangle-wave folding: repeatedly reflects anything past ±1
        y = x + bias
        return (np.abs(np.mod(y - 1.0, 4.0) - 2.0) - 1.0) - ((abs((bias - 1.0) % 4.0 - 2.0)) - 1.0)
    if mode == "tube":
        # asymmetric soft clip -> even harmonics
        y = x + bias
        pos = 1.0 - np.exp(-np.maximum(y, 0.0))
        neg = -(1.0 - np.exp(np.minimum(y, 0.0) * 1.4)) / 1.4
        return np.where(y >= 0, pos, neg)
    raise ValueError(f"unknown distortion mode {mode!r}")


def distortion(x: Audio, sr: int, drive_db: float = 12.0, mode: str = "soft", mix: float = 1.0,
               output_db: float = 0.0, tone_hz: float = 0.0, bias: float = 0.0, oversample: int = 4,
               keep_sub_hz: float = 0.0) -> Audio:
    """Waveshaping distortion.

    ``keep_sub_hz`` splits the signal and leaves everything below that frequency
    clean (common for 808s: grit on top, solid sub underneath).
    ``tone_hz`` > 0 low-passes the distorted signal to tame fizz.
    """
    dry = x
    src = x
    sub = None
    if keep_sub_hz > 0:
        lp = np.vstack([_biquad("lowpass", keep_sub_hz, sr), _biquad("lowpass", keep_sub_hz, sr)])
        sub = sos_filter(x, lp)
        src = x - sub
    k = db_to_lin(drive_db)
    wet = _oversampled(src, int(oversample), lambda s: _shape(s * k, mode, bias))
    if tone_hz > 0:
        wet = filter_(wet, sr, "lowpass", tone_hz, 0.707, 12)
    # loose level matching so "more drive" isn't just "louder"
    rms_in = float(np.sqrt(np.mean(src ** 2)) + 1e-12)
    rms_out = float(np.sqrt(np.mean(wet ** 2)) + 1e-12)
    wet = wet * min(rms_in / rms_out * 1.6, 4.0) if rms_out > 0 else wet
    if sub is not None:
        wet = wet + sub
    y = (1.0 - mix) * dry + mix * wet
    return y * db_to_lin(output_db)


def saturation(x: Audio, sr: int, amount: float = 0.4, mix: float = 1.0) -> Audio:
    """Gentle tape/tube-style saturation (amount 0..1)."""
    amount = float(np.clip(amount, 0.0, 1.0))
    drive = 1.0 + amount * 5.0
    wet = _oversampled(x, 2, lambda s: _shape(s * drive, "tube", 0.0) / _shape(np.array([drive]), "tube", 0.0)[0])
    return (1.0 - mix) * x + mix * wet


def clipper(x: Audio, sr: int, ceiling_db: float = -0.3, drive_db: float = 0.0, softness: float = 0.0, oversample: int = 4) -> Audio:
    """Intentional clipper (hard by default). ``softness`` 0..1 blends toward tanh."""
    c = db_to_lin(ceiling_db)
    k = db_to_lin(drive_db)

    def fn(s):
        s = s * k / c
        hard = np.clip(s, -1.0, 1.0)
        if softness <= 0:
            return hard * c
        return ((1 - softness) * hard + softness * np.tanh(s)) * c

    y = _oversampled(x, int(oversample), fn)
    return np.clip(y, -c, c)  # resampling filters can ring past the ceiling


def compressor(x: Audio, sr: int, threshold_db: float = -18.0, ratio: float = 4.0, attack_ms: float = 10.0,
               release_ms: float = 120.0, makeup_db: float = 0.0, knee_db: float = 6.0, mix: float = 1.0,
               sidechain: Audio | None = None) -> Audio:
    det = sidechain if sidechain is not None else x
    level = np.max(np.abs(det), axis=0)
    env = _block_envelope(level, sr, attack_ms, release_ms)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    over = env_db - threshold_db
    if knee_db > 0:
        gr = np.where(over <= -knee_db / 2, 0.0,
             np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                      (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    else:
        gr = np.maximum(over, 0.0) * (1 - 1 / ratio)
    g = 10.0 ** ((-gr + makeup_db) / 20.0)
    wet = x * g[None, :]
    return (1.0 - mix) * x + mix * wet


def limiter(x: Audio, sr: int, ceiling_db: float = -1.0, release_ms: float = 80.0, lookahead_ms: float = 3.0,
            input_db: float = 0.0) -> Audio:
    """Look-ahead brickwall peak limiter (sample-peak)."""
    x = x * db_to_lin(input_db)
    c = db_to_lin(ceiling_db)
    la = max(1, int(lookahead_ms * 1e-3 * sr))
    peak = np.max(np.abs(x), axis=0)
    need = np.minimum(1.0, c / np.maximum(peak, 1e-12))
    # gain must already be down when the peak arrives: min over the look-ahead window
    need = minimum_filter1d(need, size=2 * la + 1, mode="nearest")
    reduction = 1.0 - need  # 0 = no limiting
    smooth = _block_envelope(reduction, sr, attack_ms=0.01, release_ms=release_ms, block=16)
    g = 1.0 - np.maximum(smooth, reduction)
    y = x * g[None, :]
    return np.clip(y, -c, c)


def _reverb_ir(sr: int, decay_s: float, damping: float, predelay_ms: float, width: float, seed: int) -> Audio:
    rng = np.random.default_rng(seed)
    length = int(sr * min(max(decay_s * 1.2, 0.2), 8.0))
    t = np.arange(length) / sr
    ir = np.zeros((2, length))
    # three bands with different decay: highs die faster as damping rises
    bands = [("lowpass", 800.0, 1.0), ("bandpass", 2500.0, 1.0 - 0.45 * damping), ("highpass", 6000.0, 1.0 - 0.8 * damping)]
    for kind, f, decay_scale in bands:
        noise = rng.standard_normal((2, length))
        noise = sos_filter(noise, _biquad(kind, f, sr, 0.7))
        rt = max(decay_s * max(decay_scale, 0.08), 0.05)
        ir += noise * np.exp(-6.91 * t / rt)[None, :]  # -60 dB at rt
    # decorrelation control: blend toward mono for narrow reverbs
    mid = ir.mean(axis=0, keepdims=True)
    ir = mid + (ir - mid) * width
    # early reflections
    for k in range(6):
        d = int(sr * (0.007 + 0.011 * k + rng.uniform(0, 0.004)))
        if d < length:
            ir[k % 2, d] += 0.6 * (0.8 ** k)
    pre = int(sr * predelay_ms * 1e-3)
    ir = np.pad(ir, ((0, 0), (pre, 0)))
    ir /= np.sqrt(np.sum(ir ** 2) / 2) + 1e-12
    return ir


def reverb(x: Audio, sr: int, mix: float = 0.25, decay_s: float = 1.8, damping: float = 0.5, predelay_ms: float = 15.0,
           width: float = 1.0, seed: int = 7, keep_tail: bool = False) -> Audio:
    ir = _reverb_ir(sr, decay_s, damping, predelay_ms, width, seed)
    wet = np.vstack([signal.oaconvolve(x[0], ir[0]), signal.oaconvolve(x[1], ir[1])])
    if not keep_tail:
        wet = wet[:, : x.shape[-1]]
    else:
        x = np.pad(x, ((0, 0), (0, wet.shape[-1] - x.shape[-1])))
    return (1.0 - mix) * x + mix * wet * 0.5


def delay(x: Audio, sr: int, time_ms: float = 375.0, feedback: float = 0.35, mix: float = 0.25,
          ping_pong: bool = True, tone_hz: float = 5000.0, bpm: float | None = None, beats: float | None = None) -> Audio:
    if beats is not None and bpm:
        time_ms = 60000.0 / float(bpm) * float(beats)
    d = max(1, int(time_ms * 1e-3 * sr))
    n = x.shape[-1]
    wet = np.zeros_like(x)
    tap = x
    amp = 1.0
    feedback = float(np.clip(feedback, 0.0, 0.95))
    k = 0
    lp = _biquad("lowpass", tone_hz, sr) if tone_hz > 0 else None
    while amp > 1e-3 and (k + 1) * d < n and k < 64:
        k += 1
        if lp is not None:
            tap = sos_filter(tap, lp)
        shifted = np.zeros_like(x)
        shifted[:, k * d:] = tap[:, : n - k * d]
        if ping_pong and k % 2 == 1:
            shifted = shifted[::-1]
        wet += shifted * amp
        amp *= feedback
    return (1.0 - mix) * x + mix * wet


def stereo(x: Audio, sr: int, width: float = 1.0, pan: float = 0.0) -> Audio:
    """M/S width (0 = mono, 1 = unchanged, 2 = extra wide) then constant-power pan (-1..1)."""
    m = (x[0] + x[1]) * 0.5
    s = (x[0] - x[1]) * 0.5 * width
    y = np.vstack([m + s, m - s])
    return pan_audio(y, pan)


def pan_audio(x: Audio, pan: float) -> Audio:
    pan = float(np.clip(pan, -1.0, 1.0))
    if pan == 0.0:
        return x
    ang = (pan + 1.0) * math.pi / 4.0
    lg, rg = math.cos(ang) * math.sqrt(2), math.sin(ang) * math.sqrt(2)
    return np.vstack([x[0] * lg, x[1] * rg])


def bitcrush(x: Audio, sr: int, bits: int = 8, downsample: int = 1, mix: float = 1.0) -> Audio:
    q = 2.0 ** (int(bits) - 1)
    y = np.round(x * q) / q
    if downsample > 1:
        y = np.repeat(y[:, ::downsample], downsample, axis=1)[:, : x.shape[-1]]
    return (1 - mix) * x + mix * y


# ---------------------------------------------------------------------------
# registry (the action system validates against this)
# ---------------------------------------------------------------------------

EFFECTS: dict[str, dict[str, Any]] = {
    "gain": {"fn": gain, "params": {"db": (-60.0, 24.0, 0.0)}},
    "eq": {"fn": eq, "params": {}},  # bands validated separately
    "filter": {"fn": filter_, "params": {"cutoff": (20.0, 20000.0, 1000.0), "resonance": (0.3, 12.0, 0.707), "slope": (12, 48, 12)},
               "enums": {"mode": ["lowpass", "highpass", "bandpass"]}},
    "distortion": {"fn": distortion, "params": {"drive_db": (0.0, 48.0, 12.0), "mix": (0.0, 1.0, 1.0), "output_db": (-24.0, 12.0, 0.0),
                                                 "tone_hz": (0.0, 20000.0, 0.0), "bias": (-0.5, 0.5, 0.0), "oversample": (1, 8, 4),
                                                 "keep_sub_hz": (0.0, 300.0, 0.0)},
                   "enums": {"mode": ["soft", "hard", "foldback", "tube"]}},
    "saturation": {"fn": saturation, "params": {"amount": (0.0, 1.0, 0.4), "mix": (0.0, 1.0, 1.0)}},
    "clipper": {"fn": clipper, "params": {"ceiling_db": (-24.0, 0.0, -0.3), "drive_db": (0.0, 36.0, 0.0), "softness": (0.0, 1.0, 0.0), "oversample": (1, 8, 4)}},
    "compressor": {"fn": compressor, "params": {"threshold_db": (-60.0, 0.0, -18.0), "ratio": (1.0, 30.0, 4.0), "attack_ms": (0.1, 200.0, 10.0),
                                                 "release_ms": (5.0, 2000.0, 120.0), "makeup_db": (-12.0, 24.0, 0.0), "knee_db": (0.0, 24.0, 6.0), "mix": (0.0, 1.0, 1.0)}},
    "limiter": {"fn": limiter, "params": {"ceiling_db": (-12.0, 0.0, -1.0), "release_ms": (5.0, 1000.0, 80.0), "lookahead_ms": (0.5, 10.0, 3.0), "input_db": (-12.0, 24.0, 0.0)}},
    "reverb": {"fn": reverb, "params": {"mix": (0.0, 1.0, 0.25), "decay_s": (0.1, 8.0, 1.8), "damping": (0.0, 1.0, 0.5), "predelay_ms": (0.0, 200.0, 15.0), "width": (0.0, 1.5, 1.0)}},
    "delay": {"fn": delay, "params": {"time_ms": (1.0, 2000.0, 375.0), "feedback": (0.0, 0.95, 0.35), "mix": (0.0, 1.0, 0.25), "tone_hz": (0.0, 20000.0, 5000.0),
                                       "beats": (0.0625, 4.0, 0.75)}, "bools": ["ping_pong"]},
    "stereo": {"fn": stereo, "params": {"width": (0.0, 2.0, 1.0), "pan": (-1.0, 1.0, 0.0)}},
    "bitcrush": {"fn": bitcrush, "params": {"bits": (2, 16, 8), "downsample": (1, 32, 1), "mix": (0.0, 1.0, 1.0)}},
}

_TEMPO_AWARE = {"delay"}


def validate_effect(fx: dict) -> dict:
    """Return a cleaned copy of an effect spec or raise ValueError."""
    kind = fx.get("type")
    if kind not in EFFECTS:
        raise ValueError(f"unknown effect {kind!r}; available: {sorted(EFFECTS)}")
    spec = EFFECTS[kind]
    params_in = dict(fx.get("params") or {})
    out: dict[str, Any] = {}
    for name, value in params_in.items():
        if name in spec["params"]:
            lo, hi, _ = spec["params"][name]
            v = float(value)
            if not (lo <= v <= hi):
                raise ValueError(f"{kind}.{name}={v} outside [{lo}, {hi}]")
            out[name] = int(v) if isinstance(lo, int) and isinstance(hi, int) else v
        elif name in spec.get("enums", {}):
            if value not in spec["enums"][name]:
                raise ValueError(f"{kind}.{name} must be one of {spec['enums'][name]}")
            out[name] = value
        elif name in spec.get("bools", []):
            out[name] = bool(value)
        elif kind == "eq" and name == "bands":
            bands = []
            for b in value or []:
                t = b.get("type", "peak")
                if t not in ("peak", "lowshelf", "highshelf", "lowpass", "highpass", "bandpass"):
                    raise ValueError(f"bad eq band type {t!r}")
                f = float(b.get("freq", 1000.0)); g = float(b.get("gain_db", 0.0)); q = float(b.get("q", 0.707))
                if not (20 <= f <= 20000 and -24 <= g <= 24 and 0.1 <= q <= 18):
                    raise ValueError(f"eq band out of range: {b}")
                bands.append({"type": t, "freq": f, "gain_db": g, "q": q})
            out["bands"] = bands
        else:
            raise ValueError(f"unknown parameter {name!r} for effect {kind!r}")
    return {"type": kind, "params": out, "enabled": bool(fx.get("enabled", True)), "id": fx.get("id")}


def apply_chain(x: Audio, sr: int, chain: list[dict], bpm: float | None = None) -> Audio:
    for fx in chain or []:
        if not fx.get("enabled", True):
            continue
        spec = EFFECTS[fx["type"]]
        params = dict(fx.get("params") or {})
        if fx["type"] in _TEMPO_AWARE and "beats" in params:
            params["bpm"] = bpm
        x = spec["fn"](x, sr, **params)
    return x
