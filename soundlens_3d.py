from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import math

import librosa
import numpy as np

EPS = 1e-9

BANDS = [
    ("sub", 20, 80),
    ("bass", 80, 250),
    ("low_mid", 250, 1000),
    ("mid", 1000, 4000),
    ("high", 4000, 10000),
    ("air", 10000, 16000),
]


def _safe(v: float, digits: int = 4) -> float:
    if not np.isfinite(v):
        return 0.0
    return round(float(v), digits)


def _norm(values: np.ndarray, low_pct: float = 5, high_pct: float = 95) -> np.ndarray:
    values = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values
    lo = float(np.percentile(values, low_pct))
    hi = float(np.percentile(values, high_pct))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _aggregate(values: np.ndarray, starts: np.ndarray, ends: np.ndarray, reducer="mean") -> np.ndarray:
    out = []
    for a, b in zip(starts, ends):
        chunk = values[int(a):max(int(a) + 1, int(b))]
        if chunk.size == 0:
            out.append(0.0)
        elif reducer == "max":
            out.append(float(np.max(chunk)))
        else:
            out.append(float(np.mean(chunk)))
    return np.asarray(out, dtype=float)


def _local_peak_prominence(spectrum: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    valid = np.where((freqs >= 80) & (freqs <= 12000))[0]
    if valid.size < 10:
        return 0.0, 0.0
    spec = spectrum[valid]
    idx_local = int(np.argmax(spec))
    idx = int(valid[idx_local])
    left = max(0, idx_local - 6)
    right = min(len(spec), idx_local + 7)
    neighborhood = np.concatenate([spec[left:idx_local], spec[idx_local + 1:right]])
    baseline = float(np.median(neighborhood)) if neighborhood.size else float(np.median(spec))
    peak = float(spec[idx_local])
    prominence = 20.0 * math.log10((peak + EPS) / (baseline + EPS))
    return float(freqs[idx]), max(0.0, prominence)


def build_visual_map(audio_path: str | Path, max_slices: int = 280) -> Dict[str, Any]:
    """Build time-resolved data used by the SoundLens 3D Lab.

    This does not claim subjective mix correctness. It exposes measurable audio
    structure through time so the frontend can literally construct the track.
    """
    path = Path(audio_path)
    y, sr = librosa.load(path, sr=22050, mono=False)
    if y.size == 0:
        raise ValueError("Audio file is empty.")

    if y.ndim == 1:
        left = right = y.astype(float)
        mono = y.astype(float)
        stereo_source = False
    else:
        left = y[0].astype(float)
        right = y[1].astype(float) if y.shape[0] > 1 else y[0].astype(float)
        mono = np.mean(y[:2], axis=0).astype(float)
        stereo_source = True

    duration = float(len(mono) / sr)
    n_fft = 2048
    hop = 512

    stft = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop))
    power = stft ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_count = stft.shape[1]

    rms = librosa.feature.rms(S=stft, frame_length=n_fft, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    onset = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(S=stft)[0]

    side = (left - right) * 0.5
    mid = (left + right) * 0.5
    side_rms = librosa.feature.rms(y=side, frame_length=n_fft, hop_length=hop)[0]
    mid_rms = librosa.feature.rms(y=mid, frame_length=n_fft, hop_length=hop)[0]
    stereo_width = side_rms / (mid_rms + side_rms + EPS)
    if not stereo_source:
        stereo_width[:] = 0.0

    clip_frame = np.zeros(frame_count, dtype=float)
    abs_mono = np.abs(mono)
    frame_samples = librosa.util.frame(abs_mono, frame_length=n_fft, hop_length=hop)
    usable = min(frame_samples.shape[1], frame_count)
    clip_frame[:usable] = np.mean(frame_samples[:, :usable] >= 0.999, axis=0)

    band_frames: Dict[str, np.ndarray] = {}
    for name, lo, hi in BANDS:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size:
            band_frames[name] = np.sum(power[idx], axis=0)
        else:
            band_frames[name] = np.zeros(frame_count)

    total_power = np.sum(power, axis=0) + EPS
    for key in list(band_frames):
        band_frames[key] = band_frames[key] / total_power

    slice_count = int(min(max_slices, max(48, math.ceil(duration * 2.2))))
    starts = np.linspace(0, frame_count, slice_count, endpoint=False).astype(int)
    ends = np.concatenate([starts[1:], [frame_count]])
    times = (starts + np.maximum(1, ends - starts) / 2.0) * hop / sr

    rms_s = _aggregate(rms, starts, ends)
    onset_s = _aggregate(onset, starts, ends, reducer="max")
    cent_s = _aggregate(centroid, starts, ends)
    flat_s = _aggregate(flatness, starts, ends)
    width_s = _aggregate(stereo_width, starts, ends)
    clip_s = _aggregate(clip_frame, starts, ends, reducer="max")

    band_s = {k: _aggregate(v, starts, ends) for k, v in band_frames.items()}

    rms_n = _norm(rms_s)
    onset_n = _norm(onset_s)
    brightness_n = np.clip(cent_s / 9000.0, 0.0, 1.0)
    width_n = np.clip(width_s * 2.1, 0.0, 1.0)

    slices: List[Dict[str, Any]] = []
    for i in range(slice_count):
        bands = {k: _safe(v[i], 5) for k, v in band_s.items()}
        slices.append({
            "i": i,
            "t": _safe(times[i], 3),
            "energy": _safe(rms_n[i], 4),
            "transient": _safe(onset_n[i], 4),
            "brightness": _safe(brightness_n[i], 4),
            "centroid_hz": _safe(cent_s[i], 1),
            "stereo": _safe(width_n[i], 4),
            "texture": _safe(flat_s[i], 5),
            "clipping": _safe(clip_s[i], 6),
            "bands": bands,
        })

    pins: List[Dict[str, Any]] = []

    def add_pin(kind: str, idx: int, title: str, detail: str, strength: float = 1.0, extra: dict | None = None):
        idx = int(np.clip(idx, 0, slice_count - 1))
        item = {
            "kind": kind,
            "index": idx,
            "time": _safe(times[idx], 2),
            "title": title,
            "detail": detail,
            "strength": _safe(strength, 3),
        }
        if extra:
            item.update(extra)
        pins.append(item)

    energy_idx = int(np.argmax(rms_n))
    add_pin("energy", energy_idx, "Energy peak", "The densest/loudest energy region in the track.", rms_n[energy_idx])

    bass_mix = band_s["sub"] + band_s["bass"]
    bass_idx = int(np.argmax(bass_mix))
    add_pin(
        "bass", bass_idx, "808 / low-end focus",
        "This region carries the strongest combined sub and bass energy.",
        float(np.clip(bass_mix[bass_idx] * 2.5, 0, 1)),
        {"sub_share": _safe(band_s["sub"][bass_idx], 4), "bass_share": _safe(band_s["bass"][bass_idx], 4)},
    )

    transient_idx = int(np.argmax(onset_n))
    add_pin("transient", transient_idx, "Hardest transient", "The sharpest detected attack / rhythmic hit.", onset_n[transient_idx])

    if stereo_source:
        width_idx = int(np.argmax(width_n))
        if width_n[width_idx] > 0.12:
            add_pin("stereo", width_idx, "Widest stereo moment", "The left/right field opens most at this point.", width_n[width_idx])

    if np.max(clip_s) > 0:
        clip_idx = int(np.argmax(clip_s))
        add_pin("clip", clip_idx, "Clipping location", "Samples reach the digital ceiling in this region.", min(1.0, clip_s[clip_idx] * 50))

    # strongest narrow spectral peak across coarse slices
    resonance_candidates = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        segment_spec = np.mean(stft[:, int(a):max(int(a)+1, int(b))], axis=1)
        hz, prominence = _local_peak_prominence(segment_spec, freqs)
        if prominence >= 7.0:
            resonance_candidates.append((prominence, i, hz))
    resonance_candidates.sort(reverse=True)
    if resonance_candidates:
        prom, idx, hz = resonance_candidates[0]
        add_pin(
            "resonance", idx, "Narrow frequency focus",
            f"A narrow spectral peak stands out around {hz:.0f} Hz here.",
            min(1.0, prom / 18.0),
            {"frequency_hz": round(hz, 1), "prominence_db": round(prom, 2)},
        )

    # largest brightness change, not just brightest point
    if slice_count > 3:
        delta = np.abs(np.diff(brightness_n, prepend=brightness_n[0]))
        bright_idx = int(np.argmax(delta))
        if delta[bright_idx] > 0.08:
            direction = "brighter" if brightness_n[bright_idx] >= brightness_n[max(0, bright_idx-1)] else "darker"
            add_pin("brightness", bright_idx, "Tone shift", f"The spectrum moves noticeably {direction} here.", min(1.0, delta[bright_idx] * 4))

    # Dedupe pins clustered at nearly the same time, while preserving different high-value types.
    pins = sorted(pins, key=lambda p: p["strength"], reverse=True)[:8]
    pins = sorted(pins, key=lambda p: p["time"])

    return {
        "version": 1,
        "duration": _safe(duration, 3),
        "sample_rate": int(sr),
        "stereo_source": stereo_source,
        "slice_count": slice_count,
        "bands": [{"id": name, "low_hz": lo, "high_hz": hi} for name, lo, hi in BANDS],
        "slices": slices,
        "pins": pins,
        "legend": {
            "x": "time",
            "height": "energy + frequency-band intensity",
            "depth": "stereo width / frequency layer",
            "glow": "transient intensity",
            "surface_detail": "spectral texture",
        },
    }
