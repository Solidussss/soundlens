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

MAX_PINS = 28


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
    left = max(0, idx_local - 7)
    right = min(len(spec), idx_local + 8)
    neighborhood = np.concatenate([spec[left:idx_local], spec[idx_local + 1:right]])
    baseline = float(np.median(neighborhood)) if neighborhood.size else float(np.median(spec))
    peak = float(spec[idx_local])
    prominence = 20.0 * math.log10((peak + EPS) / (baseline + EPS))
    return float(freqs[idx]), max(0.0, prominence)


def _candidate_peaks(values: np.ndarray, min_strength: float, max_count: int, min_gap: int = 3) -> List[int]:
    """Return strongest local peaks without allowing a cluster of near-duplicates."""
    if values.size == 0:
        return []
    candidates = []
    for i in range(1, len(values) - 1):
        if values[i] >= min_strength and values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            candidates.append(i)
    candidates.sort(key=lambda i: float(values[i]), reverse=True)
    chosen: List[int] = []
    for idx in candidates:
        if all(abs(idx - old) >= min_gap for old in chosen):
            chosen.append(idx)
        if len(chosen) >= max_count:
            break
    return chosen


def _candidate_changes(values: np.ndarray, min_delta: float, max_count: int, min_gap: int = 3) -> List[int]:
    if values.size < 3:
        return []
    delta = np.abs(np.diff(values, prepend=values[0]))
    ranked = [i for i in np.argsort(delta)[::-1] if delta[i] >= min_delta]
    chosen: List[int] = []
    for idx in ranked:
        idx = int(idx)
        if all(abs(idx - old) >= min_gap for old in chosen):
            chosen.append(idx)
        if len(chosen) >= max_count:
            break
    return chosen



def _build_ai_timeline_from_existing_3d(duration, sr, hop, rms, centroid, rolloff, onset, zcr):
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    n = min(len(times), len(rms), len(centroid), len(rolloff), len(onset), len(zcr))
    if n <= 0:
        return {"enabled": False, "reason": "No aligned 3D frames."}
    times = times[:n]
    rms = np.asarray(rms[:n], dtype=float)
    centroid = np.asarray(centroid[:n], dtype=float)
    rolloff = np.asarray(rolloff[:n], dtype=float)
    onset = np.asarray(onset[:n], dtype=float)
    zcr = np.asarray(zcr[:n], dtype=float)

    def seg(a, b):
        idx = np.where((times >= duration*a) & (times <= duration*b))[0]
        if idx.size == 0:
            return {}
        return {
            "range_sec": [round(duration*a, 2), round(duration*b, 2)],
            "avg_rms": round(float(np.mean(rms[idx])), 6),
            "peak_rms": round(float(np.max(rms[idx])), 6),
            "brightness_hz": round(float(np.mean(centroid[idx])), 2),
            "rolloff_hz": round(float(np.mean(rolloff[idx])), 2),
            "zero_crossing": round(float(np.mean(zcr[idx])), 6),
            "onset_strength": round(float(np.mean(onset[idx])), 4),
        }

    loudest = int(np.argmax(rms))
    quietest = int(np.argmin(rms))
    return {
        "enabled": True,
        "duration_analyzed_sec": round(float(duration), 2),
        "overall": {
            "avg_rms": round(float(np.mean(rms)), 6),
            "rms_variation": round(float(np.std(rms)), 6),
            "avg_brightness_hz": round(float(np.mean(centroid)), 2),
            "avg_rolloff_hz": round(float(np.mean(rolloff)), 2),
            "avg_zero_crossing": round(float(np.mean(zcr)), 6),
            "avg_onset_strength": round(float(np.mean(onset)), 4),
        },
        "segments": {
            "intro": seg(0, .2),
            "early_middle": seg(.2, .45),
            "middle": seg(.45, .7),
            "ending": seg(.7, 1),
        },
        "moments": {
            "loudest_sec": round(float(times[loudest]), 2),
            "quietest_sec": round(float(times[quietest]), 2),
        },
        "source": "visual_map_reuse",
    }

def build_visual_map(audio_path: str | Path, max_slices: int = 320) -> Dict[str, Any]:
    """Build dense, time-resolved data for the SoundLens 3D experience.

    Pins describe measurable events in the WAV. They intentionally avoid claims
    about whether a creative choice is "good" or "bad". Up to ~28 high-value
    events are selected, with spacing so the 3D view does not become cluttered.
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

    complex_stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
    stft = np.abs(complex_stft)
    power = stft ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_count = stft.shape[1]

    rms = librosa.feature.rms(S=stft, frame_length=n_fft, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
    onset = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    zcr = librosa.feature.zero_crossing_rate(mono, frame_length=n_fft, hop_length=hop)[0]

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
        band_frames[name] = np.sum(power[idx], axis=0) if idx.size else np.zeros(frame_count)

    total_power = np.sum(power, axis=0) + EPS
    for key in list(band_frames):
        band_frames[key] = band_frames[key] / total_power

    # Around 2.5 visual samples/sec while keeping browser payload reasonable.
    slice_count = int(min(max_slices, max(64, math.ceil(duration * 2.5))))
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
    texture_n = _norm(flat_s)
    bass_mix = np.clip(band_s["sub"] + band_s["bass"], 0.0, 1.0)
    sub_n = _norm(band_s["sub"])
    bass_n = _norm(band_s["bass"])

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

    candidates: List[Dict[str, Any]] = []

    def candidate(kind: str, idx: int, title: str, detail: str, strength: float = 1.0,
                  priority: float = 1.0, extra: dict | None = None):
        idx = int(np.clip(idx, 0, slice_count - 1))
        item = {
            "kind": kind,
            "index": idx,
            "time": _safe(times[idx], 2),
            "title": title,
            "detail": detail,
            "strength": _safe(np.clip(strength, 0, 1), 3),
            "importance": _safe(np.clip(strength, 0, 1) * priority, 4),
        }
        if extra:
            item.update(extra)
        candidates.append(item)

    # 1) Energy: strongest regions + major jumps/drops.
    energy_peaks = _candidate_peaks(rms_n, 0.58, 5, min_gap=max(3, slice_count // 45))
    if not energy_peaks:
        energy_peaks = [int(np.argmax(rms_n))]
    for rank, idx in enumerate(energy_peaks[:4]):
        candidate(
            "energy", idx,
            "Energy peak" if rank == 0 else "High-energy region",
            f"Measured track energy is especially dense here ({rms_n[idx]*100:.0f}% relative intensity).",
            rms_n[idx], 1.15,
        )

    energy_delta = np.diff(rms_n, prepend=rms_n[0])
    for idx in _candidate_changes(rms_n, 0.22, 4, min_gap=max(4, slice_count // 40)):
        direction = "rises" if energy_delta[idx] >= 0 else "drops"
        candidate(
            "energy", idx, f"Energy {direction}",
            f"The overall level {direction} sharply at this point.",
            min(1.0, abs(energy_delta[idx]) * 2.3), 0.92,
        )

    # 2) Low end / 808: separate sub-heavy and bass-body moments, plus transitions.
    for rank, idx in enumerate(_candidate_peaks(bass_mix, float(np.percentile(bass_mix, 68)), 6,
                                                min_gap=max(3, slice_count // 48))[:4]):
        sub_share = float(band_s["sub"][idx])
        bass_share = float(band_s["bass"][idx])
        label = "808 / low-end focus" if rank == 0 else "Low-end event"
        candidate(
            "bass", idx, label,
            f"Sub + bass energy is concentrated here. Sub {sub_share*100:.1f}% · bass/body {bass_share*100:.1f}% of measured spectrum.",
            max(sub_n[idx], bass_n[idx]), 1.18,
            {"sub_share": _safe(sub_share, 4), "bass_share": _safe(bass_share, 4)},
        )

    bass_delta = np.diff(bass_mix, prepend=bass_mix[0])
    for idx in _candidate_changes(bass_mix, max(0.035, float(np.std(bass_mix) * 0.85)), 3,
                                  min_gap=max(4, slice_count // 42)):
        direction = "enters / expands" if bass_delta[idx] > 0 else "pulls back"
        candidate(
            "bass", idx, f"Low end {direction}",
            f"The 20–250 Hz share changes noticeably here.",
            min(1.0, abs(bass_delta[idx]) * 8.0), 0.95,
            {"sub_share": _safe(band_s["sub"][idx], 4), "bass_share": _safe(band_s["bass"][idx], 4)},
        )

    # 3) Transients: multiple genuinely strong rhythmic attacks.
    for rank, idx in enumerate(_candidate_peaks(onset_n, 0.60, 6, min_gap=max(3, slice_count // 55))[:4]):
        candidate(
            "transient", idx,
            "Hardest transient" if rank == 0 else "Strong transient",
            f"A sharp attack / rhythmic hit stands out here ({onset_n[idx]*100:.0f}% relative onset strength).",
            onset_n[idx], 1.02,
        )

    # 4) Stereo: strongest width + major opening/closing moments.
    if stereo_source and float(np.max(width_n)) > 0.12:
        for rank, idx in enumerate(_candidate_peaks(width_n, max(0.16, float(np.percentile(width_n, 70))), 4,
                                                    min_gap=max(4, slice_count // 44))[:3]):
            candidate(
                "stereo", idx,
                "Widest stereo moment" if rank == 0 else "Wide stereo region",
                f"Side information is strongest here ({width_n[idx]*100:.0f}% relative width).",
                width_n[idx], 1.03,
            )

        width_delta = np.diff(width_n, prepend=width_n[0])
        for idx in _candidate_changes(width_n, 0.18, 3, min_gap=max(5, slice_count // 40))[:2]:
            direction = "opens" if width_delta[idx] > 0 else "narrows"
            candidate(
                "stereo", idx, f"Stereo field {direction}",
                f"The left/right image {direction} noticeably at this point.",
                min(1.0, abs(width_delta[idx]) * 2.5), 0.92,
            )

    # 5) Tone / brightness changes. These are more useful than simply flagging the brightest frame.
    bright_delta = np.diff(brightness_n, prepend=brightness_n[0])
    for idx in _candidate_changes(brightness_n, 0.075, 5, min_gap=max(4, slice_count // 45))[:4]:
        direction = "brighter" if bright_delta[idx] >= 0 else "darker"
        candidate(
            "brightness", idx, "Tone shift",
            f"The spectral center moves {direction} here (centroid ~{cent_s[idx]:.0f} Hz).",
            min(1.0, abs(bright_delta[idx]) * 4.2), 0.93,
            {"centroid_hz": _safe(cent_s[idx], 1)},
        )

    # 6) Clipping: only report real digital-ceiling hits, and group nearby occurrences.
    clip_indices = np.where(clip_s > 0)[0]
    if clip_indices.size:
        groups: List[List[int]] = []
        current = [int(clip_indices[0])]
        for raw in clip_indices[1:]:
            idx = int(raw)
            if idx - current[-1] <= max(2, slice_count // 100):
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
        groups.append(current)
        ranked_groups = sorted(groups, key=lambda g: max(float(clip_s[i]) for i in g), reverse=True)
        for group in ranked_groups[:3]:
            idx = max(group, key=lambda i: float(clip_s[i]))
            candidate(
                "clip", idx, "Digital clipping detected",
                f"Samples touch the digital ceiling in this region ({clip_s[idx]*100:.3f}% of analyzed frame samples).",
                min(1.0, float(clip_s[idx]) * 80.0 + 0.35), 1.12,
                {"clip_fraction": _safe(clip_s[idx], 7)},
            )

    # 7) Narrow spectral concentrations / resonance-like focuses.
    resonance_candidates = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        segment_spec = np.mean(stft[:, int(a):max(int(a) + 1, int(b))], axis=1)
        hz, prominence = _local_peak_prominence(segment_spec, freqs)
        if prominence >= 8.5:
            resonance_candidates.append((prominence, i, hz))
    resonance_candidates.sort(reverse=True)
    chosen_res: List[int] = []
    for prom, idx, hz in resonance_candidates:
        if all(abs(idx - old) >= max(4, slice_count // 45) for old in chosen_res):
            chosen_res.append(idx)
            candidate(
                "resonance", idx, "Narrow frequency focus",
                f"A narrow spectral concentration stands ~{prom:.1f} dB above nearby frequencies around {hz:.0f} Hz.",
                min(1.0, prom / 18.0), 1.08,
                {"frequency_hz": round(hz, 1), "prominence_db": round(prom, 2)},
            )
        if len(chosen_res) >= 4:
            break

    # 8) Texture changes: measurable noisier/smoother spectral behavior. Reuse brightness pin style.
    texture_delta = np.diff(texture_n, prepend=texture_n[0])
    for idx in _candidate_changes(texture_n, 0.26, 3, min_gap=max(5, slice_count // 42))[:2]:
        direction = "grittier / noisier" if texture_delta[idx] > 0 else "smoother / more tonal"
        candidate(
            "brightness", idx, "Texture shift",
            f"Spectral texture becomes {direction} here.",
            min(1.0, abs(texture_delta[idx]) * 2.8), 0.78,
            {"spectral_flatness": _safe(flat_s[idx], 5)},
        )

    # Select only the most important events while keeping good coverage across time.
    # Start by ranking significance, then suppress near-identical time clusters.
    candidates.sort(key=lambda p: p["importance"], reverse=True)
    selected: List[Dict[str, Any]] = []
    min_time_gap = max(0.7, min(2.2, duration / 95.0))

    for item in candidates:
        # Different event types may coexist in the same musical moment, but avoid a pile-up.
        nearby = [p for p in selected if abs(float(p["time"]) - float(item["time"])) < min_time_gap]
        same_kind_nearby = [p for p in nearby if p["kind"] == item["kind"]]
        if same_kind_nearby:
            continue
        if len(nearby) >= 2:
            continue
        selected.append(item)
        if len(selected) >= MAX_PINS:
            break

    # Guarantee core measurements appear when actually detectable.
    core_kinds = ["energy", "bass", "transient"] + (["stereo"] if stereo_source else [])
    for kind in core_kinds:
        if not any(p["kind"] == kind for p in selected):
            fallback = next((p for p in candidates if p["kind"] == kind), None)
            if fallback:
                selected.append(fallback)

    # Final cap and chronological order. Remove internal ranking field from API output.
    selected = sorted(selected, key=lambda p: p["importance"], reverse=True)[:MAX_PINS]
    pins = []
    for item in sorted(selected, key=lambda p: p["time"]):
        item = dict(item)
        item.pop("importance", None)
        pins.append(item)

    ai_timeline_summary = _build_ai_timeline_from_existing_3d(
        duration, sr, hop, rms, centroid, rolloff, onset, zcr
    )

    return {
        "ai_timeline_summary": ai_timeline_summary,
        "version": 2,
        "duration": _safe(duration, 3),
        "sample_rate": int(sr),
        "stereo_source": stereo_source,
        "slice_count": slice_count,
        "bands": [{"id": name, "low_hz": lo, "high_hz": hi} for name, lo, hi in BANDS],
        "slices": slices,
        "pins": pins,
        "pin_count": len(pins),
        "legend": {
            "x": "time",
            "height": "energy + frequency-band intensity",
            "depth": "stereo width / frequency layer",
            "glow": "transient intensity",
            "surface_detail": "spectral texture",
        },
    }
