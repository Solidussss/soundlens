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

MAX_PINS = 14


def _safe(v: float, digits: int = 4) -> float:
    if not np.isfinite(v):
        return 0.0
    return round(float(v), digits)


def _norm(values: np.ndarray, low_pct: float = 5, high_pct: float = 95) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values
    lo = float(np.percentile(values, low_pct))
    hi = float(np.percentile(values, high_pct))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _smooth(values: np.ndarray, radius: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 3 or radius <= 0:
        return values.copy()
    width = radius * 2 + 1
    kernel = np.ones(width, dtype=float) / width
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _aggregate(values: np.ndarray, starts: np.ndarray, ends: np.ndarray, reducer: str = "mean") -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out: List[float] = []
    for a, b in zip(starts, ends):
        a = int(np.clip(a, 0, max(0, len(values) - 1)))
        b = int(np.clip(max(a + 1, b), a + 1, len(values)))
        chunk = values[a:b]
        if chunk.size == 0:
            out.append(0.0)
        elif reducer == "max":
            out.append(float(np.max(chunk)))
        else:
            out.append(float(np.mean(chunk)))
    return np.asarray(out, dtype=float)


def _local_delta(values: np.ndarray, idx: int, lookback: int) -> float:
    if idx <= 0 or values.size == 0:
        return 0.0
    a = max(0, idx - lookback)
    baseline = float(np.mean(values[a:idx])) if idx > a else float(values[idx])
    return float(values[idx] - baseline)


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
    if values.size == 0:
        return []
    candidates: List[int] = []
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


def _detect_sections(feature_matrix: np.ndarray, times: np.ndarray, duration: float) -> tuple[List[Dict[str, Any]], List[int]]:
    """Detect coarse musical sections from multi-feature change, not fixed percentages."""
    n = feature_matrix.shape[1] if feature_matrix.ndim == 2 else 0
    if n < 12:
        return [{"id": "section_1", "label": "Section 1", "start": 0.0, "end": _safe(duration, 2)}], []

    normed = []
    for row in feature_matrix:
        normed.append(_norm(row, 8, 92))
    mat = np.vstack(normed)
    mat = np.vstack([_smooth(row, 2) for row in mat])

    win = max(2, min(7, n // 35))
    novelty = np.zeros(n, dtype=float)
    for i in range(win, n - win):
        before = np.mean(mat[:, i - win:i], axis=1)
        after = np.mean(mat[:, i:i + win], axis=1)
        novelty[i] = float(np.mean(np.abs(after - before)))
    novelty = _smooth(novelty, 1)

    if float(np.max(novelty)) <= EPS:
        boundaries: List[int] = []
    else:
        threshold = max(float(np.percentile(novelty, 82)), float(np.mean(novelty) + 0.7 * np.std(novelty)))
        ranked = [int(i) for i in np.argsort(novelty)[::-1] if novelty[i] >= threshold]
        min_gap = max(5, n // 10)
        boundaries = []
        for idx in ranked:
            if idx < min_gap or idx > n - min_gap:
                continue
            if all(abs(idx - old) >= min_gap for old in boundaries):
                boundaries.append(idx)
            if len(boundaries) >= 6:
                break
        boundaries.sort()

    edges = [0] + boundaries + [n - 1]
    sections: List[Dict[str, Any]] = []
    for j in range(len(edges) - 1):
        a = edges[j]
        b = edges[j + 1]
        start = 0.0 if j == 0 else float(times[a])
        end = duration if j == len(edges) - 2 else float(times[b])
        sections.append({
            "id": f"section_{j + 1}",
            "label": f"Section {j + 1}",
            "start": _safe(start, 2),
            "end": _safe(max(start, end), 2),
            "boundary_strength": _safe(novelty[a] if a < len(novelty) else 0.0, 4),
        })
    return sections, boundaries


def _section_for_index(sections: List[Dict[str, Any]], t: float) -> Dict[str, Any]:
    for section in sections:
        if float(section["start"]) <= t <= float(section["end"]) + 1e-6:
            return section
    return sections[-1] if sections else {"id": "section_1", "label": "Section 1", "start": 0.0, "end": t}


def _fuse_candidates(
    candidates: List[Dict[str, Any]],
    times: np.ndarray,
    sections: List[Dict[str, Any]],
    duration: float,
) -> List[Dict[str, Any]]:
    """Fuse nearby measurements into one musically meaningful event."""
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda x: float(x["time"]))
    fuse_window = max(0.45, min(1.15, duration / 150.0))
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [candidates[0]]

    for item in candidates[1:]:
        if float(item["time"]) - float(current[-1]["time"]) <= fuse_window:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    groups.append(current)

    fused: List[Dict[str, Any]] = []
    type_weight = {
        "section": 1.35,
        "clip": 1.30,
        "bass": 1.18,
        "energy": 1.12,
        "transient": 1.00,
        "stereo": 0.95,
        "brightness": 0.90,
        "resonance": 0.95,
        "texture": 0.80,
    }

    for group in groups:
        by_kind: Dict[str, Dict[str, Any]] = {}
        for item in group:
            kind = str(item["kind"])
            score = float(item.get("evidence", 0.0)) * type_weight.get(kind, 1.0)
            old = by_kind.get(kind)
            if old is None or score > float(old.get("_weighted", 0.0)):
                clone = dict(item)
                clone["_weighted"] = score
                by_kind[kind] = clone

        evidence = list(by_kind.values())
        weights = np.asarray([max(0.05, float(e.get("_weighted", 0.0))) for e in evidence], dtype=float)
        event_time = float(np.average([float(e["time"]) for e in evidence], weights=weights))
        idx = int(np.argmin(np.abs(times - event_time)))
        section = _section_for_index(sections, event_time)

        kinds = set(by_kind)
        if "clip" in kinds:
            title = "Digital ceiling hit"
            primary_kind = "clip"
        elif "section" in kinds and ({"energy", "bass", "transient"} & kinds):
            title = "Section impact"
            primary_kind = "energy" if "energy" in kinds else "bass"
        elif "bass" in kinds and "energy" in kinds and "transient" in kinds:
            title = "Low-end impact"
            primary_kind = "bass"
        elif "bass" in kinds and "energy" in kinds:
            title = "Low-end energy shift"
            primary_kind = "bass"
        elif "energy" in kinds and "transient" in kinds:
            title = "Energy impact"
            primary_kind = "energy"
        elif "stereo" in kinds and ("energy" in kinds or "section" in kinds):
            title = "Spatial transition"
            primary_kind = "stereo"
        elif "brightness" in kinds and ("section" in kinds or "energy" in kinds):
            title = "Tonal transition"
            primary_kind = "brightness"
        else:
            strongest = max(evidence, key=lambda e: float(e.get("_weighted", 0.0)))
            title = str(strongest.get("title") or "Measured event")
            primary_kind = str(strongest["kind"])

        independent = len(kinds)
        strongest_score = max(float(e.get("evidence", 0.0)) for e in evidence)
        agreement_bonus = min(0.28, max(0, independent - 1) * 0.09)
        confidence = float(np.clip(0.48 + 0.38 * strongest_score + agreement_bonus, 0.0, 0.99))
        importance = float(np.clip(
            0.50 * strongest_score
            + 0.12 * min(independent, 4)
            + (0.12 if "section" in kinds else 0.0)
            + (0.06 if {"bass", "energy"} <= kinds else 0.0),
            0.0,
            1.0,
        ))

        parts: List[str] = []
        for kind in ("section", "bass", "energy", "transient", "stereo", "brightness", "clip", "resonance", "texture"):
            item = by_kind.get(kind)
            if item and item.get("summary"):
                parts.append(str(item["summary"]))
        detail = " · ".join(parts[:4])
        if not detail:
            detail = str(max(evidence, key=lambda e: float(e.get("_weighted", 0.0))).get("detail") or "")

        fused.append({
            "kind": primary_kind,
            "index": idx,
            "time": _safe(event_time, 2),
            "title": title,
            "detail": detail,
            "strength": _safe(strongest_score, 3),
            "confidence": _safe(confidence, 3),
            "importance": _safe(importance, 4),
            "section_id": section.get("id"),
            "section_label": section.get("label"),
            "evidence_types": sorted(kinds),
            "evidence_count": independent,
            "evidence": [
                {
                    "kind": e["kind"],
                    "value": _safe(float(e.get("evidence", 0.0)), 3),
                    "summary": e.get("summary", ""),
                }
                for e in sorted(evidence, key=lambda x: float(x.get("_weighted", 0.0)), reverse=True)
            ],
        })

    ranked = sorted(fused, key=lambda e: (float(e["importance"]), float(e["confidence"])), reverse=True)
    chosen: List[Dict[str, Any]] = []
    min_gap = max(1.2, min(3.0, duration / 65.0))
    section_counts: Dict[str, int] = {}

    for event in ranked:
        if float(event["confidence"]) < 0.58:
            continue
        if any(abs(float(event["time"]) - float(old["time"])) < min_gap for old in chosen):
            continue
        sid = str(event.get("section_id") or "")
        if section_counts.get(sid, 0) >= 3:
            continue
        chosen.append(event)
        section_counts[sid] = section_counts.get(sid, 0) + 1
        if len(chosen) >= MAX_PINS:
            break

    return sorted(chosen, key=lambda e: float(e["time"]))


def _build_ai_timeline(
    duration: float,
    sections: List[Dict[str, Any]],
    pins: List[Dict[str, Any]],
    slices: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "duration_analyzed_sec": _safe(duration, 2),
        "sections": sections,
        "events": pins,
        "event_count": len(pins),
        "source": "musical_event_model_v3",
        "guidance": (
            "Use these fused events as the authoritative timeline. Each event combines aligned "
            "measurements and includes confidence, evidence types, and section context."
        ),
    }


def build_visual_map(audio_path: str | Path, max_slices: int = 320) -> Dict[str, Any]:
    """Build SoundLens 3D data with section-aware fused musical events."""
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

    stft_complex = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
    stft = np.abs(stft_complex)
    power = stft ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_count = stft.shape[1]

    rms = librosa.feature.rms(S=stft, frame_length=n_fft, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
    onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(stft + EPS, ref=np.max), sr=sr, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(S=stft)[0]

    side = (left - right) * 0.5
    mid = (left + right) * 0.5
    side_rms = librosa.feature.rms(y=side, frame_length=n_fft, hop_length=hop)[0]
    mid_rms = librosa.feature.rms(y=mid, frame_length=n_fft, hop_length=hop)[0]
    stereo_width = side_rms / (mid_rms + side_rms + EPS)
    if not stereo_source:
        stereo_width[:] = 0.0

    aligned_n = min(frame_count, len(rms), len(centroid), len(rolloff), len(onset), len(flatness), len(stereo_width))
    stft = stft[:, :aligned_n]
    power = power[:, :aligned_n]
    rms = rms[:aligned_n]
    centroid = centroid[:aligned_n]
    rolloff = rolloff[:aligned_n]
    onset = onset[:aligned_n]
    flatness = flatness[:aligned_n]
    stereo_width = stereo_width[:aligned_n]
    frame_count = aligned_n

    clip_frame = np.zeros(frame_count, dtype=float)
    abs_mono = np.abs(mono)
    if len(abs_mono) >= n_fft:
        frame_samples = librosa.util.frame(abs_mono, frame_length=n_fft, hop_length=hop)
        usable = min(frame_samples.shape[1], frame_count)
        clip_frame[:usable] = np.mean(frame_samples[:, :usable] >= 0.999, axis=0)

    band_frames: Dict[str, np.ndarray] = {}
    total_power = np.sum(power, axis=0) + EPS
    for name, lo, hi in BANDS:
        band_idx = np.where((freqs >= lo) & (freqs < hi))[0]
        raw = np.sum(power[band_idx], axis=0) if band_idx.size else np.zeros(frame_count)
        band_frames[name] = raw / total_power

    slice_count = int(min(max_slices, max(64, math.ceil(max(duration, 1.0) * 2.5))))
    slice_count = min(slice_count, max(1, frame_count))
    starts = np.linspace(0, frame_count, slice_count, endpoint=False).astype(int)
    ends = np.concatenate([starts[1:], [frame_count]])
    times = (starts + np.maximum(1, ends - starts) / 2.0) * hop / sr

    rms_s = _aggregate(rms, starts, ends)
    onset_s = _aggregate(onset, starts, ends, reducer="max")
    cent_s = _aggregate(centroid, starts, ends)
    rolloff_s = _aggregate(rolloff, starts, ends)
    flat_s = _aggregate(flatness, starts, ends)
    width_s = _aggregate(stereo_width, starts, ends)
    clip_s = _aggregate(clip_frame, starts, ends, reducer="max")
    band_s = {k: _aggregate(v, starts, ends) for k, v in band_frames.items()}

    rms_n = _norm(_smooth(rms_s, 1))
    onset_n = _norm(onset_s)
    brightness_n = _norm(_smooth(cent_s, 1))
    width_n = _norm(_smooth(width_s, 1)) if stereo_source else np.zeros_like(width_s)
    texture_n = _norm(_smooth(flat_s, 1))
    bass_mix = np.clip(band_s["sub"] + band_s["bass"], 0.0, 1.0)
    bass_n = _norm(_smooth(bass_mix, 1))

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
            "rolloff_hz": _safe(rolloff_s[i], 1),
            "stereo": _safe(width_n[i], 4),
            "texture": _safe(texture_n[i], 5),
            "clipping": _safe(clip_s[i], 6),
            "bands": bands,
        })

    feature_matrix = np.vstack([
        rms_n,
        bass_n,
        onset_n,
        brightness_n,
        width_n,
        texture_n,
    ])
    sections, boundary_indices = _detect_sections(feature_matrix, times, duration)

    candidates: List[Dict[str, Any]] = []

    def add(kind: str, idx: int, title: str, evidence: float, summary: str, detail: str = "") -> None:
        idx = int(np.clip(idx, 0, slice_count - 1))
        candidates.append({
            "kind": kind,
            "index": idx,
            "time": _safe(times[idx], 3),
            "title": title,
            "evidence": float(np.clip(evidence, 0.0, 1.0)),
            "summary": summary,
            "detail": detail or summary,
        })

    for idx in boundary_indices:
        add("section", idx, "Section transition", 0.82, "structural transition")

    lookback = max(3, slice_count // 50)

    for idx in _candidate_peaks(rms_n, max(0.56, float(np.percentile(rms_n, 72))), 8, max(3, slice_count // 55)):
        local = _local_delta(rms_n, idx, lookback)
        score = max(float(rms_n[idx]) * 0.72, min(1.0, max(0.0, local) * 2.8))
        add("energy", idx, "Energy peak", score, f"energy {rms_n[idx]*100:.0f}%")

    energy_change = np.asarray([_local_delta(rms_n, i, lookback) for i in range(slice_count)])
    for idx in np.argsort(np.abs(energy_change))[::-1][:8]:
        if abs(float(energy_change[idx])) < 0.16:
            break
        direction = "rises" if energy_change[idx] > 0 else "drops"
        add("energy", int(idx), f"Energy {direction}", min(1.0, abs(float(energy_change[idx])) * 2.7),
            f"energy {direction} {abs(float(energy_change[idx]))*100:.0f}% vs local context")

    for idx in _candidate_peaks(bass_n, max(0.58, float(np.percentile(bass_n, 72))), 8, max(3, slice_count // 55)):
        local = _local_delta(bass_n, idx, lookback)
        score = max(float(bass_n[idx]) * 0.70, min(1.0, max(0.0, local) * 2.7))
        add("bass", idx, "Low-end focus", score,
            f"low end {bass_n[idx]*100:.0f}% · sub {band_s['sub'][idx]*100:.1f}% · bass {band_s['bass'][idx]*100:.1f}%")

    bass_change = np.asarray([_local_delta(bass_n, i, lookback) for i in range(slice_count)])
    for idx in np.argsort(np.abs(bass_change))[::-1][:7]:
        if abs(float(bass_change[idx])) < 0.17:
            break
        direction = "enters" if bass_change[idx] > 0 else "pulls back"
        add("bass", int(idx), f"Low end {direction}", min(1.0, abs(float(bass_change[idx])) * 2.8),
            f"low end {direction} {abs(float(bass_change[idx]))*100:.0f}% vs local context")

    for idx in _candidate_peaks(onset_n, max(0.62, float(np.percentile(onset_n, 78))), 8, max(3, slice_count // 60)):
        add("transient", idx, "Transient impact", float(onset_n[idx]), f"transient {onset_n[idx]*100:.0f}%")

    if stereo_source and float(np.max(width_n)) > 0.08:
        width_change = np.asarray([_local_delta(width_n, i, lookback) for i in range(slice_count)])
        for idx in np.argsort(np.abs(width_change))[::-1][:6]:
            if abs(float(width_change[idx])) < 0.18:
                break
            direction = "opens" if width_change[idx] > 0 else "narrows"
            add("stereo", int(idx), f"Stereo {direction}", min(1.0, abs(float(width_change[idx])) * 2.5),
                f"stereo {direction} {abs(float(width_change[idx]))*100:.0f}%")

    bright_change = np.asarray([_local_delta(brightness_n, i, lookback) for i in range(slice_count)])
    for idx in np.argsort(np.abs(bright_change))[::-1][:6]:
        if abs(float(bright_change[idx])) < 0.16:
            break
        direction = "brighter" if bright_change[idx] > 0 else "darker"
        add("brightness", int(idx), "Tone shift", min(1.0, abs(float(bright_change[idx])) * 2.7),
            f"tone {direction} · centroid {cent_s[idx]:.0f} Hz")

    texture_change = np.asarray([_local_delta(texture_n, i, lookback) for i in range(slice_count)])
    for idx in np.argsort(np.abs(texture_change))[::-1][:4]:
        if abs(float(texture_change[idx])) < 0.22:
            break
        direction = "noisier" if texture_change[idx] > 0 else "more tonal"
        add("texture", int(idx), "Texture shift", min(1.0, abs(float(texture_change[idx])) * 2.3),
            f"texture becomes {direction}")

    clip_indices = np.where(clip_s > 0)[0]
    for idx in sorted(clip_indices, key=lambda i: float(clip_s[i]), reverse=True)[:4]:
        add("clip", int(idx), "Digital clipping detected",
            min(1.0, 0.55 + float(clip_s[idx]) * 60.0),
            f"digital ceiling hit · {clip_s[idx]*100:.3f}% frame samples")

    resonance: List[tuple[float, int, float]] = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        segment_spec = np.mean(stft[:, int(a):max(int(a) + 1, int(b))], axis=1)
        hz, prominence = _local_peak_prominence(segment_spec, freqs)
        if prominence >= 9.0:
            resonance.append((prominence, i, hz))
    resonance.sort(reverse=True)
    used_res: List[int] = []
    for prominence, idx, hz in resonance:
        if all(abs(idx - old) >= max(5, slice_count // 45) for old in used_res):
            used_res.append(idx)
            add("resonance", idx, "Narrow frequency focus", min(1.0, prominence / 18.0),
                f"narrow focus {hz:.0f} Hz · +{prominence:.1f} dB")
        if len(used_res) >= 3:
            break

    pins = _fuse_candidates(candidates, times, sections, duration)
    ai_timeline = _build_ai_timeline(duration, sections, pins, slices)

    return {
        "ai_timeline_summary": ai_timeline,
        "version": 3,
        "event_model": "section_aware_fusion_v3",
        "duration": _safe(duration, 3),
        "sample_rate": int(sr),
        "stereo_source": stereo_source,
        "slice_count": slice_count,
        "bands": [{"id": name, "low_hz": lo, "high_hz": hi} for name, lo, hi in BANDS],
        "sections": sections,
        "slices": slices,
        "pins": pins,
        "events": pins,
        "pin_count": len(pins),
        "legend": {
            "x": "time",
            "height": "smoothed energy + frequency-band intensity",
            "depth": "stereo width / frequency layer",
            "glow": "transient intensity",
            "surface_detail": "spectral texture",
            "pins": "section-aware fused musical events",
        },
    }
