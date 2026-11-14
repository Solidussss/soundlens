"""Measurable quality checks.

These numbers support listening tests; they don't replace them. In particular,
clipping and distortion are *reported*, never automatically "fixed": for
some characters (e.g. rage) they are the point.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE
from .theory import NOTE_NAMES

# Krumhansl–Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# scale -> (parent tonic offset, parent mode) so modes are compared fairly with major/minor estimates
_MODE_PARENT = {"major": (0, "major"), "minor": (0, "minor"), "dorian": (10, "major"), "phrygian": (8, "major"),
                "lydian": (7, "major"), "mixolydian": (5, "major"), "locrian": (1, "major"), "harmonic_minor": (0, "minor"),
                "minor_pentatonic": (0, "minor"), "major_pentatonic": (0, "major")}


def mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.ndim == 2 else x


def loudness_lufs(x: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    v = meter.integrated_loudness(x.T if x.ndim == 2 else x)
    return float(v) if np.isfinite(v) else -120.0


def true_peak_db(x: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    from scipy import signal
    up = signal.resample_poly(x, 4, 1, axis=-1)
    p = float(np.max(np.abs(up)))
    return 20 * math.log10(p) if p > 0 else -120.0


def clipping(x: np.ndarray, threshold: float = 0.999, min_run: int = 3) -> dict:
    """Runs of ≥min_run consecutive samples at/above full scale (what a DAC would flatten)."""
    a = np.abs(x) >= threshold
    runs = 0
    clipped = 0
    for ch in np.atleast_2d(a):
        d = np.diff(np.concatenate([[0], ch.astype(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        lens = ends - starts
        long = lens >= min_run
        runs += int(long.sum())
        clipped += int(lens[long].sum())
    overs = int(np.sum(np.abs(x) > 1.0))
    return {"clip_runs": runs, "clipped_ratio": clipped / max(x.size, 1), "samples_over_full_scale": overs}


def silence_gaps(x: np.ndarray, sr: int = SAMPLE_RATE, threshold_db: float = -60.0, min_s: float = 1.5, ignore_tail_s: float = 2.0) -> list[tuple[float, float]]:
    m = mono(x)
    hop = 1024
    n = len(m) - int(ignore_tail_s * sr)
    frames = max(0, n // hop)
    if frames == 0:
        return []
    rms = np.sqrt(np.mean(m[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-20)
    quiet = 20 * np.log10(rms) < threshold_db
    gaps, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        if (not q or i == frames - 1) and start is not None:
            end = i if not q else i + 1
            if (end - start) * hop / sr >= min_s:
                gaps.append((round(start * hop / sr, 2), round(end * hop / sr, 2)))
            start = None
    return gaps


def estimate_bpm(x: np.ndarray, sr: int = SAMPLE_RATE, expected: float | None = None) -> dict:
    import librosa
    m = mono(x).astype(np.float32)
    y = librosa.resample(m, orig_sr=sr, target_sr=22050)
    onset = librosa.onset.onset_strength(y=y, sr=22050)
    tempo = librosa.feature.tempo(onset_envelope=onset, sr=22050, start_bpm=expected or 120.0, aggregate=np.median)
    est = float(np.atleast_1d(tempo)[0])
    out = {"bpm_estimate": round(est, 2)}
    if expected:
        cands = [est, est * 2, est / 2, est * 1.5, est / 1.5]
        best = min(cands, key=lambda b: abs(b - expected))
        out["bpm_matched"] = round(best, 2)
        out["bpm_error_pct"] = round(abs(best - expected) / expected * 100, 2)
        out["octave_relation"] = {0: "same", 1: "half", 2: "double", 3: "2/3", 4: "3/2"}[cands.index(best)]
    return out


def estimate_key(x: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    import librosa
    m = mono(x).astype(np.float32)
    y = librosa.resample(m, orig_sr=sr, target_sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050).mean(axis=1)
    best = None
    for tonic in range(12):
        for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
            r = float(np.corrcoef(np.roll(prof, tonic), chroma)[0, 1])
            if best is None or r > best[0]:
                best = (r, tonic, mode)
    r, tonic, mode = best
    return {"key_estimate": f"{NOTE_NAMES[tonic]} {mode}", "tonic_pc": tonic, "mode": mode, "correlation": round(r, 3)}


def key_matches(estimate: dict, key: str, scale: str) -> dict:
    """Compare a major/minor estimate with a target key that may be modal.

    A song in F phrygian has the same pitch collection as C# major; profile-based
    estimators will often report the parent or relative key. We report all three
    relationships instead of a single pass/fail.
    """
    from .theory import pitch_class
    tpc = pitch_class(key)
    off, parent_mode = _MODE_PARENT.get(scale, (0, "minor"))
    parent_pc = (tpc + off) % 12
    rel_pc = (parent_pc + 9) % 12 if parent_mode == "major" else (parent_pc + 3) % 12
    rel_mode = "minor" if parent_mode == "major" else "major"
    e_pc, e_mode = estimate["tonic_pc"], estimate["mode"]
    exact = e_pc == tpc and (e_mode == ("minor" if scale in ("minor", "harmonic_minor", "dorian", "phrygian", "locrian", "minor_pentatonic") else "major"))
    same_collection = (e_pc, e_mode) in ((parent_pc, parent_mode), (rel_pc, rel_mode))
    return {"exact_tonic_and_mode": bool(exact), "same_pitch_collection": bool(same_collection or exact)}


def spectral_bands(x: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    m = mono(x)
    spec = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2 if len(m) < 2 ** 22 else None
    if spec is None:
        m = m[: 2 ** 22]
        spec = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
    freqs = np.fft.rfftfreq(len(m), 1 / sr)
    total = spec.sum() + 1e-20
    bands = {"sub_20_60": (20, 60), "bass_60_250": (60, 250), "lowmid_250_2k": (250, 2000), "highmid_2k_6k": (2000, 6000), "high_6k_20k": (6000, 20000)}
    return {k: round(float(spec[(freqs >= lo) & (freqs < hi)].sum() / total), 4) for k, (lo, hi) in bands.items()}


def stereo_stats(x: np.ndarray) -> dict:
    if x.ndim != 2:
        return {"correlation": 1.0, "side_to_mid_db": -120.0}
    l, r = x[0], x[1]
    c = float(np.corrcoef(l, r)[0, 1]) if np.std(l) > 0 and np.std(r) > 0 else 1.0
    mid, side = (l + r) / 2, (l - r) / 2
    smr = 10 * math.log10((np.mean(side ** 2) + 1e-20) / (np.mean(mid ** 2) + 1e-20))
    return {"correlation": round(c, 3), "side_to_mid_db": round(smr, 1)}


def distortion_profile(x: np.ndarray, sr: int = SAMPLE_RATE, max_seconds: float = 60.0) -> dict:
    """Describe how distorted a bass/808 signal is.

    * ``harmonic_ratio_db``: energy 150 Hz–5 kHz relative to 25–150 Hz (grit vs sub)
    * ``thd_pct``: total harmonic distortion estimated on frames with a stable f0
    * ``crest_db``: peak-to-RMS; heavy clipping drives it toward ~3 dB (a square wave)
    * ``flat_top_ratio``: share of samples within 1% of the signal's own peak (clip flattening)
    """
    import librosa
    m = mono(x)[: int(max_seconds * sr)]
    peak = float(np.max(np.abs(m)) + 1e-20)
    # crest on *active* frames only, so rests between 808 hits don't inflate it
    hop = 1024
    k = len(m) // hop
    frames = m[: k * hop].reshape(k, hop)
    frms = np.sqrt(np.mean(frames ** 2, axis=1))
    active = frames[frms > frms.max() * 10 ** (-30 / 20)] if k else m[None, :]
    rms = float(np.sqrt(np.mean(active ** 2)) + 1e-20)
    out = {"crest_db": round(20 * math.log10(peak / rms), 2), "flat_top_ratio": round(float(np.mean(np.abs(m) >= 0.99 * peak)), 5)}
    spec = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / sr)
    sub = spec[(f >= 25) & (f < 150)].sum() + 1e-20
    top = spec[(f >= 150) & (f < 5000)].sum() + 1e-20
    out["harmonic_ratio_db"] = round(10 * math.log10(top / sub), 2)
    y = librosa.resample(m.astype(np.float32), orig_sr=sr, target_sr=11025)
    f0, voiced, _ = librosa.pyin(y, fmin=28, fmax=200, sr=11025, frame_length=2048, hop_length=512)
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))
    freqs = librosa.fft_frequencies(sr=11025, n_fft=4096)
    thds = []
    for i in np.where(voiced & np.isfinite(f0))[0][::2]:
        if i >= S.shape[1]:
            break
        f0i = f0[i]
        def mag(fr):
            j = int(np.argmin(np.abs(freqs - fr)))
            return float(S[max(j - 1, 0): j + 2, i].max())
        fund = mag(f0i)
        if fund <= 1e-6:
            continue
        harm = [mag(k * f0i) for k in range(2, 11) if k * f0i < 5000]
        thds.append(math.sqrt(sum(h * h for h in harm)) / fund)
    out["thd_pct"] = round(float(np.median(thds) * 100), 1) if thds else None
    out["voiced_frames"] = len(thds)
    return out


def summarize(x: np.ndarray, sr: int = SAMPLE_RATE, bpm: float | None = None, key: str | None = None, scale: str | None = None,
              expected_seconds: float | None = None) -> dict:
    rep = {
        "duration_s": round(x.shape[-1] / sr, 3),
        "sample_peak_db": round(20 * math.log10(float(np.max(np.abs(x))) + 1e-20), 2),
        "true_peak_db": round(true_peak_db(x, sr), 2),
        "lufs": round(loudness_lufs(x, sr), 2),
        "clipping": clipping(x),
        "silence_gaps": silence_gaps(x, sr),
        "stereo": stereo_stats(x),
        "bands": spectral_bands(x, sr),
        "finite": bool(np.isfinite(x).all()),
    }
    if expected_seconds is not None:
        rep["duration_error_s"] = round(rep["duration_s"] - expected_seconds, 3)
    if bpm:
        rep["tempo"] = estimate_bpm(x, sr, bpm)
    k = estimate_key(x, sr)
    rep["key"] = k
    if key and scale:
        rep["key"].update(key_matches(k, key, scale))
    return rep


def timing_alignment(stem: np.ndarray, expected_beats: list[float], bpm: float, sr: int = SAMPLE_RATE, tol_ms: float = 30.0) -> dict:
    """Detected onsets of a (drum) stem vs the note times it was rendered from."""
    import librosa
    y = librosa.resample(mono(stem).astype(np.float32), orig_sr=sr, target_sr=22050)
    det = librosa.onset.onset_detect(y=y, sr=22050, hop_length=128, units="time", backtrack=True)
    exp = np.array(sorted(b * 60.0 / bpm for b in expected_beats))
    if len(det) == 0 or len(exp) == 0:
        return {"matched_pct": 0.0, "expected": int(len(exp)), "detected": int(len(det))}
    errs = np.array([np.min(np.abs(det - t)) for t in exp]) * 1000.0
    hit = errs <= tol_ms
    return {"expected": int(len(exp)), "detected": int(len(det)), "matched_pct": round(float(hit.mean() * 100), 1),
            "median_abs_error_ms": round(float(np.median(errs[hit])) if hit.any() else float("nan"), 2),
            "p95_abs_error_ms": round(float(np.percentile(errs[hit], 95)) if hit.any() else float("nan"), 2)}


def null_test(a: np.ndarray, b: np.ndarray) -> dict:
    """How different two renders are. ``identical`` means bit-for-bit equal."""
    n = min(a.shape[-1], b.shape[-1])
    d = a[..., :n].astype(np.float64) - b[..., :n].astype(np.float64)
    ref = float(np.sqrt(np.mean(a[..., :n].astype(np.float64) ** 2)) + 1e-20)
    resid = float(np.sqrt(np.mean(d ** 2)))
    return {"identical": bool(np.array_equal(a[..., :n], b[..., :n]) and a.shape == b.shape),
            "max_abs_diff": float(np.max(np.abs(d))) if d.size else 0.0,
            "residual_db": round(20 * math.log10(resid / ref), 1) if resid > 0 else -300.0}


def export_integrity(path: Path, expected_sr: int, expected_channels: int, expected_seconds: float, tol_s: float = 0.05) -> dict:
    info = sf.info(str(path))
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    ok = (sr == expected_sr and info.channels == expected_channels and abs(info.duration - expected_seconds) <= tol_s
          and np.isfinite(data).all() and float(np.max(np.abs(data))) > 1e-4)
    return {"ok": bool(ok), "sr": sr, "channels": info.channels, "duration_s": round(info.duration, 3), "subtype": info.subtype,
            "bytes": Path(path).stat().st_size}


def leakage(stem: np.ndarray, others: list[np.ndarray], sr: int = SAMPLE_RATE) -> dict:
    """Crude leakage measure for *separated* stems: max |corr| of the stem's envelope with each other stem's envelope.

    With generated-by-design stems this is a musical-overlap measure, not leakage, so
    only use it to compare separation outputs.
    """
    def env(x):
        m = np.abs(mono(x))
        hop = 512
        k = len(m) // hop
        return m[: k * hop].reshape(k, hop).mean(axis=1)
    e = env(stem)
    out = []
    for o in others:
        eo = env(o)
        n = min(len(e), len(eo))
        out.append(round(float(np.corrcoef(e[:n], eo[:n])[0, 1]), 3) if n > 2 and np.std(e[:n]) > 0 and np.std(eo[:n]) > 0 else 0.0)
    return {"envelope_correlation": out}
