"""
SoundLens Pro - producer-focused audio analysis tool

What this version does:
- Reads WAV/MP3/FLAC/M4A audio files
- Detects BPM, key, loudness, clipping, frequency balance, rhythm activity
- Estimates arrangement sections based on BPM/bar length
- Adds producer-focused feedback instead of only raw numbers
- Gives top problems, suggested fixes, release readiness, and profile scores
- Saves a clean .txt report automatically

Install needed packages:
    pip install librosa numpy soundfile

Run:
    python soundlens_pro.py

Optional direct file run:
    python soundlens_pro.py "mybeat.wav"
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import subprocess
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

EPSILON = 1e-9
NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif"}

# Krumhansl-Schmuckler style key profiles. They are not perfect, but they are
# much better than simply picking the loudest chroma note and calling it minor.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

FREQUENCY_BANDS = {
    "Sub": (20, 80),
    "Bass / 808": (80, 250),
    "Mud": (250, 500),
    "Low Mids": (500, 1000),
    "Mids / Melody": (1000, 4000),
    "Harsh Zone": (2000, 5000),
    "Highs": (4000, 10000),
    "Air": (10000, 16000),
    "Vocal Range": (300, 3400),
}

REFERENCE_TARGETS = {
    "trap_rage": {
        "rms_min": -15.5,
        "rms_max": -7.0,
        "peak_max": -0.1,
        "dynamic_min": 6.0,
        "dynamic_max": 16.0,
        "sub_min": 8.0,
        "sub_max": 25.0,
        "bass_min": 12.0,
        "bass_max": 35.0,
        "mud_max": 12.0,
        "harsh_max": 18.0,
        "high_min": 8.0,
        "high_max": 35.0,
        "intro_bars_max": 8,
    }
}

STYLE_PRESETS = {
    "rage_trap": {
        "low_end_warning": 60,
        "low_end_problem": 72,
        "bass_808_problem": 48,
        "rms_min": -15.5,
        "rms_max": -6.0,
        "allow_heavy_bass": True,
    },
    "general": {
        "low_end_warning": 40,
        "low_end_problem": 50,
        "bass_808_problem": 38,
        "rms_min": -16.0,
        "rms_max": -8.0,
        "allow_heavy_bass": False,
    },
}

DEFAULT_STYLE = "rage_trap"


@dataclass
class BasicInfo:
    file_name: str
    file_path: str
    sample_rate: int
    duration_seconds: float
    bpm: float
    key: str
    key_note: str
    key_mode: str
    key_confidence: float


@dataclass
class LoudnessInfo:
    peak_db: float
    rms_db: float
    dynamic_range_db: float
    clipping_detected: bool
    clipping_samples: int
    clipping_percent: float
    headroom_db: float


@dataclass
class FrequencyInfo:
    band_percentages: Dict[str, float]
    brightness_centroid_hz: float
    brightness_label: str
    spectral_rolloff_hz: float
    dominant_band: str
    low_end_total_percent: float
    mid_total_percent: float
    top_total_percent: float


@dataclass
class RhythmInfo:
    onset_count: int
    onset_density: float
    drum_activity: str
    estimated_bars: int
    seconds_per_bar: float


@dataclass
class ArrangementSection:
    name: str
    start: float
    end: float
    avg_energy: float
    energy_label: str


@dataclass
class Scores:
    mix: int
    master: int
    arrangement: int
    release: int
    energy: float
    bass_strength: float
    darkness: float
    brightness: float
    drum_bounce: float
    vocal_space: float


@dataclass
class StemMetrics:
    name: str
    file_path: str
    peak_db: float
    rms_db: float
    dynamic_range_db: float
    low_end_total_percent: float
    mid_total_percent: float
    top_total_percent: float
    brightness_centroid_hz: float
    spectral_rolloff_hz: float


@dataclass
class StemBalanceInfo:
    enabled: bool
    status: str
    confidence: str
    stems: Dict[str, StemMetrics] = field(default_factory=dict)
    vocal_to_beat_db: Optional[float] = None
    bass_to_vocal_db: Optional[float] = None
    bass_to_other_db: Optional[float] = None
    drums_to_vocal_db: Optional[float] = None
    vocal_presence_score: Optional[int] = None
    bass_dominance_score: Optional[int] = None
    beat_vocal_balance_score: Optional[int] = None
    melody_presence_score: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class SoundLensReport:
    basic: BasicInfo
    loudness: LoudnessInfo
    frequency: FrequencyInfo
    fingerprint: Dict[str, float]
    rhythm: RhythmInfo
    sections: List[ArrangementSection]
    scores: Scores
    stem_balance: Optional[StemBalanceInfo] = None
    top_problems: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    artist_notes: List[str] = field(default_factory=list)
    producer_notes: List[str] = field(default_factory=list)
    master_notes: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


def format_time(seconds: float) -> str:
    seconds = max(0, float(seconds))
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def db(value: float) -> float:
    return 20 * math.log10(max(value, EPSILON))


def level_label(value: float, low: float, high: float) -> str:
    if value >= high:
        return "High"
    if value >= low:
        return "Medium"
    return "Low"


def score_label(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Needs Work"
    return "Weak"


def _ffmpeg_executable() -> Optional[str]:
    """Find system FFmpeg or the binary bundled by imageio-ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _decode_with_ffmpeg(audio_file: Path) -> Tuple[np.ndarray, int]:
    """Decode a supported file to a temporary WAV used only for analysis."""
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("This audio format needs FFmpeg, but FFmpeg is unavailable on the server.")

    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_name = tmp.name

        completed = subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(audio_file),
             "-vn", "-ac", "1", "-c:a", "pcm_f32le", temp_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or "FFmpeg could not decode the file.").strip()
            raise ValueError(f"Could not decode audio file: {detail[-800:]}")

        y, sr = librosa.load(temp_name, mono=True, sr=None)
        return np.asarray(y, dtype=np.float32), int(sr)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def load_audio(audio_file: Path) -> Tuple[np.ndarray, int]:
    if not audio_file.exists():
        raise FileNotFoundError(f"File not found: {audio_file}")

    suffix = audio_file.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{suffix or 'unknown'}'. "
            "SoundLens supports WAV, MP3, FLAC, M4A, AAC, OGG, AIFF and AIF."
        )

    try:
        y, sr = librosa.load(audio_file, mono=True, sr=None)
        y = np.asarray(y, dtype=np.float32)
    except Exception as primary_error:
        print(f"Primary decoder failed for {audio_file.name}: {primary_error}")
        y, sr = _decode_with_ffmpeg(audio_file)

    if y.size == 0:
        raise ValueError("Audio file loaded empty.")
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if float(np.max(np.abs(y))) <= EPSILON:
        raise ValueError("Audio file contains no audible signal.")

    y = y.astype(np.float32)
    y = y - float(np.mean(y))
    return y, int(sr)

def memory_safe_audio(y: np.ndarray, sr: int, target_sr: int = 22050, max_seconds: int = 120) -> Tuple[np.ndarray, int]:
    """Downsample and trim an analysis-only copy so Railway does not run out of RAM."""
    y_work = y.astype(np.float32)

    if sr != target_sr:
        try:
            y_work = librosa.resample(y_work, orig_sr=sr, target_sr=target_sr)
            work_sr = target_sr
        except Exception:
            work_sr = sr
    else:
        work_sr = sr

    max_samples = int(work_sr * max_seconds)
    if len(y_work) > max_samples:
        start = max(0, (len(y_work) // 2) - (max_samples // 2))
        y_work = y_work[start:start + max_samples]

    y_work = y_work - float(np.mean(y_work))
    return y_work.astype(np.float32), work_sr


def detect_bpm(y: np.ndarray, sr: int) -> float:
    """Estimate BPM from multi-window onset autocorrelation.

    This version was benchmarked on the SoundLens/TuneBat validation set. It
    scores every BPM from 55-190 against rhythmic autocorrelation rather than
    letting one beat-track estimate dominate, then checks several song regions.
    Half/double-time pulses are kept as related evidence without forcing 140 BPM.
    """
    try:
        target_sr = 16000
        y_work = np.asarray(y, dtype=np.float32)
        if y_work.ndim > 1:
            y_work = np.mean(y_work, axis=1)
        if sr != target_sr:
            y_work = librosa.resample(y_work, orig_sr=sr, target_sr=target_sr)
            bpm_sr = target_sr
        else:
            bpm_sr = sr
        y_work = y_work - float(np.mean(y_work))
        duration = len(y_work) / max(bpm_sr, 1)
        if duration < 4.0:
            return 0.0

        # Keep analysis bounded for Railway while sampling the song broadly.
        max_seconds = 80.0
        if duration > max_seconds:
            n = int(max_seconds * bpm_sr)
            st = max(0, (len(y_work) - n) // 2)
            y_work = y_work[st:st+n]
            duration = len(y_work) / bpm_sr

        win_seconds = min(28.0, duration)
        win_len = int(win_seconds * bpm_sr)
        centers = (0.20, 0.50, 0.80) if duration > 40.0 else (0.50,)
        scores = np.zeros(191, dtype=np.float64)
        direct_votes: List[float] = []

        for center_ratio in centers:
            center = int(len(y_work) * center_ratio)
            st = max(0, min(center - win_len // 2, len(y_work) - win_len))
            piece = y_work[st:st + win_len]
            if piece.size < bpm_sr * 4:
                continue

            onset_env = librosa.onset.onset_strength(y=piece, sr=bpm_sr, hop_length=256)
            onset_env = np.asarray(onset_env, dtype=np.float64)
            if onset_env.size < 16 or float(np.max(onset_env)) <= EPSILON:
                continue
            onset_env = np.maximum(onset_env - float(np.mean(onset_env)), 0.0)

            # Autocorrelation gives an explicit score for every possible pulse.
            max_lag = min(len(onset_env) - 1, int((bpm_sr / 256.0) * 1.2))
            ac = librosa.autocorrelate(onset_env, max_size=max_lag)
            if ac.size < 4 or float(ac[0]) <= EPSILON:
                continue
            ac = ac / (float(ac[0]) + EPSILON)

            try:
                tempo = librosa.feature.tempo(
                    onset_envelope=onset_env,
                    sr=bpm_sr,
                    hop_length=256,
                    aggregate=np.median,
                )
                tempo = float(np.ravel(tempo)[0])
                if np.isfinite(tempo) and tempo > 0:
                    direct_votes.append(tempo)
            except Exception:
                pass

            for bpm in range(55, 191):
                lag = (60.0 * bpm_sr) / (256.0 * bpm)
                idx = int(round(lag))
                if idx <= 0 or idx >= len(ac):
                    continue
                score = max(0.0, float(ac[idx]))
                # Real beats usually retain periodicity over 2 and 4 beats.
                for mult, weight in ((2, 0.45), (4, 0.20)):
                    j = int(round(lag * mult))
                    if j < len(ac):
                        score += weight * max(0.0, float(ac[j]))
                # Subdivisions are useful evidence but intentionally weak.
                j = int(round(lag / 2.0))
                if 0 < j < len(ac):
                    score += 0.08 * max(0.0, float(ac[j]))
                scores[bpm] += score

        if not np.any(scores[55:191] > 0):
            return 0.0

        # Add a small direct-tempo vote. Autocorrelation remains the main signal.
        for tempo in direct_votes:
            while tempo < 55:
                tempo *= 2.0
            while tempo > 190:
                tempo /= 2.0
            for bpm in range(55, 191):
                distance = abs(tempo - bpm)
                scores[bpm] += 0.35 * max(0.0, 1.0 - distance / 5.0)

        # Smooth 1-BPM quantization noise.
        smoothed = scores.copy()
        for bpm in range(56, 190):
            smoothed[bpm] = (0.20 * scores[bpm - 1]) + (0.60 * scores[bpm]) + (0.20 * scores[bpm + 1])

        best_bpm = int(np.argmax(smoothed[55:191]) + 55)
        if smoothed[best_bpm] <= 0:
            return 0.0
        return float(best_bpm)
    except Exception:
        return 0.0

def detect_key(y: np.ndarray, sr: int) -> Tuple[str, str, str, float]:
    """Estimate musical key with a genre-calibrated multi-window HPCP/chroma vote.

    The profiles below were calibrated from the SoundLens validation library and
    blended with Krumhansl-Schmuckler profiles.  They are *not* artist-specific
    corrections: they describe how major/minor pitch-class energy tends to look
    in the rap/trap material SoundLens is built for.  Multiple song regions and
    full/high-passed/bass views reduce domination by one 808 or one sparse loop.
    """
    # 70% SoundLens validation profile + 30% established key profile.
    major_profile = np.array([
        0.126231, 0.077852, 0.080908, 0.073026, 0.093841, 0.094862,
        0.073267, 0.099948, 0.070900, 0.070191, 0.060347, 0.078628,
    ], dtype=np.float64)
    minor_profile = np.array([
        0.113968, 0.076979, 0.076314, 0.094303, 0.067253, 0.084145,
        0.070048, 0.095466, 0.098610, 0.073940, 0.077731, 0.071243,
    ], dtype=np.float64)

    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + EPSILON
        return float(np.dot(a, b) / denom)

    def pitch_views(piece: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        piece = np.asarray(piece, dtype=np.float32)
        piece = piece - float(np.mean(piece))
        views: List[np.ndarray] = []

        # Full harmonic picture.
        try:
            chroma_full = librosa.feature.chroma_cqt(
                y=piece, sr=sample_rate, bins_per_octave=24, hop_length=512
            )
            vec = np.mean(chroma_full, axis=1)
            if np.any(vec > 0):
                views.append(np.asarray(vec, dtype=np.float64) * 1.00)
        except Exception:
            pass

        # High-passed picture: reduces the chance that a single sub/808 note wins.
        try:
            sos = butter(4, 90.0, btype="highpass", fs=sample_rate, output="sos")
            high = sosfiltfilt(sos, piece).astype(np.float32)
            chroma_high = librosa.feature.chroma_cqt(
                y=high, sr=sample_rate, bins_per_octave=24, hop_length=512
            )
            vec = np.mean(chroma_high, axis=1)
            if np.any(vec > 0):
                views.append(np.asarray(vec, dtype=np.float64) * 0.50)
        except Exception:
            pass

        # Bass pitch-class evidence is useful, but deliberately weak so an 808
        # cannot decide major/minor by itself.
        try:
            cqt_bass = np.abs(librosa.cqt(
                piece,
                sr=sample_rate,
                hop_length=512,
                fmin=librosa.note_to_hz("C1"),
                n_bins=36,
                bins_per_octave=12,
            ))
            bass = np.array([np.mean(cqt_bass[i::12]) for i in range(12)], dtype=np.float64)
            if np.any(bass > 0):
                views.append(bass * 0.20)
        except Exception:
            pass

        return views

    try:
        target_sr = 16000
        y_work = np.asarray(y, dtype=np.float32)
        if y_work.ndim > 1:
            y_work = np.mean(y_work, axis=1)
        if sr != target_sr:
            y_work = librosa.resample(y_work, orig_sr=sr, target_sr=target_sr)
            key_sr = target_sr
        else:
            key_sr = sr
        y_work = y_work - float(np.mean(y_work))
        duration = len(y_work) / max(key_sr, 1)
        if duration < 4.0:
            return "Uncertain", "Uncertain", "Uncertain", 0.0

        # Four short regions are more robust than one long middle excerpt.
        window_seconds = min(14.0, duration)
        win_len = int(window_seconds * key_sr)
        centers = (0.20, 0.40, 0.60, 0.80) if duration > 22.0 else (0.50,)

        aggregate: Dict[Tuple[str, str], float] = {}
        first_votes: Dict[Tuple[str, str], int] = {}
        usable_windows = 0

        for center_ratio in centers:
            center = int(len(y_work) * center_ratio)
            st = max(0, min(center - win_len // 2, len(y_work) - win_len))
            piece = y_work[st:st + win_len]
            if piece.size < key_sr * 4:
                continue
            views = pitch_views(piece, key_sr)
            if not views:
                continue
            usable_windows += 1

            # Add the weighted views together before ranking this region.
            combined = np.zeros(12, dtype=np.float64)
            for view in views:
                total = float(np.sum(np.abs(view))) + EPSILON
                combined += view / total

            window_scores: List[Tuple[float, str, str]] = []
            for root_idx, note in enumerate(NOTES_SHARP):
                window_scores.append((
                    correlation(combined, np.roll(major_profile, root_idx)), note, "Major"
                ))
                window_scores.append((
                    correlation(combined, np.roll(minor_profile, root_idx)), note, "Minor"
                ))
            window_scores.sort(key=lambda item: item[0], reverse=True)
            if not window_scores:
                continue

            winner = (window_scores[0][1], window_scores[0][2])
            first_votes[winner] = first_votes.get(winner, 0) + 1

            # Graded top-candidate support prevents one nearly tied window from
            # overruling the rest of the song.
            low = window_scores[-1][0]
            high = window_scores[0][0]
            span = max(high - low, EPSILON)
            for rank, (score, note, mode) in enumerate(window_scores[:6]):
                rank_weight = (1.00, 0.62, 0.38, 0.23, 0.14, 0.08)[rank]
                normalized = (score - low) / span
                key_id = (note, mode)
                aggregate[key_id] = aggregate.get(key_id, 0.0) + normalized * rank_weight

        if not aggregate or usable_windows == 0:
            return "Uncertain", "Uncertain", "Uncertain", 0.0

        ranked = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
        (best_note, best_mode), best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        third_score = ranked[2][1] if len(ranked) > 2 else 0.0

        vote_ratio = first_votes.get((best_note, best_mode), 0) / max(usable_windows, 1)
        gap_ratio = (best_score - second_score) / max(best_score, EPSILON)
        spread_ratio = (best_score - third_score) / max(best_score, EPSILON)
        confidence = clamp(
            18.0 + (vote_ratio * 24.0) + (gap_ratio * 30.0) + (spread_ratio * 12.0),
            15.0,
            75.0,
        )
        if gap_ratio < 0.08:
            confidence = min(confidence, 55.0)
        if vote_ratio < 0.50:
            confidence = min(confidence, 50.0)

        return f"{best_note} {best_mode}", best_note, best_mode, round(float(confidence), 1)
    except Exception:
        return "Uncertain", "Uncertain", "Uncertain", 0.0

def analyze_loudness(y: np.ndarray) -> LoudnessInfo:
    peak = float(np.max(np.abs(y)))
    peak_db = db(peak)
    rms = float(np.sqrt(np.mean(np.square(y))))
    rms_db = db(rms)
    dynamic_range = peak_db - rms_db
    clipping_samples = int(np.sum(np.abs(y) >= 0.98))
    clipping_percent = float(clipping_samples / max(len(y), 1) * 100)
    return LoudnessInfo(
        peak_db=peak_db,
        rms_db=rms_db,
        dynamic_range_db=dynamic_range,
        clipping_detected=clipping_samples > 0,
        clipping_samples=clipping_samples,
        clipping_percent=clipping_percent,
        headroom_db=0 - peak_db,
    )


def analyze_frequency(y: np.ndarray, sr: int) -> FrequencyInfo:
    """Memory-safe frequency analysis for Railway."""
    y_freq, freq_sr = memory_safe_audio(y, sr, target_sr=22050, max_seconds=120)

    n_fft = 2048
    hop_length = 1024

    stft = np.abs(librosa.stft(y_freq, n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
    freqs = librosa.fft_frequencies(sr=freq_sr, n_fft=n_fft)

    band_raw: Dict[str, float] = {}
    for name, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        if np.any(mask):
            energy = float(np.sum(stft[mask] ** 2))
            band_raw[name] = float(np.log10(energy + 1))
        else:
            band_raw[name] = 0.0

    del stft
    gc.collect()

    primary_names = ["Sub", "Bass / 808", "Mud", "Low Mids", "Mids / Melody", "Highs", "Air"]
    total_primary = sum(band_raw[name] for name in primary_names) + EPSILON
    band_percentages = {name: (value / total_primary) * 100 for name, value in band_raw.items()}

    centroid = librosa.feature.spectral_centroid(y=y_freq, sr=freq_sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y_freq, sr=freq_sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]

    brightness_centroid = float(np.mean(centroid))
    spectral_rolloff = float(np.mean(rolloff))
    brightness = level_label(brightness_centroid, 1800, 3500)

    dominant_band = max(primary_names, key=lambda name: band_percentages[name])
    low_end_total = band_percentages["Sub"] + band_percentages["Bass / 808"]
    mid_total = band_percentages["Mud"] + band_percentages["Low Mids"] + band_percentages["Mids / Melody"]
    top_total = band_percentages["Highs"] + band_percentages["Air"]

    return FrequencyInfo(
        band_percentages=band_percentages,
        brightness_centroid_hz=brightness_centroid,
        brightness_label=brightness,
        spectral_rolloff_hz=spectral_rolloff,
        dominant_band=dominant_band,
        low_end_total_percent=low_end_total,
        mid_total_percent=mid_total,
        top_total_percent=top_total,
    )

def analyze_audio_fingerprint(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Stronger SoundLens sonic fingerprint.

    This keeps the same output keys your current profile/compare system already
    expects, but fills them with more stable values.

    Important:
    - It does NOT change loudness, scores, Stripe, accounts, website, or report shape.
    - It normalizes a copy of the audio only for fingerprinting so Artist Match
      is less fooled by volume/mastering level.
    - It uses multiple windows across the song instead of only one middle chunk.
    - It always tries to return non-zero embed_* values.
    """
    def finite_float(value: float) -> float:
        try:
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return 0.0
            return number
        except Exception:
            return 0.0

    def summarize_feature(name: str, array: np.ndarray, fingerprint: Dict[str, float], limit: Optional[int] = None) -> None:
        array = np.asarray(array, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        rows = array.shape[0] if limit is None else min(limit, array.shape[0])

        for i in range(rows):
            row = array[i]
            fingerprint[f"{name}_{i+1}_mean"] = finite_float(np.mean(row))
            fingerprint[f"{name}_{i+1}_std"] = finite_float(np.std(row))

    try:
        # Use three sections instead of one: early, middle, late.
        # This gives the fingerprint a better sense of the full song while
        # still staying Railway/laptop safe.
        target_sr = 22050
        max_total_seconds = 120
        window_seconds = 40

        if sr > target_sr:
            y_work = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            fp_sr = target_sr
        else:
            y_work = y.astype(np.float32)
            fp_sr = sr

        y_work = y_work - float(np.mean(y_work))
        duration_seconds = len(y_work) / max(fp_sr, 1)

        max_samples = int(max_total_seconds * fp_sr)
        window_samples = int(window_seconds * fp_sr)

        chunks: List[np.ndarray] = []

        if len(y_work) <= max_samples:
            chunks.append(y_work)
        else:
            starts = [
                int(len(y_work) * 0.12),
                int(len(y_work) * 0.50) - window_samples // 2,
                int(len(y_work) * 0.82) - window_samples,
            ]

            for start in starts:
                start = max(0, min(start, max(0, len(y_work) - window_samples)))
                chunk = y_work[start:start + window_samples]
                if len(chunk) > fp_sr * 5:
                    chunks.append(chunk)

        if not chunks:
            chunks = [y_work[:max_samples]]

        y_fp = np.concatenate(chunks)

        # Normalize fingerprint audio so comparison is about sonic character,
        # not just who was mastered louder.
        peak = float(np.max(np.abs(y_fp))) + EPSILON
        y_fp = (y_fp / peak) * 0.95
        y_fp = y_fp.astype(np.float32)

        # Add light pre-emphasis for timbre/brightness features without affecting
        # normal loudness/frequency analysis elsewhere in the report.
        try:
            y_timbre = librosa.effects.preemphasis(y_fp)
        except Exception:
            y_timbre = y_fp

        fingerprint: Dict[str, float] = {}

        # Core timbre/harmony features.
        mfcc = librosa.feature.mfcc(y=y_timbre, sr=fp_sr, n_mfcc=20)
        try:
            chroma = librosa.feature.chroma_cqt(y=y_fp, sr=fp_sr)
        except Exception:
            chroma = librosa.feature.chroma_stft(y=y_fp, sr=fp_sr)

        contrast = librosa.feature.spectral_contrast(y=y_fp, sr=fp_sr)
        flatness = librosa.feature.spectral_flatness(y=y_fp)
        zcr = librosa.feature.zero_crossing_rate(y_fp)
        centroid = librosa.feature.spectral_centroid(y=y_fp, sr=fp_sr)
        rolloff = librosa.feature.spectral_rolloff(y=y_fp, sr=fp_sr, roll_percent=0.85)
        rms = librosa.feature.rms(y=y_fp)
        onset_env = librosa.onset.onset_strength(y=y_fp, sr=fp_sr)

        # Delta motion helps distinguish static loops from bouncy/moving songs.
        mfcc_delta = librosa.feature.delta(mfcc)
        chroma_delta = librosa.feature.delta(chroma)

        # Backward-compatible fields already used by older profiles.
        for i in range(13):
            fingerprint[f"mfcc_{i+1}"] = finite_float(np.mean(mfcc[i]))

        for i in range(12):
            fingerprint[f"chroma_{i+1}"] = finite_float(np.mean(chroma[i]))

        fingerprint["spectral_contrast"] = finite_float(np.mean(contrast))
        fingerprint["spectral_flatness"] = finite_float(np.mean(flatness))
        fingerprint["zero_crossing_rate"] = finite_float(np.mean(zcr))

        # Existing embedding keys your builder/compare code expects.
        summarize_feature("embed_mfcc", mfcc, fingerprint, limit=20)
        summarize_feature("embed_chroma", chroma, fingerprint, limit=12)
        summarize_feature("embed_contrast", contrast, fingerprint, limit=7)

        chroma_energy = np.mean(chroma, axis=1)
        chroma_energy = chroma_energy / (np.sum(chroma_energy) + EPSILON)
        chroma_entropy = -float(np.sum(chroma_energy * np.log(chroma_energy + EPSILON)))

        fingerprint["embed_chroma_entropy"] = finite_float(chroma_entropy)
        fingerprint["embed_centroid_mean"] = finite_float(np.mean(centroid))
        fingerprint["embed_centroid_std"] = finite_float(np.std(centroid))
        fingerprint["embed_rolloff_mean"] = finite_float(np.mean(rolloff))
        fingerprint["embed_rolloff_std"] = finite_float(np.std(rolloff))
        fingerprint["embed_flatness_mean"] = finite_float(np.mean(flatness))
        fingerprint["embed_flatness_std"] = finite_float(np.std(flatness))
        fingerprint["embed_zcr_mean"] = finite_float(np.mean(zcr))
        fingerprint["embed_zcr_std"] = finite_float(np.std(zcr))
        fingerprint["embed_rms_mean"] = finite_float(np.mean(rms))
        fingerprint["embed_rms_std"] = finite_float(np.std(rms))
        fingerprint["embed_onset_mean"] = finite_float(np.mean(onset_env))
        fingerprint["embed_onset_std"] = finite_float(np.std(onset_env))

        # Extra future-proof fields. These do not break anything if older compare
        # code ignores them, but they make reports more useful later.
        fingerprint["fingerprint_version"] = 2.0
        fingerprint["fingerprint_duration_seconds"] = finite_float(duration_seconds)
        fingerprint["embed_mfcc_delta_mean"] = finite_float(np.mean(mfcc_delta))
        fingerprint["embed_mfcc_delta_std"] = finite_float(np.std(mfcc_delta))
        fingerprint["embed_chroma_delta_mean"] = finite_float(np.mean(chroma_delta))
        fingerprint["embed_chroma_delta_std"] = finite_float(np.std(chroma_delta))
        fingerprint["embed_onset_peak"] = finite_float(np.max(onset_env)) if onset_env.size else 0.0
        fingerprint["embed_onset_p95"] = finite_float(np.percentile(onset_env, 95)) if onset_env.size else 0.0
        fingerprint["embed_rms_p10"] = finite_float(np.percentile(rms, 10)) if rms.size else 0.0
        fingerprint["embed_rms_p50"] = finite_float(np.percentile(rms, 50)) if rms.size else 0.0
        fingerprint["embed_rms_p90"] = finite_float(np.percentile(rms, 90)) if rms.size else 0.0

        # Health check: compare scripts can ignore this, but you can grep it.
        non_zero_embed_values = [
            value for key, value in fingerprint.items()
            if key.startswith("embed_") and isinstance(value, (int, float)) and abs(float(value)) > 1e-9
        ]
        fingerprint["embedding_nonzero_count"] = float(len(non_zero_embed_values))

        return fingerprint

    except Exception as error:
        # Keep SoundLens alive, but still return the strongest old fingerprint
        # possible instead of crashing the whole product.
        try:
            y_safe = y.astype(np.float32)
            y_safe = y_safe - float(np.mean(y_safe))
            peak = float(np.max(np.abs(y_safe))) + EPSILON
            y_safe = (y_safe / peak) * 0.95

            if sr > 22050:
                y_safe = librosa.resample(y_safe, orig_sr=sr, target_sr=22050)
                safe_sr = 22050
            else:
                safe_sr = sr

            mfcc = librosa.feature.mfcc(y=y_safe, sr=safe_sr, n_mfcc=20)
            chroma = librosa.feature.chroma_stft(y=y_safe, sr=safe_sr)
            contrast = librosa.feature.spectral_contrast(y=y_safe, sr=safe_sr)
            flatness = librosa.feature.spectral_flatness(y=y_safe)
            zcr = librosa.feature.zero_crossing_rate(y_safe)
            centroid = librosa.feature.spectral_centroid(y=y_safe, sr=safe_sr)
            rolloff = librosa.feature.spectral_rolloff(y=y_safe, sr=safe_sr)
            rms = librosa.feature.rms(y=y_safe)
            onset_env = librosa.onset.onset_strength(y=y_safe, sr=safe_sr)

            fingerprint: Dict[str, float] = {}

            for i in range(13):
                fingerprint[f"mfcc_{i+1}"] = finite_float(np.mean(mfcc[i]))

            for i in range(12):
                fingerprint[f"chroma_{i+1}"] = finite_float(np.mean(chroma[i]))

            fingerprint["spectral_contrast"] = finite_float(np.mean(contrast))
            fingerprint["spectral_flatness"] = finite_float(np.mean(flatness))
            fingerprint["zero_crossing_rate"] = finite_float(np.mean(zcr))

            summarize_feature("embed_mfcc", mfcc, fingerprint, limit=20)
            summarize_feature("embed_chroma", chroma, fingerprint, limit=12)
            summarize_feature("embed_contrast", contrast, fingerprint, limit=7)

            fingerprint["embed_chroma_entropy"] = 0.0
            fingerprint["embed_centroid_mean"] = finite_float(np.mean(centroid))
            fingerprint["embed_centroid_std"] = finite_float(np.std(centroid))
            fingerprint["embed_rolloff_mean"] = finite_float(np.mean(rolloff))
            fingerprint["embed_rolloff_std"] = finite_float(np.std(rolloff))
            fingerprint["embed_flatness_mean"] = finite_float(np.mean(flatness))
            fingerprint["embed_flatness_std"] = finite_float(np.std(flatness))
            fingerprint["embed_zcr_mean"] = finite_float(np.mean(zcr))
            fingerprint["embed_zcr_std"] = finite_float(np.std(zcr))
            fingerprint["embed_rms_mean"] = finite_float(np.mean(rms))
            fingerprint["embed_rms_std"] = finite_float(np.std(rms))
            fingerprint["embed_onset_mean"] = finite_float(np.mean(onset_env))
            fingerprint["embed_onset_std"] = finite_float(np.std(onset_env))
            fingerprint["fingerprint_version"] = 2.0
            fingerprint["embedding_nonzero_count"] = float(
                len([v for k, v in fingerprint.items() if k.startswith("embed_") and isinstance(v, (int, float)) and abs(float(v)) > 1e-9])
            )
            fingerprint["fingerprint_fallback_used"] = 1.0

            return fingerprint

        except Exception:
            return {
                "fingerprint_version": 2.0,
                "embedding_nonzero_count": 0.0,
                "fingerprint_failed": 1.0,
            }

def analyze_rhythm(y: np.ndarray, sr: int, duration: float, bpm: float) -> RhythmInfo:
    y_rhythm, rhythm_sr = memory_safe_audio(y, sr, target_sr=22050, max_seconds=120)

    onset_times = librosa.onset.onset_detect(
        y=y_rhythm,
        sr=rhythm_sr,
        units="time",
        backtrack=False,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.12,
        wait=2,
    )
    onset_count = len(onset_times)

    analyzed_duration = max(len(y_rhythm) / max(rhythm_sr, 1), 1)
    onset_density = onset_count / analyzed_duration

    # The previous High threshold (3 hits/sec) classified almost every trap song
    # as overcrowded. These wider bands keep the value descriptive instead.
    drum_activity = level_label(onset_density, 2.0, 5.5)

    if bpm > 0:
        seconds_per_bar = (60 / bpm) * 4
        estimated_bars = int(round(duration / max(seconds_per_bar, EPSILON)))
    else:
        seconds_per_bar = 0.0
        estimated_bars = 0

    return RhythmInfo(
        onset_count=onset_count,
        onset_density=onset_density,
        drum_activity=drum_activity,
        estimated_bars=estimated_bars,
        seconds_per_bar=seconds_per_bar,
    )

def section_energy(y: np.ndarray, sr: int, start: float, end: float) -> float:
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    part = y[start_sample:end_sample]
    if part.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(part))))


def estimate_arrangement(y: np.ndarray, sr: int, duration: float, rhythm: RhythmInfo) -> List[ArrangementSection]:
    """Estimate musical sections without assuming a fixed verse/chorus order.

    Boundaries use both energy and tonal/timbral novelty.  Labels are conservative:
    SoundLens calls something a likely hook/chorus only when a similar section
    actually repeats later in the song.  This avoids manufacturing a conventional
    structure that the audio does not support.
    """
    if duration <= 0:
        return []

    y_arr, arr_sr = memory_safe_audio(y, sr, target_sr=22050, max_seconds=180)
    hop_length = 512
    frame_length = 2048

    try:
        rms = librosa.feature.rms(y=y_arr, frame_length=frame_length, hop_length=hop_length)[0]
    except Exception:
        rms = np.array([], dtype=np.float32)

    if rms.size < 8:
        return [ArrangementSection(
            name="Full Track",
            start=0.0,
            end=duration,
            avg_energy=section_energy(y, sr, 0.0, duration),
            energy_label="Medium",
        )]

    times = librosa.frames_to_time(np.arange(len(rms)), sr=arr_sr, hop_length=hop_length)
    rms = np.nan_to_num(np.asarray(rms, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    energy = rms / (float(np.max(rms)) + EPSILON) if float(np.max(rms)) > EPSILON else rms

    smooth_seconds = 2.0
    smooth_frames = max(5, int(round(smooth_seconds / max(hop_length / arr_sr, EPSILON))))
    if smooth_frames % 2 == 0:
        smooth_frames += 1
    kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
    smooth = np.convolve(energy, kernel, mode="same")

    rms_derivative = np.abs(np.diff(smooth, prepend=smooth[0]))
    rms_change = rms_derivative / (float(np.percentile(rms_derivative, 95)) + EPSILON)
    rms_change = np.clip(rms_change, 0.0, 2.0)

    # Harmonic/timbral novelty catches transitions that do not create a large
    # volume dip (for example a beat switch or verse-to-hook instrument change).
    spectral_change = np.zeros_like(smooth, dtype=np.float32)
    try:
        chroma = librosa.feature.chroma_stft(
            y=y_arr,
            sr=arr_sr,
            n_fft=frame_length,
            hop_length=hop_length,
        )
        chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
        chroma_delta = np.linalg.norm(np.diff(chroma, axis=1, prepend=chroma[:, :1]), axis=0)

        centroid = librosa.feature.spectral_centroid(
            y=y_arr,
            sr=arr_sr,
            n_fft=frame_length,
            hop_length=hop_length,
        )[0]
        centroid = np.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0)
        centroid_delta = np.abs(np.diff(centroid, prepend=centroid[0]))
        centroid_delta = centroid_delta / (float(np.percentile(centroid_delta, 95)) + EPSILON)

        chroma_delta = chroma_delta / (float(np.percentile(chroma_delta, 95)) + EPSILON)
        spectral_change = np.clip((chroma_delta * 0.72) + (centroid_delta * 0.28), 0.0, 2.0).astype(np.float32)
    except Exception:
        pass

    combined_change = (rms_change * 0.52) + (spectral_change * 0.48)

    bar_seconds = max(float(rhythm.seconds_per_bar), 1.5)
    min_section_seconds = max(8.0, min(12.0, bar_seconds * 4.0))
    preferred_section_seconds = max(14.0, min(28.0, bar_seconds * 8.0))
    max_section_seconds = max(24.0, min(46.0, bar_seconds * 16.0))
    if duration < 120:
        max_section_seconds = min(max_section_seconds, 34.0)

    candidates: List[Tuple[float, float]] = []
    valley_threshold = float(np.percentile(smooth, 44))
    change_threshold = max(float(np.percentile(combined_change, 80)), 0.12)
    look_seconds = 5.0
    look_frames = max(4, int(round(look_seconds / max(hop_length / arr_sr, EPSILON))))

    for idx in range(look_frames, len(smooth) - look_frames):
        time = float(times[idx])
        if time < 4.0 or time > duration - 4.0:
            continue

        left_avg = float(np.mean(smooth[idx - look_frames:idx]))
        right_avg = float(np.mean(smooth[idx:idx + look_frames]))
        energy_shift = abs(right_avg - left_avg)
        is_valley = smooth[idx] <= smooth[idx - 1] and smooth[idx] <= smooth[idx + 1]
        is_low = smooth[idx] <= valley_threshold
        novelty = float(combined_change[idx])

        # Accept either a meaningful energy transition or a strong tonal/timbral
        # transition. Valley points get a bonus because section changes often land
        # on a short dip or breath.
        meaningful_energy = energy_shift >= 0.045
        meaningful_novelty = novelty >= change_threshold
        if (meaningful_energy and (is_valley or is_low)) or meaningful_novelty:
            valley_strength = 1.0 - float(smooth[idx])
            score = (energy_shift * 3.2) + (novelty * 1.45) + (valley_strength * (0.55 if is_valley else 0.20))
            candidates.append((time, score))

    # Collapse candidate clusters so one transition does not produce several
    # boundaries a fraction of a second apart.
    candidates.sort(key=lambda item: item[0])
    clustered: List[Tuple[float, float]] = []
    for time, score in candidates:
        if clustered and time - clustered[-1][0] < 2.0:
            if score > clustered[-1][1]:
                clustered[-1] = (time, score)
        else:
            clustered.append((time, score))
    candidates = clustered

    boundaries = [0.0]
    intro_candidates = [(t, sc) for t, sc in candidates if 6.0 <= t <= min(24.0, duration * 0.28)]
    if intro_candidates:
        boundaries.append(round(float(max(intro_candidates, key=lambda item: item[1])[0]), 2))

    while duration - boundaries[-1] > max_section_seconds:
        last = boundaries[-1]
        target = last + preferred_section_seconds
        search_start = last + min_section_seconds
        search_end = min(last + max_section_seconds, duration - min_section_seconds)
        available = [(t, sc) for t, sc in candidates if search_start <= t <= search_end]

        if available:
            def candidate_score(item):
                t, sc = item
                distance_penalty = abs(t - target) / max(preferred_section_seconds, 1.0)
                return sc - (distance_penalty * 0.48)
            boundary = max(available, key=candidate_score)[0]
        else:
            target_time = min(target, duration - min_section_seconds)
            idx = int(np.argmin(np.abs(times - target_time)))
            radius = max(4, int(round(4.0 / max(hop_length / arr_sr, EPSILON))))
            left = max(0, idx - radius)
            right = min(len(smooth), idx + radius + 1)
            # If there is no clear candidate, use a local minimum as a conservative
            # fallback rather than inventing an exact fixed-time boundary.
            valley_idx = left + int(np.argmin(smooth[left:right]))
            boundary = float(times[valley_idx])

        if boundary <= boundaries[-1] + min_section_seconds:
            boundary = boundaries[-1] + preferred_section_seconds
        if boundary >= duration - min_section_seconds:
            break
        boundaries.append(round(float(boundary), 2))

    max_sections = 8 if duration >= 130 else 7

    # A short song may contain several clear transitions even when no section
    # exceeds max_section_seconds. Add strong interior candidates when they split
    # an existing span into two musically plausible sections.
    candidate_strength_cutoff = float(np.percentile([sc for _, sc in candidates], 68)) if candidates else float("inf")
    changed = True
    while changed and len(boundaries) < max_sections:
        changed = False
        trial_boundaries = sorted(boundaries + [duration])
        best_extra = None
        best_extra_score = -float("inf")
        for left_b, right_b in zip(trial_boundaries[:-1], trial_boundaries[1:]):
            if right_b - left_b < min_section_seconds * 2.15:
                continue
            for t, sc in candidates:
                if sc < candidate_strength_cutoff:
                    continue
                if t - left_b < min_section_seconds or right_b - t < min_section_seconds:
                    continue
                # Prefer strong changes near an 8/16-bar-style subdivision.
                midpoint_penalty = abs((t - left_b) - preferred_section_seconds) / max(preferred_section_seconds, 1.0)
                adjusted = sc - midpoint_penalty * 0.20
                if adjusted > best_extra_score:
                    best_extra_score = adjusted
                    best_extra = t
        if best_extra is not None:
            boundaries.append(round(float(best_extra), 2))
            boundaries = sorted(set(boundaries))
            changed = True

    boundaries.append(duration)
    boundaries = sorted(set(boundaries))
    cleaned = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary - cleaned[-1] >= min_section_seconds or abs(boundary - duration) < 0.1:
            cleaned.append(boundary)
    if cleaned[-1] < duration:
        cleaned.append(duration)
    if len(cleaned) >= 3 and cleaned[-1] - cleaned[-2] < 7.0:
        cleaned.pop(-2)

    boundaries = cleaned
    while len(boundaries) - 1 > max_sections:
        lengths = [(boundaries[i + 1] - boundaries[i], i) for i in range(1, len(boundaries) - 1)]
        if not lengths:
            break
        _, remove_index = min(lengths, key=lambda item: item[0])
        boundaries.pop(remove_index)

    raw_energies = [section_energy(y, sr, a, b) for a, b in zip(boundaries[:-1], boundaries[1:])]
    avg_energy = float(np.mean(raw_energies)) + EPSILON
    max_energy = float(np.max(raw_energies)) + EPSILON

    # Build compact timbre/harmony signatures for repeated-section detection.
    # This is only used to decide whether "hook/chorus" is justified.
    signatures: List[Optional[np.ndarray]] = []
    for start_t, end_t in zip(boundaries[:-1], boundaries[1:]):
        try:
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            part = np.asarray(y[start_sample:end_sample], dtype=np.float32)
            if part.size < sr * 3:
                signatures.append(None)
                continue
            if sr != 22050:
                part = librosa.resample(part, orig_sr=sr, target_sr=22050)
                sig_sr = 22050
            else:
                sig_sr = sr
            max_len = int(sig_sr * 24)
            if len(part) > max_len:
                middle = len(part) // 2
                part = part[max(0, middle - max_len // 2):max(0, middle - max_len // 2) + max_len]
            peak = float(np.max(np.abs(part))) + EPSILON
            part = (part / peak).astype(np.float32)
            try:
                tonal = librosa.effects.harmonic(part, margin=2.0)
                if float(np.max(np.abs(tonal))) <= 1e-5:
                    tonal = part
            except Exception:
                tonal = part

            mfcc = np.mean(librosa.feature.mfcc(y=part, sr=sig_sr, n_mfcc=10), axis=1)
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + EPSILON)
            chroma = np.mean(librosa.feature.chroma_stft(y=tonal, sr=sig_sr, n_fft=2048, hop_length=1024), axis=1)
            chroma = chroma / (np.linalg.norm(chroma) + EPSILON)
            contrast = np.mean(librosa.feature.spectral_contrast(y=part, sr=sig_sr), axis=1)
            contrast = (contrast - np.mean(contrast)) / (np.std(contrast) + EPSILON)
            signature = np.concatenate([mfcc, chroma, contrast]).astype(np.float32)
            signature = signature / (np.linalg.norm(signature) + EPSILON)
            signatures.append(signature)
        except Exception:
            signatures.append(None)

    repeated_indices = set()
    for i in range(1, max(1, len(signatures) - 1)):
        sig_i = signatures[i]
        if sig_i is None:
            continue
        for j in range(i + 2, len(signatures) - 1):
            sig_j = signatures[j]
            if sig_j is None or len(sig_i) != len(sig_j):
                continue
            similarity = float(np.dot(sig_i, sig_j))
            # Require strong timbral/harmonic similarity and roughly comparable
            # section energy. This is much stronger evidence of a repeated hook
            # than position in the song alone.
            e1 = raw_energies[i] / avg_energy
            e2 = raw_energies[j] / avg_energy
            if similarity >= 0.88 and abs(e1 - e2) <= 0.28:
                repeated_indices.add(i)
                repeated_indices.add(j)

    sections: List[ArrangementSection] = []
    for idx, (start_t, end_t) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        energy_value = raw_energies[idx]
        rel_avg = energy_value / avg_energy
        rel_max = energy_value / max_energy
        length = end_t - start_t

        if rel_avg >= 1.10 or rel_max >= 0.86:
            energy_label = "High"
        elif rel_avg <= 0.82 or rel_max <= 0.52:
            energy_label = "Low"
        else:
            energy_label = "Medium"

        is_first = idx == 0
        is_last = idx == len(boundaries) - 2

        section_total = len(boundaries) - 1
        if section_total == 1:
            name = "Full Track"
        elif is_first:
            # Only call it an intro when it is actually a short opening section.
            # Otherwise avoid pretending the first long block has a known role.
            intro_limit = min(24.0, max(10.0, duration * 0.28))
            name = "Intro" if length <= intro_limit else "Opening Section"
        elif is_last and (energy_label == "Low" or length <= 24):
            name = "Outro"
        elif idx in repeated_indices and energy_label in {"High", "Medium"}:
            name = "Likely Hook / Chorus"
        elif energy_label == "High":
            name = "Peak Section"
        elif energy_label == "Low":
            name = "Breakdown / Low Energy"
        else:
            name = "Main Section"

        sections.append(ArrangementSection(
            name=name,
            start=float(start_t),
            end=float(end_t),
            avg_energy=float(energy_value),
            energy_label=energy_label,
        ))

    return sections

def calculate_scores(
    duration: float,
    loudness: LoudnessInfo,
    frequency: FrequencyInfo,
    rhythm: RhythmInfo,
    sections: List[ArrangementSection],
) -> Scores:
    """Measurement-driven scoring with continuous evidence instead of broad buckets."""
    bands = frequency.band_percentages
    style = STYLE_PRESETS[DEFAULT_STYLE]

    primary_values = np.array([
        bands["Sub"], bands["Bass / 808"], bands["Mud"], bands["Low Mids"],
        bands["Mids / Melody"], bands["Highs"], bands["Air"],
    ], dtype=float)
    primary_average = float(np.mean(primary_values))
    mud_excess = float(bands["Mud"] - primary_average)

    def closeness(value: float, center: float, soft_radius: float, hard_radius: float) -> float:
        distance = abs(float(value) - float(center))
        if distance <= soft_radius:
            return 1.0 - 0.25 * (distance / max(soft_radius, EPSILON))
        if distance <= hard_radius:
            return 0.75 * (1.0 - (distance - soft_radius) / max(hard_radius - soft_radius, EPSILON))
        overshoot = min((distance - hard_radius) / max(hard_radius, EPSILON), 1.0)
        return -0.35 * overshoot

    def range_fit(value: float, low: float, high: float, outer_low: float, outer_high: float) -> float:
        v = float(value)
        if low <= v <= high:
            center = (low + high) / 2.0
            half = max((high - low) / 2.0, EPSILON)
            return 1.0 - 0.15 * abs(v - center) / half
        if outer_low <= v < low:
            return (v - outer_low) / max(low - outer_low, EPSILON)
        if high < v <= outer_high:
            return (outer_high - v) / max(outer_high - high, EPSILON)
        if v < outer_low:
            return -min((outer_low - v) / max(abs(outer_low), 1.0), 1.0)
        return -min((v - outer_high) / max(abs(outer_high), 1.0), 1.0)

    # MIX: many continuous measurements must agree before the score can move a lot.
    bass_808 = float(bands["Bass / 808"])
    low_end = float(frequency.low_end_total_percent)
    mud = float(bands["Mud"])
    low_mids = float(bands["Low Mids"])
    mids = float(bands["Mids / Melody"])
    highs = float(bands["Highs"])
    harsh = float(bands["Harsh Zone"])
    mid_total = float(frequency.mid_total_percent)
    top_total = float(frequency.top_total_percent)

    mix_components = {
        "bass_808": range_fit(bass_808, 13.0, 22.0, 7.0, 40.0),
        "low_end": range_fit(low_end, 25.0, 42.0, 15.0, 62.0),
        "mud": range_fit(mud, 9.0, 14.5, 5.0, 21.0),
        "low_mids": range_fit(low_mids, 10.0, 19.0, 5.0, 28.0),
        "mids": range_fit(mids, 11.0, 23.0, 6.0, 34.0),
        "highs": range_fit(highs, 9.0, 21.0, 4.0, 40.0),
        "harsh": range_fit(harsh, 8.0, 15.0, 5.0, 24.0),
        "mid_total": range_fit(mid_total, 30.0, 48.0, 20.0, 60.0),
        "top_total": range_fit(top_total, 17.0, 32.0, 8.0, 43.0),
        "mud_excess": closeness(mud_excess, 0.0, 1.5, 5.0),
    }
    mix_weights = {
        "bass_808": 0.13, "low_end": 0.13, "mud": 0.12, "low_mids": 0.08,
        "mids": 0.10, "highs": 0.09, "harsh": 0.11, "mid_total": 0.08,
        "top_total": 0.08, "mud_excess": 0.08,
    }
    spectral_quality = sum(mix_components[k] * mix_weights[k] for k in mix_weights)
    dynamic_fit = range_fit(float(loudness.dynamic_range_db), 6.0, 14.5, 3.0, 21.0)
    rms_fit = range_fit(float(loudness.rms_db), style["rms_min"], style["rms_max"], style["rms_min"] - 5.0, style["rms_max"] + 4.0)

    clip = float(loudness.clipping_percent)
    if clip <= 0.002:
        clipping_fit = 1.0
    elif clip <= 0.01:
        clipping_fit = 0.75
    elif clip <= 0.05:
        clipping_fit = 0.35
    elif clip <= 0.10:
        clipping_fit = 0.0
    elif clip <= 0.25:
        clipping_fit = -0.45
    else:
        clipping_fit = -0.85

    technical_quality = spectral_quality * 0.72 + dynamic_fit * 0.14 + rms_fit * 0.06 + clipping_fit * 0.08
    mix_score = 63.0 + technical_quality * 27.0
    strong_count = sum(1 for v in mix_components.values() if v >= 0.75)
    weak_count = sum(1 for v in mix_components.values() if v < 0.20)
    if strong_count >= 8 and weak_count == 0 and dynamic_fit >= 0.65 and clipping_fit >= 0.35:
        mix_score += 4.0
    elif strong_count <= 4:
        mix_score -= 4.0
    if weak_count >= 3:
        mix_score -= min(9.0, 2.5 * weak_count)

    master_quality = dynamic_fit * 0.42 + rms_fit * 0.33 + clipping_fit * 0.25
    master_score = 65.0 + master_quality * 27.0

    # ARRANGEMENT: score quality only as strongly as SoundLens trusts its own section map.
    section_count = len(sections)
    energies = np.array([float(s.avg_energy) for s in sections], dtype=float) if sections else np.array([])
    lengths = np.array([max(0.0, float(s.end - s.start)) for s in sections], dtype=float) if sections else np.array([])
    confidence_parts = []
    if 4 <= section_count <= 8:
        confidence_parts.append(0.95)
    elif section_count == 3 or section_count == 9:
        confidence_parts.append(0.70)
    elif section_count in (1, 2) or section_count > 10:
        confidence_parts.append(0.35)
    else:
        confidence_parts.append(0.55)

    if lengths.size >= 2 and float(np.mean(lengths)) > EPSILON:
        length_cv = float(np.std(lengths) / (np.mean(lengths) + EPSILON))
        confidence_parts.append(0.90 if 0.10 <= length_cv <= 0.45 else 0.70 if 0.05 <= length_cv < 0.10 or 0.45 < length_cv <= 0.70 else 0.45)
    else:
        length_cv = 0.0
        confidence_parts.append(0.35)

    if energies.size >= 2 and float(np.mean(energies)) > EPSILON:
        energy_cv = float(np.std(energies) / (np.mean(energies) + EPSILON))
        confidence_parts.append(0.95 if 0.08 <= energy_cv <= 0.40 else 0.70 if 0.04 <= energy_cv < 0.08 or 0.40 < energy_cv <= 0.55 else 0.40)
    else:
        energy_cv = 0.0
        confidence_parts.append(0.35)

    repeated_indices = [i for i, sec in enumerate(sections) if "Likely Hook" in sec.name or "Chorus" in sec.name]
    peak_indices = [i for i, sec in enumerate(sections) if sec.name == "Peak Section"]
    confidence_parts.append(0.95 if len(repeated_indices) >= 2 else 0.75 if len(repeated_indices) == 1 or len(peak_indices) >= 1 else 0.55)
    structure_confidence = float(np.mean(confidence_parts)) if confidence_parts else 0.35

    energy_quality = range_fit(energy_cv, 0.10, 0.30, 0.03, 0.58) if energies.size >= 2 else 0.0
    pacing_quality = range_fit(length_cv, 0.10, 0.38, 0.02, 0.75) if lengths.size >= 2 else 0.0
    repeated_quality = 1.0 if len(repeated_indices) >= 2 else 0.55 if len(repeated_indices) == 1 else 0.30 if len(peak_indices) >= 1 else 0.0

    hook_energies = [s.avg_energy for s in sections if "Hook" in s.name or "Chorus" in s.name]
    main_energies = [s.avg_energy for s in sections if s.name in {"Main Section", "Peak Section"} and "Hook" not in s.name and "Chorus" not in s.name]
    contrast_quality = 0.0
    if hook_energies and main_energies:
        contrast_ratio = float(np.mean(hook_energies)) / max(float(np.mean(main_energies)), EPSILON)
        contrast_quality = range_fit(contrast_ratio, 1.05, 1.38, 0.90, 1.75)
    elif energies.size >= 3:
        interior = energies[1:-1]
        if interior.size >= 2 and float(np.mean(interior)) > EPSILON:
            movement = float(np.std(interior) / (np.mean(interior) + EPSILON))
            contrast_quality = range_fit(movement, 0.07, 0.22, 0.015, 0.50)

    edge_quality = 0.5
    if sections:
        intro_len = float(sections[0].end - sections[0].start)
        intro_fit = range_fit(intro_len, 5.0, 18.0, 0.0, 36.0)
        last = sections[-1]
        outro_len = float(last.end - last.start)
        outro_fit = range_fit(outro_len, 4.0, 22.0, 0.0, 45.0) if last.name == "Outro" else 0.25
        edge_quality = intro_fit * 0.60 + outro_fit * 0.40

    onset_fit = range_fit(float(rhythm.onset_density), 1.4, 5.8, 0.45, 8.5)
    arrangement_quality = energy_quality * 0.28 + pacing_quality * 0.18 + repeated_quality * 0.16 + contrast_quality * 0.18 + edge_quality * 0.08 + onset_fit * 0.12
    raw_arrangement = 60.0 + arrangement_quality * 31.0
    neutral = 72.0
    arrangement_score = neutral + (raw_arrangement - neutral) * structure_confidence
    if arrangement_quality >= 0.82 and structure_confidence >= 0.82:
        arrangement_score += 3.0
    if structure_confidence < 0.50:
        arrangement_score = neutral + (arrangement_score - neutral) * 0.55

    mix_score = int(round(clamp(mix_score, 35, 96)))
    master_score = int(round(clamp(master_score, 35, 96)))
    arrangement_score = int(round(clamp(arrangement_score, 35, 96)))

    release_score = (mix_score * 0.58) + (arrangement_score * 0.42)
    weakest_public_score = min(mix_score, arrangement_score)
    if weakest_public_score < 60:
        release_score = min(release_score, weakest_public_score + 8)
    elif weakest_public_score < 70:
        release_score = min(release_score, weakest_public_score + 10)
    elif weakest_public_score < 80:
        release_score = min(release_score, weakest_public_score + 12)
    release_score = int(round(clamp(release_score, 30, 95)))

    energy_score = clamp((loudness.rms_db + 24.0) / 1.6, 0, 10)
    bass_strength = clamp(frequency.low_end_total_percent / 4.5, 0, 10)
    brightness_score = clamp(frequency.brightness_centroid_hz / 500, 0, 10)
    darkness_score = clamp(10 - brightness_score, 0, 10)
    drum_bounce = clamp((rhythm.onset_density - 0.5) * 1.35, 0, 10)
    vocal_space = clamp(10 - max(0.0, mud_excess) * 1.5 - max(0.0, bands["Low Mids"] - 17.0) * 0.5, 0, 10)

    return Scores(
        mix=mix_score, master=master_score, arrangement=arrangement_score, release=release_score,
        energy=energy_score, bass_strength=bass_strength, darkness=darkness_score,
        brightness=brightness_score, drum_bounce=drum_bounce, vocal_space=vocal_space,
    )

def build_feedback(
    basic: BasicInfo,
    loudness: LoudnessInfo,
    frequency: FrequencyInfo,
    rhythm: RhythmInfo,
    sections: List[ArrangementSection],
    scores: Scores,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Build evidence-based feedback without forcing five generic problems."""
    bands = frequency.band_percentages
    style = STYLE_PRESETS[DEFAULT_STYLE]
    problems: List[Tuple[int, str, str]] = []
    artist_notes: List[str] = []
    producer_notes: List[str] = []
    master_notes: List[str] = []

    primary_values = [
        bands["Sub"], bands["Bass / 808"], bands["Mud"], bands["Low Mids"],
        bands["Mids / Melody"], bands["Highs"], bands["Air"],
    ]
    primary_average = float(np.mean(primary_values))
    mud_excess = bands["Mud"] - primary_average

    def add_problem(priority: int, problem: str, fix: str) -> None:
        problems.append((priority, problem, fix))

    # Clipping, limiting, RMS loudness, and dynamic range remain available as
    # measurements, but they no longer become the overview's main issue or fix.
    # SoundLens now prioritizes production, balance, rhythm, and arrangement.
    if loudness.clipping_detected:
        master_notes.append(
            f"Clipping measurement: {loudness.clipping_percent:.4f}% of samples near full scale."
        )
    else:
        master_notes.append("No near-full-scale samples were detected.")
    master_notes.append(
        f"Loudness measurement: {loudness.rms_db:.1f} dB RMS with {loudness.dynamic_range_db:.1f} dB peak-to-average range."
    )

    if frequency.low_end_total_percent > style["low_end_problem"]:
        add_problem(
            74,
            f"Low end dominates {frequency.low_end_total_percent:.1f}% of the measured balance.",
            "Check whether the 808 masks the vocal or melody before lowering it; use level, envelope, or selective EQ instead of a blind bass cut.",
        )
    elif frequency.low_end_total_percent < 13:
        add_problem(
            69,
            f"Low-end presence is light for rage/trap ({frequency.low_end_total_percent:.1f}%).",
            "Check the 808 level, octave, saturation, and compatibility with the kick.",
        )
    else:
        producer_notes.append(f"Low-end balance is usable for the selected style ({frequency.low_end_total_percent:.1f}%).")

    if bands["Mud"] > 17.0 and mud_excess > 2.0:
        add_problem(
            76,
            f"The 250-500 Hz area stands about {mud_excess:.1f} points above the track's average band level.",
            "Solo likely contributors and make a small cut only where the cloudiness is actually coming from.",
        )

    if bands["Low Mids"] > 20.0:
        add_problem(
            63,
            f"Low mids are unusually concentrated ({bands['Low Mids']:.1f}%).",
            "Check 500-1000 Hz on stacked melodies and vocals for masking or boxiness.",
        )
    if bands["Harsh Zone"] > 20.0:
        add_problem(
            73,
            f"Upper-mid energy is strong around 2-5 kHz ({bands['Harsh Zone']:.1f}%).",
            "Use dynamic EQ on the specific lead, vocal, clap, or hat that becomes sharp—not automatically on the full master.",
        )
    if bands["Highs"] < 6.0:
        add_problem(
            56,
            f"The top end is restrained ({bands['Highs']:.1f}%).",
            "Compare against a reference before adding hats, excitation, or a gentle high shelf.",
        )
    elif bands["Highs"] > 38.0:
        add_problem(
            64,
            f"The top end is unusually dominant ({bands['Highs']:.1f}%).",
            "Check hats and bright leads on headphones and tame only the source that feels sharp.",
        )

    if rhythm.onset_density < 0.7:
        add_problem(
            48,
            f"Rhythmic movement is sparse ({rhythm.onset_density:.2f} detected hits/sec).",
            "Add variation only if the song feels empty; silence may be part of the intended pocket.",
        )
    elif rhythm.onset_density > 7.5:
        add_problem(
            50,
            f"Rhythmic movement is extremely dense ({rhythm.onset_density:.2f} detected hits/sec).",
            "Check whether hats or percussion blur together, and remove parts only in sections that feel crowded.",
        )

    hook_energies = [s.avg_energy for s in sections if "Hook" in s.name or "Chorus" in s.name]
    main_energies = [s.avg_energy for s in sections if s.name in {"Main Section", "Peak Section"}]
    if hook_energies and main_energies:
        hook_energy = float(np.mean(hook_energies))
        main_energy = float(np.mean(main_energies))
        if hook_energy <= main_energy * 0.96:
            add_problem(
                47,
                "A repeated section that looks hook-like is quieter than the surrounding main sections.",
                "Listen at the estimated transitions and add contrast only if that repeated section is intended to be the hook.",
            )
        elif hook_energy <= main_energy * 1.01:
            producer_notes.append("The repeated hook-like section and surrounding sections have similar energy; this may be intentional.")

    intro = sections[0] if sections else None
    if intro and rhythm.seconds_per_bar > 0:
        intro_bars = intro.end / rhythm.seconds_per_bar
        if intro_bars > 12:
            add_problem(
                52,
                f"The detected intro is approximately {intro_bars:.1f} bars long.",
                "Consider introducing a defining vocal, drum, 808, or melodic moment earlier if retention feels slow.",
            )

    artist_notes.append(
        f"Detected Auto-Tune starting point: {basic.key} ({basic.key_confidence:.1f}% confidence). Verify by ear while key detection continues improving."
    )

    artist_notes.append(f"Beat energy: {scores.energy:.1f}/10. Darkness: {scores.darkness:.1f}/10.")
    artist_notes.append(f"Vocal space estimate: {scores.vocal_space:.1f}/10. Higher suggests less measured low-mid masking.")

    producer_notes.append(f"Selected style preset: {DEFAULT_STYLE}.")
    producer_notes.append(f"Dominant measured frequency area: {frequency.dominant_band}.")
    producer_notes.append(f"Detected rhythmic activity: {rhythm.drum_activity.lower()} ({rhythm.onset_density:.2f} hits/sec).")
    if basic.bpm > 0 and rhythm.estimated_bars > 0:
        producer_notes.append(f"Estimated length: {rhythm.estimated_bars} bars at {basic.bpm:.1f} BPM.")
    else:
        producer_notes.append("BPM was uncertain, so bar count was not estimated.")

    master_notes.append(
        f"Peak: {loudness.peak_db:.2f} dB. RMS: {loudness.rms_db:.2f} dB. Peak-to-average range: {loudness.dynamic_range_db:.2f} dB."
    )

    # Keep one problem per category and only return evidence-backed items.
    sorted_items = sorted(problems, key=lambda item: item[0], reverse=True)
    top_items = sorted_items[:5]
    top_problems = [problem for _, problem, _ in top_items]
    suggested_fixes = []
    for _, _, fix in sorted_items:
        if fix not in suggested_fixes:
            suggested_fixes.append(fix)

    next_steps = [fix for _, _, fix in top_items[:3]]
    if not top_problems:
        top_problems = ["No strong technical red flags were detected. Use a reference track and your ears for creative decisions."]
        next_steps = ["Compare the loudest hook and busiest verse against one reference at matched volume before changing anything."]

    if scores.release >= 85:
        next_steps.append("The technical reading is strong; avoid changing the mix unless a reference or listening test reveals a clear issue.")
    elif scores.release >= 70:
        next_steps.append("Address only the highest-confidence issue, export again, and compare at matched loudness.")
    else:
        next_steps.append("Fix the clearest technical issue first, then re-run SoundLens before making smaller changes.")

    return top_problems, suggested_fixes, artist_notes, producer_notes, master_notes, next_steps

def stem_metrics_from_file(stem_path: Path, name: str) -> StemMetrics:
    y, sr = load_audio(stem_path)
    loudness = analyze_loudness(y)
    frequency = analyze_frequency(y, sr)
    return StemMetrics(
        name=name,
        file_path=str(stem_path.resolve()),
        peak_db=loudness.peak_db,
        rms_db=loudness.rms_db,
        dynamic_range_db=loudness.dynamic_range_db,
        low_end_total_percent=frequency.low_end_total_percent,
        mid_total_percent=frequency.mid_total_percent,
        top_total_percent=frequency.top_total_percent,
        brightness_centroid_hz=frequency.brightness_centroid_hz,
        spectral_rolloff_hz=frequency.spectral_rolloff_hz,
    )


def find_demucs_stem_folder(audio_file: Path, demucs_output_dir: Path) -> Optional[Path]:
    song_stem = audio_file.stem
    matches = list(demucs_output_dir.glob(f"**/{song_stem}"))
    for match in matches:
        if match.is_dir() and (match / "vocals.wav").exists():
            return match
    return None


def run_demucs(audio_file: Path, demucs_output_dir: Path) -> Tuple[bool, str, Optional[Path]]:
    """
    Runs Demucs from Python. If Demucs is not installed, SoundLens keeps working
    and returns a readable status instead of crashing.
    """
    demucs_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        command = [
            sys.executable,
            "-m",
            "demucs",
            "--out",
            str(demucs_output_dir),
            str(audio_file),
        ]

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "Demucs failed with no message."
            return False, message, None

        stem_folder = find_demucs_stem_folder(audio_file, demucs_output_dir)
        if not stem_folder:
            return False, "Demucs ran, but SoundLens could not find the stem folder.", None

        return True, "Demucs stem separation complete.", stem_folder

    except Exception as error:
        return False, f"Demucs could not run: {error}", None


def analyze_stem_balance(audio_file: Path, demucs_output_dir: Path = Path("stems")) -> StemBalanceInfo:
    ok, status, stem_folder = run_demucs(audio_file, demucs_output_dir)

    if not ok or stem_folder is None:
        return StemBalanceInfo(
            enabled=True,
            status=status,
            confidence="None",
            warnings=[
                "Stem analysis did not run. Install with: pip install demucs",
                "Master WAV analysis still works, but vocal/beat balance will be missing.",
            ],
        )

    wanted = {
        "vocals": stem_folder / "vocals.wav",
        "drums": stem_folder / "drums.wav",
        "bass": stem_folder / "bass.wav",
        "other": stem_folder / "other.wav",
    }

    stems: Dict[str, StemMetrics] = {}
    warnings: List[str] = []

    for name, path in wanted.items():
        if path.exists():
            stems[name] = stem_metrics_from_file(path, name)
        else:
            warnings.append(f"Missing {name}.wav from Demucs output.")

    required = {"vocals", "drums", "bass", "other"}
    if not required.issubset(stems.keys()):
        return StemBalanceInfo(
            enabled=True,
            status="Stem folder found, but not all expected stems were created.",
            confidence="Low",
            stems=stems,
            warnings=warnings,
        )

    vocals = stems["vocals"]
    drums = stems["drums"]
    bass = stems["bass"]
    other = stems["other"]

    # Approximate beat loudness by combining drum, bass, and other RMS powers.
    beat_power = sum(10 ** (stem.rms_db / 10) for stem in [drums, bass, other])
    beat_rms_db = 10 * math.log10(max(beat_power, EPSILON))

    vocal_to_beat_db = round(vocals.rms_db - beat_rms_db, 2)
    bass_to_vocal_db = round(bass.rms_db - vocals.rms_db, 2)
    bass_to_other_db = round(bass.rms_db - other.rms_db, 2)
    drums_to_vocal_db = round(drums.rms_db - vocals.rms_db, 2)

    # These are diagnosis scores, not "quality" scores.
    # 100 means the relationship looks easier to mix; low means likely masking/burial.
    vocal_presence_score = int(clamp(100 - abs(vocal_to_beat_db + 3) * 10, 0, 100))
    bass_dominance_score = int(clamp(50 + bass_to_vocal_db * 6, 0, 100))
    beat_vocal_balance_score = int(clamp(100 - max(0, bass_to_vocal_db - 2) * 9 - max(0, -vocal_to_beat_db - 7) * 8, 0, 100))
    melody_presence_score = int(clamp(50 + (other.rms_db - bass.rms_db) * 5, 0, 100))

    if bass_to_vocal_db > 4:
        warnings.append("Bass/808 stem is much louder than the vocal stem. Vocal masking is likely.")
    if vocal_to_beat_db < -8:
        warnings.append("Vocal stem is sitting far behind the full beat estimate.")
    if other.rms_db < bass.rms_db - 10:
        warnings.append("Melody/other stem is much quieter than the bass stem. Melody may feel small or hidden.")

    return StemBalanceInfo(
        enabled=True,
        status=status,
        confidence="Medium - stem separation is AI-estimated, so listen for bleed/artifacts.",
        stems=stems,
        vocal_to_beat_db=vocal_to_beat_db,
        bass_to_vocal_db=bass_to_vocal_db,
        bass_to_other_db=bass_to_other_db,
        drums_to_vocal_db=drums_to_vocal_db,
        vocal_presence_score=vocal_presence_score,
        bass_dominance_score=bass_dominance_score,
        beat_vocal_balance_score=beat_vocal_balance_score,
        melody_presence_score=melody_presence_score,
        warnings=warnings,
    )

def analyze_audio(audio_file: Path, use_stems: bool = False, demucs_output_dir: Path = Path("stems")) -> SoundLensReport:
    print("\n[1/8] Loading audio...")
    y, sr = load_audio(audio_file)
    duration = float(librosa.get_duration(y=y, sr=sr))
    print("[1/8] Audio loaded")

    print("[2/8] Detecting BPM...")
    bpm = detect_bpm(y, sr)
    print("[2/8] BPM detected")

    print("[3/8] Detecting key...")
    key, key_note, key_mode, key_confidence = detect_key(y, sr)
    print("[3/8] Key analysis complete")

    basic = BasicInfo(
        file_name=audio_file.name,
        file_path=str(audio_file.resolve()),
        sample_rate=sr,
        duration_seconds=duration,
        bpm=bpm,
        key=key,
        key_note=key_note,
        key_mode=key_mode,
        key_confidence=key_confidence,
    )

    print("[4/8] Analyzing loudness...")
    loudness = analyze_loudness(y)
    print("[4/8] Loudness analysis complete")

    print("[5/8] Analyzing frequency balance...")
    frequency = analyze_frequency(y, sr)
    print("[5/8] Frequency analysis complete")

    print("[6/8] Building audio fingerprint...")
    fingerprint = analyze_audio_fingerprint(y, sr)
    print("[6/8] Audio fingerprint built")

    print("[7/8] Analyzing rhythm and arrangement...")
    rhythm = analyze_rhythm(y, sr, duration, bpm)
    sections = estimate_arrangement(y, sr, duration, rhythm)
    print("[7/8] Arrangement analysis complete")

    print("[8/8] Calculating scores and feedback...")
    scores = calculate_scores(duration, loudness, frequency, rhythm, sections)

    top_problems, fixes, artist_notes, producer_notes, master_notes, next_steps = build_feedback(
        basic,
        loudness,
        frequency,
        rhythm,
        sections,
        scores,
    )
    stem_balance = None
    if use_stems:
        print("[Stem] Running Demucs and analyzing separated stems...")
        stem_balance = analyze_stem_balance(audio_file, demucs_output_dir=demucs_output_dir)
        print(f"[Stem] {stem_balance.status}")

        if stem_balance.vocal_to_beat_db is not None:
            producer_notes.append(
                f"Stem balance: vocal-to-beat {stem_balance.vocal_to_beat_db:+.2f} dB, "
                f"bass-to-vocal {stem_balance.bass_to_vocal_db:+.2f} dB."
            )
            artist_notes.append(
                f"Vocal/beat balance score: {stem_balance.beat_vocal_balance_score}/100. "
                "This is based on AI-separated stems, not perfect project stems."
            )

            for warning in stem_balance.warnings[:3]:
                top_problems.insert(0, warning)
            top_problems = top_problems[:5]

            if stem_balance.beat_vocal_balance_score is not None and stem_balance.beat_vocal_balance_score < 60:
                next_steps.insert(0, "Use the separated stems to check whether the vocal is buried by the 808/beat before changing the whole master.")
            if stem_balance.melody_presence_score is not None and stem_balance.melody_presence_score < 40:
                next_steps.insert(0, "Check the melody/other stem. If it is clean but quiet, the melody may need more upper-mid presence or level.")
        else:
            producer_notes.append("Stem analysis was requested, but Demucs did not produce usable stems.")

    print("[8/8] Report generated")

    return SoundLensReport(
        basic=basic,
        loudness=loudness,
        frequency=frequency,
        fingerprint=fingerprint,
        rhythm=rhythm,
        sections=sections,
        scores=scores,
        stem_balance=stem_balance,
        top_problems=top_problems,
        suggested_fixes=fixes,
        artist_notes=artist_notes,
        producer_notes=producer_notes,
        master_notes=master_notes,
        next_steps=next_steps,
    )


def report_status(score: int) -> str:
    if score >= 85:
        return "Close to release-ready"
    if score >= 70:
        return "Good, but needs small fixes"
    if score >= 50:
        return "Needs work before release"
    return "Not ready yet"


def render_report(report: SoundLensReport) -> str:
    b = report.basic
    l = report.loudness
    f = report.frequency
    r = report.rhythm
    s = report.scores
    bands = f.band_percentages

    lines: List[str] = []
    add = lines.append

    add("\n=== SoundLens Pro Report ===")
    add(f"File: {b.file_name}")
    add(f"BPM: {b.bpm:.0f}")
    add(f"Key: {b.key}")
    add(f"Key Confidence: {b.key_confidence:.1f}%")
    add(f"Duration: {format_time(b.duration_seconds)} ({b.duration_seconds:.2f} sec)")
    add(f"Sample Rate: {b.sample_rate} Hz")

    add("\n=== Quick Verdict ===")
    add(f"Release Score: {s.release}/100 - {report_status(s.release)}")
    add(f"Mix Score: {s.mix}/100 - {score_label(s.mix)}")
    add(f"Master Score: {s.master}/100 - {score_label(s.master)}")
    add(f"Arrangement Score: {s.arrangement}/100 - {score_label(s.arrangement)}")

    add("\n=== Top Problems ===")
    for i, problem in enumerate(report.top_problems, 1):
        add(f"{i}. {problem}")

    add("\n=== Best Next Fixes ===")
    for i, fix in enumerate(report.next_steps, 1):
        add(f"{i}. {fix}")

    add("\n=== Arrangement Estimate ===")
    for section in report.sections:
        add(
            f"{section.name}: {format_time(section.start)} - {format_time(section.end)} "
            f"| Energy: {section.energy_label}"
        )

    add("\n=== Mix / Loudness Analysis ===")
    add(f"Peak Level: {l.peak_db:.2f} dB")
    add(f"Headroom: {l.headroom_db:.2f} dB")
    add(f"RMS Loudness: {l.rms_db:.2f} dB")
    add(f"Dynamic Range: {l.dynamic_range_db:.2f} dB")
    add(f"Clipping: {'Yes' if l.clipping_detected else 'No'}")
    add(f"Clipping Samples: {l.clipping_samples} ({l.clipping_percent:.5f}%)")

    add("\n=== Frequency Balance ===")
    add(f"Dominant Band: {f.dominant_band}")
    add(f"Low End Total: {f.low_end_total_percent:.2f}%")
    add(f"Mid Total: {f.mid_total_percent:.2f}%")
    add(f"Top End Total: {f.top_total_percent:.2f}%")
    for name in ["Sub", "Bass / 808", "Mud", "Low Mids", "Mids / Melody", "Harsh Zone", "Highs", "Air", "Vocal Range"]:
        add(f"{name}: {bands[name]:.2f}%")
    add(f"Brightness: {f.brightness_label} ({f.brightness_centroid_hz:.0f} Hz centroid)")
    add(f"Spectral Rolloff: {f.spectral_rolloff_hz:.0f} Hz")

    add("\n=== Rhythm / Drum Analysis ===")
    add(f"Onsets Detected: {r.onset_count}")
    add(f"Onset Density: {r.onset_density:.2f} hits/sec")
    add(f"Drum Activity: {r.drum_activity}")
    add(f"Estimated Bars: {r.estimated_bars}")
    add(f"Seconds Per Bar: {r.seconds_per_bar:.2f}")

    add("\n=== Beat Profile ===")
    add(f"Energy: {s.energy:.1f}/10")
    add(f"Bass Strength: {s.bass_strength:.1f}/10")
    add(f"Darkness: {s.darkness:.1f}/10")
    add(f"Brightness: {s.brightness:.1f}/10")
    add(f"Drum Bounce: {s.drum_bounce:.1f}/10")
    add(f"Vocal Space: {s.vocal_space:.1f}/10")

    if report.stem_balance:
        sb = report.stem_balance
        add("\n=== Stem Balance / Vocal vs Beat ===")
        add(f"Status: {sb.status}")
        add(f"Confidence: {sb.confidence}")
        if sb.vocal_to_beat_db is not None:
            add(f"Vocal to Beat: {sb.vocal_to_beat_db:+.2f} dB")
            add(f"Bass to Vocal: {sb.bass_to_vocal_db:+.2f} dB")
            add(f"Bass to Melody/Other: {sb.bass_to_other_db:+.2f} dB")
            add(f"Drums to Vocal: {sb.drums_to_vocal_db:+.2f} dB")
            add(f"Vocal Presence Score: {sb.vocal_presence_score}/100")
            add(f"Bass Dominance Score: {sb.bass_dominance_score}/100")
            add(f"Beat/Vocal Balance Score: {sb.beat_vocal_balance_score}/100")
            add(f"Melody Presence Score: {sb.melody_presence_score}/100")
        if sb.warnings:
            add("Stem Warnings:")
            for warning in sb.warnings:
                add(f"- {warning}")

    add("\n=== Artist Notes ===")
    for note in report.artist_notes:
        add(f"- {note}")

    add("\n=== Producer Notes ===")
    for note in report.producer_notes:
        add(f"- {note}")

    add("\n=== Master Notes ===")
    for note in report.master_notes:
        add(f"- {note}")

    add("\n=== Suggested Mix Moves ===")
    if report.suggested_fixes:
        for fix in report.suggested_fixes:
            add(f"- {fix}")
    else:
        add("- No huge technical mix move detected. Use reference tracks and make taste-based adjustments.")

    add("\n=== Suggested Master Chain ===")
    chain = [
        "1. Gain staging: make sure tracks and master are not clipping.",
        "2. EQ cleanup: remove mud/harshness only where needed.",
        "3. Saturation or soft clipper: add controlled loudness and 808 energy.",
        "4. Compression: control peaks if the beat feels uneven.",
        "5. Limiter: final loudness with safe output ceiling.",
        "6. Reference check: compare against one released track in the same style.",
    ]
    for step in chain:
        add(f"- {step}")

    add("\n=== SoundLens Reminder ===")
    add("Numbers help you find likely problems. Your ears still make the final call.")
    return "\n".join(lines)


def save_outputs(report: SoundLensReport, output_dir: Path, save_json: bool = True) -> Tuple[Path, Optional[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = Path(report.basic.file_name).stem.replace(" ", "_")
    txt_path = output_dir / f"{safe_stem}_soundlens_report.txt"
    txt_path.write_text(render_report(report), encoding="utf-8")

    json_path: Optional[Path] = None
    if save_json:
        json_path = output_dir / f"{safe_stem}_soundlens_report.json"
        json_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return txt_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoundLens Pro audio analyzer")
    parser.add_argument("audio", nargs="?", help="Path to audio file")
    parser.add_argument("--output", "-o", default="soundlens_reports", help="Folder to save reports")
    parser.add_argument("--no-json", action="store_true", help="Do not save JSON report")
    parser.add_argument("--stems", action="store_true", help="Run Demucs stem separation and add vocal/beat balance analysis")
    return parser.parse_args()

def loading_animation(stop_event):
    frames = ["|", "/", "-", "\\"]

    i = 0

    while not stop_event.is_set():
        print(
            f"\rAnalyzing track... {frames[i % len(frames)]}",
            end="",
            flush=True,
        )

        time.sleep(0.15)
        i += 1

    print("\rAnalysis complete.          ")

def main() -> int:
    args = parse_args()
    audio_name = args.audio or input("Enter audio file name: ").strip().strip('"')
    audio_file = Path(audio_name).expanduser()

    try:
        stop_event = threading.Event()

        loader = threading.Thread(
            target=loading_animation,
            args=(stop_event,),
        )

        loader.start()

        try:
            report = analyze_audio(audio_file, use_stems=args.stems)
        finally:
            stop_event.set()
            loader.join()

        text = render_report(report)
        print(text)

        txt_path, json_path = save_outputs(
            report,
            Path(args.output),
            save_json=not args.no_json,
        )

        print("\n=== Saved Files ===")
        print(f"Text Report: {txt_path}")

        if json_path:
            print(f"JSON Report: {json_path}")

        return 0

    except FileNotFoundError as error:
        print(f"Error: {error}")
        return 1

    except Exception as error:
        print(f"SoundLens crashed while analyzing the file: {error}")
        print("Try a WAV export first if the file type is weird.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
