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
import json
import math
import sys
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

EPSILON = 1e-9
NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}

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


def load_audio(audio_file: Path) -> Tuple[np.ndarray, int]:
    if not audio_file.exists():
        raise FileNotFoundError(f"File not found: {audio_file}")
    if audio_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Warning: {audio_file.suffix} is not in the common supported list, but SoundLens will try to load it.")
    y, sr = librosa.load(audio_file, mono=True, sr=None)
    if y.size == 0:
        raise ValueError("Audio file loaded empty.")
    # Remove DC offset and normalize only for analysis stability, not for loudness numbers.
    y = y.astype(np.float32)
    y = y - float(np.mean(y))
    return y, sr


def detect_bpm(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])

    bpm = float(tempo)

    # Trap/rage often gets detected too slow.
    # Push suspicious low-mid tempos into a more useful rage/trap range.
    if bpm < 70:
        bpm *= 2
    elif 90 <= bpm <= 115:
        bpm *= 1.5
    elif 70 <= bpm < 90:
        bpm *= 2
    elif bpm > 190:
        bpm /= 2

    return round(bpm, 2)


def detect_key(y: np.ndarray, sr: int) -> Tuple[str, str, str, float]:
    """
    Railway-safe key detection.
    Avoids librosa.effects.harmonic()/HPSS because it can crash small containers.
    """
    def chroma_vector(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate, bins_per_octave=12)
        except Exception:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)

        chroma_mean = np.mean(chroma, axis=1)
        chroma_mean = np.maximum(chroma_mean - np.median(chroma_mean) * 0.25, 0)
        return chroma_mean / (np.linalg.norm(chroma_mean) + EPSILON)

    def key_candidates(chroma_norm: np.ndarray) -> List[Tuple[float, str, str]]:
        major_norm = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
        minor_norm = MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE)
        results = []

        for i, note in enumerate(NOTES_SHARP):
            results.append((float(np.dot(chroma_norm, np.roll(major_norm, i))), note, "Major"))
            results.append((float(np.dot(chroma_norm, np.roll(minor_norm, i))), note, "Minor"))

        return sorted(results, key=lambda x: x[0], reverse=True)

    try:
        max_seconds = 75

        if len(y) > sr * max_seconds:
            start = max(0, (len(y) // 2) - (sr * max_seconds // 2))
            y_key = y[start:start + (sr * max_seconds)]
        else:
            y_key = y

        target_sr = 22050

        if sr > target_sr:
            y_key = librosa.resample(y_key, orig_sr=sr, target_sr=target_sr)
            key_sr = target_sr
        else:
            key_sr = sr

        y_key = y_key.astype(np.float32)
        y_key = y_key - float(np.mean(y_key))

        chroma_norm = chroma_vector(y_key, key_sr)
        ranked = key_candidates(chroma_norm)

        best_score, best_note, best_mode = ranked[0]
        second_score, _, _ = ranked[1]

        gap = best_score - second_score
        confidence = clamp(42 + (gap * 90), 35, 88)

        return f"{best_note} {best_mode}", best_note, best_mode, confidence

    except Exception:
        return "Unknown", "Unknown", "Unknown", 0.0


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
    stft = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    band_raw: Dict[str, float] = {}
    for name, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)

        if np.any(mask):
            energy = np.sum(stft[mask] ** 2)
            band_raw[name] = float(np.log10(energy + 1))
        else:
            band_raw[name] = 0.0

    # Avoid double-counting vocal range and harsh zone in overall percent totals.
    primary_names = ["Sub", "Bass / 808", "Mud", "Low Mids", "Mids / Melody", "Highs", "Air"]
    total_primary = sum(band_raw[name] for name in primary_names) + EPSILON
    band_percentages = {name: (value / total_primary) * 100 for name, value in band_raw.items()}

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
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
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    fingerprint = {}

    for i in range(13):
        fingerprint[f"mfcc_{i+1}"] = float(np.mean(mfcc[i]))

    for i in range(12):
        fingerprint[f"chroma_{i+1}"] = float(np.mean(chroma[i]))

    fingerprint["spectral_contrast"] = float(np.mean(contrast))
    fingerprint["spectral_flatness"] = float(np.mean(flatness))
    fingerprint["zero_crossing_rate"] = float(np.mean(zcr))

    return fingerprint

def analyze_rhythm(y: np.ndarray, sr: int, duration: float, bpm: float) -> RhythmInfo:
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    onset_count = len(onset_times)
    onset_density = onset_count / max(duration, 1)
    drum_activity = level_label(onset_density, 1.5, 3.0)
    seconds_per_bar = (60 / max(bpm, 1)) * 4
    estimated_bars = int(round(duration / max(seconds_per_bar, EPSILON)))
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
    seconds_per_bar = rhythm.seconds_per_bar
    section_length = max(seconds_per_bar * 8, 16.0)

    boundaries = [0.0]
    current = section_length

    while current < duration - section_length:
        boundaries.append(current)
        current += section_length

    boundaries.append(duration)

    sections: List[ArrangementSection] = []
    energies = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        energies.append(section_energy(y, sr, start, end))

    avg_energy = float(np.mean(energies)) + EPSILON

    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        energy = energies[idx]
        ratio = energy / avg_energy

        if ratio >= 1.15:
            energy_label = "High"
        elif ratio <= 0.85:
            energy_label = "Low"
        else:
            energy_label = "Medium"

        last_idx = len(boundaries) - 2

        if idx == 0:
            name = "Intro"
        elif idx == last_idx:
            name = "Outro"
        elif idx == 1:
            name = "Chorus / Hook"
        elif idx == 2:
            name = "Verse"
        elif idx == 3:
            name = "Chorus / Hook"
        elif idx == 4:
            name = "Verse"
        elif energy_label == "High":
            name = "Hook / Drop"
        elif energy_label == "Low":
            name = "Bridge / Breakdown"
        else:
            name = "Verse"

        sections.append(
            ArrangementSection(
                name=name,
                start=start,
                end=end,
                avg_energy=energy,
                energy_label=energy_label,
            )
        )

    return sections

def calculate_scores(
    duration: float,
    loudness: LoudnessInfo,
    frequency: FrequencyInfo,
    rhythm: RhythmInfo,
    sections: List[ArrangementSection],
) -> Scores:
    """
    Scores the track using the default style preset.

    Important SoundLens rule:
    For rage/underground trap, heavy bass is not automatically a problem.
    It only becomes a penalty when it is extreme enough to likely mask the rest
    of the beat or vocal space.
    """
    bands = frequency.band_percentages
    targets = REFERENCE_TARGETS["trap_rage"]
    style = STYLE_PRESETS[DEFAULT_STYLE]

    mix_score = 100

    # Commercial trap/rage tracks often have light clipping/limiting.
    # Small clipping = small warning. Heavy clipping = bigger penalty.
    if loudness.clipping_percent > 1.0:
        mix_score -= 20
    elif loudness.clipping_percent > 0.1:
        mix_score -= 8
    elif loudness.clipping_detected:
        mix_score -= 3

    if loudness.peak_db > -0.1:
        mix_score -= 5
    elif loudness.peak_db > -0.3:
        mix_score -= 2

    # Use the style preset instead of judging every genre by one loudness range.
    if loudness.rms_db < style["rms_min"] or loudness.rms_db > style["rms_max"]:
        mix_score -= 8

    if loudness.dynamic_range_db < targets["dynamic_min"] or loudness.dynamic_range_db > targets["dynamic_max"]:
        mix_score -= 8

    # Rage/trap can be bass-heavy. Penalize lightly only when Bass/808 is extreme.
    if bands["Bass / 808"] > style["bass_808_problem"]:
        mix_score -= 3 if style["allow_heavy_bass"] else 10
    elif bands["Bass / 808"] < 10:
        mix_score -= 10

    if bands["Highs"] < targets["high_min"] or bands["Highs"] > targets["high_max"]:
        mix_score -= 8

    if bands["Mud"] > targets["mud_max"]:
        mix_score -= 6

    if bands["Harsh Zone"] > targets["harsh_max"]:
        mix_score -= 8

    master_score = 100

    if loudness.rms_db < style["rms_min"]:
        master_score -= 10
    elif loudness.rms_db > style["rms_max"]:
        master_score -= 6

    if loudness.peak_db > -0.1:
        master_score -= 5
    elif loudness.peak_db > -0.3:
        master_score -= 2

    if loudness.clipping_percent > 1.0:
        master_score -= 20
    elif loudness.clipping_percent > 0.1:
        master_score -= 8
    elif loudness.clipping_detected:
        master_score -= 3

    if loudness.dynamic_range_db < 6:
        master_score -= 10
    elif loudness.dynamic_range_db > 16:
        master_score -= 6

    # Low end is expected for rage/trap. Only punish it when it is past the style problem threshold.
    if frequency.low_end_total_percent > style["low_end_problem"]:
        master_score -= 5 if style["allow_heavy_bass"] else 12
    elif frequency.low_end_total_percent < 15:
        master_score -= 8

    if bands["Mud"] > 12:
        master_score -= 6

    if bands["Harsh Zone"] > 18:
        master_score -= 8

    if bands["Highs"] > 35:
        master_score -= 8
    elif bands["Highs"] < 8:
        master_score -= 5

    arrangement_score = 85

    if duration < 60:
        arrangement_score -= 25
    elif duration < 100:
        arrangement_score -= 10
    elif duration > 240:
        arrangement_score -= 5

    if rhythm.drum_activity == "High":
        arrangement_score -= 4
    elif rhythm.drum_activity == "Low":
        arrangement_score -= 4

    # Do not over-punish hook/verse estimates because the current arrangement model is still rough.
    if len(sections) >= 3:
        verse = sections[1].avg_energy if len(sections) > 1 else 0
        hook = sections[2].avg_energy if len(sections) > 2 else 0
        if hook <= verse * 1.03:
            arrangement_score -= 6

    mix_score = int(clamp(mix_score, 0, 100))
    master_score = int(clamp(master_score, 0, 100))
    arrangement_score = int(clamp(arrangement_score, 0, 100))
    release_score = int(round((mix_score * 0.35) + (master_score * 0.35) + (arrangement_score * 0.30)))

    energy_score = clamp(abs(loudness.rms_db + 25) / 2.0, 0, 10)
    bass_strength = clamp(frequency.low_end_total_percent / 5.0, 0, 10)
    brightness_score = clamp(frequency.brightness_centroid_hz / 500, 0, 10)
    darkness_score = clamp(10 - brightness_score, 0, 10)
    drum_bounce = clamp(rhythm.onset_density * 2.5, 0, 10)
    vocal_space = clamp(10 - ((bands["Mud"] / 2.5) + (bands["Low Mids"] / 7)), 0, 10)

    return Scores(
        mix=mix_score,
        master=master_score,
        arrangement=arrangement_score,
        release=release_score,
        energy=energy_score,
        bass_strength=bass_strength,
        darkness=darkness_score,
        brightness=brightness_score,
        drum_bounce=drum_bounce,
        vocal_space=vocal_space,
    )

def build_feedback(
    basic: BasicInfo,
    loudness: LoudnessInfo,
    frequency: FrequencyInfo,
    rhythm: RhythmInfo,
    sections: List[ArrangementSection],
    scores: Scores,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    bands = frequency.band_percentages
    style = STYLE_PRESETS[DEFAULT_STYLE]
    problems: List[Tuple[int, str]] = []
    fixes: List[str] = []
    artist_notes: List[str] = []
    producer_notes: List[str] = []
    master_notes: List[str] = []
    next_steps: List[str] = []

    if loudness.clipping_detected:
        problems.append((95, f"Clipping detected: {loudness.clipping_samples} samples are near/over the limit."))
        fixes.append("Lower the master/output, then use a soft clipper or limiter with safer ceiling.")
        master_notes.append("Clipping is the first thing to fix before judging the rest of the mix.")
    elif loudness.peak_db > -0.3:
        problems.append((75, "Peak is extremely close to 0 dB, so the export has almost no headroom."))
        fixes.append("Set limiter/output ceiling around -0.3 to -1.0 dB before exporting.")
    else:
        master_notes.append("No obvious clipping problem detected.")

    if loudness.rms_db < style["rms_min"]:
        problems.append((55, "Track is quiet for the selected style."))
        fixes.append("Add controlled loudness with gain staging, saturation, soft clipping, then limiting.")
    elif loudness.rms_db > style["rms_max"]:
        problems.append((65, "Track is extremely loud for the selected style and may be crushed."))
        fixes.append("Back off limiting/clipping only if the punch or clarity disappears.")
    else:
        master_notes.append("Loudness is in a usable range for the selected style.")

    if loudness.dynamic_range_db < 6:
        problems.append((60, "Dynamic range is tight, which can make the beat feel flattened."))
        fixes.append("Reduce over-compression or clipping on the master and let transients breathe.")
    elif loudness.dynamic_range_db > 16:
        problems.append((55, "Dynamic range is wide, so some sections may feel uneven."))
        fixes.append("Use automation or compression to make sections feel more controlled.")

    if frequency.low_end_total_percent > style["low_end_problem"]:
        problems.append((72, "Low end is extremely dominant, even for rage/underground trap."))
        fixes.append("Do not automatically turn the 808 down. First check whether the melody, vocal range, or upper mids are being masked.")
        producer_notes.append("Extreme low-end profile detected. This can be stylistically correct if the melody/vocal still cuts through.")
    elif frequency.low_end_total_percent > style["low_end_warning"]:
        producer_notes.append("Heavy bass detected, but this is normal for rage and underground trap. Judge masking before calling it a problem.")
    elif frequency.low_end_total_percent < 15:
        problems.append((70, "Low end is weak for rage/trap; the beat may not hit hard enough."))
        fixes.append("Raise the 808/bass, add saturation, or choose a stronger 808 sample.")
    else:
        producer_notes.append("Low-end presence looks usable for the selected style.")

    if bands["Mud"] > 12:
        problems.append((80, "Mud range is elevated around 250-500 Hz."))
        fixes.append("Gently cut 250-500 Hz on melodies, pads, or the master if the mix feels cloudy.")
    if bands["Low Mids"] > 24:
        problems.append((62, "Low mids are crowded, which can fight vocals and make the beat feel boxy."))
        fixes.append("Make room around 500-1000 Hz, especially if vocals will be added.")
    if bands["Harsh Zone"] > 18:
        problems.append((78, "Harsh zone is strong around 2k-5k."))
        fixes.append("Use dynamic EQ around 2k-5k on harsh leads, vocals, claps, or hats.")
    if bands["Highs"] < 8:
        problems.append((55, "High end is dull, so the beat may lack shine or air."))
        fixes.append("Add brightness carefully with hats, open hats, exciter, or a gentle high shelf.")
    elif bands["Highs"] > 35:
        problems.append((65, "High end is very strong and could become sharp on headphones."))
        fixes.append("Turn down harsh hats/leads or tame 6k-10k with EQ.")

    if rhythm.drum_activity == "Low":
        problems.append((45, "Drum movement is low; the beat may need more bounce or percussion variation."))
        fixes.append("Add hat rolls, percussion fills, or small drum changes every 4-8 bars.")
    elif rhythm.drum_activity == "High":
        problems.append((45, "Drum movement is very active; the beat could become overcrowded."))
        fixes.append("Mute extra percussion in some sections so the hook/drop feels bigger.")

    if len(sections) >= 3:
        verse = sections[1].avg_energy
        hook = sections[2].avg_energy
        if hook <= verse * 1.03:
            problems.append((45, "Hook/verse contrast looks low, but arrangement detection is still an estimate."))
            fixes.append("Check the waveform/sections by ear. If the hook feels flat, add contrast with melody layers, drum changes, or vocal energy.")

    intro = sections[0] if sections else None
    if intro and intro.end > rhythm.seconds_per_bar * 8.5:
        problems.append((50, "Intro may be long for short-form listener retention."))
        fixes.append("Consider bringing drums, 808, or a strong tag/moment in earlier.")

    artist_notes.append(f"Autotune/key starting point: {basic.key}. Confidence: {basic.key_confidence:.1f}%.")
    artist_notes.append(f"Beat energy: {scores.energy:.1f}/10. Darkness: {scores.darkness:.1f}/10.")
    artist_notes.append(f"Vocal space estimate: {scores.vocal_space:.1f}/10. Higher means easier space for vocals.")
    if frequency.low_end_total_percent > style["low_end_warning"]:
        artist_notes.append("Low end is very strong. For rage/trap this can be correct, but vocals may need low-mid cleanup to sit right.")
    if frequency.brightness_label == "Low":
        artist_notes.append("Beat leans dark, so brighter vocals/adlibs may cut through well.")

    producer_notes.append(f"Selected style preset: {DEFAULT_STYLE}.")
    producer_notes.append(f"Dominant frequency area: {frequency.dominant_band}.")
    producer_notes.append(f"Drum activity is {rhythm.drum_activity.lower()} at {rhythm.onset_density:.2f} hits/sec.")
    producer_notes.append(f"Estimated length: {rhythm.estimated_bars} bars at {basic.bpm:.0f} BPM.")

    master_notes.append(f"Peak: {loudness.peak_db:.2f} dB. RMS: {loudness.rms_db:.2f} dB. Dynamic range: {loudness.dynamic_range_db:.2f} dB.")

    sorted_problems = [text for _, text in sorted(problems, key=lambda item: item[0], reverse=True)]
    top_problems = sorted_problems[:5]
    unique_fixes = []
    for fix in fixes:
        if fix not in unique_fixes:
            unique_fixes.append(fix)

    if top_problems:
        next_steps.extend(unique_fixes[:3])
    else:
        top_problems.append("No major technical red flags. Focus on taste, arrangement, and reference comparison.")
        next_steps.append("Compare it to one reference track and adjust by ear, not just by numbers.")

    if scores.release >= 85:
        next_steps.append("This is close. Only make small changes unless you hear a clear problem.")
    elif scores.release >= 70:
        next_steps.append("Fix the top 1-2 issues, export again, then re-run SoundLens.")
    else:
        next_steps.append("Fix the technical problems first before worrying about tiny creative details.")

    return top_problems, unique_fixes, artist_notes, producer_notes, master_notes, next_steps



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
