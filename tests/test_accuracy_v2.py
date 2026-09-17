import math
import numpy as np

import accuracy_v2


def _click_track(bpm: float, seconds: float = 24.0, sr: int = 16000) -> np.ndarray:
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    step = 60.0 / bpm
    for t in np.arange(0.25, seconds, step):
        i = int(t * sr)
        width = min(180, n - i)
        if width <= 0:
            continue
        env = np.exp(-np.linspace(0, 7, width)).astype(np.float32)
        y[i:i + width] += env
        # quiet subdivision gives realistic half/double ambiguity
        j = int((t + step / 2.0) * sr)
        if j < n:
            width2 = min(90, n - j)
            y[j:j + width2] += 0.22 * np.exp(-np.linspace(0, 7, width2)).astype(np.float32)
    return y


def _triad(root_midi: int, minor: bool, seconds: float = 18.0, sr: int = 22050) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    intervals = [0, 3 if minor else 4, 7]
    y = np.zeros_like(t)
    for interval, amp in zip(intervals, [1.0, 0.72, 0.62]):
        hz = 440.0 * (2.0 ** ((root_midi + interval - 69) / 12.0))
        y += amp * np.sin(2 * np.pi * hz * t)
        y += 0.18 * amp * np.sin(2 * np.pi * hz * 2.0 * t)
    # slow amplitude movement makes windows non-identical without changing tonality
    y *= (0.72 + 0.28 * np.sin(2 * np.pi * 0.12 * t) ** 2)
    return (y / max(np.max(np.abs(y)), 1e-9) * 0.7).astype(np.float32)


def test_bpm_known_tempos():
    for bpm in [68, 75, 92, 120, 140, 155, 174]:
        estimate = accuracy_v2.detect_bpm_v2(_click_track(bpm), 16000)
        assert min(abs(estimate - bpm), abs(estimate - bpm * 2), abs(estimate * 2 - bpm)) <= 2.0, (bpm, estimate)


def test_key_major_minor_examples():
    cases = [
        (60, False, "C", "Major"),
        (57, True, "A", "Minor"),
        (66, True, "F#", "Minor"),
        (63, False, "D#", "Major"),
    ]
    for midi, minor, note, mode in cases:
        label, got_note, got_mode, confidence = accuracy_v2.detect_key_v2(_triad(midi, minor), 22050)
        assert got_note == note, (label, note)
        assert got_mode == mode, (label, mode)
        assert 0 <= confidence <= 100


def test_artist_distance_is_key_transposition_tolerant():
    # Embedding layout: 40 MFCC values, then 24 chroma mean/std values, then extras.
    rng = np.random.default_rng(7)
    song = rng.normal(size=90).tolist()
    proto = list(song)
    # Circularly transpose only chroma pairs by five semitones.
    chroma = [proto[40 + i * 2:42 + i * 2] for i in range(12)]
    chroma = chroma[-5:] + chroma[:-5]
    proto[40:64] = [v for pair in chroma for v in pair]
    library = [song, proto]
    means = [float(np.mean([a, b])) for a, b in zip(*library)]
    stdevs = [max(float(np.std([a, b])), 1.0) for a, b in zip(*library)]
    weights = accuracy_v2.vector_weights_v2(len(song))
    distance = accuracy_v2.standardized_distance_v2(song, proto, means, stdevs, weights)
    assert distance is not None
    assert distance < 1e-6


def test_vector_weights_keep_chroma_below_timbre():
    weights = accuracy_v2.vector_weights_v2(90)
    assert np.mean(weights[:40]) > np.mean(weights[40:64])
    assert np.mean(weights[64:78]) > np.mean(weights[40:64])
