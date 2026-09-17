from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

# Import through sitecustomize-equivalent production module behavior when tests
# run normally from the repo. The core assertions intentionally target measured
# event families rather than exact timestamps/titles so calibration can evolve.
from soundlens_3d import build_visual_map


SR = 22050


def _tone(seconds: float, hz: float = 220.0, amp: float = 0.15, sr: int = SR) -> np.ndarray:
    n = max(1, int(seconds * sr))
    t = np.arange(n, dtype=float) / sr
    return amp * np.sin(2 * np.pi * hz * t)


def _stereo(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    right = left if right is None else right
    n = min(len(left), len(right))
    return np.vstack([left[:n], right[:n]])


def _analyze(audio: np.ndarray, sr: int = SR) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "fixture.wav"
        sf.write(path, audio.T if audio.ndim == 2 else audio, sr)
        return build_visual_map(path)


def _event_kinds(result: dict) -> set[str]:
    kinds: set[str] = set()
    for event in result.get("pins") or []:
        kinds.add(str(event.get("kind") or ""))
        kinds.update(str(k) for k in (event.get("evidence_types") or []))
    return kinds


def test_low_end_entrance_is_detected():
    quiet = _tone(6.0, 900.0, 0.08)
    body = _tone(6.0, 900.0, 0.10) + _tone(6.0, 55.0, 0.34)
    audio = _stereo(np.concatenate([quiet, body]))
    result = _analyze(audio)
    assert "bass" in _event_kinds(result)


def test_energy_drop_is_detected():
    loud = _tone(6.0, 220.0, 0.42) + _tone(6.0, 880.0, 0.18)
    quiet = _tone(6.0, 220.0, 0.06)
    audio = _stereo(np.concatenate([loud, quiet]))
    result = _analyze(audio)
    assert "energy" in _event_kinds(result)


def test_stereo_opening_is_detected():
    mono = _tone(6.0, 330.0, 0.18)
    wide_l = _tone(6.0, 330.0, 0.18) + _tone(6.0, 660.0, 0.12)
    wide_r = _tone(6.0, 330.0, 0.18) - _tone(6.0, 660.0, 0.12)
    left = np.concatenate([mono, wide_l])
    right = np.concatenate([mono, wide_r])
    result = _analyze(_stereo(left, right))
    assert "stereo" in _event_kinds(result)


def test_transient_burst_is_detected():
    base = _tone(12.0, 220.0, 0.08)
    # Short impulses spaced around the midpoint; Hann shapes avoid pathological
    # single-sample clipping while still producing strong onset evidence.
    for center_sec in (5.7, 6.0, 6.3):
        center = int(center_sec * SR)
        width = 320
        start = max(0, center - width // 2)
        end = min(len(base), start + width)
        burst = np.hanning(end - start) * 0.75
        base[start:end] += burst
    result = _analyze(_stereo(base))
    assert "transient" in _event_kinds(result)


def test_map_is_sparse_and_ranked():
    a = _tone(4.0, 440.0, 0.08)
    b = _tone(4.0, 55.0, 0.30) + _tone(4.0, 440.0, 0.20)
    c = _tone(4.0, 440.0, 0.05)
    result = _analyze(_stereo(np.concatenate([a, b, c])))
    pins = result.get("pins") or []
    assert len(pins) <= 14
    assert all(float(pins[i].get("time") or 0) <= float(pins[i + 1].get("time") or 0) for i in range(len(pins) - 1))
    assert all(0.0 <= float(p.get("confidence") or 0.0) <= 1.0 for p in pins)
