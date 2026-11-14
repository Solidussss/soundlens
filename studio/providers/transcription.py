"""Audio → MIDI with Spotify Basic Pitch (Apache-2.0; model bundled in the pip package).

basic-pitch 0.4.0 declares TensorFlow as a dependency on Linux/Python ≥3.11, but
it runs on its bundled ONNX model with only ``onnxruntime``; install with
``pip install --no-deps basic-pitch`` plus pretty-midi, mir-eval, resampy<0.4.3,
scikit-learn, onnxruntime.

Transcribed notes are marked ``"source": "transcribed"`` and must be shown as
suggestions: our measurements (docs/studio/poc_results.md) show they are far
from exact, especially on distorted 808s.
"""
from __future__ import annotations

from pathlib import Path

from .base import Provider


class BasicPitchProvider(Provider):
    name = "basic_pitch"
    capabilities = {"transcribe"}
    cost_note = "CPU, faster than real time"

    def available(self) -> tuple[bool, str]:
        try:
            import basic_pitch  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError:
            return False, "pip install basic-pitch onnxruntime (see module docstring)"
        return True, "ONNX model bundled"

    def transcribe(self, audio: Path, bpm: float, onset_threshold: float = 0.5, frame_threshold: float = 0.3,
                   min_hz: float | None = None, max_hz: float | None = None) -> list[dict]:
        self.require()
        import logging
        logging.getLogger().setLevel(logging.ERROR)
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict
        _, _, events = predict(str(audio), ICASSP_2022_MODEL_PATH, onset_threshold=onset_threshold,
                               frame_threshold=frame_threshold, minimum_frequency=min_hz, maximum_frequency=max_hz)
        spb = 60.0 / bpm
        notes = []
        for start, end, pitch, amp, _bends in events:
            notes.append({"t": round(float(start) / spb, 4), "d": round(max(float(end - start) / spb, 1 / 64), 4),
                          "p": int(pitch), "v": round(float(min(1.0, max(0.05, amp))), 3)})
        return sorted(notes, key=lambda n: (n["t"], n["p"]))
