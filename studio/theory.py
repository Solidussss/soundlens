"""Music theory helpers: pitch classes, scales, diatonic chords."""
from __future__ import annotations

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_ALIASES = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}

SCALES: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],          # natural minor / aeolian
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
}

# Scales with fewer than 7 degrees borrow chords from their parent 7-note scale.
_PARENT = {"minor_pentatonic": "minor", "major_pentatonic": "major"}


def pitch_class(name: str) -> int:
    n = name.strip()
    if not n:
        raise ValueError("empty key name")
    n = n[0].upper() + n[1:]
    n = _FLAT_ALIASES.get(n, n)
    if n not in NOTE_NAMES:
        raise ValueError(f"unknown key: {name!r}")
    return NOTE_NAMES.index(n)


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def scale_intervals(scale: str) -> list[int]:
    if scale not in SCALES:
        raise ValueError(f"unknown scale: {scale!r}; choose from {sorted(SCALES)}")
    return SCALES[scale]


def scale_pitches(root_pc: int, scale: str, low: int, high: int) -> list[int]:
    """All MIDI notes in [low, high] belonging to the scale."""
    pcs = {(root_pc + i) % 12 for i in scale_intervals(scale)}
    return [m for m in range(low, high + 1) if m % 12 in pcs]


def diatonic_chord(root_pc: int, scale: str, degree: int, size: int = 3) -> list[int]:
    """Pitch-class offsets (relative to C=0) of a stacked-thirds chord on a 0-based degree."""
    parent = _PARENT.get(scale, scale)
    steps = scale_intervals(parent)
    n = len(steps)
    out = []
    for k in range(size):
        idx = degree + 2 * k
        out.append((root_pc + steps[idx % n] + 12 * (idx // n)))
    return out  # absolute semitones above C0-of-root; caller places in octave


def chord_quality(chord: list[int]) -> str:
    rel = [(c - chord[0]) for c in chord[:3]]
    if rel[1:3] == [4, 7]:
        return "maj"
    if rel[1:3] == [3, 7]:
        return "min"
    if rel[1:3] == [3, 6]:
        return "dim"
    if rel[1:3] == [4, 8]:
        return "aug"
    return "other"


def key_label(root_pc: int, scale: str) -> str:
    return f"{NOTE_NAMES[root_pc]} {scale.replace('_', ' ')}"
