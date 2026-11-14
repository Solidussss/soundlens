"""Interchangeable back ends.

Each provider reports ``available()`` with a human-readable reason, so the UI
can show exactly why a feature is off instead of pretending it works.

* ``procedural`` – the built-in composer + synths (always available, CPU)
* ``acestep``    – ACE-Step 1.5 REST server (GPU); song generation, repaint, lego (add a track), extract
* ``demucs``     – source separation of uploaded or generated mixes (Architecture A)
* ``basic_pitch``– audio → MIDI transcription (results are suggestions, never ground truth)
"""
from __future__ import annotations

from .base import Provider
from .acestep import AceStepProvider
from .separation import DemucsProvider
from .transcription import BasicPitchProvider


def registry() -> dict[str, Provider]:
    return {p.name: p for p in (AceStepProvider(), DemucsProvider(), BasicPitchProvider())}


def status() -> list[dict]:
    out = [{"name": "procedural", "available": True, "reason": "built in", "capabilities": ["structure", "midi_parts", "render"]}]
    for p in registry().values():
        ok, reason = p.available()
        out.append({"name": p.name, "available": ok, "reason": reason, "capabilities": sorted(p.capabilities),
                    "cost_note": p.cost_note})
    return out
