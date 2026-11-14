"""MIDI import/export for MIDI tracks (via ``mido``)."""
from __future__ import annotations

import io

import mido

from .project import track_notes_absolute

PPQ = 480
GM_DRUM_NOTES = {"kick": 36, "snare": 38, "clap": 39, "hat": 42, "perc": 37}


def track_to_midi_bytes(project: dict, track: dict) -> bytes:
    mid = mido.MidiFile(ticks_per_beat=PPQ)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("track_name", name=track["name"], time=0))
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(project["params"]["bpm"])), time=0))
    is_drum = track["role"] in GM_DRUM_NOTES
    channel = 9 if is_drum else 0
    events = []
    for n in track_notes_absolute(track):
        pitch = n["p"]
        if is_drum and track["role"] != "hat":
            pitch = GM_DRUM_NOTES[track["role"]]
        vel = max(1, min(127, int(round(n.get("v", 0.8) * 127))))
        on = int(round(n["t"] * PPQ))
        off = int(round((n["t"] + n["d"]) * PPQ))
        events.append((on, 1, mido.Message("note_on", note=pitch, velocity=vel, channel=channel)))
        events.append((max(off, on + 1), 0, mido.Message("note_off", note=pitch, velocity=0, channel=channel)))
    events.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for tick, _, msg in events:
        msg.time = tick - last
        last = tick
        tr.append(msg)
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def midi_bytes_to_notes(data: bytes) -> tuple[list[dict], float | None]:
    """Parse a MIDI file into notes (beats). Returns (notes, file_bpm)."""
    mid = mido.MidiFile(file=io.BytesIO(data))
    ppq = mid.ticks_per_beat
    bpm = None
    notes: list[dict] = []
    for track in mid.tracks:
        tick = 0
        active: dict[tuple[int, int], tuple[int, int]] = {}
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo" and bpm is None:
                bpm = mido.tempo2bpm(msg.tempo)
            elif msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (tick, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                k = (msg.channel, msg.note)
                if k in active:
                    t0, vel = active.pop(k)
                    notes.append({"t": t0 / ppq, "d": max((tick - t0) / ppq, 1 / 64), "p": msg.note, "v": round(vel / 127, 3)})
    return sorted(notes, key=lambda n: (n["t"], n["p"])), bpm
