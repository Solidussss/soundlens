"""Character profiles: original fictional musical identities stored as data.

Adding a character means adding a JSON file to ``studio/characters/`` (or to
``<data dir>/studio/characters/`` for admin-created ones). The composer reads
these profiles; nothing in the generation engine is character-specific.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import dsp, synth
from .theory import SCALES, pitch_class

BUILTIN_DIR = Path(__file__).with_name("characters")
KICK_PATTERNS = {"rage", "plugg", "sparse", "boombap", "jersey", "trap"}
BASS_STYLES = {"follow_kick", "sustained"}
REQUIRED = ["id", "name", "tagline", "visual", "genre", "tempo", "keys", "drums", "bass", "melody", "harmony", "arrangement", "mix", "vocals"]


def validate_character(c: dict) -> dict:
    for k in REQUIRED:
        if k not in c:
            raise ValueError(f"character {c.get('id')!r} missing {k!r}")
    t = c["tempo"]
    if not (40 <= t["min"] <= t["default"] <= t["max"] <= 240):
        raise ValueError(f"{c['id']}: tempo range invalid")
    for s in c["keys"]["preferred_scales"]:
        if s not in SCALES:
            raise ValueError(f"{c['id']}: unknown scale {s}")
    for r in c["keys"].get("preferred_roots", []):
        pitch_class(r)
    for role, spec in c["drums"].items():
        if spec is None:
            continue
        synth.preset(spec["preset"])
        if role == "kick" and spec.get("pattern", "trap") not in KICK_PATTERNS:
            raise ValueError(f"{c['id']}: unknown kick pattern {spec.get('pattern')}")
    synth.preset(c["bass"]["preset"])
    if c["bass"].get("style", "follow_kick") not in BASS_STYLES:
        raise ValueError(f"{c['id']}: unknown bass style")
    synth.preset(c["melody"]["preset"])
    if c["harmony"].get("preset"):
        synth.preset(c["harmony"]["preset"])
    for e in c.get("extra", []):
        synth.preset(e["preset"])
    for part in ("bass", "melody"):
        for fx in c[part].get("fx", []):
            dsp.validate_effect(fx)
    for fx in c["mix"].get("master_fx", []):
        dsp.validate_effect(fx)
    for item in c["arrangement"]["template"]:
        name, bars = item.split(":")
        if int(bars) <= 0:
            raise ValueError(f"{c['id']}: bad arrangement item {item}")
    return c


def _dirs(extra_dir: Path | None) -> list[Path]:
    return [BUILTIN_DIR] + ([extra_dir] if extra_dir and extra_dir.exists() else [])


def load_all(extra_dir: Path | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in _dirs(extra_dir):
        for f in sorted(d.glob("*.json")):
            c = validate_character(json.loads(f.read_text()))
            out[c["id"]] = c
    return out


def get(character_id: str, extra_dir: Path | None = None) -> dict:
    chars = load_all(extra_dir)
    if character_id not in chars:
        raise ValueError(f"unknown character {character_id!r}; available: {sorted(chars)}")
    return chars[character_id]


def public_view(c: dict) -> dict:
    """What the character picker needs (no engine internals)."""
    return {
        "id": c["id"], "name": c["name"], "tagline": c["tagline"], "description": c.get("description", ""),
        "genre": c["genre"], "subgenre": c.get("subgenre"), "visual": c["visual"],
        "tempo": c["tempo"], "scales": c["keys"]["preferred_scales"], "vocals": c["vocals"],
        "avatar_svg": avatar_svg(c),
    }


def avatar_svg(c: dict, size: int = 160) -> str:
    """A simple original emblem drawn from the character's palette (no likeness of anyone)."""
    bg, main, accent = (c["visual"].get("palette", []) + ["#111", "#888", "#fff"])[:3]
    emblem = c["visual"].get("emblem", "circle")
    s = size
    shapes = {
        "visor": f'<path d="M{s*.2} {s*.62} Q{s*.5} {s*.12} {s*.8} {s*.62} Z" fill="{main}"/><rect x="{s*.27}" y="{s*.46}" width="{s*.46}" height="{s*.1}" rx="{s*.05}" fill="{accent}"/><path d="M{s*.5} {s*.46} l{s*.04} {s*.05} l-{s*.06} {s*.03} l{s*.03} {s*.02}" stroke="{bg}" stroke-width="2" fill="none"/>',
        "halo": f'<ellipse cx="{s*.5}" cy="{s*.28}" rx="{s*.24}" ry="{s*.07}" fill="none" stroke="{accent}" stroke-width="{s*.03}"/><circle cx="{s*.5}" cy="{s*.6}" r="{s*.2}" fill="{main}"/>',
        "static": "".join(f'<rect x="{s*(.25+(i%6)*.085):.1f}" y="{s*(.3+(i//6)*.085):.1f}" width="{s*.07:.1f}" height="{s*.07:.1f}" fill="{main if (i*7)%3 else accent}" opacity="{0.4+((i*13)%6)/10:.1f}"/>' for i in range(36)),
        "moon": f'<circle cx="{s*.5}" cy="{s*.5}" r="{s*.26}" fill="{main}"/><circle cx="{s*.6}" cy="{s*.44}" r="{s*.22}" fill="{bg}"/><circle cx="{s*.36}" cy="{s*.66}" r="{s*.04}" fill="{accent}"/>',
        "bolt": f'<path d="M{s*.56} {s*.18} L{s*.32} {s*.55} L{s*.48} {s*.55} L{s*.42} {s*.84} L{s*.7} {s*.42} L{s*.53} {s*.42} Z" fill="{main}" stroke="{accent}" stroke-width="3"/>',
    }
    body = shapes.get(emblem, f'<circle cx="{s/2}" cy="{s/2}" r="{s*.3}" fill="{main}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {s} {s}" width="{s}" height="{s}" role="img" aria-label="{c["name"]}">'
            f'<rect width="{s}" height="{s}" rx="{s*.18}" fill="{bg}"/>{body}</svg>')
