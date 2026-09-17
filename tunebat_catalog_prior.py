from __future__ import annotations

"""TuneBat catalog context for SoundLens Artist Match.

This layer intentionally stays small. Raw SoundLens audio embeddings remain the
primary identity signal. TuneBat metadata is used to:
1) repair historically broken profile BPM summaries,
2) remove the old prototype BPM-style bias, and
3) provide a low-weight catalog tempo/key prior for ranking.

The catalog lives in tunebat_catalog.json and can be expanded without changing
matching code.
"""

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import compare_to_profile_pro as compare

CATALOG_PATH = Path(__file__).with_name("tunebat_catalog.json")
EPS = 1e-9

_base_compare_library = compare.compare_against_track_library
_base_load_track_library = compare.load_track_library
_base_track_style_similarity = compare.track_style_similarity


def _load_catalog() -> Dict[str, Any]:
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return data.get("artists", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


CATALOG = _load_catalog()


def _norm_name(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _catalog_entry(artist_name: Any) -> Tuple[str | None, Dict[str, Any] | None]:
    target = _norm_name(artist_name)
    if not target:
        return None, None
    for canonical, entry in CATALOG.items():
        aliases = [canonical] + list((entry or {}).get("aliases") or [])
        if any(_norm_name(alias) == target for alias in aliases):
            return canonical, entry
    return None, None


def _numeric(values: List[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            x = float(value)
            if math.isfinite(x):
                out.append(x)
        except Exception:
            pass
    return out


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    avg = sum(values) / len(values)
    variance = sum((x - avg) ** 2 for x in values) / max(1, len(values) - 1)
    return {
        "avg": round(avg, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "stdev": round(math.sqrt(max(0.0, variance)), 4),
    }


def load_track_library_with_catalog(profile_files):
    library, profiles = _base_load_track_library(profile_files)
    # Repair profile-level BPM metadata when TuneBat has at least two direct
    # artist tracks. Keep the original analyzer values under legacy_bpm.
    for artist_name, profile in profiles.items():
        canonical, entry = _catalog_entry(artist_name)
        tracks = list((entry or {}).get("tracks") or []) if entry else []
        bpms = _numeric([track.get("bpm") for track in tracks])
        if len(bpms) >= 2 and isinstance(profile, dict):
            averages = profile.setdefault("averages", {})
            if isinstance(averages, dict):
                old = copy.deepcopy(averages.get("bpm"))
                averages["legacy_bpm"] = old
                averages["bpm"] = _summary(bpms)
            profile["catalog_context"] = {
                "source": "TuneBat",
                "canonical_artist": canonical,
                "track_count": len(tracks),
                "bpm": _summary(bpms),
                "keys": [str(track.get("key")) for track in tracks if track.get("key")],
            }
    return library, profiles


def track_style_similarity_without_legacy_bpm(report_dict, prototype):
    # Old track prototypes were created when BPM frequently collapsed to 140.
    # Remove bpm_style from that legacy style bonus. Tempo now comes from the
    # independent catalog prior below and from the newly improved BPM detector.
    if not isinstance(prototype, dict):
        return _base_track_style_similarity(report_dict, prototype)
    clone = dict(prototype)
    style = dict(clone.get("style_fingerprint") or {})
    style.pop("bpm_style", None)
    clone["style_fingerprint"] = style
    return _base_track_style_similarity(report_dict, clone)


def _fold_bpm(bpm: float) -> List[float]:
    values = set()
    for factor in (0.5, 1.0, 2.0):
        value = float(bpm) * factor
        while value < 55:
            value *= 2.0
        while value > 190:
            value /= 2.0
        if 55 <= value <= 190:
            values.add(round(value, 4))
    return list(values)


def _bpm_distance(a: float, b: float) -> float:
    return min(abs(x - y) for x in _fold_bpm(a) for y in _fold_bpm(b))


def _bpm_fit(song_bpm: float, tracks: List[Dict[str, Any]]) -> float | None:
    bpms = _numeric([track.get("bpm") for track in tracks])
    if not bpms or song_bpm <= 0:
        return None
    distances = sorted(_bpm_distance(song_bpm, value) for value in bpms)
    # Use several closest catalog songs rather than one lucky neighbor.
    nearest = distances[: min(4, len(distances))]
    d = sum(nearest) / len(nearest)
    return max(0.0, min(100.0, 100.0 * math.exp(-((d / 14.0) ** 2))))


def _parse_key(value: Any) -> Tuple[str, str] | None:
    text = str(value or "").strip().replace("♯", "#").replace("♭", "b")
    if not text:
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    note = parts[0].upper().replace("DB", "C#").replace("EB", "D#").replace("GB", "F#").replace("AB", "G#").replace("BB", "A#")
    mode = parts[1].lower()
    mode = "Major" if mode.startswith("maj") else ("Minor" if mode.startswith("min") else parts[1].title())
    return note, mode


def _relative_key(note: str, mode: str) -> Tuple[str, str] | None:
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if note not in notes:
        return None
    idx = notes.index(note)
    if mode == "Major":
        return notes[(idx + 9) % 12], "Minor"
    if mode == "Minor":
        return notes[(idx + 3) % 12], "Major"
    return None


def _key_fit(song_key: Any, tracks: List[Dict[str, Any]]) -> float | None:
    parsed = _parse_key(song_key)
    catalog_keys = [_parse_key(track.get("key")) for track in tracks]
    catalog_keys = [item for item in catalog_keys if item]
    if not parsed or not catalog_keys:
        return None
    scores = []
    relative = _relative_key(*parsed)
    for key in catalog_keys:
        if key == parsed:
            scores.append(100.0)
        elif relative and key == relative:
            scores.append(78.0)
        elif key[0] == parsed[0]:
            scores.append(62.0)
        else:
            scores.append(35.0)
    scores.sort(reverse=True)
    top = scores[: min(4, len(scores))]
    return sum(top) / len(top)


def compare_against_track_library_with_catalog(report_dict, profile_files, top_n):
    ranked, profiles, nearest = _base_compare_library(report_dict, profile_files, max(top_n, 12))
    if not ranked:
        return ranked, profiles, nearest

    basic = report_dict.get("basic", {}) or {}
    try:
        song_bpm = float(basic.get("bpm") or 0.0)
    except Exception:
        song_bpm = 0.0
    song_key = basic.get("key") or ""

    for item in ranked:
        artist = item.get("profile_name")
        canonical, entry = _catalog_entry(artist)
        tracks = list((entry or {}).get("tracks") or []) if entry else []
        if not tracks:
            continue

        bpm_fit = _bpm_fit(song_bpm, tracks)
        key_fit = _key_fit(song_key, tracks)
        pieces = []
        if bpm_fit is not None:
            pieces.append((bpm_fit, 0.82))
        if key_fit is not None:
            pieces.append((key_fit, 0.18))
        if not pieces:
            continue
        prior = sum(score * weight for score, weight in pieces) / sum(weight for _, weight in pieces)

        # Coverage limits how much a small TuneBat sample can influence ranking.
        coverage = min(1.0, len(tracks) / 6.0)
        # Maximum movement is only about +/-4 points at full coverage. The audio
        # identity engine remains primary.
        adjustment = ((prior - 50.0) / 50.0) * 4.0 * coverage
        raw = float(item.get("match_score") or 0.0)
        item["match_score"] = round(max(0.0, min(96.0, raw + adjustment)), 2)
        components = item.setdefault("score_components", {})
        components["tunebat_catalog"] = round(prior, 2)
        components["tunebat_bpm_fit"] = round(bpm_fit, 2) if bpm_fit is not None else None
        components["tunebat_key_support"] = round(key_fit, 2) if key_fit is not None else None
        components["tunebat_catalog_tracks"] = len(tracks)
        item["catalog_context_source"] = "TuneBat"
        item["catalog_artist"] = canonical

    ranked.sort(key=lambda row: float(row.get("match_score") or 0.0), reverse=True)

    # Recompute confidence and labels after the small catalog adjustment.
    for index, item in enumerate(ranked):
        score = float(item.get("match_score") or 0.0)
        competitor = float(ranked[1]["match_score"] if index == 0 and len(ranked) > 1 else ranked[0]["match_score"])
        item["confidence"] = compare.confidence_from_gap(
            score, competitor,
            int(item.get("compared_fields") or 0),
            int(item.get("track_count") or 0),
        )
        item["confidence_percent"] = compare.confidence_percent(
            score, competitor,
            int(item.get("compared_fields") or 0),
            int(item.get("track_count") or 0),
        )
        item["match_label"] = compare.label_for_score(score, item.get("confidence"))

    return ranked[:top_n], profiles, nearest


def install() -> None:
    compare.load_track_library = load_track_library_with_catalog
    compare.track_style_similarity = track_style_similarity_without_legacy_bpm
    compare.compare_against_track_library = compare_against_track_library_with_catalog
    print(f"[soundlens] TuneBat catalog context loaded for {len(CATALOG)} artists")
