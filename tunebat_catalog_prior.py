from __future__ import annotations

"""TuneBat catalog context for SoundLens Artist Match.

Raw SoundLens audio embeddings remain the primary identity signal. TuneBat metadata is used to:
1) repair historically broken profile BPM summaries,
2) remove the old prototype BPM-style bias,
3) provide a low-weight catalog tempo/key prior for ranking, and
4) enrich artist database entries with provisional song-level metadata until
   full audio fingerprints are available.
"""

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import compare_to_profile_pro as compare

CATALOG_PATH = Path(__file__).with_name("tunebat_catalog.json")
EXPANSION_PATH = Path(__file__).with_name("tunebat_catalog_expansion.json")
EPS = 1e-9

_base_compare_library = compare.compare_against_track_library
_base_load_track_library = compare.load_track_library
_base_track_style_similarity = compare.track_style_similarity

DEFAULT_DISPLAY_LABELS = {
    "bpm": "Tempo",
    "key": "Tonal Center",
    "energy": "Intensity",
    "danceability": "Groove",
    "loudness_db": "Master Level",
    "speechiness": "Vocal Density",
    "acousticness": "Acoustic Character",
    "instrumentalness": "Instrumental Lean",
    "liveness": "Live Feel",
    "happiness": "Mood Lift",
    "popularity": "Catalog Reach",
    "duration": "Length",
}


def _read_catalog_file(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _merge_catalogs() -> Tuple[Dict[str, Any], Dict[str, str]]:
    artists: Dict[str, Any] = {}
    labels = dict(DEFAULT_DISPLAY_LABELS)
    for path in (CATALOG_PATH, EXPANSION_PATH):
        data = _read_catalog_file(path)
        labels.update(data.get("display_labels") or {})
        for canonical, entry in (data.get("artists") or {}).items():
            if canonical not in artists:
                artists[canonical] = copy.deepcopy(entry)
                continue
            target = artists[canonical]
            target_aliases = list(target.get("aliases") or [])
            for alias in list((entry or {}).get("aliases") or []):
                if alias not in target_aliases:
                    target_aliases.append(alias)
            target["aliases"] = target_aliases
            existing_titles = {str(t.get("title") or "").strip().lower() for t in target.get("tracks") or []}
            for track in list((entry or {}).get("tracks") or []):
                title = str(track.get("title") or "").strip().lower()
                if title and title not in existing_titles:
                    target.setdefault("tracks", []).append(copy.deepcopy(track))
                    existing_titles.add(title)
    return artists, labels


CATALOG, DISPLAY_LABELS = _merge_catalogs()


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


def _public_track(track: Dict[str, Any]) -> Dict[str, Any]:
    """Expose SoundLens wording while preserving raw numeric meaning."""
    row = {"Song": track.get("title")}
    for raw_key, label in DISPLAY_LABELS.items():
        if raw_key in track and track.get(raw_key) is not None:
            row[label] = track.get(raw_key)
    return row


def load_track_library_with_catalog(profile_files):
    library, profiles = _base_load_track_library(profile_files)
    for artist_name, profile in profiles.items():
        canonical, entry = _catalog_entry(artist_name)
        tracks = list((entry or {}).get("tracks") or []) if entry else []
        bpms = _numeric([track.get("bpm") for track in tracks])
        if tracks and isinstance(profile, dict):
            # Keep catalog metadata separate from measured audio features.
            profile["catalog_context"] = {
                "source": "TuneBat public track metadata",
                "status": "metadata_profile",
                "canonical_artist": canonical,
                "track_count": len(tracks),
                "songs": [_public_track(track) for track in tracks],
                "display_labels": DISPLAY_LABELS,
                "bpm": _summary(bpms) if bpms else None,
                "keys": [str(track.get("key")) for track in tracks if track.get("key")],
            }
        if len(bpms) >= 2 and isinstance(profile, dict):
            averages = profile.setdefault("averages", {})
            if isinstance(averages, dict):
                old = copy.deepcopy(averages.get("bpm"))
                averages["legacy_bpm"] = old
                averages["bpm"] = _summary(bpms)
    return library, profiles


def track_style_similarity_without_legacy_bpm(report_dict, prototype):
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

        coverage = min(1.0, len(tracks) / 6.0)
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
