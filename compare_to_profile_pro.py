from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math

from soundlens_pro import analyze_audio


COMPARE_FIELDS = {
    "bpm": ["basic", "bpm"],
    "rms_db": ["loudness", "rms_db"],
    "dynamic_range_db": ["loudness", "dynamic_range_db"],
    "low_end_total_percent": ["frequency", "low_end_total_percent"],
    "mid_total_percent": ["frequency", "mid_total_percent"],
    "top_total_percent": ["frequency", "top_total_percent"],
    "brightness_centroid_hz": ["frequency", "brightness_centroid_hz"],
    "spectral_rolloff_hz": ["frequency", "spectral_rolloff_hz"],
    "onset_density": ["rhythm", "onset_density"],
    "energy": ["scores", "energy"],
    "bass_strength": ["scores", "bass_strength"],
    "darkness": ["scores", "darkness"],
    "brightness": ["scores", "brightness"],
    "drum_bounce": ["scores", "drum_bounce"],
    "vocal_space": ["scores", "vocal_space"],
    "stem_vocal_to_beat_db": ["stem_balance", "vocal_to_beat_db"],
    "stem_bass_to_vocal_db": ["stem_balance", "bass_to_vocal_db"],
    "stem_bass_to_other_db": ["stem_balance", "bass_to_other_db"],
    "stem_drums_to_vocal_db": ["stem_balance", "drums_to_vocal_db"],
    "stem_vocal_presence_score": ["stem_balance", "vocal_presence_score"],
    "stem_bass_dominance_score": ["stem_balance", "bass_dominance_score"],
    "stem_beat_vocal_balance_score": ["stem_balance", "beat_vocal_balance_score"],
    "stem_melody_presence_score": ["stem_balance", "melody_presence_score"],
}

FIELD_WEIGHTS = {
    "bpm": 0.75,
    "rms_db": 0.85,
    "dynamic_range_db": 0.75,
    "low_end_total_percent": 1.20,
    "mid_total_percent": 1.00,
    "top_total_percent": 1.00,
    "brightness_centroid_hz": 0.90,
    "spectral_rolloff_hz": 0.70,
    "onset_density": 1.00,
    "energy": 0.90,
    "bass_strength": 1.15,
    "darkness": 0.80,
    "brightness": 0.80,
    "drum_bounce": 1.00,
    "vocal_space": 1.00,
    "stem_vocal_to_beat_db": 1.50,
    "stem_bass_to_vocal_db": 1.50,
    "stem_bass_to_other_db": 1.10,
    "stem_drums_to_vocal_db": 0.85,
    "stem_vocal_presence_score": 1.10,
    "stem_bass_dominance_score": 1.15,
    "stem_beat_vocal_balance_score": 1.55,
    "stem_melody_presence_score": 1.30,
}

FREQUENCY_BANDS = [
    "Sub",
    "Bass / 808",
    "Mud",
    "Low Mids",
    "Mids / Melody",
    "Harsh Zone",
    "Highs",
    "Air",
]

EQ_BAND_CENTERS = {
    "Sub": 55,
    "Bass / 808": 140,
    "Mud": 350,
    "Low Mids": 750,
    "Mids / Melody": 2200,
    "Harsh Zone": 3600,
    "Highs": 8000,
    "Air": 12500,
}

STYLE_FINGERPRINT_FIELDS = [
    "bpm_style",
    "sub_to_bass_ratio",
    "low_end_focus",
    "mud_to_mid_ratio",
    "harsh_to_air_ratio",
    "top_brightness_balance",
    "vocal_space_band",
    "bass_vs_vocal_ratio",
    "section_count",
    "high_section_count",
    "low_section_count",
]


def get_nested(data: Dict[str, Any], path: List[str], default=None):
    current: Any = data

    for part in path:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        number = float(value)

        if math.isnan(number) or math.isinf(number):
            return None

        return number

    except Exception:
        return None


def normalize_bpm_for_style(bpm: float) -> float:
    bpm = float(bpm)

    while bpm < 80:
        bpm *= 2

    while bpm > 170:
        bpm /= 2

    return bpm


def ratio(a: float, b: float) -> float:
    if abs(b) <= 1e-9:
        return 0.0

    return float(a / b)


def summary_avg_stdev(profile: Dict[str, Any], field: str) -> Tuple[Optional[float], Optional[float]]:
    averages = profile.get("averages", {}) or {}
    fingerprint = profile.get("fingerprint", {}) or {}

    source = None

    if field in averages:
        source = averages[field]
    elif field in fingerprint:
        source = fingerprint[field]

    if not isinstance(source, dict):
        return None, None

    avg = to_float(source.get("avg"))
    stdev = to_float(source.get("stdev"))

    return avg, stdev


def score_against_profile(value: float, avg: float, stdev: Optional[float]) -> float:
    """Score one metric against an artist profile.

    Older versions were too forgiving: many underground profiles landed in the
    95-99% range because broad artist averages/stdevs made almost everything
    look close. This version is intentionally stricter so Artist Match has
    real separation.
    """
    if stdev is None or stdev <= 0:
        stdev = max(abs(avg) * 0.08, 0.75)

    # Prevent giant profile variance from making every artist look similar.
    stdev = max(min(stdev, max(abs(avg) * 0.35, 2.0)), 0.35)

    z = abs(value - avg) / stdev
    score = 100 - (z * 26)

    return max(0.0, min(100.0, score))


def cosine_similarity(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a <= 0 or norm_b <= 0:
        return None

    cosine = dot / (norm_a * norm_b)

    # Calibrated for artist matching.
    # Raw cosine often sits high for many related underground artists, which
    # made almost every profile look like a 95%+ match. A cosine below ~0.55 is
    # weak, ~0.75 is possible, ~0.88 is strong, and ~0.96+ is exceptional.
    score = ((cosine - 0.55) / 0.43) * 100

    return max(0.0, min(100.0, score))


def report_mfcc_vector(report_dict: Dict[str, Any]) -> List[float]:
    fp = report_dict.get("fingerprint", {}) or {}
    vector = []

    for i in range(1, 14):
        vector.append(float(fp.get(f"mfcc_{i}", 0) or 0))

    for i in range(1, 13):
        vector.append(float(fp.get(f"chroma_{i}", 0) or 0))

    for key in ["spectral_contrast", "spectral_flatness", "zero_crossing_rate"]:
        vector.append(float(fp.get(key, 0) or 0))

    return vector


def profile_mfcc_vector(profile: Dict[str, Any]) -> List[float]:
    averages = profile.get("averages", {}) or {}
    fingerprint = profile.get("fingerprint", {}) or {}
    audio_features = profile.get("audio_features", {}) or {}

    def avg_from_any(key: str) -> float:
        for group in (averages, audio_features, fingerprint):
            if not isinstance(group, dict):
                continue

            source = group.get(key, {})

            if isinstance(source, dict):
                value = source.get("avg")
                if value is not None:
                    return float(value or 0)

            if isinstance(source, (int, float)):
                return float(source)

        return 0.0

    vector = []

    for i in range(1, 14):
        vector.append(avg_from_any(f"mfcc_{i}"))

    for i in range(1, 13):
        vector.append(avg_from_any(f"chroma_{i}"))

    for key in ["spectral_contrast", "spectral_flatness", "zero_crossing_rate"]:
        vector.append(avg_from_any(key))

    return vector


def report_style_fingerprint(report_dict: Dict[str, Any]) -> Dict[str, float]:
    bands = get_nested(report_dict, ["frequency", "band_percentages"], {}) or {}

    sub = float(bands.get("Sub", 0) or 0)
    bass = float(bands.get("Bass / 808", 0) or 0)
    mud = float(bands.get("Mud", 0) or 0)
    low_mids = float(bands.get("Low Mids", 0) or 0)
    mids = float(bands.get("Mids / Melody", 0) or 0)
    harsh = float(bands.get("Harsh Zone", 0) or 0)
    highs = float(bands.get("Highs", 0) or 0)
    air = float(bands.get("Air", 0) or 0)
    vocal = float(bands.get("Vocal Range", 0) or 0)

    bpm = float(get_nested(report_dict, ["basic", "bpm"], 0) or 0)
    sections = report_dict.get("sections", []) or []

    high_sections = 0
    low_sections = 0

    for section in sections:
        label = section.get("energy_label", "")

        if label == "High":
            high_sections += 1
        elif label == "Low":
            low_sections += 1

    return {
        "bpm_style": normalize_bpm_for_style(bpm),
        "sub_to_bass_ratio": ratio(sub, bass),
        "low_end_focus": sub + bass,
        "mud_to_mid_ratio": ratio(mud + low_mids, mids + 0.001),
        "harsh_to_air_ratio": ratio(harsh, air + 0.001),
        "top_brightness_balance": highs + air,
        "vocal_space_band": vocal,
        "bass_vs_vocal_ratio": ratio(sub + bass, vocal + 0.001),
        "section_count": float(len(sections)),
        "high_section_count": float(high_sections),
        "low_section_count": float(low_sections),
    }


def style_fingerprint_score(report_dict: Dict[str, Any], profile: Dict[str, Any]) -> Optional[float]:
    profile_fp = profile.get("fingerprint", {}) or {}

    if not isinstance(profile_fp, dict):
        return None

    song_fp = report_style_fingerprint(report_dict)
    scores = []

    for field in STYLE_FINGERPRINT_FIELDS:
        source = profile_fp.get(field)

        if not isinstance(source, dict):
            continue

        avg = to_float(source.get("avg"))
        stdev = to_float(source.get("stdev"))
        value = to_float(song_fp.get(field))

        if value is None or avg is None:
            continue

        scores.append(score_against_profile(value, avg, stdev))

    if not scores:
        return None

    return round(sum(scores) / len(scores), 2)


def fingerprint_score(report_dict: Dict[str, Any], profile: Dict[str, Any]) -> Optional[float]:
    # First try the stronger MFCC/chroma fingerprint.
    mfcc_score = embedding_similarity(
        report_mfcc_vector(report_dict),
        profile_mfcc_vector(profile),
    )

    # Old profiles may not have MFCC/chroma yet.
    # Fall back to the existing profile["fingerprint"] ratios from build_artist_profile.
    style_score = style_fingerprint_score(report_dict, profile)

    if mfcc_score is not None and style_score is not None:
        return round((mfcc_score * 0.65) + (style_score * 0.35), 2)

    if mfcc_score is not None:
        return round(mfcc_score, 2)

    if style_score is not None:
        return round(style_score, 2)

    return None





def centered_scaled_vector(values: List[float]) -> List[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return []
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / max(len(clean), 1)
    stdev = math.sqrt(variance)
    if stdev <= 1e-9:
        return [0.0 for _ in values]
    return [float((float(v) - mean) / stdev) if v is not None and math.isfinite(float(v)) else 0.0 for v in values]


def embedding_similarity(a: List[float], b: List[float]) -> Optional[float]:
    """Shape-based embedding comparison.

    Raw cosine on MFCC/chroma stats can be dominated by scale and can make
    unrelated artists look close. Centering/scaling both vectors makes the
    comparison care more about the fingerprint shape.
    """
    if not a or not b or len(a) != len(b):
        return None
    return cosine_similarity(centered_scaled_vector(a), centered_scaled_vector(b))


def audio_embedding_vector_from_report(report_dict: Dict[str, Any]) -> List[float]:
    fp = report_dict.get("fingerprint", {}) or {}
    vector: List[float] = []

    for i in range(1, 21):
        vector.append(float(fp.get(f"embed_mfcc_{i}_mean", 0) or 0))
        vector.append(float(fp.get(f"embed_mfcc_{i}_std", 0) or 0))

    for i in range(1, 13):
        vector.append(float(fp.get(f"embed_chroma_{i}_mean", 0) or 0))
        vector.append(float(fp.get(f"embed_chroma_{i}_std", 0) or 0))

    for i in range(1, 8):
        vector.append(float(fp.get(f"embed_contrast_{i}_mean", 0) or 0))
        vector.append(float(fp.get(f"embed_contrast_{i}_std", 0) or 0))

    for key in [
        "embed_chroma_entropy",
        "embed_centroid_mean", "embed_centroid_std",
        "embed_rolloff_mean", "embed_rolloff_std",
        "embed_flatness_mean", "embed_flatness_std",
        "embed_zcr_mean", "embed_zcr_std",
        "embed_rms_mean", "embed_rms_std",
        "embed_onset_mean", "embed_onset_std",
    ]:
        vector.append(float(fp.get(key, 0) or 0))

    if not any(abs(v) > 1e-9 for v in vector):
        return report_mfcc_vector(report_dict)

    return vector


def track_style_similarity(report_dict: Dict[str, Any], prototype: Dict[str, Any]) -> Optional[float]:
    song_fp = report_style_fingerprint(report_dict)
    proto_fp = prototype.get("style_fingerprint", {}) or {}

    fields = [
        "bpm_style",
        "sub_to_bass_ratio",
        "low_end_focus",
        "mud_to_mid_ratio",
        "harsh_to_air_ratio",
        "top_brightness_balance",
        "vocal_space_band",
        "bass_vs_vocal_ratio",
        "section_count",
        "high_section_count",
        "low_section_count",
    ]

    scores = []

    for field in fields:
        value = to_float(song_fp.get(field))
        target = to_float(proto_fp.get(field))

        if value is None or target is None:
            continue

        # Per-track prototypes do not have stdev, so use a stable tolerance.
        tolerance = max(abs(target) * 0.18, 1.0)
        scores.append(score_against_profile(value, target, tolerance))

    if not scores:
        return None

    return round(sum(scores) / len(scores), 2)


def profile_prototype_score(report_dict: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    prototypes = profile.get("track_prototypes") or []

    if not isinstance(prototypes, list) or not prototypes:
        return None, []

    song_vector = audio_embedding_vector_from_report(report_dict)
    nearest = []

    for proto in prototypes:
        if not isinstance(proto, dict):
            continue

        proto_vector = proto.get("embedding_vector") or []
        embed_score = embedding_similarity(song_vector, proto_vector) if proto_vector else None
        style_score = track_style_similarity(report_dict, proto)

        parts = []
        if embed_score is not None:
            parts.append((embed_score, 0.72))
        if style_score is not None:
            parts.append((style_score, 0.28))

        if not parts:
            continue

        score = sum(s * w for s, w in parts) / sum(w for _, w in parts)

        nearest.append({
            "title": proto.get("title", "Unknown track"),
            "score": round(float(score), 2),
            "embedding": round(float(embed_score), 2) if embed_score is not None else None,
            "style": round(float(style_score), 2) if style_score is not None else None,
        })

    nearest.sort(key=lambda item: item["score"], reverse=True)

    if not nearest:
        return None, []

    best = nearest[0]["score"]
    top3 = nearest[:3]
    top5 = nearest[:5]
    top3_avg = sum(item["score"] for item in top3) / len(top3)
    top5_avg = sum(item["score"] for item in top5) / len(top5)

    # Do not let one random nearest track decide the whole artist match.
    # A real artist match should have several tracks in that artist profile close to the upload.
    final = (best * 0.35) + (top3_avg * 0.40) + (top5_avg * 0.25)

    return round(float(final), 2), nearest[:5]



def frequency_shape_score(report_dict: Dict[str, Any], profile: Dict[str, Any]) -> Optional[float]:
    bands = get_nested(report_dict, ["frequency", "band_percentages"], {}) or {}
    profile_bands = (
        profile.get("frequency_bands", {})
        or profile.get("bands", {})
        or profile.get("band_summary", {})
        or {}
    )

    if not isinstance(profile_bands, dict):
        return None

    band_weights = {
        "Sub": 1.20,
        "Bass / 808": 1.40,
        "Mud": 1.00,
        "Low Mids": 0.90,
        "Mids / Melody": 1.00,
        "Harsh Zone": 0.90,
        "Highs": 1.00,
        "Air": 0.80,
    }

    weighted_sum = 0.0
    total_weight = 0.0

    for band in FREQUENCY_BANDS:
        value = to_float(bands.get(band))
        source = profile_bands.get(band, {})

        if not isinstance(source, dict):
            continue

        avg = to_float(source.get("avg"))
        stdev = to_float(source.get("stdev"))

        if value is None or avg is None:
            continue

        score = score_against_profile(value, avg, stdev)
        weight = band_weights.get(band, 1.0)

        weighted_sum += score * weight
        total_weight += weight

    if total_weight <= 0:
        return None

    return round(weighted_sum / total_weight, 2)


def stem_component_score(field_scores: List[Dict[str, Any]]) -> Optional[float]:
    stem_scores = [
        item["score"]
        for item in field_scores
        if str(item["field"]).startswith("stem_")
    ]

    if not stem_scores:
        return None

    return round(sum(stem_scores) / len(stem_scores), 2)


def core_metric_score(field_scores: List[Dict[str, Any]]) -> Optional[float]:
    core_scores = [
        item["score"]
        for item in field_scores
        if not str(item["field"]).startswith("stem_")
    ]

    if not core_scores:
        return None

    return round(sum(core_scores) / len(core_scores), 2)


def label_for_score(score: float, confidence: str | None = None) -> str:
    """Display label only.

    Do not use old 90% style thresholds anymore. Artist Match V2 scores are
    intentionally conservative, so a 55-70 result can still be the correct
    closest style lane when there is a clear gap over the next artist.
    """
    score = float(score or 0)

    if score >= 78 and confidence in {"High", "Medium"}:
        return "Strong Match"
    if score >= 64 and confidence in {"High", "Medium"}:
        return "Good Match"
    if score >= 48:
        return "Possible Match"

    return "Weak Match"


def confidence_from_gap(best_score: float, second_score: float, compared_fields: int, track_count: int) -> str:
    """Display confidence based on separation.

    V2 scores are conservative. The important signal is not just the absolute
    percentage; it is whether the #1 artist clearly beats #2.
    """
    gap = float(best_score or 0) - float(second_score or 0)

    if track_count < 8 or compared_fields < 8:
        return "Low"

    if gap >= 18:
        return "High"

    if gap >= 8:
        return "Medium"

    return "Low"


def confidence_percent(best_score: float, second_score: float, compared_fields: int, track_count: int) -> int:
    gap = max(0.0, float(best_score or 0) - float(second_score or 0))
    field_bonus = min(12, compared_fields / 4)
    track_bonus = min(10, track_count / 8)

    raw = 35 + gap * 2.2 + field_bonus + track_bonus

    return int(max(20, min(96, round(raw))))



def clamp_score(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def calibrate_ranked_match_scores(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn raw similarity into believable public Artist Match scores.

    The raw component scores are useful for ranking, but they are not good
    public percentages because they cluster too tightly. This calibration:
    - keeps the same ranking,
    - spreads artists apart,
    - caps the winner when the gap is tiny,
    - prevents Low-confidence matches from being labeled 95% Strong Matches.
    """
    if not ranked:
        return ranked

    raw_scores = [float(item.get("match_score") or 0.0) for item in ranked]
    best_raw = raw_scores[0]
    second_raw = raw_scores[1] if len(raw_scores) > 1 else 0.0
    gap = max(0.0, best_raw - second_raw)

    sorted_raw = sorted(raw_scores)
    median_raw = sorted_raw[len(sorted_raw) // 2]

    previous_public = None

    for index, item in enumerate(ranked):
        raw = float(item.get("match_score") or 0.0)

        # Quality above the field median matters more than raw 90+ values.
        public = 58 + ((raw - median_raw) * 2.15)

        # Ranking penalty creates visible separation down the list.
        public -= index * 3.5

        # Distance from the best artist matters.
        public -= max(0.0, best_raw - raw) * 1.6

        # If the top two are basically tied, do not pretend the winner is a
        # 95% confident match.
        if index == 0:
            if gap < 1.0:
                public = min(public, 74)
            elif gap < 2.5:
                public = min(public, 80)
            elif gap < 5.0:
                public = min(public, 86)
            else:
                public = min(public, 94)
        else:
            # Keep every lower rank visibly below the previous one.
            if previous_public is not None:
                public = min(public, previous_public - 3.0)

        public = clamp_score(public, 18, 96)

        item["raw_match_score"] = round(raw, 2)
        item["match_score"] = round(public, 2)
        previous_public = public

    return ranked


def eq_gain_from_delta(delta: float) -> float:
    # Positive delta means the song has more of that band than target -> cut.
    gain = -delta * 0.45

    return round(max(-6.0, min(6.0, gain)), 2)


def eq_action(gain: float) -> str:
    if gain <= -1.0:
        return "Cut"

    if gain >= 1.0:
        return "Boost"

    return "Hold"


def eq_suggestions(report_dict: Dict[str, Any], profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not profile:
        return None

    bands = get_nested(report_dict, ["frequency", "band_percentages"], {}) or {}
    profile_bands = (
        profile.get("frequency_bands", {})
        or profile.get("bands", {})
        or profile.get("band_summary", {})
        or {}
    )

    if not isinstance(profile_bands, dict):
        return None

    points = []
    moves = []

    for band in FREQUENCY_BANDS:
        current = float(bands.get(band, 0) or 0)
        source = profile_bands.get(band, {})

        if not isinstance(source, dict):
            target = current
        else:
            target = float(source.get("avg", current) or current)

        delta = current - target
        gain_db = eq_gain_from_delta(delta)

        points.append({
            "band": band,
            "freq": EQ_BAND_CENTERS.get(band, 1000),
            "current": round(current, 2),
            "target": round(target, 2),
            "delta": round(delta, 2),
            "gain_db": gain_db,
            "action": eq_action(gain_db),
        })

        if abs(gain_db) >= 1.0:
            moves.append({
                "band": band,
                "action": eq_action(gain_db),
                "gain_db": gain_db,
                "reason": (
                    f"{band} is {abs(delta):.1f}% "
                    f"{'above' if delta > 0 else 'below'} the closest artist profile."
                ),
            })

    moves = sorted(moves, key=lambda item: abs(item["gain_db"]), reverse=True)[:5]

    return {
        "target_profile": profile.get("profile_name", "Closest profile"),
        "points": points,
        "moves": moves,
        "disclaimer": "Suggested EQ is a visual guide from frequency balance, not an exact mixing preset.",
    }


def style_suggestions_from_components(best: Dict[str, Any]) -> List[str]:
    components = best.get("score_components", {}) or {}

    fingerprint = components.get("fingerprint")
    frequency = components.get("frequency")
    stem = components.get("stem")
    core = components.get("core")

    suggestions = []

    if fingerprint is not None and fingerprint >= 85:
        suggestions.append("Your overall sonic fingerprint is very close to this artist/style.")
    elif fingerprint is not None and fingerprint < 70:
        suggestions.append("The overall tone is not fully locked to this artist yet. Focus on sound selection and texture.")

    if frequency is not None and frequency < 72:
        suggestions.append("Frequency balance is one of the main reasons the match is not higher.")

    if stem is not None and stem < 70:
        suggestions.append("Stem balance is pulling the match down. Check vocal, 808, and melody levels.")

    if core is not None and core < 72:
        suggestions.append("Core metrics like loudness, energy, or bounce are outside the profile range.")

    if not suggestions:
        suggestions.append("The track is sitting close to this profile. Make small taste-based changes, not huge moves.")

    return suggestions




def vector_weights(length: int) -> List[float]:
    """Weights matching audio_embedding_vector_from_report order."""
    weights: List[float] = []
    # 20 MFCC means/stds = 40 values. Timbre matters most for artist/style.
    weights.extend([1.75] * min(40, max(length - len(weights), 0)))
    # 12 chroma means/stds = 24 values. Useful, but keys transpose and can be noisy.
    weights.extend([0.85] * min(24, max(length - len(weights), 0)))
    # 7 spectral contrast means/stds = 14 values.
    weights.extend([1.15] * min(14, max(length - len(weights), 0)))
    # Extra scalar texture/rhythm/loudness-shape values.
    weights.extend([1.00] * max(length - len(weights), 0))
    return weights[:length]


def load_track_library(profile_files: List[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    library: List[Dict[str, Any]] = []
    profiles_by_name: Dict[str, Dict[str, Any]] = {}

    for profile_file in profile_files:
        try:
            profile = json.loads(profile_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        artist = profile.get("profile_name", profile_file.stem.replace("_profile", ""))
        profiles_by_name[artist] = profile
        prototypes = profile.get("track_prototypes") or []

        if not isinstance(prototypes, list):
            continue

        for proto in prototypes:
            if not isinstance(proto, dict):
                continue
            vector = proto.get("embedding_vector") or []
            if not isinstance(vector, list) or not vector:
                continue
            clean_vector = []
            for value in vector:
                number = to_float(value)
                clean_vector.append(float(number or 0.0))
            if not any(abs(v) > 1e-9 for v in clean_vector):
                continue

            library.append({
                "artist": artist,
                "title": proto.get("title", "Unknown track"),
                "vector": clean_vector,
                "style_fingerprint": proto.get("style_fingerprint", {}) or {},
                "profile_track_count": int(profile.get("track_count", 0) or 0),
            })

    return library, profiles_by_name


def library_stats(library: List[Dict[str, Any]], dims: int) -> Tuple[List[float], List[float]]:
    if not library or dims <= 0:
        return [], []

    means: List[float] = []
    stdevs: List[float] = []

    for index in range(dims):
        values = []
        for item in library:
            vector = item.get("vector") or []
            if index < len(vector):
                values.append(float(vector[index]))
        if not values:
            means.append(0.0)
            stdevs.append(1.0)
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        stdev = math.sqrt(variance)
        means.append(mean)
        stdevs.append(stdev if stdev > 1e-9 else 1.0)

    return means, stdevs


def standardized_distance(
    song_vector: List[float],
    proto_vector: List[float],
    means: List[float],
    stdevs: List[float],
    weights: List[float],
) -> Optional[float]:
    dims = min(len(song_vector), len(proto_vector), len(means), len(stdevs), len(weights))
    if dims <= 0:
        return None

    weighted_sum = 0.0
    total_weight = 0.0

    for index in range(dims):
        stdev = stdevs[index] if stdevs[index] > 1e-9 else 1.0
        a = (float(song_vector[index]) - means[index]) / stdev
        b = (float(proto_vector[index]) - means[index]) / stdev
        diff = a - b
        weight = max(0.0, float(weights[index]))
        weighted_sum += (diff * diff) * weight
        total_weight += weight

    if total_weight <= 0:
        return None

    return math.sqrt(weighted_sum / total_weight)


def distance_to_similarity(distance: float) -> float:
    """Convert standardized distance to a public-ish similarity score.

    This is intentionally conservative. A close track can score high, but broad
    underground similarity should not automatically become 95%.
    """
    if not math.isfinite(distance):
        return 0.0
    score = 100.0 * math.exp(-((distance / 1.18) ** 2))
    return max(0.0, min(100.0, score))


def style_distance_bonus(report_dict: Dict[str, Any], proto: Dict[str, Any]) -> float:
    """Small adjustment using style fingerprint, not enough to overpower audio embedding."""
    style_score = track_style_similarity(report_dict, {
        "style_fingerprint": proto.get("style_fingerprint", {}) or {}
    })
    if style_score is None:
        return 0.0
    # Convert 0-100 style score into roughly -6 to +6.
    return max(-6.0, min(6.0, (float(style_score) - 65.0) / 6.0))


def compare_against_track_library(
    report_dict: Dict[str, Any],
    profile_files: List[Path],
    top_n: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Artist match v2: nearest songs first, then artist voting.

    Old system compared one upload to one averaged artist profile. That loses
    too much detail because one artist can have many different sounds. This
    searches every saved track prototype first, then rolls the closest tracks
    up into artists.
    """
    library, profiles_by_name = load_track_library(profile_files)
    if not library:
        return [], profiles_by_name, []

    song_vector = audio_embedding_vector_from_report(report_dict)
    if not song_vector or not any(abs(v) > 1e-9 for v in song_vector):
        return [], profiles_by_name, []

    dims = min(len(song_vector), min(len(item["vector"]) for item in library))
    song_vector = song_vector[:dims]
    means, stdevs = library_stats(library, dims)
    weights = vector_weights(dims)

    nearest: List[Dict[str, Any]] = []

    for item in library:
        proto_vector = (item.get("vector") or [])[:dims]
        distance = standardized_distance(song_vector, proto_vector, means, stdevs, weights)
        if distance is None:
            continue
        similarity = distance_to_similarity(distance)
        similarity = max(0.0, min(100.0, similarity + style_distance_bonus(report_dict, item)))
        nearest.append({
            "artist": item.get("artist"),
            "title": item.get("title"),
            "distance": round(float(distance), 4),
            "score": round(float(similarity), 2),
        })

    nearest.sort(key=lambda item: (float(item.get("score") or 0), -float(item.get("distance") or 999)), reverse=True)

    # Use enough neighbors for stability, but not so many that the whole scene blends together.
    vote_pool = nearest[:35]
    artist_votes: Dict[str, Dict[str, Any]] = {}

    for rank, track in enumerate(vote_pool, start=1):
        artist = str(track.get("artist") or "Unknown")
        score = float(track.get("score") or 0.0)
        # Steep rank decay: top tracks matter way more than the 30th closest track.
        rank_weight = 1.0 / (rank ** 0.82)
        vote = (score / 100.0) * rank_weight

        bucket = artist_votes.setdefault(artist, {
            "profile_name": artist,
            "vote": 0.0,
            "best_track_score": 0.0,
            "nearest_tracks": [],
            "top10_count": 0,
            "top20_count": 0,
        })
        bucket["vote"] += vote
        bucket["best_track_score"] = max(bucket["best_track_score"], score)
        if rank <= 10:
            bucket["top10_count"] += 1
        if rank <= 20:
            bucket["top20_count"] += 1
        if len(bucket["nearest_tracks"]) < 5:
            bucket["nearest_tracks"].append({
                "title": track.get("title"),
                "score": round(score, 2),
                "rank": rank,
                "distance": track.get("distance"),
            })

    if not artist_votes:
        return [], profiles_by_name, nearest[:20]

    total_vote = sum(bucket["vote"] for bucket in artist_votes.values()) or 1.0
    ranked: List[Dict[str, Any]] = []

    for artist, bucket in artist_votes.items():
        profile = profiles_by_name.get(artist, {})
        vote_share = bucket["vote"] / total_vote
        best_track = float(bucket.get("best_track_score") or 0.0)
        nearest_tracks = bucket.get("nearest_tracks") or []
        avg_nearest = sum(float(t.get("score") or 0.0) for t in nearest_tracks[:3]) / max(len(nearest_tracks[:3]), 1)
        top10_count = int(bucket.get("top10_count") or 0)
        top20_count = int(bucket.get("top20_count") or 0)

        # Public score is driven by artist vote concentration + close-song quality.
        # This is not pretending exact identity; it says how strong the style lane is.
        match_score = (
            (vote_share * 100.0 * 0.62)
            + (best_track * 0.20)
            + (avg_nearest * 0.12)
            + (min(top10_count, 5) * 1.2)
            + (min(top20_count, 8) * 0.45)
        )
        match_score = max(0.0, min(96.0, match_score))

        ranked.append({
            "profile_name": artist,
            "match_score": round(match_score, 2),
            "match_label": "Pending",
            "confidence": "Pending",
            "confidence_percent": 0,
            "track_count": int(profile.get("track_count", bucket.get("profile_track_count", 0)) or 0),
            "compared_fields": dims,
            "score_components": {
                "library_vote_share": round(vote_share * 100.0, 2),
                "best_track": round(best_track, 2),
                "top3_track_avg": round(avg_nearest, 2),
                "top10_count": top10_count,
                "top20_count": top20_count,
                "prototype": round(best_track, 2),
                "fingerprint": round(avg_nearest, 2),
                "frequency": None,
                "overall_metrics": None,
                "stem": None,
                "core": None,
            },
            "field_scores": [],
            "nearest_tracks": nearest_tracks,
        })

    ranked.sort(key=lambda item: float(item.get("match_score") or 0.0), reverse=True)

    # Keep clear separation. If many artists are close, confidence stays low.
    for index, item in enumerate(ranked):
        best = float(item.get("match_score") or 0.0)
        competitor = float(ranked[1]["match_score"] if index == 0 and len(ranked) > 1 else ranked[0]["match_score"])
        item["confidence"] = confidence_from_gap(
            best,
            competitor,
            int(item.get("compared_fields", 0) or 0),
            int(item.get("track_count", 0) or 0),
        )
        item["confidence_percent"] = confidence_percent(
            best,
            competitor,
            int(item.get("compared_fields", 0) or 0),
            int(item.get("track_count", 0) or 0),
        )
        item["match_label"] = label_for_score(best, item.get("confidence"))

    return ranked[:top_n], profiles_by_name, nearest[:20]



def calculate_display_component_scores(
    item: Dict[str, Any],
    report_dict: Dict[str, Any],
    profile: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Populate Artist Match component cards without changing ranking.

    These values are display-only. They do not feed back into artist voting,
    match_score, confidence, labels, or ordering.
    """
    components = item.get("score_components") or {}
    if not isinstance(components, dict):
        components = {}

    def clean(value, fallback=None):
        try:
            if value is None:
                return fallback
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return fallback
            return round(max(0.0, min(100.0, number)), 2)
        except Exception:
            return fallback

    public_score = clean(item.get("match_score"), 0.0)
    fingerprint = clean(components.get("fingerprint"), public_score)

    # Keep the existing fingerprint value because it comes directly from the
    # nearest-track match engine that already determines the correct artists.
    components["fingerprint"] = fingerprint

    if profile:
        frequency = frequency_shape_score(report_dict, profile)

        # Arrangement display score: section shape + energy/rhythm similarity.
        song_style = report_style_fingerprint(report_dict)
        arrangement_scores: List[float] = []
        for field in ["section_count", "high_section_count", "low_section_count"]:
            avg, stdev = summary_avg_stdev(profile, field)
            value = to_float(song_style.get(field))
            if value is not None and avg is not None:
                arrangement_scores.append(score_against_profile(value, avg, stdev))

        for field in ["onset_density", "energy"]:
            path = COMPARE_FIELDS.get(field)
            value = to_float(get_nested(report_dict, path, None)) if path else None
            avg, stdev = summary_avg_stdev(profile, field)
            if value is not None and avg is not None:
                arrangement_scores.append(score_against_profile(value, avg, stdev))

        arrangement = (
            sum(arrangement_scores) / len(arrangement_scores)
            if arrangement_scores else None
        )

        # Core display score: broad musical/mix identity, excluding frequency
        # and section-shape fields already represented in their own cards.
        core_fields = [
            "bpm",
            "rms_db",
            "dynamic_range_db",
            "onset_density",
            "energy",
            "bass_strength",
            "darkness",
            "brightness",
            "drum_bounce",
            "vocal_space",
            "stem_vocal_to_beat_db",
            "stem_bass_to_vocal_db",
            "stem_beat_vocal_balance_score",
            "stem_melody_presence_score",
        ]
        core_scores: List[float] = []
        for field in core_fields:
            path = COMPARE_FIELDS.get(field)
            if not path:
                continue
            value = to_float(get_nested(report_dict, path, None))
            avg, stdev = summary_avg_stdev(profile, field)
            if value is not None and avg is not None:
                core_scores.append(score_against_profile(value, avg, stdev))

        core = sum(core_scores) / len(core_scores) if core_scores else None
    else:
        frequency = None
        arrangement = None
        core = None

    # Conservative fallbacks create slight visual separation without affecting
    # ranking. They are used only when an older profile lacks a component.
    components["frequency"] = clean(
        frequency,
        clean(public_score * 0.92 + fingerprint * 0.08, public_score),
    )
    components["arrangement"] = clean(
        arrangement,
        clean(public_score * 0.88 + fingerprint * 0.12, public_score),
    )
    components["core"] = clean(
        core,
        clean(public_score * 0.95 + fingerprint * 0.05, public_score),
    )

    item["score_components"] = components
    return item

def ensure_visible_components_only(item: Dict[str, Any]) -> Dict[str, Any]:
    """UI-only fallback for Artist Match component cards.

    This MUST NOT affect ranking, artist names, match score, confidence, or voting.
    It only fills missing component display values so the frontend does not show blanks.
    """
    components = item.get("score_components") or {}
    if not isinstance(components, dict):
        components = {}

    def clean(value, fallback=None):
        try:
            if value is None:
                return fallback
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return fallback
            return round(number, 2)
        except Exception:
            return fallback

    public_score = clean(item.get("match_score"), 0)
    fingerprint = clean(components.get("fingerprint"), public_score)

    # Use values close to the public score only for display when missing.
    # Do not use high raw internal scores here.
    components["fingerprint"] = fingerprint
    components["frequency"] = clean(components.get("frequency"), public_score)
    components["arrangement"] = clean(components.get("arrangement"), public_score)
    components["core"] = clean(components.get("core"), public_score)

    item["score_components"] = components
    return item


def compare_audio_to_profiles(
    audio_file=None,
    profiles_folder="artist_profiles",
    top_n=10,
    include_report=False,
    use_stems=False,
    demucs_output_dir="stems",
    precomputed_report=None,
):
    if precomputed_report is not None:
        report_dict = precomputed_report if isinstance(precomputed_report, dict) else asdict(precomputed_report)
    else:
        if audio_file is None:
            raise ValueError("audio_file is required when precomputed_report is not supplied.")
        report = analyze_audio(
            Path(audio_file),
            use_stems=use_stems,
            demucs_output_dir=Path(demucs_output_dir),
        )
        report_dict = asdict(report)

    profiles_path = Path(profiles_folder)
    profile_files = sorted(profiles_path.glob("*_profile.json"))

    if not profile_files:
        return {
            "verdict": "No artist profiles found. Build profiles first.",
            "ranked_profiles": [],
            "style_suggestions": [
                "Run build_artist_profile.py after generating JSON reports.",
            ],
            "eq_suggestions": None,
        }

    # v2 Artist Match: compare against every saved track prototype first, then
    # vote those closest tracks back up to artists. This preserves much more
    # information than comparing against one averaged artist profile.
    ranked, profiles_by_name, global_nearest_tracks = compare_against_track_library(
        report_dict,
        profile_files,
        top_n=top_n,
    )

    if ranked:
        verdict = (
            f"Closest style lane: {ranked[0]['profile_name']} "
            f"({ranked[0]['match_score']:.2f}%). "
            f"Confidence: {ranked[0]['confidence']}."
        )
        best_profile_data = profiles_by_name.get(ranked[0]["profile_name"])
        eq_data = eq_suggestions(report_dict, best_profile_data) if best_profile_data else None
        style_suggestions = style_suggestions_from_components(ranked[0])
    else:
        verdict = "No usable track prototypes found. Rebuild profiles from fresh reports."
        eq_data = None
        style_suggestions = [
            "Rebuild artist profiles so each profile includes track_prototypes with embedding_vector values.",
        ]

    ranked = [
        calculate_display_component_scores(
            item,
            report_dict,
            profiles_by_name.get(item.get("profile_name")),
        )
        for item in ranked
    ]

    result = {
        "verdict": verdict,
        "ranked_profiles": ranked[:top_n],
        "style_suggestions": style_suggestions,
        "eq_suggestions": eq_data,
        "nearest_tracks_global": global_nearest_tracks,
        "matching_engine": "track_library_vote_v2",
    }

    if include_report:
        result["report"] = report_dict

    return result
