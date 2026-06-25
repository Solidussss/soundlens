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

EMBEDDING_FIELDS = [
    "embed_mfcc_1_mean",
    "embed_mfcc_1_std",
    "embed_mfcc_2_mean",
    "embed_mfcc_2_std",
    "embed_mfcc_3_mean",
    "embed_mfcc_3_std",
    "embed_mfcc_4_mean",
    "embed_mfcc_4_std",
    "embed_mfcc_5_mean",
    "embed_mfcc_5_std",
    "embed_mfcc_6_mean",
    "embed_mfcc_6_std",
    "embed_mfcc_7_mean",
    "embed_mfcc_7_std",
    "embed_mfcc_8_mean",
    "embed_mfcc_8_std",
    "embed_mfcc_9_mean",
    "embed_mfcc_9_std",
    "embed_mfcc_10_mean",
    "embed_mfcc_10_std",
    "embed_mfcc_11_mean",
    "embed_mfcc_11_std",
    "embed_mfcc_12_mean",
    "embed_mfcc_12_std",
    "embed_mfcc_13_mean",
    "embed_mfcc_13_std",
    "embed_mfcc_14_mean",
    "embed_mfcc_14_std",
    "embed_mfcc_15_mean",
    "embed_mfcc_15_std",
    "embed_mfcc_16_mean",
    "embed_mfcc_16_std",
    "embed_mfcc_17_mean",
    "embed_mfcc_17_std",
    "embed_mfcc_18_mean",
    "embed_mfcc_18_std",
    "embed_mfcc_19_mean",
    "embed_mfcc_19_std",
    "embed_mfcc_20_mean",
    "embed_mfcc_20_std",
    "embed_chroma_1_mean",
    "embed_chroma_1_std",
    "embed_chroma_2_mean",
    "embed_chroma_2_std",
    "embed_chroma_3_mean",
    "embed_chroma_3_std",
    "embed_chroma_4_mean",
    "embed_chroma_4_std",
    "embed_chroma_5_mean",
    "embed_chroma_5_std",
    "embed_chroma_6_mean",
    "embed_chroma_6_std",
    "embed_chroma_7_mean",
    "embed_chroma_7_std",
    "embed_chroma_8_mean",
    "embed_chroma_8_std",
    "embed_chroma_9_mean",
    "embed_chroma_9_std",
    "embed_chroma_10_mean",
    "embed_chroma_10_std",
    "embed_chroma_11_mean",
    "embed_chroma_11_std",
    "embed_chroma_12_mean",
    "embed_chroma_12_std",
    "embed_contrast_1_mean",
    "embed_contrast_1_std",
    "embed_contrast_2_mean",
    "embed_contrast_2_std",
    "embed_contrast_3_mean",
    "embed_contrast_3_std",
    "embed_contrast_4_mean",
    "embed_contrast_4_std",
    "embed_contrast_5_mean",
    "embed_contrast_5_std",
    "embed_contrast_6_mean",
    "embed_contrast_6_std",
    "embed_contrast_7_mean",
    "embed_contrast_7_std",
    "embed_chroma_entropy",
    "embed_centroid_mean",
    "embed_centroid_std",
    "embed_rolloff_mean",
    "embed_rolloff_std",
    "embed_flatness_mean",
    "embed_flatness_std",
    "embed_zcr_mean",
    "embed_zcr_std",
    "embed_rms_mean",
    "embed_rms_std",
    "embed_onset_mean",
    "embed_onset_std",
]

LEGACY_EMBEDDING_FIELDS = [
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "mfcc_6",
    "mfcc_7",
    "mfcc_8",
    "mfcc_9",
    "mfcc_10",
    "mfcc_11",
    "mfcc_12",
    "mfcc_13",
    "chroma_1",
    "chroma_2",
    "chroma_3",
    "chroma_4",
    "chroma_5",
    "chroma_6",
    "chroma_7",
    "chroma_8",
    "chroma_9",
    "chroma_10",
    "chroma_11",
    "chroma_12",
    "spectral_contrast",
    "spectral_flatness",
    "zero_crossing_rate",
]

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
    if stdev is None or stdev <= 0:
        stdev = max(abs(avg) * 0.12, 1.0)

    z = abs(value - avg) / stdev
    score = 100 - (z * 14)

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

    return max(0.0, min(100.0, ((cosine + 1) / 2) * 100))


def report_mfcc_vector(report_dict: Dict[str, Any]) -> List[float]:
    fp = report_dict.get("fingerprint", {}) or {}

    fields = EMBEDDING_FIELDS if any(key in fp for key in EMBEDDING_FIELDS) else LEGACY_EMBEDDING_FIELDS
    return [float(fp.get(key, 0) or 0) for key in fields]


def profile_mfcc_vector(profile: Dict[str, Any]) -> List[float]:
    averages = profile.get("averages", {}) or {}
    fingerprint = profile.get("fingerprint", {}) or {}
    audio_features = profile.get("audio_features", {}) or {}

    use_new_embedding = any(key in averages or key in fingerprint or key in audio_features for key in EMBEDDING_FIELDS)
    fields = EMBEDDING_FIELDS if use_new_embedding else LEGACY_EMBEDDING_FIELDS

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

    return [avg_from_any(key) for key in fields]


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
    # Audio embedding similarity is the closest V1.5 layer to "this feels like this artist."
    # It compares a vector of timbre, harmony, spectral texture, movement, and energy features.
    embedding_score = cosine_similarity(
        report_mfcc_vector(report_dict),
        profile_mfcc_vector(profile),
    )

    # Style ratios still help keep the match grounded in SoundLens-specific profile traits.
    style_score = style_fingerprint_score(report_dict, profile)

    if embedding_score is not None and style_score is not None:
        return round((embedding_score * 0.78) + (style_score * 0.22), 2)

    if embedding_score is not None:
        return round(embedding_score, 2)

    if style_score is not None:
        return round(style_score, 2)

    return None


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


def label_for_score(score: float) -> str:
    if score >= 85:
        return "Strong Match"
    if score >= 72:
        return "Good Match"
    if score >= 58:
        return "Possible Match"

    return "Weak Match"


def confidence_from_gap(best_score: float, second_score: float, compared_fields: int, track_count: int) -> str:
    gap = best_score - second_score

    if track_count < 8 or compared_fields < 8:
        return "Low"

    if gap >= 8 and best_score >= 78:
        return "High"

    if gap >= 4 and best_score >= 70:
        return "Medium"

    return "Low"


def confidence_percent(best_score: float, second_score: float, compared_fields: int, track_count: int) -> int:
    gap = max(0.0, best_score - second_score)
    field_bonus = min(15, compared_fields)
    track_bonus = min(15, track_count / 3)

    raw = 35 + gap * 4 + field_bonus + track_bonus

    if best_score < 60:
        raw -= 15

    return int(max(20, min(96, round(raw))))


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


def compare_audio_to_profiles(
    audio_file,
    profiles_folder="artist_profiles",
    top_n=10,
    include_report=False,
    use_stems=False,
    demucs_output_dir="stems",
):
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

    ranked = []

    for profile_file in profile_files:
        try:
            profile = json.loads(profile_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        field_scores = []
        total_weight = 0.0
        weighted_sum = 0.0

        compare_fields = profile.get("website_compare_fields") or list(COMPARE_FIELDS.keys())

        for field in compare_fields:
            if field not in COMPARE_FIELDS:
                continue

            value = to_float(get_nested(report_dict, COMPARE_FIELDS[field]))
            avg, stdev = summary_avg_stdev(profile, field)

            if field == "bpm" and value is not None:
                value = normalize_bpm_for_style(value)

            if value is None or avg is None:
                continue

            field_score = score_against_profile(value, avg, stdev)
            weight = FIELD_WEIGHTS.get(field, 1.0)

            weighted_sum += field_score * weight
            total_weight += weight

            field_scores.append({
                "field": field,
                "score": round(field_score, 2),
                "value": round(value, 4),
                "profile_avg": round(avg, 4),
            })

        metric_score = round(weighted_sum / total_weight, 2) if total_weight else None
        freq_score = frequency_shape_score(report_dict, profile)
        fp_score = fingerprint_score(report_dict, profile)
        stem_score = stem_component_score(field_scores)
        core_score = core_metric_score(field_scores)

        weighted_components = []

        if metric_score is not None:
            weighted_components.append((metric_score, 0.16))
        if freq_score is not None:
            weighted_components.append((freq_score, 0.20))
        if fp_score is not None:
            weighted_components.append((fp_score, 0.49))
        if stem_score is not None:
            weighted_components.append((stem_score, 0.15))

        if weighted_components:
            match_score = sum(score * weight for score, weight in weighted_components) / sum(
                weight for _, weight in weighted_components
            )
        else:
            match_score = 0.0

        match_score = round(float(match_score or 0), 2)

        ranked.append({
            "profile_name": profile.get("profile_name", profile_file.stem.replace("_profile", "")),
            "match_score": match_score,
            "match_label": label_for_score(match_score),
            "confidence": "Pending",
            "confidence_percent": 0,
            "track_count": int(profile.get("track_count", 0) or 0),
            "compared_fields": len(field_scores),
            "score_components": {
                "overall_metrics": round(metric_score, 2) if metric_score is not None else None,
                "frequency": round(freq_score, 2) if freq_score is not None else None,
                "fingerprint": round(fp_score, 2) if fp_score is not None else None,
                "stem": round(stem_score, 2) if stem_score is not None else None,
                "core": round(core_score, 2) if core_score is not None else None,
            },
            "field_scores": sorted(field_scores, key=lambda item: item["score"])[:8],
        })

    ranked.sort(key=lambda item: float(item.get("match_score") or 0), reverse=True)

    if ranked:
        best_score = float(ranked[0]["match_score"])
        second_score = float(ranked[1]["match_score"]) if len(ranked) > 1 else 0.0

        for index, item in enumerate(ranked):
            comparison_score = second_score if index == 0 else best_score

            item["confidence"] = confidence_from_gap(
                float(item["match_score"]),
                comparison_score,
                int(item.get("compared_fields", 0) or 0),
                int(item.get("track_count", 0) or 0),
            )

            item["confidence_percent"] = confidence_percent(
                float(item["match_score"]),
                comparison_score,
                int(item.get("compared_fields", 0) or 0),
                int(item.get("track_count", 0) or 0),
            )

        verdict = (
            f"Closest style: {ranked[0]['profile_name']} "
            f"({ranked[0]['match_score']:.2f}%). "
            f"Confidence: {ranked[0]['confidence']}."
        )
    else:
        verdict = "No usable profiles found."

    best_profile_data = None

    if ranked:
        best_name = ranked[0]["profile_name"]

        for profile_file in profile_files:
            try:
                profile = json.loads(profile_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            profile_name = profile.get("profile_name", profile_file.stem.replace("_profile", ""))

            if profile_name == best_name:
                best_profile_data = profile
                break

    eq_data = eq_suggestions(report_dict, best_profile_data) if best_profile_data else None
    style_suggestions = style_suggestions_from_components(ranked[0]) if ranked else []

    result = {
        "verdict": verdict,
        "ranked_profiles": ranked[:top_n],
        "style_suggestions": style_suggestions,
        "eq_suggestions": eq_data,
    }

    if include_report:
        result["report"] = report_dict

    return result
