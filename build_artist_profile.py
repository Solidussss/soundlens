"""
SoundLens Artist/Profile Builder Pro+

Builds stronger artist/style profiles from SoundLens JSON reports.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


NUMERIC_FIELDS = {
    "bpm": ["basic", "bpm"],
    "duration_seconds": ["basic", "duration_seconds"],
    "key_confidence": ["basic", "key_confidence"],

    "mfcc_1": ["fingerprint", "mfcc_1"],
    "mfcc_2": ["fingerprint", "mfcc_2"],
    "mfcc_3": ["fingerprint", "mfcc_3"],
    "mfcc_4": ["fingerprint", "mfcc_4"],
    "mfcc_5": ["fingerprint", "mfcc_5"],
    "mfcc_6": ["fingerprint", "mfcc_6"],
    "mfcc_7": ["fingerprint", "mfcc_7"],
    "mfcc_8": ["fingerprint", "mfcc_8"],
    "mfcc_9": ["fingerprint", "mfcc_9"],
    "mfcc_10": ["fingerprint", "mfcc_10"],
    "mfcc_11": ["fingerprint", "mfcc_11"],
    "mfcc_12": ["fingerprint", "mfcc_12"],
    "mfcc_13": ["fingerprint", "mfcc_13"],
    "chroma_1": ["fingerprint", "chroma_1"],
    "chroma_2": ["fingerprint", "chroma_2"],
    "chroma_3": ["fingerprint", "chroma_3"],
    "chroma_4": ["fingerprint", "chroma_4"],
    "chroma_5": ["fingerprint", "chroma_5"],
    "chroma_6": ["fingerprint", "chroma_6"],
    "chroma_7": ["fingerprint", "chroma_7"],
    "chroma_8": ["fingerprint", "chroma_8"],
    "chroma_9": ["fingerprint", "chroma_9"],
    "chroma_10": ["fingerprint", "chroma_10"],
    "chroma_11": ["fingerprint", "chroma_11"],
    "chroma_12": ["fingerprint", "chroma_12"],
    "spectral_contrast": ["fingerprint", "spectral_contrast"],
    "spectral_flatness": ["fingerprint", "spectral_flatness"],
    "zero_crossing_rate": ["fingerprint", "zero_crossing_rate"],

    "embed_mfcc_1_mean": ["fingerprint", "embed_mfcc_1_mean"],
    "embed_mfcc_1_std": ["fingerprint", "embed_mfcc_1_std"],
    "embed_mfcc_2_mean": ["fingerprint", "embed_mfcc_2_mean"],
    "embed_mfcc_2_std": ["fingerprint", "embed_mfcc_2_std"],
    "embed_mfcc_3_mean": ["fingerprint", "embed_mfcc_3_mean"],
    "embed_mfcc_3_std": ["fingerprint", "embed_mfcc_3_std"],
    "embed_mfcc_4_mean": ["fingerprint", "embed_mfcc_4_mean"],
    "embed_mfcc_4_std": ["fingerprint", "embed_mfcc_4_std"],
    "embed_mfcc_5_mean": ["fingerprint", "embed_mfcc_5_mean"],
    "embed_mfcc_5_std": ["fingerprint", "embed_mfcc_5_std"],
    "embed_mfcc_6_mean": ["fingerprint", "embed_mfcc_6_mean"],
    "embed_mfcc_6_std": ["fingerprint", "embed_mfcc_6_std"],
    "embed_mfcc_7_mean": ["fingerprint", "embed_mfcc_7_mean"],
    "embed_mfcc_7_std": ["fingerprint", "embed_mfcc_7_std"],
    "embed_mfcc_8_mean": ["fingerprint", "embed_mfcc_8_mean"],
    "embed_mfcc_8_std": ["fingerprint", "embed_mfcc_8_std"],
    "embed_mfcc_9_mean": ["fingerprint", "embed_mfcc_9_mean"],
    "embed_mfcc_9_std": ["fingerprint", "embed_mfcc_9_std"],
    "embed_mfcc_10_mean": ["fingerprint", "embed_mfcc_10_mean"],
    "embed_mfcc_10_std": ["fingerprint", "embed_mfcc_10_std"],
    "embed_mfcc_11_mean": ["fingerprint", "embed_mfcc_11_mean"],
    "embed_mfcc_11_std": ["fingerprint", "embed_mfcc_11_std"],
    "embed_mfcc_12_mean": ["fingerprint", "embed_mfcc_12_mean"],
    "embed_mfcc_12_std": ["fingerprint", "embed_mfcc_12_std"],
    "embed_mfcc_13_mean": ["fingerprint", "embed_mfcc_13_mean"],
    "embed_mfcc_13_std": ["fingerprint", "embed_mfcc_13_std"],
    "embed_mfcc_14_mean": ["fingerprint", "embed_mfcc_14_mean"],
    "embed_mfcc_14_std": ["fingerprint", "embed_mfcc_14_std"],
    "embed_mfcc_15_mean": ["fingerprint", "embed_mfcc_15_mean"],
    "embed_mfcc_15_std": ["fingerprint", "embed_mfcc_15_std"],
    "embed_mfcc_16_mean": ["fingerprint", "embed_mfcc_16_mean"],
    "embed_mfcc_16_std": ["fingerprint", "embed_mfcc_16_std"],
    "embed_mfcc_17_mean": ["fingerprint", "embed_mfcc_17_mean"],
    "embed_mfcc_17_std": ["fingerprint", "embed_mfcc_17_std"],
    "embed_mfcc_18_mean": ["fingerprint", "embed_mfcc_18_mean"],
    "embed_mfcc_18_std": ["fingerprint", "embed_mfcc_18_std"],
    "embed_mfcc_19_mean": ["fingerprint", "embed_mfcc_19_mean"],
    "embed_mfcc_19_std": ["fingerprint", "embed_mfcc_19_std"],
    "embed_mfcc_20_mean": ["fingerprint", "embed_mfcc_20_mean"],
    "embed_mfcc_20_std": ["fingerprint", "embed_mfcc_20_std"],
    "embed_chroma_1_mean": ["fingerprint", "embed_chroma_1_mean"],
    "embed_chroma_1_std": ["fingerprint", "embed_chroma_1_std"],
    "embed_chroma_2_mean": ["fingerprint", "embed_chroma_2_mean"],
    "embed_chroma_2_std": ["fingerprint", "embed_chroma_2_std"],
    "embed_chroma_3_mean": ["fingerprint", "embed_chroma_3_mean"],
    "embed_chroma_3_std": ["fingerprint", "embed_chroma_3_std"],
    "embed_chroma_4_mean": ["fingerprint", "embed_chroma_4_mean"],
    "embed_chroma_4_std": ["fingerprint", "embed_chroma_4_std"],
    "embed_chroma_5_mean": ["fingerprint", "embed_chroma_5_mean"],
    "embed_chroma_5_std": ["fingerprint", "embed_chroma_5_std"],
    "embed_chroma_6_mean": ["fingerprint", "embed_chroma_6_mean"],
    "embed_chroma_6_std": ["fingerprint", "embed_chroma_6_std"],
    "embed_chroma_7_mean": ["fingerprint", "embed_chroma_7_mean"],
    "embed_chroma_7_std": ["fingerprint", "embed_chroma_7_std"],
    "embed_chroma_8_mean": ["fingerprint", "embed_chroma_8_mean"],
    "embed_chroma_8_std": ["fingerprint", "embed_chroma_8_std"],
    "embed_chroma_9_mean": ["fingerprint", "embed_chroma_9_mean"],
    "embed_chroma_9_std": ["fingerprint", "embed_chroma_9_std"],
    "embed_chroma_10_mean": ["fingerprint", "embed_chroma_10_mean"],
    "embed_chroma_10_std": ["fingerprint", "embed_chroma_10_std"],
    "embed_chroma_11_mean": ["fingerprint", "embed_chroma_11_mean"],
    "embed_chroma_11_std": ["fingerprint", "embed_chroma_11_std"],
    "embed_chroma_12_mean": ["fingerprint", "embed_chroma_12_mean"],
    "embed_chroma_12_std": ["fingerprint", "embed_chroma_12_std"],
    "embed_contrast_1_mean": ["fingerprint", "embed_contrast_1_mean"],
    "embed_contrast_1_std": ["fingerprint", "embed_contrast_1_std"],
    "embed_contrast_2_mean": ["fingerprint", "embed_contrast_2_mean"],
    "embed_contrast_2_std": ["fingerprint", "embed_contrast_2_std"],
    "embed_contrast_3_mean": ["fingerprint", "embed_contrast_3_mean"],
    "embed_contrast_3_std": ["fingerprint", "embed_contrast_3_std"],
    "embed_contrast_4_mean": ["fingerprint", "embed_contrast_4_mean"],
    "embed_contrast_4_std": ["fingerprint", "embed_contrast_4_std"],
    "embed_contrast_5_mean": ["fingerprint", "embed_contrast_5_mean"],
    "embed_contrast_5_std": ["fingerprint", "embed_contrast_5_std"],
    "embed_contrast_6_mean": ["fingerprint", "embed_contrast_6_mean"],
    "embed_contrast_6_std": ["fingerprint", "embed_contrast_6_std"],
    "embed_contrast_7_mean": ["fingerprint", "embed_contrast_7_mean"],
    "embed_contrast_7_std": ["fingerprint", "embed_contrast_7_std"],
    "embed_chroma_entropy": ["fingerprint", "embed_chroma_entropy"],
    "embed_centroid_mean": ["fingerprint", "embed_centroid_mean"],
    "embed_centroid_std": ["fingerprint", "embed_centroid_std"],
    "embed_rolloff_mean": ["fingerprint", "embed_rolloff_mean"],
    "embed_rolloff_std": ["fingerprint", "embed_rolloff_std"],
    "embed_flatness_mean": ["fingerprint", "embed_flatness_mean"],
    "embed_flatness_std": ["fingerprint", "embed_flatness_std"],
    "embed_zcr_mean": ["fingerprint", "embed_zcr_mean"],
    "embed_zcr_std": ["fingerprint", "embed_zcr_std"],
    "embed_rms_mean": ["fingerprint", "embed_rms_mean"],
    "embed_rms_std": ["fingerprint", "embed_rms_std"],
    "embed_onset_mean": ["fingerprint", "embed_onset_mean"],
    "embed_onset_std": ["fingerprint", "embed_onset_std"],

    "peak_db": ["loudness", "peak_db"],
    "rms_db": ["loudness", "rms_db"],
    "dynamic_range_db": ["loudness", "dynamic_range_db"],
    "clipping_percent": ["loudness", "clipping_percent"],

    "low_end_total_percent": ["frequency", "low_end_total_percent"],
    "mid_total_percent": ["frequency", "mid_total_percent"],
    "top_total_percent": ["frequency", "top_total_percent"],
    "brightness_centroid_hz": ["frequency", "brightness_centroid_hz"],
    "spectral_rolloff_hz": ["frequency", "spectral_rolloff_hz"],

    "onset_density": ["rhythm", "onset_density"],
    "estimated_bars": ["rhythm", "estimated_bars"],

    "mix_score": ["scores", "mix"],
    "master_score": ["scores", "master"],
    "arrangement_score": ["scores", "arrangement"],
    "release_score": ["scores", "release"],
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

BAND_NAMES = [
    "Sub",
    "Bass / 808",
    "Mud",
    "Low Mids",
    "Mids / Melody",
    "Harsh Zone",
    "Highs",
    "Air",
    "Vocal Range",
]


def get_nested(data: Dict[str, Any], path: List[str], default=None):
    current = data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def safe_values(values: Iterable[Any]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float))]


def safe_mean(values: Iterable[Any]) -> float:
    values = safe_values(values)
    return round(statistics.mean(values), 4) if values else 0.0


def safe_min(values: Iterable[Any]) -> float:
    values = safe_values(values)
    return round(min(values), 4) if values else 0.0


def safe_max(values: Iterable[Any]) -> float:
    values = safe_values(values)
    return round(max(values), 4) if values else 0.0


def safe_stdev(values: Iterable[Any]) -> float:
    values = safe_values(values)
    return round(statistics.stdev(values), 4) if len(values) >= 2 else 0.0


def summarize(values: Iterable[Any]) -> Dict[str, float]:
    values = safe_values(values)
    return {
        "avg": safe_mean(values),
        "min": safe_min(values),
        "max": safe_max(values),
        "stdev": safe_stdev(values),
    }


def normalize_bpm_for_style(bpm: float) -> float:
    bpm = float(bpm)
    while bpm < 80:
        bpm *= 2
    while bpm > 170:
        bpm /= 2
    return bpm


def ratio(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return round(a / b, 4)


def find_report_files(reports_dir: Path) -> List[Path]:
    return sorted(reports_dir.glob("**/*_soundlens_report.json"))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def category_for_report(path: Path, reports_dir: Path) -> str:
    try:
        relative = path.relative_to(reports_dir)
        if len(relative.parts) > 1:
            return relative.parts[0]
    except ValueError:
        pass
    return "Uncategorized"


def build_fingerprint(report: Dict[str, Any]) -> Dict[str, Any]:
    bands = get_nested(report, ["frequency", "band_percentages"], {}) or {}

    sub = float(bands.get("Sub", 0) or 0)
    bass = float(bands.get("Bass / 808", 0) or 0)
    mud = float(bands.get("Mud", 0) or 0)
    low_mids = float(bands.get("Low Mids", 0) or 0)
    mids = float(bands.get("Mids / Melody", 0) or 0)
    harsh = float(bands.get("Harsh Zone", 0) or 0)
    highs = float(bands.get("Highs", 0) or 0)
    air = float(bands.get("Air", 0) or 0)
    vocal = float(bands.get("Vocal Range", 0) or 0)

    bpm = get_nested(report, ["basic", "bpm"], 0) or 0
    sections = report.get("sections", []) or []

    intro_len = 0.0
    outro_len = 0.0
    hook_count = 0
    verse_count = 0
    high_sections = 0
    low_sections = 0

    for section in sections:
        name = str(section.get("name", "")).lower()
        start = float(section.get("start", 0) or 0)
        end = float(section.get("end", 0) or 0)
        length = max(0.0, end - start)

        if "intro" in name:
            intro_len += length
        if "outro" in name:
            outro_len += length
        if "hook" in name or "drop" in name:
            hook_count += 1
        if "verse" in name or "build" in name:
            verse_count += 1

        label = section.get("energy_label", "")
        if label == "High":
            high_sections += 1
        elif label == "Low":
            low_sections += 1

    stem = report.get("stem_balance", {}) or {}

    return {
        "bpm_style": normalize_bpm_for_style(float(bpm)),
        "sub_to_bass_ratio": ratio(sub, bass),
        "low_end_focus": round(sub + bass, 4),
        "mud_to_mid_ratio": ratio(mud + low_mids, mids + 0.001),
        "harsh_to_air_ratio": ratio(harsh, air + 0.001),
        "top_brightness_balance": round(highs + air, 4),
        "vocal_space_band": vocal,
        "bass_vs_vocal_ratio": ratio(sub + bass, vocal + 0.001),
        "stem_vocal_to_beat_db": stem.get("vocal_to_beat_db", 0) or 0,
        "stem_bass_to_vocal_db": stem.get("bass_to_vocal_db", 0) or 0,
        "stem_bass_to_other_db": stem.get("bass_to_other_db", 0) or 0,
        "stem_beat_vocal_balance_score": stem.get("beat_vocal_balance_score", 0) or 0,
        "stem_melody_presence_score": stem.get("melody_presence_score", 0) or 0,
        "section_count": len(sections),
        "intro_length": round(intro_len, 4),
        "outro_length": round(outro_len, 4),
        "hook_count": hook_count,
        "verse_count": verse_count,
        "high_section_count": high_sections,
        "low_section_count": low_sections,
    }




# Track-level prototypes keep each song's sonic fingerprint inside the artist profile.
# This is stronger than only comparing to an artist average, because artist averages
# can wash out the exact sound of individual songs.
def report_embedding_vector(report: Dict[str, Any]) -> List[float]:
    fp = report.get("fingerprint", {}) or {}
    vector: List[float] = []

    # New embedding-style fields if reports were rebuilt with Audio Embedding V1.5.
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

    # Fallback fields for older reports.
    if not any(abs(v) > 1e-9 for v in vector):
        for i in range(1, 14):
            vector.append(float(fp.get(f"mfcc_{i}", 0) or 0))
        for i in range(1, 13):
            vector.append(float(fp.get(f"chroma_{i}", 0) or 0))
        for key in ["spectral_contrast", "spectral_flatness", "zero_crossing_rate"]:
            vector.append(float(fp.get(key, 0) or 0))

    return [round(float(v), 6) for v in vector]


def build_track_prototype(report: Dict[str, Any]) -> Dict[str, Any]:
    basic = report.get("basic", {}) or {}
    loudness = report.get("loudness", {}) or {}
    frequency = report.get("frequency", {}) or {}
    rhythm = report.get("rhythm", {}) or {}
    scores = report.get("scores", {}) or {}

    return {
        "title": basic.get("file_name") or basic.get("file_path") or "Unknown track",
        "embedding_vector": report_embedding_vector(report),
        "style_fingerprint": build_fingerprint(report),
        "summary": {
            "bpm": basic.get("bpm"),
            "key": basic.get("key"),
            "rms_db": loudness.get("rms_db"),
            "dynamic_range_db": loudness.get("dynamic_range_db"),
            "low_end_total_percent": frequency.get("low_end_total_percent"),
            "mid_total_percent": frequency.get("mid_total_percent"),
            "top_total_percent": frequency.get("top_total_percent"),
            "onset_density": rhythm.get("onset_density"),
            "energy": scores.get("energy"),
            "bass_strength": scores.get("bass_strength"),
            "darkness": scores.get("darkness"),
            "brightness": scores.get("brightness"),
            "drum_bounce": scores.get("drum_bounce"),
            "vocal_space": scores.get("vocal_space"),
        },
    }



def build_profile(category: str, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_summary = {}

    for output_name, path in NUMERIC_FIELDS.items():
        values = [get_nested(report, path) for report in reports]
        numeric_summary[output_name] = summarize(values)

    band_summary = {}

    for band in BAND_NAMES:
        values = [
            get_nested(report, ["frequency", "band_percentages", band])
            for report in reports
        ]
        band_summary[band] = summarize(values)

    fingerprints = [build_fingerprint(report) for report in reports]
    fingerprint_summary = {}

    for key in fingerprints[0].keys() if fingerprints else []:
        fingerprint_summary[key] = summarize([fp.get(key) for fp in fingerprints])

    key_counter = Counter(
        get_nested(report, ["basic", "key"], "Unknown") for report in reports
    )
    key_mode_counter = Counter(
        get_nested(report, ["basic", "key_mode"], "Unknown") for report in reports
    )
    brightness_counter = Counter(
        get_nested(report, ["frequency", "brightness_label"], "Unknown") for report in reports
    )
    drum_counter = Counter(
        get_nested(report, ["rhythm", "drum_activity"], "Unknown") for report in reports
    )
    dominant_band_counter = Counter(
        get_nested(report, ["frequency", "dominant_band"], "Unknown") for report in reports
    )

    section_name_counter = Counter()
    section_energy_counter = Counter()
    common_problems = Counter()

    for report in reports:
        for section in report.get("sections", []) or []:
            section_name_counter[section.get("name", "Unknown")] += 1
            section_energy_counter[section.get("energy_label", "Unknown")] += 1

        for problem in report.get("top_problems", [])[:5]:
            common_problems[problem] += 1

    profile = {
        "profile_name": category,
        "track_count": len(reports),
        "description": f"SoundLens Pro+ profile built from {len(reports)} analyzed track(s).",
        "averages": numeric_summary,
        "frequency_bands": band_summary,
        "fingerprint": fingerprint_summary,
        "track_prototypes": [build_track_prototype(report) for report in reports],
        "most_common": {
            "keys": key_counter.most_common(5),
            "key_modes": key_mode_counter.most_common(3),
            "brightness_labels": brightness_counter.most_common(3),
            "drum_activity": drum_counter.most_common(3),
            "dominant_bands": dominant_band_counter.most_common(5),
            "section_names": section_name_counter.most_common(10),
            "section_energy_labels": section_energy_counter.most_common(5),
            "top_problems": common_problems.most_common(10),
        },
        "website_compare_fields": [
            "bpm",
            "bpm_style",
            "rms_db",
            "dynamic_range_db",
            "low_end_total_percent",
            "mid_total_percent",
            "top_total_percent",
            "brightness_centroid_hz",
            "spectral_rolloff_hz",
            "onset_density",
            "energy",
            "bass_strength",
            "darkness",
            "brightness",
            "drum_bounce",
            "vocal_space",
            "sub_to_bass_ratio",
            "low_end_focus",
            "mud_to_mid_ratio",
            "harsh_to_air_ratio",
            "top_brightness_balance",
            "vocal_space_band",
            "bass_vs_vocal_ratio",
            "stem_vocal_to_beat_db",
            "stem_bass_to_vocal_db",
            "stem_bass_to_other_db",
            "stem_beat_vocal_balance_score",
            "stem_melody_presence_score",
            "intro_length",
            "hook_count",
            "verse_count",
            "high_section_count",
            "low_section_count",
        ],
    }

    return profile


def safe_filename(name: str) -> str:
    return "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in name
    ).strip("_") or "profile"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SoundLens artist/style profiles from JSON reports."
    )
    parser.add_argument(
        "--reports",
        "-r",
        default="reports",
        help="Folder containing SoundLens JSON reports.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="artist_profiles",
        help="Folder to save profile JSON files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports_dir = Path(args.reports).expanduser()
    output_dir = Path(args.output).expanduser()

    if not reports_dir.exists():
        print(f"Reports folder not found: {reports_dir}")
        return 1

    report_files = find_report_files(reports_dir)

    if not report_files:
        print(f"No SoundLens JSON reports found in: {reports_dir}")
        return 1

    grouped: dict[str, list[dict]] = defaultdict(list)

    for path in report_files:
        try:
            category = category_for_report(path, reports_dir)
            grouped[category].append(load_json(path))
        except Exception as error:
            print(f"Skipping {path}: {error}")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_index = {}

    for category, category_reports in sorted(grouped.items()):
        profile = build_profile(category, category_reports)
        profile_path = output_dir / f"{safe_filename(category)}_profile.json"
        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        combined_index[category] = str(profile_path)
        print(f"Saved profile: {profile_path} ({len(category_reports)} tracks)")

    index_path = output_dir / "profile_index.json"
    index_path.write_text(json.dumps(combined_index, indent=2), encoding="utf-8")

    print(f"Saved index: {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())