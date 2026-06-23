"""
SoundLens Batch Analyzer

Examples:
    python batch_analyze.py --input tracks --output reports
    python batch_analyze.py --input "tracks/Cheromani" --output "reports/Cheromani"
    python batch_analyze.py --input tracks --output reports --stems
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

from soundlens_pro import SUPPORTED_EXTENSIONS, analyze_audio, save_outputs


def find_audio_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def relative_report_folder(audio_file: Path, input_dir: Path, output_dir: Path) -> Path:
    try:
        relative_parent = audio_file.parent.relative_to(input_dir)
    except ValueError:
        relative_parent = Path()
    return output_dir / relative_parent


def write_summary_csv(rows: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "soundlens_batch_summary.csv"

    columns = [
        "file", "category", "bpm", "key", "duration_sec",
        "release_score", "mix_score", "master_score", "arrangement_score",
        "peak_db", "rms_db", "dynamic_range_db", "clipping",
        "low_end_total", "mid_total", "top_total",
        "sub", "bass_808", "mud", "low_mids", "mids_melody", "harsh_zone", "highs", "air",
        "brightness", "drum_activity", "onset_density", "top_problem",
        "stem_vocal_to_beat_db", "stem_bass_to_vocal_db", "stem_melody_presence_score",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


def flatten_report_for_csv(report, audio_file: Path, input_dir: Path) -> dict:
    bands = report.frequency.band_percentages
    try:
        category = audio_file.parent.relative_to(input_dir).parts[0]
    except Exception:
        category = audio_file.parent.name

    stem = report.stem_balance

    return {
        "file": audio_file.name,
        "category": category,
        "bpm": round(report.basic.bpm, 2),
        "key": report.basic.key,
        "duration_sec": round(report.basic.duration_seconds, 2),
        "release_score": report.scores.release,
        "mix_score": report.scores.mix,
        "master_score": report.scores.master,
        "arrangement_score": report.scores.arrangement,
        "peak_db": round(report.loudness.peak_db, 2),
        "rms_db": round(report.loudness.rms_db, 2),
        "dynamic_range_db": round(report.loudness.dynamic_range_db, 2),
        "clipping": report.loudness.clipping_detected,
        "low_end_total": round(report.frequency.low_end_total_percent, 2),
        "mid_total": round(report.frequency.mid_total_percent, 2),
        "top_total": round(report.frequency.top_total_percent, 2),
        "sub": round(bands.get("Sub", 0), 2),
        "bass_808": round(bands.get("Bass / 808", 0), 2),
        "mud": round(bands.get("Mud", 0), 2),
        "low_mids": round(bands.get("Low Mids", 0), 2),
        "mids_melody": round(bands.get("Mids / Melody", 0), 2),
        "harsh_zone": round(bands.get("Harsh Zone", 0), 2),
        "highs": round(bands.get("Highs", 0), 2),
        "air": round(bands.get("Air", 0), 2),
        "brightness": report.frequency.brightness_label,
        "drum_activity": report.rhythm.drum_activity,
        "onset_density": round(report.rhythm.onset_density, 2),
        "top_problem": report.top_problems[0] if report.top_problems else "",
        "stem_vocal_to_beat_db": getattr(stem, "vocal_to_beat_db", "") if stem else "",
        "stem_bass_to_vocal_db": getattr(stem, "bass_to_vocal_db", "") if stem else "",
        "stem_melody_presence_score": getattr(stem, "melody_presence_score", "") if stem else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a folder of audio files with SoundLens.")
    parser.add_argument("--input", "-i", default="tracks", help="Folder containing audio files or artist folders.")
    parser.add_argument("--output", "-o", default="reports", help="Folder where reports will be saved.")
    parser.add_argument("--no-recursive", action="store_true", help="Only analyze files directly inside the input folder.")
    parser.add_argument("--no-json", action="store_true", help="Do not save JSON reports.")
    parser.add_argument("--stems", action="store_true", help="Run Demucs stem analysis for each file. Slower but better data.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    recursive = not args.no_recursive

    try:
        audio_files = find_audio_files(input_dir, recursive=recursive)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return 1

    if not audio_files:
        print(f"No audio files found in: {input_dir}")
        print(f"Supported files: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return 1

    print(f"Found {len(audio_files)} audio file(s).")
    summary_rows = []
    failed = []

    for index, audio_file in enumerate(audio_files, 1):
        print(f"\n[{index}/{len(audio_files)}] Analyzing: {audio_file}")
        try:
            report = analyze_audio(audio_file, use_stems=args.stems)
            report_folder = relative_report_folder(audio_file, input_dir, output_dir)
            txt_path, json_path = save_outputs(report, report_folder, save_json=not args.no_json)
            summary_rows.append(flatten_report_for_csv(report, audio_file, input_dir))
            print(f"Saved TXT: {txt_path}")
            if json_path:
                print(f"Saved JSON: {json_path}")
        except Exception as error:
            failed.append((audio_file, str(error)))
            print(f"Failed: {audio_file} -> {error}")

    if summary_rows:
        csv_path = write_summary_csv(summary_rows, output_dir)
        print(f"\nBatch summary saved: {csv_path}")

    if failed:
        print("\nSome files failed:")
        for path, error in failed:
            print(f"- {path}: {error}")
        return 2

    print("\nBatch analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
