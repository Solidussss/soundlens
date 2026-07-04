from __future__ import annotations

import argparse
import csv
from pathlib import Path

from compare_to_profile_pro import compare_audio_to_profiles


def main():
    parser = argparse.ArgumentParser(description="Validate SoundLens Artist Match against labeled folders.")
    parser.add_argument("--input", required=True, help="Folder of tracks organized by artist folder.")
    parser.add_argument("--profiles", default="artist_profiles")
    parser.add_argument("--output", default="artist_match_validation.csv")
    parser.add_argument("--limit-per-artist", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.input)
    rows = []
    total = top1 = top3 = top5 = 0
    exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}

    for artist_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [p for p in sorted(artist_dir.rglob("*")) if p.suffix.lower() in exts]
        if args.limit_per_artist > 0:
            files = files[:args.limit_per_artist]
        for audio in files:
            result = compare_audio_to_profiles(audio, profiles_folder=args.profiles, top_n=10)
            ranked = result.get("ranked_profiles", [])
            names = [r.get("profile_name", "") for r in ranked]
            expected = artist_dir.name.replace("_", " ").lower()
            lowered = [n.lower() for n in names]
            total += 1
            hit1 = bool(lowered[:1] and lowered[0] == expected)
            hit3 = expected in lowered[:3]
            hit5 = expected in lowered[:5]
            top1 += int(hit1)
            top3 += int(hit3)
            top5 += int(hit5)
            rows.append({
                "file": str(audio),
                "expected": artist_dir.name,
                "top1": names[0] if names else "",
                "top2": names[1] if len(names) > 1 else "",
                "top3": names[2] if len(names) > 2 else "",
                "hit1": hit1,
                "hit3": hit3,
                "hit5": hit5,
            })
            print(f"{audio.name}: expected={artist_dir.name} top1={names[0] if names else 'NONE'}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "expected", "top1", "top2", "top3", "hit1", "hit3", "hit5"])
        writer.writeheader()
        writer.writerows(rows)

    if total:
        print(f"Top-1: {top1}/{total} = {top1/total*100:.1f}%")
        print(f"Top-3: {top3}/{total} = {top3/total*100:.1f}%")
        print(f"Top-5: {top5}/{total} = {top5/total*100:.1f}%")
    else:
        print("No audio files found.")

if __name__ == "__main__":
    main()
