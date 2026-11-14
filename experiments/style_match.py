"""Does a learned character sound like the artist it learned from?

For each reference artist: learn a character from their songs, generate songs from it,
then measure each generated song with the *same* feature extractor and check which
artist's reference songs it lands closest to. A character that captured a style lands
closest to its own artist. The same test is run with an untouched built-in character as
a baseline.

    python experiments/style_match.py

Reference songs are local analyzer test files (not in git). Results are reported per
artist label only and nothing from the audio is stored.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio import SAMPLE_RATE, characters, composer, learn  # noqa: E402
from studio.render import Renderer  # noqa: E402

REFS = {
    "A (Nettspend refs)": sorted((ROOT / "tracks" / "Songs").glob("Nettspend*.wav")),
    "B (Ken Carson refs)": sorted((ROOT / "tracks" / "Songs").glob("Ken Carson*.wav")),
    "C (Bktharula refs)": [ROOT / "uploads" / "Bktharula - Forever.wav"],
}
FEATURES = ["bpm", "half"] + learn.TARGET_KEYS


def vec(t: dict) -> list[float]:
    m = learn.measure(t)
    return [t["bpm"], 1.0 if t["feel"] == "half" else 0.0] + [m[k] if m.get(k) is not None else np.nan for k in learn.TARGET_KEYS]


def main(seeds=(1, 2, 3), seconds=60):
    tmp = Path(tempfile.mkdtemp())

    class _R:  # a fresh renderer per song so the cache never grows
        def __getattr__(self, name):
            return getattr(Renderer(tmp / "cache"), name)
    r = _R()
    ref_vecs, learned, unfitted = {}, {}, {}
    for label, files in REFS.items():
        files = [f for f in files if f.exists()]
        if not files:
            print("missing refs for", label); return
        tracks = [learn.analyze_track(f) for f in files]
        ok = [t for t in tracks if "error" not in t]
        ref_vecs[label] = np.array([vec(t) for t in ok])
        style = learn.aggregate(tracks)
        base = learn.character_from_style(style, f"x{len(learned)}", label, {"look": {"seed": label}})
        fit = learn.fit_character(base, learn.targets_from(ok), iterations=5, work_dir=tmp / "fit")
        print(label, "fit:", fit["report"].get("start_distance"), "->", fit["report"].get("best_distance"))
        learned[label] = fit["character"]
        unfitted[label] = base
        print(label, "learned:", json.dumps({"bpm": style["bpm"], "feel": style["feel"], "808": learned[label]["bass"]["preset"],
                                             "melody": learned[label]["melody"]["preset"]}))
    allref = np.vstack(list(ref_vecs.values()))
    mu, sd = np.nanmean(allref, 0), np.nanstd(allref, 0) + 1e-9
    cents = {k: np.nanmean((v - mu) / sd, 0) for k, v in ref_vecs.items()}

    def nearest(t: dict) -> tuple[str, dict]:
        z = (np.array(vec(t)) - mu) / sd
        d = {k: float(np.sqrt(np.nansum((z - c) ** 2))) for k, c in cents.items()}
        return min(d, key=d.get), d

    def generate_and_measure(char, seed):
        p = composer.generate_project(char, {"duration_s": seconds}, seed=seed)
        r.calibrate_master(p)
        y = np.clip(r.mix(p), -1, 1)
        path = tmp / f"{char['id']}_{seed}.wav"
        sf.write(str(path), y.T, SAMPLE_RATE)
        t = learn.analyze_track(path)
        path.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(tmp / "cache", ignore_errors=True)
        return t

    rows, rows_unfit = [], []
    for label, char in learned.items():
        for s in seeds:
            for which, ch, bucket in (("fitted", char, rows), ("unfitted", unfitted[label], rows_unfit)):
                t = generate_and_measure(ch, s)
                got, d = nearest(t)
                bucket.append({"character": label, "version": which, "seed": s, "closest_refs": got, "correct": got == label,
                               "distances": {k: round(v, 2) for k, v in d.items()}})
                print(json.dumps(bucket[-1]))
    baseline = []
    for cid in ("vanta", "mercury_halo", "static_saint"):
        t = generate_and_measure(characters.get(cid), 1)
        got, d = nearest(t)
        baseline.append({"builtin": cid, "closest_refs": got, "distances": {k: round(v, 2) for k, v in d.items()}})
        print("baseline", json.dumps(baseline[-1]))
    acc = sum(r_["correct"] for r_ in rows) / len(rows)
    summary = {"generated": len(rows), "closest_to_own_artist": sum(r_["correct"] for r_ in rows), "accuracy": round(acc, 3),
               "unfitted_accuracy": round(sum(r_["correct"] for r_ in rows_unfit) / len(rows_unfit), 3),
               "chance": round(1 / len(REFS), 3), "features": FEATURES}
    print("SUMMARY", json.dumps(summary))
    out = ROOT / "experiments" / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "style_match.json").write_text(json.dumps({"rows": rows, "rows_unfitted": rows_unfit, "baseline": baseline, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
