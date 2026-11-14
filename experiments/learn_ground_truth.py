"""How accurate is the style learner? Test it on songs where the answer is known.

Renders procedural songs (exact kick/snare/808 patterns, tempo and feel are known),
runs ``studio.learn.analyze_track`` on the *mixed* audio (no stems, like a user upload)
and scores: tempo error, feel, and how much of the learned kick/snare/808 pattern
library matches the true patterns.

    python experiments/learn_ground_truth.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio import SAMPLE_RATE, characters, composer, learn  # noqa: E402
from studio.project import track_by_ref, track_notes_absolute  # noqa: E402
from studio.render import Renderer  # noqa: E402


def true_bars(project, role, nbars):
    try:
        notes = track_notes_absolute(track_by_ref(project, role))
    except ValueError:
        return []
    grid = np.zeros((nbars, 16), dtype=bool)
    for n in notes:
        if role == "808" and n.get("slide"):
            continue
        step = n["t"] / 0.25
        if abs(step - round(step)) > 0.2:
            continue
        b, j = divmod(int(round(step)), 16)
        if b < nbars:
            grid[b, j] = True
    return ["".join("x" if v else "." for v in row) for row in grid if row.any()]


def union_bars(project, nbars):
    """True low-end groove: kick hits plus 808 attacks (slides excluded)."""
    k = true_bars(project, "kick", nbars)
    b = true_bars(project, "808", nbars)
    rows = {}
    for src in (k, b):
        pass
    grid = np.zeros((nbars, 16), dtype=bool)
    for role in ("kick", "808"):
        try:
            notes = track_notes_absolute(track_by_ref(project, role))
        except ValueError:
            continue
        for n in notes:
            if n.get("slide"):
                continue
            st = n["t"] / 0.25
            if abs(st - round(st)) <= 0.2:
                bi, j = divmod(int(round(st)), 16)
                if bi < nbars:
                    grid[bi, j] = True
    return ["".join("x" if v else "." for v in row) for row in grid if row.any()]


def library_match(learned: list[str], truth: list[str]) -> float:
    """Share of the learned pattern library (weighted by use) that exists in the true library."""
    if not learned:
        return 0.0
    tset = set(truth)
    c = Counter(learned)
    return sum(n for p, n in c.items() if p in tset) / sum(c.values())


def groove_match(learned: list[str], truth: list[str]) -> float:
    """Share of the clustered (consensus) learned grooves that are exactly a true pattern, weighted by use."""
    lib = learn._pattern_library(learned)
    if not lib:
        return 0.0
    tset = set(truth)
    return sum(p["share"] for p in lib if p["pattern"] in tset) / sum(p["share"] for p in lib)


def step_f1(learned: list[str], truth: list[str]) -> float:
    """Onset-position agreement of the per-step hit *profiles* (robust to a few bad bars)."""
    def prof(bars):
        a = np.array([[ch == "x" for ch in b] for b in bars], dtype=float)
        return a.mean(0) if len(a) else np.zeros(16)
    p, t = prof(learned), prof(truth)
    tp = np.minimum(p, t).sum()
    return float(2 * tp / (p.sum() + t.sum() + 1e-9))


def main():
    r = Renderer(Path(tempfile.mkdtemp()))
    rows = []
    for cid in ("vanta", "mercury_halo", "static_saint", "pulse_runner", "lowtide"):
        for seed in (1, 2):
            c = characters.get(cid)
            p = composer.generate_project(c, {"duration_s": 60}, seed=seed)
            y = np.clip(r.mix(p), -1, 1)
            path = Path(tempfile.mkdtemp()) / f"{cid}_{seed}.wav"
            sf.write(str(path), y.T, SAMPLE_RATE)
            res = learn.analyze_track(path)
            nb = int(sum(s["bars"] for s in p["sections"]))
            tk, ts = true_bars(p, "kick", nb), true_bars(p, "snare", nb) or true_bars(p, "clap", nb)
            tb = true_bars(p, "808", nb)
            tlow = union_bars(p, nb)
            bpm_true = p["params"]["bpm"]
            rows.append({
                "song": f"{cid}#{seed}", "bpm_true": bpm_true, "bpm_learned": res.get("bpm"),
                "bpm_err_pct": round(abs(res.get("bpm", 0) - bpm_true) / bpm_true * 100, 2),
                "feel_true": c["tempo"]["feel"], "feel_learned": res.get("feel"),
                "kick_lib_match": round(library_match(res.get("kick_bars", []), tk), 2),
                "kick_groove_match": round(groove_match(res.get("kick_bars", []), tk), 2),
                "kick_step_f1": round(step_f1(res.get("kick_bars", []), tk), 2),
                "snare_step_f1": round(step_f1(res.get("snare_bars", []), ts), 2),
                "808_step_f1": round(step_f1(res.get("bass_bars", []), tb), 2),
                "low_groove_f1": round(step_f1(res.get("low_bars", []), tlow), 2),
                "low_groove_match": round(groove_match(res.get("low_bars", []), tlow), 2),
                "realigned_blocks": res.get("realigned_blocks"),
            })
            print(json.dumps(rows[-1]))
    def mean(k):
        return round(float(np.mean([r_[k] for r_ in rows])), 3)
    summary = {"songs": len(rows), "tempo_within_2pct": int(sum(r_["bpm_err_pct"] <= 2 for r_ in rows)),
               "tempo_exact": int(sum(r_["bpm_err_pct"] <= 0.2 for r_ in rows)),
               "feel_correct": int(sum(r_["feel_true"] == r_["feel_learned"] for r_ in rows)),
               "kick_lib_match": mean("kick_lib_match"), "kick_groove_match": mean("kick_groove_match"), "kick_step_f1": mean("kick_step_f1"),
               "snare_step_f1": mean("snare_step_f1"), "808_step_f1": mean("808_step_f1"),
               "low_groove_f1": mean("low_groove_f1"), "low_groove_match": mean("low_groove_match")}
    print("SUMMARY", json.dumps(summary))
    (ROOT / "experiments" / "output").mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments" / "output" / "learn_ground_truth.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
