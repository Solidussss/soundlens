"""Phase 2 proof of concept: generate → separate layers → change one layer → verify the rest → play/export.

Run from the repo root:

    python experiments/poc_phase2.py [--out experiments/output] [--seconds 32] [--quick]

Writes ``results.json`` plus audio (WAV/MP3) and spectrogram PNGs to ``--out``.
Every claim in docs/studio/poc_results.md comes from this script's output.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio import SAMPLE_RATE, analysis, characters, composer, dsp, synth  # noqa: E402
from studio.actions import apply_actions  # noqa: E402
from studio.orchestrator import interpret  # noqa: E402
from studio.project import beats_to_seconds, section_beats, section_by_ref, track_by_ref  # noqa: E402
from studio.render import Renderer, song_samples  # noqa: E402


def spectrogram_png(y: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    m = analysis.mono(y).astype(np.float32)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(m, n_fft=4096, hop_length=1024)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=90)
    librosa.display.specshow(S, sr=SAMPLE_RATE, hop_length=1024, x_axis="time", y_axis="log", ax=ax, vmin=-80, vmax=0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pre_master_mix(r: Renderer, p: dict) -> np.ndarray:
    return r.mix(p, apply_master=False)


def edit_and_verify(r: Renderer, project: dict, character: dict, actions: list[dict], label: str, out: Path,
                    section_ref: str | None = None) -> tuple[dict, dict]:
    before = project
    t0 = time.time()
    after, info = apply_actions(before, actions, character)
    stems_before = {t["id"]: r.stem(before, t) for t in before["tracks"]}
    r.stats = {"rendered": 0, "cached": 0}
    stems_after = {t["id"]: r.stem(after, t) for t in after["tracks"]}
    render_stats = dict(r.stats)
    elapsed = time.time() - t0
    changed_ids = {c["id"] for c in info["diff"]["sound_changed"]}
    untouched = []
    for tid, y in stems_after.items():
        if tid in changed_ids or tid not in stems_before:
            continue
        nt = analysis.null_test(stems_before[tid], y)
        untouched.append({"track": next(t["name"] for t in after["tracks"] if t["id"] == tid), **nt})
    # pre-master mix residual must equal the changed tracks' own difference
    mb, ma = pre_master_mix(r, before), pre_master_mix(r, after)
    expected = np.zeros_like(mb)
    for t in after["tracks"]:
        if t["id"] in changed_ids:
            g = dsp.db_to_lin(t["gain_db"])
            tb = next((x for x in before["tracks"] if x["id"] == t["id"]), None)
            gb = dsp.db_to_lin(tb["gain_db"]) if tb else 0.0
            expected += dsp.pan_audio(stems_after[t["id"]].astype(np.float64) * g, t["pan"])
            if tb is not None:
                expected -= dsp.pan_audio(stems_before[t["id"]].astype(np.float64) * gb, tb["pan"])
    interaction = analysis.null_test(ma - mb, expected)
    # the master bus (clipper/limiter/compressor) is non-linear, so changing one track can change how
    # the *others* are clipped in the final mix. Measure that coupling instead of assuming it away.
    post_b, post_a = r.mix(before), r.mix(after)
    post_interaction = analysis.null_test(post_a - post_b, expected * dsp.db_to_lin(after["master"].get("gain_db", 0.0)))
    changed_report = []
    for t in after["tracks"]:
        if t["id"] in changed_ids and t["id"] in stems_before:
            d = analysis.null_test(stems_before[t["id"]], stems_after[t["id"]])
            entry = {"track": t["name"], "residual_db_vs_before": d["residual_db"]}
            if section_ref:
                sec = section_by_ref(after, section_ref)
                lo, hi = section_beats(after, sec)
                a = int(beats_to_seconds(after, lo) * SAMPLE_RATE)
                b = int(beats_to_seconds(after, hi) * SAMPLE_RATE)
                yb_, ya_ = stems_before[t["id"]], stems_after[t["id"]]
                diff_idx = np.where(np.max(np.abs(yb_.astype(np.float64) - ya_), axis=0) > 1e-6)[0]
                entry["section_seconds"] = [round(a / SAMPLE_RATE, 3), round(b / SAMPLE_RATE, 3)]
                entry["before_section_identical"] = bool(np.array_equal(yb_[:, :a], ya_[:, :a]))
                if len(diff_idx):
                    entry["first_difference_s"] = round(diff_idx[0] / SAMPLE_RATE, 3)
                    entry["last_difference_s"] = round(diff_idx[-1] / SAMPLE_RATE, 3)
                    entry["difference_extends_past_section_s"] = round(max(0.0, (diff_idx[-1] - b) / SAMPLE_RATE), 3)
                entry["explanation"] = "differences after the section are release + reverb/delay tails of the new notes"
            changed_report.append(entry)
    y_after = r.mix(after)
    r.export_wav(after, out / f"{label}.wav")
    try:
        r.export_mp3(after, out / f"{label}.mp3", bitrate="160k")
    except Exception as e:  # ffmpeg missing is reported, not fatal
        print("mp3 export failed:", e)
    spectrogram_png(y_after, out / f"{label}.png", label)
    return after, {
        "label": label, "actions": actions, "diff": info["diff"], "seconds_to_apply_and_render": round(elapsed, 2),
        "render_stats": render_stats, "untouched_tracks_null_test": untouched,
        "all_untouched_identical": all(u["identical"] for u in untouched),
        "mix_difference_explained_by_changed_tracks": interaction,
        "post_master_difference_vs_changed_tracks": post_interaction, "changed_tracks": changed_report,
        "qa": analysis.summarize(y_after, SAMPLE_RATE, after["params"]["bpm"], after["params"]["key"], after["params"]["scale"]),
    }


def transcription_test(r: Renderer, project: dict, roles: list[str], out: Path) -> list[dict]:
    """Basic Pitch on stems where we *know* the true notes."""
    import mir_eval
    import soundfile as sf
    from studio.providers.transcription import BasicPitchProvider
    from studio.project import track_notes_absolute
    bp = BasicPitchProvider()
    ok, why = bp.available()
    if not ok:
        return [{"skipped": why}]
    res = []
    spb = 60.0 / project["params"]["bpm"]
    for role in roles:
        try:
            t = track_by_ref(project, role)
        except ValueError:
            continue
        y = r.stem(project, t)
        path = out / f"stem_{role}.wav"
        sf.write(str(path), (y / max(1.0, float(np.max(np.abs(y))))).T, SAMPLE_RATE)
        t0 = time.time()
        est = bp.transcribe(path, project["params"]["bpm"])  # note: passing minimum_frequency made basic-pitch 0.4.0 return nothing
        dt = time.time() - t0
        ref = track_notes_absolute(t)
        ref_iv = np.array([[n["t"] * spb, (n["t"] + n["d"]) * spb] for n in ref])
        ref_hz = np.array([440 * 2 ** ((n["p"] - 69) / 12) for n in ref])
        if not est:
            res.append({"role": role, "preset": t["instrument"]["preset"], "ref_notes": len(ref), "est_notes": 0, "f1_onset_pitch": 0.0})
            continue
        est_iv = np.array([[n["t"] * spb, (n["t"] + n["d"]) * spb] for n in est])
        est_hz = np.array([440 * 2 ** ((n["p"] - 69) / 12) for n in est])
        p_, r_, f_, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_iv, ref_hz, est_iv, est_hz, offset_ratio=None)
        # pitch-class-only (octave errors forgiven) to see whether failures are octave confusions
        ref_pc = 440 * 2 ** (((np.array([n["p"] for n in ref]) % 12) + 60 - 69) / 12)
        est_pc = 440 * 2 ** (((np.array([n["p"] for n in est]) % 12) + 60 - 69) / 12)
        _, _, f_pc, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_iv, ref_pc, est_iv, est_pc, offset_ratio=None)
        on_p, on_r, on_f = mir_eval.transcription.onset_precision_recall_f1(ref_iv, est_iv)
        res.append({"role": role, "preset": t["instrument"]["preset"], "ref_notes": len(ref), "est_notes": len(est),
                    "precision": round(p_, 3), "recall": round(r_, 3), "f1_onset_pitch": round(f_, 3),
                    "f1_ignoring_octave": round(f_pc, 3), "f1_onset_only": round(on_f, 3), "seconds": round(dt, 2)})
    return res


def reference_808_study(out: Path) -> dict:
    """Distortion profile of bass stems the analyzer already produced (htdemucs) vs our 808 renders.

    Reference files are existing SoundLens analyzer test material; they are used for
    measurement only, reported anonymously, and are not used to train anything.
    """
    import soundfile as sf
    stem_root = ROOT / "stems" / "htdemucs"
    refs = []
    for i, folder in enumerate(sorted(p for p in stem_root.iterdir() if p.is_dir()) if stem_root.exists() else []):
        bass = folder / "bass.wav"
        if not bass.exists():
            continue
        y, sr = sf.read(str(bass), dtype="float32", always_2d=True)
        y = y.T
        mid = y[:, y.shape[1] // 3: y.shape[1] // 3 + sr * 40]
        if sr != SAMPLE_RATE:
            import librosa
            mid = librosa.resample(mid, orig_sr=sr, target_sr=SAMPLE_RATE)
        prof = analysis.distortion_profile(mid)
        stems = {s: folder / f"{s}.wav" for s in ("vocals", "drums", "bass", "other")}
        mix_candidates = list((ROOT / "uploads").glob(f"{folder.name}.wav"))
        recon = None
        if mix_candidates and all(p.exists() for p in stems.values()):
            mix, msr = sf.read(str(mix_candidates[0]), dtype="float32", always_2d=True)
            parts = [sf.read(str(p), dtype="float32", always_2d=True)[0] for p in stems.values()]
            n = min([mix.shape[0]] + [q.shape[0] for q in parts])
            summed = sum(q[:n] for q in parts)
            recon = analysis.null_test(mix[:n].T, summed.T)["residual_db"]
            env_corr = analysis.leakage(parts[2][:n].T, [parts[1][:n].T, parts[3][:n].T])["envelope_correlation"]
        else:
            env_corr = None
        refs.append({"ref": f"ref_{i + 1}", **prof, "sum_of_stems_vs_mix_residual_db": recon,
                     "bass_envelope_corr_with_drums_other": env_corr})
    ours = []
    bpm = 140.0
    notes = [{"t": 0, "d": 1.5, "p": 31, "v": 0.95}, {"t": 2, "d": 1.0, "p": 31, "v": 0.95}, {"t": 3, "d": 1.0, "p": 34, "v": 0.9},
             {"t": 4, "d": 2.0, "p": 29, "v": 0.95}, {"t": 6, "d": 1.5, "p": 31, "v": 0.95}]
    notes = [dict(n, t=n["t"] + 8 * k) for k in range(4) for n in notes]
    n = int(SAMPLE_RATE * 32 * 60 / bpm) + SAMPLE_RATE
    chains = {
        "808_clean": [], "808_warm": [], "808_distorted": [], "808_blown": [],
        "808_clean + DSP hard distortion (sub kept)": [{"type": "distortion", "params": {"drive_db": 24, "mode": "hard", "keep_sub_hz": 70, "tone_hz": 6500}}],
        "808_clean + DSP clipper +12 dB": [{"type": "clipper", "params": {"ceiling_db": -1, "drive_db": 12}}],
        "808_warm + DSP tube saturation + clipper": [{"type": "distortion", "params": {"drive_db": 18, "mode": "tube"}}, {"type": "clipper", "params": {"ceiling_db": -1, "drive_db": 6}}],
    }
    for name, chain in chains.items():
        base = name.split(" ")[0]
        y = synth.render_notes(synth.preset(base), notes, bpm, n, SAMPLE_RATE, seed=1)
        y = dsp.apply_chain(y, SAMPLE_RATE, [dsp.validate_effect(f) for f in chain], bpm=bpm)
        y = y / max(1e-9, float(np.max(np.abs(y)))) * 0.9
        ours.append({"render": name, **analysis.distortion_profile(y)})
    return {"references": refs, "ours": ours}


def consistency_study(r: Renderer, seeds: list[int], seconds: float) -> dict:
    """Does each character keep a recognisable identity across seeds? Feature spread within vs between characters."""
    rows = []
    for cid, c in characters.load_all().items():
        for s in seeds:
            p = composer.generate_project(c, {"duration_s": seconds}, seed=s)
            r.calibrate_master(p)
            y = r.mix(p)
            bands = analysis.spectral_bands(y)
            bass = r.stem(p, track_by_ref(p, "808"))
            dp = analysis.distortion_profile(bass, max_seconds=20)
            hat_notes = sum(len(cl.get("notes", [])) for t in p["tracks"] if t["role"] == "hat" for cl in t["clips"])
            rows.append({"character": cid, "seed": s, "bpm": p["params"]["bpm"], "key": f"{p['params']['key']} {p['params']['scale']}",
                         "lufs": round(analysis.loudness_lufs(y), 2), "sub": bands["sub_20_60"], "bass": bands["bass_60_250"],
                         "high": bands["high_6k_20k"], "808_thd_pct": dp["thd_pct"], "808_harmonic_ratio_db": dp["harmonic_ratio_db"],
                         "hat_notes_per_bar": round(hat_notes / max(1, sum(sx["bars"] for sx in p["sections"])), 2),
                         "tracks": [t["role"] for t in p["tracks"]]})
    # LUFS is excluded: it is set by each character's calibration target, so it would trivially identify them
    feats = ["sub", "high", "808_harmonic_ratio_db", "808_thd_pct", "hat_notes_per_bar", "bpm"]
    X = np.array([[row[f] if row[f] is not None else 0.0 for f in feats] for row in rows], dtype=float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    labels = np.array([row["character"] for row in rows])
    within, between = [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d = float(np.linalg.norm(Z[i] - Z[j]))
            (within if labels[i] == labels[j] else between).append(d)
    # nearest-centroid identification (leave-one-out)
    correct = 0
    for i in range(len(rows)):
        cents = {c: Z[(labels == c) & (np.arange(len(rows)) != i)].mean(0) for c in set(labels)}
        guess = min(cents, key=lambda c: np.linalg.norm(Z[i] - cents[c]))
        correct += guess == labels[i]
    return {"rows": rows, "features": feats, "mean_within_character_distance": round(float(np.mean(within)), 3),
            "mean_between_character_distance": round(float(np.mean(between)), 3),
            "leave_one_out_identification_accuracy": round(correct / len(rows), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments" / "output"))
    ap.add_argument("--seconds", type=float, default=32.0)
    ap.add_argument("--quick", action="store_true", help="skip the multi-seed consistency study")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r = Renderer(out / "cache")
    results: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": args.seconds}

    # 1. generate a short instrumental (structured: every layer exists by design)
    vanta = characters.get("vanta")
    t0 = time.time()
    p0 = composer.generate_project(vanta, {"duration_s": args.seconds, "prompt": "dark aggressive rage beat"}, seed=11)
    gen_s = time.time() - t0
    t0 = time.time()
    calib = r.calibrate_master(p0)
    y0 = r.mix(p0)
    render_s = time.time() - t0
    r.export_wav(p0, out / "01_original.wav")
    r.export_mp3(p0, out / "01_original.mp3", bitrate="160k")
    spectrogram_png(y0, out / "01_original.png", "VANTA original")
    expected_s = song_samples(p0) / SAMPLE_RATE
    results["generation"] = {
        "character": "vanta", "params": p0["params"], "sections": [(s["label"], s["bars"]) for s in p0["sections"]],
        "tracks": [{"name": t["name"], "role": t["role"], "instrument": t["instrument"]["preset"],
                    "notes": sum(len(c.get("notes", [])) for c in t["clips"])} for t in p0["tracks"]],
        "compose_seconds": round(gen_s, 3), "render_seconds": round(render_s, 2), "audio_seconds": round(expected_s, 2),
        "qa": analysis.summarize(y0, SAMPLE_RATE, p0["params"]["bpm"], p0["params"]["key"], p0["params"]["scale"], expected_s),
        "export_integrity": analysis.export_integrity(out / "01_original.wav", SAMPLE_RATE, 2, expected_s),
        "prompt_hints": p0["generation_notes"]["prompt_hints"],
        "master_calibration": calib,
        "timing_alignment": {role: analysis.timing_alignment(r.stem(p0, track_by_ref(p0, role)),
                                                             [n["t"] for c in track_by_ref(p0, role)["clips"] for n in c["notes"]],
                                                             p0["params"]["bpm"]) for role in ("kick", "snare", "clap")},
    }

    edits = []
    # 2a. 808 replacement (instrument swap: same notes, new sound)
    p1, rep = edit_and_verify(r, p0, vanta, [{"op": "set_instrument", "track": "808", "preset": "808_clean"},
                                             {"op": "remove_effect", "track": "808", "type": "distortion"}], "02_808_replaced_clean", out)
    edits.append(rep)
    # 2b. 808 via natural language
    nl = interpret(p1, "make the 808 a lot more distorted", character=vanta)
    p2, rep = edit_and_verify(r, p1, vanta, nl["actions"], "03_808_more_distorted_nl", out)
    rep["nl_explanation"] = nl["explanation"]
    edits.append(rep)
    # 2c. melodic instrument modification
    nl = interpret(p2, "change the melody to a bell", character=vanta)
    p3, rep = edit_and_verify(r, p2, vanta, nl["actions"], "04_melody_to_bell_nl", out)
    rep["nl_explanation"] = nl["explanation"]
    edits.append(rep)
    # 2d. targeted regeneration of one section
    sec = next(s for s in p3["sections"] if s["name"] == "verse")
    p4, rep = edit_and_verify(r, p3, vanta, [{"op": "regenerate_part", "track": "melody", "sections": [sec["id"]]}],
                              "05_regenerate_verse_melody", out, section_ref=sec["id"])
    edits.append(rep)
    # 2e. structural edit: drums out of the first hook
    nl = interpret(p4, "take the drums out of the first hook", character=vanta)
    p5, rep = edit_and_verify(r, p4, vanta, nl["actions"], "06_no_drums_first_hook", out)
    rep["nl_explanation"] = nl["explanation"]
    edits.append(rep)
    results["edits"] = edits

    # export of the final arrangement
    r.export_stems_zip(p5, out / "06_stems.zip")
    import zipfile
    with zipfile.ZipFile(out / "06_stems.zip") as z:
        results["stems_zip"] = {"files": len(z.namelist()), "midi_files": sum(1 for n in z.namelist() if n.endswith(".mid")),
                                "bytes": (out / "06_stems.zip").stat().st_size}

    # 3. transcription accuracy vs ground truth
    results["transcription_vs_ground_truth"] = transcription_test(r, p0, ["melody", "808", "chords"], out) + \
        transcription_test(r, p1, ["808"], out)
    # 4. 808 study vs existing reference stems
    results["808_study"] = reference_808_study(out)
    # 5. styles + identity consistency
    if not args.quick:
        results["consistency"] = consistency_study(r, [1, 2, 3], min(args.seconds, 24.0))
    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: v for k, v in results.items() if k in ("generation",)}, indent=1, default=str)[:3000])
    for e in edits:
        print(e["label"], "untouched identical:", e["all_untouched_identical"], "changed:", [c["name"] for c in e["diff"]["sound_changed"]],
              "mix residual vs explained:", e["mix_difference_explained_by_changed_tracks"]["residual_db"], "render:", e["render_stats"],
              e.get("changed_tracks"))
    print("transcription:", json.dumps(results["transcription_vs_ground_truth"], indent=0))


if __name__ == "__main__":
    main()
