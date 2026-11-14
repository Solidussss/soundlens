"""Reference-music collections for characters, with explicit provenance.

A collection is a set of recordings plus a *rights record*. Analysis turns the
audio into a feature profile (loudness, spectral balance, 808 distortion,
tempo, stereo) that can calibrate a character without copying any audio.

Permitted uses are explicit and checked:

* ``analysis``     – measure features (default; what this module does)
* ``conditioning`` – pass audio as a reference to a generative model at inference time
* ``training``     – fine-tune / LoRA on the audio

Altering recordings does not change their licensing status, so nothing here
infers rights from what was done to the files. ``training`` must be granted by
whoever holds the rights and is recorded with evidence.

CLI::

    python -m studio.references create --id dark_refs --title "Dark refs" --rights-holder "SoundLens (own productions)" \
        --license "owned" --uses analysis,conditioning --files a.wav b.wav
    python -m studio.references analyze dark_refs
    python -m studio.references attach dark_refs static_saint
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE, analysis

USES = {"analysis", "conditioning", "training"}


def _root() -> Path:
    base = Path(os.getenv("SOUNDLENS_DATA_DIR", "/data" if Path("/data").exists() else "."))
    d = base / "studio" / "references"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def create(collection_id: str, title: str, rights_holder: str, license_: str, uses: list[str], files: list[Path],
           evidence: str = "", root: Path | None = None) -> dict:
    bad = set(uses) - USES
    if bad:
        raise ValueError(f"unknown uses {bad}")
    if not rights_holder or not license_:
        raise ValueError("a rights holder and licence are required")
    root = root or _root()
    d = root / collection_id
    if d.exists():
        raise ValueError("collection exists")
    d.mkdir(parents=True)
    items = []
    for f in files:
        f = Path(f)
        items.append({"file": str(f.resolve()), "sha256": _sha(f), "bytes": f.stat().st_size})
    rec = {"id": collection_id, "title": title, "created_at": time.time(),
           "rights": {"holder": rights_holder, "license": license_, "permitted_uses": sorted(set(uses)) or ["analysis"], "evidence": evidence},
           "items": items, "profile": None}
    (d / "collection.json").write_text(json.dumps(rec, indent=2))
    return rec


def load(collection_id: str, root: Path | None = None) -> dict:
    return json.loads(((root or _root()) / collection_id / "collection.json").read_text())


def require_use(collection: dict, use: str) -> None:
    if use not in collection["rights"]["permitted_uses"]:
        raise PermissionError(f"collection {collection['id']} is not cleared for {use!r} "
                              f"(cleared for {collection['rights']['permitted_uses']})")


def analyze_file(path: Path, max_seconds: float = 90.0, bass_stem: Path | None = None) -> dict:
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.T
    if sr != SAMPLE_RATE:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    start = max(0, y.shape[1] // 4 - int(max_seconds * SAMPLE_RATE) // 2)
    seg = y[:, start:start + int(max_seconds * SAMPLE_RATE)]
    if seg.shape[0] == 1:
        seg = np.vstack([seg, seg])
    feats = {
        "lufs": round(analysis.loudness_lufs(seg), 2),
        "true_peak_db": round(analysis.true_peak_db(seg), 2),
        "clip_runs_per_min": round(analysis.clipping(seg)["clip_runs"] / (seg.shape[1] / SAMPLE_RATE / 60), 2),
        "bands": analysis.spectral_bands(seg),
        "stereo": analysis.stereo_stats(seg),
        "tempo": analysis.estimate_bpm(seg)["bpm_estimate"],
        "key": analysis.estimate_key(seg)["key_estimate"],
    }
    if bass_stem and Path(bass_stem).exists():
        b, bsr = sf.read(str(bass_stem), dtype="float32", always_2d=True)
        b = b.T[:, start:start + int(max_seconds * bsr)]
        feats["bass_distortion"] = analysis.distortion_profile(b, bsr)
    return feats


def analyze(collection_id: str, root: Path | None = None, stems_dir: Path | None = None) -> dict:
    """Measure every item and store a robust (median) profile. Needs only the 'analysis' permission."""
    col = load(collection_id, root)
    require_use(col, "analysis")
    per = []
    for it in col["items"]:
        p = Path(it["file"])
        stem = (stems_dir / p.stem / "bass.wav") if stems_dir else None
        per.append(analyze_file(p, bass_stem=stem))
    def med(get):
        vals = [v for v in (get(f) for f in per) if v is not None]
        return round(float(np.median(vals)), 3) if vals else None
    profile = {
        "n": len(per),
        "lufs": med(lambda f: f["lufs"]), "true_peak_db": med(lambda f: f["true_peak_db"]),
        "clip_runs_per_min": med(lambda f: f["clip_runs_per_min"]),
        "sub_share": med(lambda f: f["bands"]["sub_20_60"]), "high_share": med(lambda f: f["bands"]["high_6k_20k"]),
        "stereo_correlation": med(lambda f: f["stereo"]["correlation"]),
        "tempo_estimate": med(lambda f: f["tempo"]),
        "bass_thd_pct": med(lambda f: (f.get("bass_distortion") or {}).get("thd_pct")),
        "bass_harmonic_ratio_db": med(lambda f: (f.get("bass_distortion") or {}).get("harmonic_ratio_db")),
        "per_item": per,
    }
    col["profile"] = profile
    ((root or _root()) / collection_id / "collection.json").write_text(json.dumps(col, indent=2))
    return profile


def suggest_character_settings(profile: dict) -> dict:
    """Map a measured profile onto engine settings (feature-based conditioning, no audio copied)."""
    out: dict = {}
    if profile.get("lufs") is not None:
        out["mix.target_lufs"] = float(np.clip(profile["lufs"], -16, -5))
    hr = profile.get("bass_harmonic_ratio_db")
    if hr is not None:
        # calibrated against our presets' measured harmonic ratios (docs/studio/poc_results.md)
        out["bass.preset"] = "808_clean" if hr < -26 else "808_warm" if hr < -16 else "808_distorted" if hr < -12 else "808_blown"
    if profile.get("clip_runs_per_min") is not None and profile["clip_runs_per_min"] > 20:
        out["mix.master_fx"] = "add a clipper before the limiter (references clip audibly)"
    return out


def attach(collection_id: str, character_path: Path, root: Path | None = None) -> dict:
    col = load(collection_id, root)
    c = json.loads(Path(character_path).read_text())
    refs = c.setdefault("references", {}).setdefault("collections", [])
    if not any(r["id"] == collection_id for r in refs):
        refs.append({"id": collection_id, "permitted_uses": col["rights"]["permitted_uses"]})
    Path(character_path).write_text(json.dumps(c, indent=2) + "\n")
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m studio.references")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--id", required=True); c.add_argument("--title", required=True)
    c.add_argument("--rights-holder", required=True); c.add_argument("--license", required=True)
    c.add_argument("--uses", default="analysis"); c.add_argument("--evidence", default="")
    c.add_argument("--files", nargs="+", required=True)
    a = sub.add_parser("analyze"); a.add_argument("id"); a.add_argument("--stems-dir")
    t = sub.add_parser("attach"); t.add_argument("id"); t.add_argument("character")
    args = ap.parse_args(argv)
    if args.cmd == "create":
        print(json.dumps(create(args.id, args.title, args.rights_holder, args.license, args.uses.split(","), [Path(f) for f in args.files], args.evidence), indent=2))
    elif args.cmd == "analyze":
        prof = analyze(args.id, stems_dir=Path(args.stems_dir) if args.stems_dir else None)
        print(json.dumps({k: v for k, v in prof.items() if k != "per_item"}, indent=2))
        print("suggested settings:", json.dumps(suggest_character_settings(prof), indent=2))
    elif args.cmd == "attach":
        path = Path(__file__).with_name("characters") / f"{args.character}.json"
        attach(args.id, path)
        print(f"attached {args.id} to {args.character}")


if __name__ == "__main__":
    main()
