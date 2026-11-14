"""Deterministic per-track rendering, mixdown and export.

Each track renders to a *stem* (instrument + insert effects, before fader and
pan). Stems are cached by a hash of everything that affects their sound, so:

* changing one track re-renders only that track;
* untouched tracks come out of the cache **bit-identical**, which the PoC
  verifies with a null test instead of trusting it.

Mixer settings (gain, pan, mute, solo) are applied at mix time and never
invalidate a stem.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE, dsp, synth
from .project import MIXER_FIELDS, beats_to_seconds, total_beats, track_notes_absolute

RENDER_VERSION = "3"  # bump when synth/dsp changes alter output, to invalidate caches
TAIL_SECONDS = 2.0


def song_samples(project: dict, sr: int = SAMPLE_RATE) -> int:
    return int(round((beats_to_seconds(project, total_beats(project)) + TAIL_SECONDS) * sr))


def stem_key(project: dict, track: dict, sr: int = SAMPLE_RATE) -> str:
    relevant = {k: v for k, v in track.items() if k not in MIXER_FIELDS}
    payload = {"v": RENDER_VERSION, "sr": sr, "bpm": project["params"]["bpm"], "n": song_samples(project, sr), "track": relevant}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


class Renderer:
    def __init__(self, cache_dir: Path, assets_dir: Path | None = None, sr: int = SAMPLE_RATE):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = Path(assets_dir) if assets_dir else None
        self.sr = sr
        self.stats = {"rendered": 0, "cached": 0}

    # -- stems --------------------------------------------------------------
    def _render_track_uncached(self, project: dict, track: dict) -> np.ndarray:
        n = song_samples(project, self.sr)
        bpm = float(project["params"]["bpm"])
        if track["kind"] == "midi":
            y = synth.render_notes(track["instrument"], track_notes_absolute(track), bpm, n, self.sr, seed=track.get("seed", 0))
        else:
            y = np.zeros((2, n))
            for clip in track["clips"]:
                if clip.get("muted"):
                    continue
                y += self._render_audio_clip(project, clip, n)
        return dsp.apply_chain(y, self.sr, track.get("fx", []), bpm=bpm)

    def _render_audio_clip(self, project: dict, clip: dict, n: int) -> np.ndarray:
        src = clip["source"]
        path = Path(src["uri"])
        if not path.is_absolute():
            if self.assets_dir is None:
                raise FileNotFoundError("audio clip needs an assets directory")
            path = self.assets_dir / path
        audio, file_sr = sf.read(str(path), dtype="float64", always_2d=True)
        audio = dsp.as_stereo(audio.T)
        if file_sr != self.sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr)
        bpm = float(project["params"]["bpm"])
        origin = float(src.get("origin_bpm") or bpm)
        if abs(origin - bpm) > 1e-6:
            import librosa  # phase-vocoder stretch: audible artefacts on drums; reported by QA
            audio = np.vstack([librosa.effects.time_stretch(ch, rate=bpm / origin) for ch in audio])
        sec = 60.0 / bpm
        off = int(round(clip.get("offset", 0.0) * sec * self.sr))
        length = int(round(clip["length"] * sec * self.sr))
        start = int(round(clip["start"] * sec * self.sr))
        seg = audio[:, off:off + length]
        out = np.zeros((2, n))
        end = min(n, start + seg.shape[1])
        if end > start:
            seg = seg[:, : end - start].copy()
            fade = min(64, seg.shape[1] // 2)  # de-click clip edges
            if fade > 0:
                ramp = np.linspace(0, 1, fade)
                seg[:, :fade] *= ramp
                seg[:, -fade:] *= ramp[::-1]
            out[:, start:end] = seg * dsp.db_to_lin(clip.get("gain_db", 0.0))
        return out

    def stem(self, project: dict, track: dict) -> np.ndarray:
        key = stem_key(project, track, self.sr)
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            self.stats["cached"] += 1
            return np.load(path)
        y = self._render_track_uncached(project, track).astype(np.float32)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, y)
        tmp.replace(path)
        self.stats["rendered"] += 1
        return y

    def stem_wav(self, project: dict, track: dict) -> tuple[Path, float]:
        """16-bit WAV of a stem for the browser, peak-normalised; returns (path, scale).

        The browser multiplies by ``1/scale`` so its playback matches the server mix.
        """
        key = stem_key(project, track, self.sr)
        wav = self.cache_dir / f"{key}.wav"
        meta = self.cache_dir / f"{key}.json"
        if wav.exists() and meta.exists():
            return wav, json.loads(meta.read_text())["scale"]
        y = self.stem(project, track)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        scale = 0.98 / peak if peak > 0.98 else 1.0
        sf.write(str(wav), (y * scale).T, self.sr, subtype="PCM_16")
        meta.write_text(json.dumps({"scale": scale, "peak": peak}))
        return wav, scale

    # -- mix ----------------------------------------------------------------
    def audible_tracks(self, project: dict) -> list[dict]:
        tracks = project["tracks"]
        soloed = [t for t in tracks if t.get("solo")]
        return [t for t in (soloed or tracks) if not t.get("mute")]

    def mix(self, project: dict, apply_master: bool = True) -> np.ndarray:
        n = song_samples(project, self.sr)
        out = np.zeros((2, n))
        for t in self.audible_tracks(project):
            y = self.stem(project, t).astype(np.float64)
            out += dsp.pan_audio(y * dsp.db_to_lin(t.get("gain_db", 0.0)), t.get("pan", 0.0))
        if apply_master:
            m = project.get("master", {})
            out = out * dsp.db_to_lin(m.get("gain_db", 0.0))
            out = dsp.apply_chain(out, self.sr, m.get("fx", []), bpm=float(project["params"]["bpm"]))
        return out

    def calibrate_master(self, project: dict, target_lufs: float | None = None, iterations: int = 8) -> dict:
        """Set ``master.gain_db`` so the mastered mix lands near the character's loudness target.

        The master chain can be non-linear (clipper/limiter), so this iterates instead of
        assuming +1 dB in gives +1 dB out. Stems come from the cache, so only the master
        chain is recomputed per iteration.
        """
        from .analysis import loudness_lufs
        target = target_lufs if target_lufs is not None else project.get("master", {}).get("target_lufs")
        if target is None:
            return {"calibrated": False}
        pre = self.mix(project, apply_master=False)
        m = project.setdefault("master", {"gain_db": 0.0, "fx": []})
        g = float(m.get("gain_db", 0.0))
        bpm = float(project["params"]["bpm"])
        measured = None
        best = None
        for _ in range(iterations):
            y = dsp.apply_chain(pre * dsp.db_to_lin(g), self.sr, m.get("fx", []), bpm=bpm)
            measured = loudness_lufs(y, self.sr)
            err = float(target) - measured
            if best is None or abs(err) < abs(best[1]):
                best = (g, err, measured)
            if abs(err) < 0.3:
                break
            # a limiter flattens the gain->loudness curve, so push harder when we are below target
            g = float(np.clip(g + err * (1.6 if err > 0 else 0.9), -24.0, 18.0))
        g, _, measured = best
        m["gain_db"] = round(g, 2)
        m["target_lufs"] = float(target)
        return {"calibrated": True, "gain_db": m["gain_db"], "lufs": round(measured, 2) if measured is not None else None, "target": target}

    # -- export -------------------------------------------------------------
    def export_wav(self, project: dict, path: Path, subtype: str = "PCM_24") -> Path:
        y = self.mix(project)
        y = np.clip(y, -1.0, 1.0)  # a WAV can't hold more; QA reports whether this clipped
        sf.write(str(path), y.T, self.sr, subtype=subtype)
        return Path(path)

    def export_mp3(self, project: dict, path: Path, bitrate: str = "320k") -> Path:
        tmp = Path(path).with_suffix(".tmp.wav")
        self.export_wav(project, tmp, "PCM_16")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp), "-b:a", bitrate, str(path)], check=True)
        tmp.unlink(missing_ok=True)
        return Path(path)

    def export_stems_zip(self, project: dict, path: Path) -> Path:
        from .midi_io import track_to_midi_bytes
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for i, t in enumerate(project["tracks"], 1):
                y = self.stem(project, t).astype(np.float64)
                y = dsp.pan_audio(y * dsp.db_to_lin(t.get("gain_db", 0.0)), t.get("pan", 0.0))
                buf = io.BytesIO()
                sf.write(buf, y.T.astype(np.float32), self.sr, format="WAV", subtype="FLOAT")
                safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in t["name"]).strip()
                z.writestr(f"{i:02d} {safe}.wav", buf.getvalue())
                if t["kind"] == "midi":
                    z.writestr(f"midi/{i:02d} {safe}.mid", track_to_midi_bytes(project, t))
            z.writestr("project.json", json.dumps(project, indent=2))
        return Path(path)
