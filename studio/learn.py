"""Learn a character's musical style from reference songs.

Input: a handful of finished songs (mixes) by one artist, optionally with
Demucs stems. Output: a *style* — statistics and pattern libraries — that the
composer can generate from. Nothing audible is copied: no samples, no melodies,
no vocals. What is kept:

* tempo distribution and half-/full-time feel
* one-bar drum patterns (kick, snare/clap) on a 16-step grid, with frequencies
* hi-hat density, 32nd-note rolls, triplet rolls, open hats
* 808 behaviour: register, note lengths, how often it follows the kick, slides,
  distortion profile (THD / harmonic ratio / crest)
* key/mode tendencies and the common 4-bar chord-degree moves
* melodic texture: density, register, sustained vs plucked
* song structure (section lengths), loudness, clipping, stereo, brightness

Detection works on full mixes (harmonic/percussive separation keeps rap vocals
out of the drum detection) and is better with stems. It is statistical: a
single misdetected bar doesn't matter because patterns are kept only when they
recur.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from . import analysis

SR = 22050
HOP = 128          # 5.8 ms: fine enough for 32nd notes at 180 BPM
_MODES = {
    "minor": [0, 2, 3, 5, 7, 8, 10], "phrygian": [0, 1, 3, 5, 7, 8, 10], "dorian": [0, 2, 3, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11], "major": [0, 2, 4, 5, 7, 9, 11], "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load(path: Path, sr: int = SR, stereo: bool = False) -> np.ndarray:
    import librosa
    import soundfile as sf
    try:
        y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.T
    except Exception:  # mp3/m4a through audioread
        y, file_sr = librosa.load(str(path), sr=None, mono=False)
        y = np.atleast_2d(y)
    if not stereo:
        y = y.mean(axis=0)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def _band_onset(S_db: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    import librosa
    idx = (freqs >= lo) & (freqs < hi)
    env = librosa.onset.onset_strength(S=S_db[idx], sr=SR, hop_length=HOP, aggregate=np.mean)
    return env / (np.percentile(env, 99) + 1e-9)


# ---------------------------------------------------------------------------
# tempo + grid
# ---------------------------------------------------------------------------

def _comb_fit(env: np.ndarray, hop_s: float, bpm0: float, span: float = 0.04) -> tuple[float, float, float]:
    """Find (bpm, phase) whose evenly spaced grid best lines up with the onset envelope.

    Producer beats have one fixed tempo, so a straight grid is the right model; scanning
    tempo in 0.05-BPM steps gives a far more precise tempo than a beat tracker's frame grid.
    """
    t = np.arange(len(env)) * hop_s
    best = (bpm0, 0.0, -1.0)
    for bpm in np.arange(bpm0 * (1 - span), bpm0 * (1 + span), 0.05):
        period = 60.0 / bpm
        nb = max(8, int(round(period / 0.004)))
        bins = np.floor((t % period) / period * nb).astype(int)
        hist = np.bincount(bins, weights=env, minlength=nb) / np.bincount(bins, minlength=nb).clip(1)
        k = int(np.argmax(hist))
        if hist[k] > best[2]:
            best = (float(bpm), (k + 0.5) / nb * period, float(hist[k]))
    return best


def _tempo_and_beats(y_perc: np.ndarray) -> tuple[float, np.ndarray]:
    import librosa
    env = librosa.onset.onset_strength(y=y_perc, sr=SR, hop_length=512)
    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=env, sr=SR, hop_length=512, start_bpm=140, aggregate=np.median))[0])
    while tempo < 70:
        tempo *= 2
    while tempo > 185:
        tempo /= 2
    fine = librosa.onset.onset_strength(y=y_perc, sr=SR, hop_length=HOP)
    fine = fine / (np.percentile(fine, 99) + 1e-9)
    # Don't trust a single first guess (it can be off by 4:5 or 3:2): take the strongest tempogram
    # peaks plus their double/half/1.5x relatives, line a fixed grid up with each, keep the best fit.
    tg = librosa.feature.tempogram(onset_envelope=env, sr=SR, hop_length=512).mean(axis=1)
    freqs_t = librosa.tempo_frequencies(len(tg), sr=SR, hop_length=512)
    ok = (freqs_t >= 60) & (freqs_t <= 200)
    peaks = [float(freqs_t[i]) for i in np.argsort(-np.where(ok, tg, 0))[:6] if ok[i]]
    cands = set()
    for t0 in [tempo] + peaks:
        for f in (1, 2, 0.5, 1.5, 2 / 3, 4, 0.25):
            v = t0 * f
            # one octave only: a slower grid always "fits" better because it lands only on the
            # strongest hits, so comparing across octaves would always pick the slowest reading
            if 96 <= v <= 185:
                cands.add(round(v, 1))
    fits = []
    for cnd in sorted(cands):
        b, ph, sc = _comb_fit(fine, HOP / SR, cnd, span=0.015)
        fits.append((sc, b, ph))
    _, bpm, phase = max(fits)
    if abs(bpm - round(bpm)) < 0.3:  # producers set whole-number tempos
        bpm = float(round(bpm))
        bpm, phase, _ = _comb_fit(fine, HOP / SR, bpm, span=0.0005)
    period = 60.0 / bpm
    beats = phase + period * np.arange(int((len(y_perc) / SR - phase) / period))
    return round(bpm, 2), beats


def _maybe_double(bpm: float, beats: np.ndarray, hat_env: np.ndarray) -> tuple[float, np.ndarray]:
    """Half/double ambiguity: at the half count, trap hats look like non-stop 16ths. Count them at double speed."""
    if bpm >= 100:
        return bpm, beats
    g16 = _grid(beats, 4)
    dens = float(_hits(_sample_at(hat_env, g16, 0.012), 0.2).mean())
    if dens > 0.6:
        mids = (beats[:-1] + beats[1:]) / 2
        return bpm * 2, np.sort(np.concatenate([beats, mids]))
    return bpm, beats


def _grid(beats: np.ndarray, per_beat: int) -> np.ndarray:
    pts = []
    for a, b in zip(beats[:-1], beats[1:]):
        pts.extend(a + (b - a) * np.arange(per_beat) / per_beat)
    return np.asarray(pts)


def _sample_at(env: np.ndarray, times: np.ndarray, win_s: float = 0.025) -> np.ndarray:
    frames = np.round(times * SR / HOP).astype(int)
    w = max(1, int(win_s * SR / HOP))
    out = np.zeros(len(times))
    for i, f in enumerate(frames):
        lo, hi = max(0, f - w), min(len(env), f + w + 1)
        out[i] = env[lo:hi].max() if hi > lo else 0.0
    return out


def _hits(values: np.ndarray, floor: float = 0.3) -> np.ndarray:
    """Split grid samples into 'hit' and 'no hit' with 1-D two-means (finds the natural gap)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 4 or np.ptp(v) < 1e-6:
        return np.zeros(len(v), dtype=bool)
    lo, hi = np.percentile(v, 25), np.percentile(v, 95)
    for _ in range(30):
        t = (lo + hi) / 2
        a, b = v[v <= t], v[v > t]
        if not len(a) or not len(b):
            break
        nlo, nhi = a.mean(), b.mean()
        if abs(nlo - lo) + abs(nhi - hi) < 1e-6:
            break
        lo, hi = nlo, nhi
    return v > max((lo + hi) / 2, floor)


def _refine_offset(env: np.ndarray, beats: np.ndarray) -> float:
    """Beat trackers mark beats a few ms after the transient; find the shift that best lines the grid up."""
    g = _grid(beats, 4)
    shifts = np.arange(-0.07, 0.0701, 0.005)
    scores = [_sample_at(env, g + d, 0.008).mean() for d in shifts]
    return float(shifts[int(np.argmax(scores))])


def _bass_harmonic_db(sig: np.ndarray, f0: np.ndarray, voiced: np.ndarray, t_f0: np.ndarray) -> float | None:
    """How much the 808 is distorted, independent of its pitch: energy in harmonics 2–4
    relative to the fundamental, on frames where an 808 pitch was tracked."""
    import librosa
    n_fft = 8192
    S = np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=1024))
    fr = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    ts = np.arange(S.shape[1]) * 1024 / SR
    vals = []
    for i in np.where(voiced & np.isfinite(f0) & (f0 < 120))[0][::3]:
        j = int(np.argmin(np.abs(ts - t_f0[i])))
        f = f0[i]
        mag = lambda hz: S[max(int(np.argmin(np.abs(fr - hz))) - 1, 0): int(np.argmin(np.abs(fr - hz))) + 2, j].max()
        fund = mag(f)
        if fund <= 1e-6:
            continue
        harm = math.sqrt(sum(mag(k * f) ** 2 for k in (2, 3, 4)))
        vals.append(20 * math.log10(harm / fund + 1e-9))
    return round(float(np.median(vals)), 2) if len(vals) >= 10 else None


# ---------------------------------------------------------------------------
# per-track analysis
# ---------------------------------------------------------------------------

def analyze_track(path: Path, stems: dict[str, Path] | None = None, progress: Callable[[str], None] | None = None) -> dict:
    import librosa
    say = progress or (lambda m: None)
    say(f"loading {Path(path).name}")
    y = _load(path)
    drums = _load(stems["drums"]) if stems and "drums" in stems else None
    bass = _load(stems["bass"]) if stems and "bass" in stems else None
    say("separating harmonic / percussive")
    y_harm, y_perc = librosa.effects.hpss(y, margin=2.0)
    perc = drums if drums is not None else y_perc
    n = min(len(y), len(perc))
    tempo, beats = _tempo_and_beats(perc[:n])
    out: dict = {"file": Path(path).name, "duration_s": round(len(y) / SR, 2), "bpm": tempo, "beats": len(beats)}
    if len(beats) < 16:
        out["error"] = "could not find a steady beat"
        return out

    # band onset envelopes on the percussive signal
    S = np.abs(librosa.stft(perc, n_fft=1024, hop_length=HOP))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    kick_env = _band_onset(S_db, freqs, 30, 140)
    snare_env = _band_onset(S_db, freqs, 1500, 5000)
    hat_env = _band_onset(S_db, freqs, 7000, 11000)

    # downbeat phase: kick strongest on 1, snare on 3 (half-time) or 2 & 4
    tempo, beats = _maybe_double(tempo, beats, hat_env)
    out["bpm"] = tempo
    beats = beats + _refine_offset(kick_env + snare_env + hat_env, beats)
    g4 = beats
    kb, sb = _sample_at(kick_env, g4), _sample_at(snare_env, g4)
    scores = []
    for ph in range(4):
        k = kb[ph::4].mean() if len(kb[ph::4]) else 0
        s3 = sb[(ph + 2) % 4::4].mean()
        s24 = (sb[(ph + 1) % 4::4].mean() + sb[(ph + 3) % 4::4].mean()) / 2
        scores.append(k + max(s3, s24) - sb[ph::4].mean() * 0.5)
    phase = int(np.argmax(scores))
    bar_beats = beats[phase:]
    nbars = (len(bar_beats) - 1) // 4
    bar_beats = bar_beats[: nbars * 4 + 1]
    if nbars < 4:
        out["error"] = "too short"
        return out
    g16 = _grid(bar_beats, 4)
    g32 = _grid(bar_beats, 8)
    g24 = _grid(bar_beats, 6)
    snare_v = _sample_at(snare_env, g16)
    kick_v = _sample_at(kick_env, g16)
    # kicks have a click in the snare band and snares have low-end body: each band is only counted
    # where it clearly dominates the other
    kick = _hits(kick_v - 0.4 * snare_v).reshape(nbars, 16)
    snare = _hits(snare_v - 0.6 * kick_v).reshape(nbars, 16)
    # the low-end groove: every kick or 808 attack (full signal, not just the percussive part,
    # because an 808 re-trigger on the same pitch is an amplitude jump, not a pitch change)
    S_low = np.abs(librosa.stft(y[:n], n_fft=2048, hop_length=HOP))
    fr_low = librosa.fft_frequencies(sr=SR, n_fft=2048)
    low_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S_low[(fr_low >= 25) & (fr_low < 120)], ref=np.max), sr=SR, hop_length=HOP)
    low_env = low_env / (np.percentile(low_env, 99) + 1e-9)
    low = _hits(_sample_at(low_env, g16)).reshape(nbars, 16)
    hat32_v = _sample_at(hat_env, g32, 0.012)
    hat32 = _hits(hat32_v, 0.2).reshape(nbars, 32)
    hat24 = _hits(_sample_at(hat_env, g24, 0.012), 0.2).reshape(nbars, 24)
    bar_starts = bar_beats[: nbars * 4: 4]

    # Bar phase + feel: score every (rotation, feel) template over the whole song.
    #   half-time: snare on step 8 (beat 3);  full-time: snare on steps 4 and 12
    # A block of bars is only re-aligned when it fits a different rotation *clearly* better,
    # because beat trackers occasionally slip a beat but onset noise must not move bars around.
    def fit(k, sn, r, feel_):
        k2, s2 = np.roll(k, -r, axis=1), np.roll(sn, -r, axis=1)
        back = s2[:, 8] if feel_ == "half" else (s2[:, 4] + s2[:, 12]) / 2
        return float((back + 0.6 * k2[:, 0] - 0.4 * s2[:, 0] - 0.2 * k2[:, 8 if feel_ == "half" else 4]).sum())
    best = max(((r, f_) for r in (0, 4, 8, 12) for f_ in ("half", "full")), key=lambda rf: fit(kick, snare, *rf))
    g_rot, feel = best
    targets = [8] if feel == "half" else [4, 12]
    rotations = []
    for b0 in range(0, nbars, 4):
        blk = slice(b0, min(b0 + 4, nbars))
        base = fit(kick[blk], snare[blk], g_rot, feel)
        alt_r = max((0, 4, 8, 12), key=lambda r: fit(kick[blk], snare[blk], r, feel))
        r = alt_r if fit(kick[blk], snare[blk], alt_r, feel) >= base + 2.5 else g_rot
        rotations.append(r)
        kick[blk] = np.roll(kick[blk], -r, axis=1)
        snare[blk] = np.roll(snare[blk], -r, axis=1)
        hat32[blk] = np.roll(hat32[blk], -r * 2, axis=1)
        hat24[blk] = np.roll(hat24[blk], -r * 3 // 2, axis=1)
        low[blk] = np.roll(low[blk], -r, axis=1)
    out["realigned_blocks"] = int(sum(1 for r in rotations if r != g_rot))

    # bars that actually have drums (skip intros / drop-outs when learning patterns)
    active = kick.sum(1) + snare.sum(1) >= 2
    hat_on_16 = hat32[:, ::2]
    hat_off = hat32[:, 1::2]
    triplet_score = float(hat24[active].mean()) - float(hat32[active].mean()) if active.any() else 0.0
    out.update({
        "feel": feel,
        "kick_bars": ["".join("x" if v else "." for v in row) for row in kick[active]],
        "low_bars": ["".join("x" if v else "." for v in row) for row in low[active]],
        "snare_bars": ["".join("x" if v else "." for v in row) for row in snare[active]],
        "hat_density_16": round(float(hat_on_16[active].mean()) if active.any() else 0.0, 3),
        "hat_roll_rate": round(float((hat_off[active].sum(1) >= 3).mean()) if active.any() else 0.0, 3),
        "hat_triplet_score": round(triplet_score, 3),
        "drumless_bars_ratio": round(float(1 - active.mean()), 3),
    })
    # drop-outs right before a bar-group boundary (classic pre-hook mute)
    dropout = [i for i in range(1, nbars - 1) if active[i - 1] and not active[i] and active[min(i + 1, nbars - 1)]]
    out["dropouts_per_minute"] = round(len(dropout) / (out["duration_s"] / 60), 2)

    # --- 808 / bass ---------------------------------------------------------
    say("tracking the 808")
    low_src = bass if bass is not None else y_harm
    low_sig = librosa.resample(low_src, orig_sr=SR, target_sr=4000)
    import scipy.signal as sps
    sos = sps.butter(4, 180, "low", fs=4000, output="sos")
    low_sig = sps.sosfilt(sos, low_sig)
    f0, voiced, _ = librosa.pyin(low_sig, fmin=28, fmax=170, sr=4000, frame_length=1024, hop_length=128)
    t_f0 = np.arange(len(f0)) * 128 / 4000
    notes = []
    cur = None
    midi = librosa.hz_to_midi(np.where(voiced, f0, np.nan))
    for i, m in enumerate(midi):
        if np.isfinite(m) and (cur is None or abs(m - cur["last"]) < 0.8):
            if cur is None:
                cur = {"start": t_f0[i], "pitches": [m], "last": m}
            else:
                cur["pitches"].append(m); cur["last"] = m
        else:
            if cur is not None and len(cur["pitches"]) >= 4:
                notes.append({"start": cur["start"], "dur": len(cur["pitches"]) * 128 / 4000,
                              "midi": float(np.median(cur["pitches"])), "glide": float(np.ptp(cur["pitches"]))})
            cur = {"start": t_f0[i], "pitches": [m], "last": m} if np.isfinite(m) else None
    beat_s = 60.0 / tempo
    out["bass_harm_db"] = _bass_harmonic_db(y_harm if bass is None else bass, f0, voiced, t_f0)
    if notes:
        # how often a low-end hit coincides with a kick-band click (≈ "808 follows the kick")
        both = (low & kick)[active].sum()
        follow = float(both / max(low[active].sum(), 1))
        pitch_jumps = np.abs(np.diff([n["midi"] for n in notes]))
        out["bass_bars"] = out["low_bars"]
        out["bass"] = {
            "notes": len(notes),
            "median_midi": round(float(np.median([n["midi"] for n in notes])), 1),
            "median_len_beats": round(float(np.median([n["dur"] for n in notes])) / beat_s, 2),
            "follows_kick": round(follow, 3),
            "octave_jump_rate": round(float(np.mean((pitch_jumps > 10) & (pitch_jumps < 14))) if len(pitch_jumps) else 0.0, 3),
            "slide_rate": round(float(np.mean([n["glide"] > 1.5 for n in notes])), 3),
        }
    if bass is not None:
        lowband = librosa.resample(bass, orig_sr=SR, target_sr=11025)
        out["bass_distortion"] = analysis.distortion_profile(lowband[None, :].repeat(2, 0), 11025, max_seconds=60)
        out["bass_distortion"]["source"] = "bass stem"

    # --- harmony -----------------------------------------------------------------
    say("reading the chords")
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=SR, hop_length=512)
    mean_chroma = chroma.mean(1)
    best = None
    for tonic in range(12):
        for mode, steps in _MODES.items():
            tmpl = np.zeros(12); tmpl[[(tonic + s) % 12 for s in steps]] = 1.0
            tmpl[tonic] += 1.0; tmpl[(tonic + 7) % 12] += 0.5
            r = float(np.corrcoef(tmpl, mean_chroma)[0, 1])
            if best is None or r > best[0]:
                best = (r, tonic, mode)
    _, tonic, mode = best
    out["key"] = {"tonic": NOTE_NAMES[tonic], "mode": mode}
    # per-bar chord degree (root relative to tonic, snapped to the mode)
    steps = _MODES[mode]
    bar_times = bar_beats[::4]
    frames = np.round(bar_times * SR / 512).astype(int)
    degrees = []
    for a, b in zip(frames[:-1], frames[1:]):
        c = chroma[:, a:max(b, a + 1)].mean(1)
        scores = []
        for d, s in enumerate(steps):
            root = (tonic + s) % 12
            third = (tonic + steps[(d + 2) % 7]) % 12
            fifth = (tonic + steps[(d + 4) % 7]) % 12
            scores.append(c[root] * 1.2 + c[third] + c[fifth] * 0.8)
        degrees.append(int(np.argmax(scores)))
    progs = Counter(tuple(degrees[i:i + 4]) for i in range(0, len(degrees) - 3, 4))
    out["progressions"] = [list(p) for p, _ in progs.most_common(4)]

    # --- melody texture --------------------------------------------------------------
    S_h = np.abs(librosa.stft(y_harm, n_fft=2048, hop_length=512))
    fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
    mid = S_h[(fr > 400) & (fr < 4000)]
    cent = float(np.sum(fr[(fr > 400) & (fr < 4000), None] * mid) / (np.sum(mid) + 1e-9))
    mel_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(mid), sr=SR, hop_length=512)
    mel_on = librosa.onset.onset_detect(onset_envelope=mel_env, sr=SR, hop_length=512, units="time")
    # sustain: how much energy remains 150 ms after harmonic onsets (pads ≈ high, plucks/bells ≈ low)
    rms = np.sqrt(np.mean(mid ** 2, axis=0))
    sus = []
    for t in mel_on[:400]:
        f = int(t * SR / 512)
        if f + 7 < len(rms):
            sus.append(rms[f + 6] / (rms[f:f + 2].max() + 1e-9))
    out["melody"] = {"centroid_hz": round(cent, 1), "onsets_per_bar": round(len(mel_on) / max(nbars, 1), 2),
                     "sustain": round(float(np.median(sus)) if sus else 0.5, 3)}

    # --- structure ---------------------------------------------------------------------
    rms_full = librosa.feature.rms(y=y, hop_length=512)[0]
    bar_rms = np.array([rms_full[int(a):int(b)].mean() if b > a else 0 for a, b in zip(frames[:-1], frames[1:])])
    if len(bar_rms) > 8:
        z = (bar_rms - bar_rms.mean()) / (bar_rms.std() + 1e-9)
        change = np.where(np.abs(np.diff(z)) > 0.9)[0] + 1
        bounds = [0] + [int(c) for c in change] + [len(bar_rms)]
        lens = [b - a for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]
        out["section_bars"] = lens
        out["intro_bars"] = int(next((i for i, v in enumerate(active) if v), 0))

    # --- mix ---------------------------------------------------------------------------
    say("measuring the mix")
    st = _load(path, 44100, stereo=True)
    st = st[:, : 44100 * 120]
    if st.shape[0] == 1:
        st = np.vstack([st, st])
    bands = analysis.spectral_bands(st)
    if "bass_distortion" not in out:
        # no stem: a distorted 808 pushes energy from the sub (20–60 Hz) up into 60–250 Hz
        grit = bands["bass_60_250"] / max(bands["sub_20_60"], 1e-6)
        out["bass_distortion"] = {"grit_ratio": round(grit, 3), "source": "mix estimate (less reliable than a bass stem)"}
    out["mix"] = {"lufs": round(analysis.loudness_lufs(st), 2), "true_peak_db": round(analysis.true_peak_db(st), 2),
                  "clip_runs_per_min": round(analysis.clipping(st)["clip_runs"] / (st.shape[1] / 44100 / 60), 1),
                  "bands": bands, "stereo": analysis.stereo_stats(st)}
    return out


# ---------------------------------------------------------------------------
# aggregate + map to a character
# ---------------------------------------------------------------------------

def _pattern_library(bars: list[str], min_share: float = 0.04, top: int = 6, radius: int = 2) -> list[dict]:
    """Cluster bars that differ by ≤ ``radius`` hits and keep each cluster's majority-vote pattern.

    One stray detection makes a bar look 'new'; the cluster consensus is the groove a producer
    would actually have programmed.
    """
    c = Counter(b for b in bars if "x" in b)
    total = sum(c.values()) or 1
    clusters: list[dict] = []
    for pat, n in c.most_common():
        v = np.array([ch == "x" for ch in pat])
        for cl in clusters:
            if int(np.sum(cl["center"] != v)) <= radius:
                cl["members"].append((v, n)); cl["count"] += n
                w = np.array([m[1] for m in cl["members"]], dtype=float)
                cl["center"] = (np.array([m[0] for m in cl["members"]]) * w[:, None]).sum(0) / w.sum() >= 0.5
                break
        else:
            clusters.append({"center": v, "members": [(v, n)], "count": n})
    clusters.sort(key=lambda cl: -cl["count"])
    lib = [{"pattern": "".join("x" if x else "." for x in cl["center"]), "share": round(cl["count"] / total, 3)}
           for cl in clusters if cl["count"] / total >= min_share and cl["center"].any()]
    return lib[:top]


def aggregate(tracks: list[dict]) -> dict:
    ok = [t for t in tracks if "error" not in t]
    if not ok:
        raise ValueError("none of the songs could be analyzed (no steady beat found)")
    med = lambda xs: float(np.median(xs)) if xs else None
    # one artist rarely jumps tempo octaves: fold each song toward the group's consensus (x2, /2, x1.5, /1.5)
    raw = [t["bpm"] for t in ok]
    centre = float(np.median(raw))
    bpms = [min((b * f for f in (1, 2, 0.5, 1.5, 2 / 3)), key=lambda v: abs(np.log(v / centre))) for b in raw]
    style = {
        "tracks": len(ok), "skipped": [t["file"] for t in tracks if "error" in t],
        "bpm": {"median": round(med(bpms), 1), "min": round(min(bpms), 1), "max": round(max(bpms), 1)},
        "feel": Counter(t["feel"] for t in ok).most_common(1)[0][0],
        "kick_patterns": _pattern_library(sum((t["kick_bars"] for t in ok), [])),
        "low_patterns": _pattern_library(sum((t.get("low_bars", []) for t in ok), [])),
        "snare_patterns": _pattern_library(sum((t["snare_bars"] for t in ok), [])),
        "bass_patterns": _pattern_library(sum((t.get("bass_bars", []) for t in ok), [])),
        "hat_density_16": round(med([t["hat_density_16"] for t in ok]), 3),
        "hat_roll_rate": round(med([t["hat_roll_rate"] for t in ok]), 3),
        "hat_triplets": med([t["hat_triplet_score"] for t in ok]) > 0.02,
        "dropouts_per_minute": round(med([t["dropouts_per_minute"] for t in ok]), 2),
        "keys": Counter(f"{t['key']['tonic']} {t['key']['mode']}" for t in ok).most_common(),
        "modes": Counter(t["key"]["mode"] for t in ok).most_common(),
        "progressions": [list(p) for p, _ in Counter(tuple(p) for t in ok for p in t.get("progressions", [])).most_common(4)],
        "mix": {"lufs": round(med([t["mix"]["lufs"] for t in ok]), 2),
                "clip_runs_per_min": round(med([t["mix"]["clip_runs_per_min"] for t in ok]), 1),
                "sub_share": round(med([t["mix"]["bands"]["sub_20_60"] for t in ok]), 4),
                "high_share": round(med([t["mix"]["bands"]["high_6k_20k"] for t in ok]), 4),
                "stereo_correlation": round(med([t["mix"]["stereo"]["correlation"] for t in ok]), 3)},
        "melody": {k: round(med([t["melody"][k] for t in ok]), 3) for k in ("centroid_hz", "onsets_per_bar", "sustain")},
        "section_bars": [b for t in ok for b in t.get("section_bars", [])],
        "intro_bars": round(med([t.get("intro_bars", 4) for t in ok]) or 4),
    }
    basses = [t["bass"] for t in ok if t.get("bass")]
    if basses:
        style["bass"] = {k: round(med([b[k] for b in basses]), 3) for k in ("median_midi", "median_len_beats", "follows_kick", "octave_jump_rate", "slide_rate")}
    stem_d = [t["bass_distortion"] for t in ok if t.get("bass_distortion", {}).get("source") == "bass stem"]
    mix_d = [t["bass_distortion"]["grit_ratio"] for t in ok if "grit_ratio" in t.get("bass_distortion", {})]
    style["bass_distortion"] = {}
    hd = [t["bass_harm_db"] for t in ok if t.get("bass_harm_db") is not None]
    if hd:
        style["bass_distortion"]["harm_db"] = round(med(hd), 2)
    if stem_d:
        style["bass_distortion"].update({k: round(med([d[k] for d in stem_d if d.get(k) is not None]) or 0, 2) for k in ("thd_pct", "harmonic_ratio_db", "crest_db")})
    if mix_d:
        style["bass_distortion"]["grit_ratio"] = round(med(mix_d), 3)
    return style


def _arrangement(section_bars: list[int], intro: int) -> list[str]:
    sizes = [min(16, max(4, int(round(b / 4.0)) * 4)) for b in section_bars] or [8]
    body = int(Counter(sizes).most_common(1)[0][0])
    intro = 4 if intro <= 5 else 8
    return [f"intro:{intro}", f"hook:{body}", f"verse:{max(8, body)}", f"hook:{body}", f"verse:{max(8, body)}", f"hook:{body}", "outro:4"]


def character_from_style(style: dict, character_id: str, name: str, visual: dict, tagline: str = "") -> dict:
    """Map learned statistics onto the composer's character schema."""
    from .characters import validate_character
    bpm = style["bpm"]
    modes = [m for m, _ in style["modes"]]
    scales = [m for m in modes if m in _MODES][:3] or ["minor"]
    roots = []
    for k, _ in style["keys"]:
        r = k.split()[0]
        if r not in roots:
            roots.append(r)
    lo = max(60, int(bpm["min"]) - 4); hi = min(200, int(bpm["max"]) + 4)
    default = int(round(bpm["median"]))
    lo, hi = min(lo, default), max(hi, default)
    dist = style.get("bass_distortion", {})
    if "harmonic_ratio_db" in dist:  # measured on a bass stem (preferred)
        hr = dist["harmonic_ratio_db"]
        bass_preset = "808_clean" if hr < -26 else "808_warm" if hr < -16 else "808_distorted" if hr < -11 else "808_blown"
    else:
        g = dist.get("grit_ratio", 0.2)
        bass_preset = "808_clean" if g < 0.15 else "808_warm" if g < 0.25 else "808_distorted" if g < 0.4 else "808_blown"
    b = style.get("bass", {})
    loud = style["mix"]["lufs"]
    clips = style["mix"]["clip_runs_per_min"] > 20
    mel = style["melody"]
    sustained = mel["sustain"] > 0.55
    bright = mel["centroid_hz"] > 1300
    # melody gets a lead sound; the pads are left to the chords so the two never share a sound
    mel_preset = ("supersaw_lead" if sustained and bright else "dark_synth" if sustained else
                  "bell_bright" if bright else "bell_dark")
    if mel["onsets_per_bar"] > 6 and not sustained:
        mel_preset = "pluck_soft" if bright else "rage_lead"
    density = float(np.clip(mel["onsets_per_bar"] / 10.0, 0.25, 0.9))
    hat_rate = 0.25 if style["hat_density_16"] > 0.6 else 0.5
    # Without stems kick and 808 attacks are hard to tell apart, and in trap they mostly share one groove,
    # so generation drives kick + 808 from the learned low-end groove (the 808 follows it).
    kick_lib = [p["pattern"] for p in (style.get("low_patterns") or style["kick_patterns"]) if len(p["pattern"]) == 16] or ["x.......x.x....."]
    snare_lib = [p["pattern"] for p in style["snare_patterns"] if len(p["pattern"]) == 16]
    progressions = [p for p in style["progressions"] if len(p) == 4][:3] or [[0, 5, 3, 4]]
    mel_reg_lo = 64 if mel["centroid_hz"] < 900 else 70 if mel["centroid_hz"] < 1500 else 74
    character = {
        "id": character_id, "name": name, "tagline": tagline or "Learned from reference songs.",
        "description": f"Learned from {style['tracks']} song(s): around {default} BPM, {style['feel']}-time drums, "
                       f"{bass_preset.replace('_', ' ')}, mostly {scales[0].replace('_', ' ')}.",
        "visual": visual, "genre": "rap", "subgenre": "learned",
        "tempo": {"min": lo, "max": hi, "default": default, "feel": style["feel"]},
        "swing": 0.0,
        "keys": {"preferred_scales": scales, "preferred_roots": roots[:4] or ["C"]},
        "drums": {
            "kick": {"preset": "kick_hard" if clips else "kick_punch", "pattern": "learned", "patterns": kick_lib},
            "snare": {"preset": "snare_tight", "patterns": snare_lib} if snare_lib else {"preset": "snare_tight"},
            "clap": {"preset": "clap_snappy"},
            "hat": {"preset": "hat_crisp" if style["mix"]["high_share"] > 0.01 else "hat_dark", "rate": hat_rate,
                    "roll_density": float(np.clip(style["hat_roll_rate"] * 2.0, 0.05, 0.9)), "triplet_rolls": bool(style["hat_triplets"]),
                    "open_prob": 0.04},
            "perc": None,
        },
        "bass": {"preset": bass_preset,
                 "style": "follow_kick",
                 "patterns": [p["pattern"] for p in style.get("bass_patterns", [])],
                 "median_len_beats": b.get("median_len_beats"), "register_midi": b.get("median_midi"),
                 "drive": None,
                 "glide_prob": float(np.clip(b.get("slide_rate", 0.2) + b.get("octave_jump_rate", 0.1), 0.05, 0.85)),
                 "octave": 1, "fx": []},
        "melody": {"preset": mel_preset, "register": [mel_reg_lo, mel_reg_lo + 16], "density": round(density, 2),
                   "motif_bars": 2 if density < 0.5 else 1, "repetition": 0.8, "step_bias": 0.6,
                   "fx": [{"type": "reverb", "params": {"mix": 0.28 if sustained else 0.22, "decay_s": 2.2}},
                          {"type": "delay", "params": {"beats": 0.75, "feedback": 0.3, "mix": 0.16}}]},
        "harmony": {"preset": "pad_warm" if not bright else "pad_glass", "progressions": progressions, "rhythm": "sustain",
                    "register": [52, 68]},
        "extra": [{"role": "atmosphere", "preset": "atmos_dark" if not bright else "atmos_air", "gain_db": -12}],
        "arrangement": {"template": _arrangement(style["section_bars"], style["intro_bars"])},
        "mix": {"master_fx": ([{"type": "clipper", "params": {"ceiling_db": -0.3, "drive_db": 3.0}}] if clips else []) +
                [{"type": "limiter", "params": {"ceiling_db": -0.5 if clips else -1.0}}],
                "target_lufs": float(np.clip(loud, -14, -5)),
                "track_gains": {"kick": -5, "snare": -7, "clap": -8, "hat": -15, "808": -6, "melody": -9, "chords": -15, "atmosphere": -12}},
        "vocals": {"default_mode": "instrumental", "delivery": "", "energy": 0.7, "tags": []},
        "generation_hints": {"caption_tags": ["rap", "trap", f"{default} bpm", scales[0].replace("_", " "),
                                              "distorted 808" if "distorted" in bass_preset or "blown" in bass_preset else "clean 808"]},
        "learned": {"version": 1, "style": {k: v for k, v in style.items() if k not in ("section_bars",)}},
        "references": {"collections": []},
    }
    return validate_character(character)


# ---------------------------------------------------------------------------
# closed-loop fitting: make the character's *output* measure like the references
# ---------------------------------------------------------------------------

TARGET_KEYS = ["sub", "high", "hat_density_16", "hat_roll_rate", "bass_harm_db", "mel_onsets", "mel_centroid", "lufs"]
# typical spread of each measurement across real songs; used to weigh errors fairly
SCALE = {"sub": 0.12, "high": 0.008, "hat_density_16": 0.12, "hat_roll_rate": 0.08, "bass_harm_db": 4.0,
         "mel_onsets": 4.0, "mel_centroid": 250.0, "lufs": 1.5}
# (mel_onsets has a wide scale on purpose: a distorted 808's harmonics and the chords also trigger it)


def measure(t: dict) -> dict:
    m = t["mix"]
    return {"sub": m["bands"]["sub_20_60"], "high": m["bands"]["high_6k_20k"], "hat_density_16": t["hat_density_16"],
            "hat_roll_rate": t["hat_roll_rate"], "bass_harm_db": t.get("bass_harm_db"), "mel_onsets": t["melody"]["onsets_per_bar"],
            "mel_centroid": t["melody"]["centroid_hz"], "lufs": m["lufs"]}


def targets_from(tracks: list[dict]) -> dict:
    ms = [measure(t) for t in tracks if "error" not in t]
    out = {}
    for k in TARGET_KEYS:
        v = [m[k] for m in ms if m.get(k) is not None]
        if v:
            out[k] = float(np.median(v))
    return out


def style_distance(meas: dict, target: dict) -> float:
    ks = [k for k in target if meas.get(k) is not None]
    return float(np.sqrt(np.mean([((meas[k] - target[k]) / SCALE[k]) ** 2 for k in ks]))) if ks else 99.0


def fit_character(character: dict, target: dict, iterations: int = 5, seconds: float = 32.0,
                  progress: Callable[[str], None] | None = None, work_dir: Path | None = None) -> dict:
    """Generate → measure → adjust, a few times. Returns the best character found plus the fit report."""
    import copy
    import tempfile
    import soundfile as sf
    from . import SAMPLE_RATE
    from .composer import generate_project
    from .render import Renderer
    say = progress or (lambda m: None)
    tmp = Path(work_dir or tempfile.mkdtemp())
    r = Renderer(tmp / "fitcache")
    c = copy.deepcopy(character)
    gains = c["mix"]["track_gains"]
    history = []
    best = None
    for it in range(iterations):
        say(f"tuning the sound ({it + 1}/{iterations})")
        p = generate_project(c, {"duration_s": seconds}, seed=1000 + it)
        r.calibrate_master(p)
        path = tmp / f"fit_{it}.wav"
        sf.write(str(path), np.clip(r.mix(p), -1, 1).T, SAMPLE_RATE)
        t = analyze_track(path)
        # test renders are only needed for this one measurement: free the disk straight away
        path.unlink(missing_ok=True)
        import shutil as _sh
        _sh.rmtree(tmp / "fitcache", ignore_errors=True)
        r = Renderer(tmp / "fitcache")
        if "error" in t:
            continue
        m = measure(t)
        d = style_distance(m, target)
        history.append({"iteration": it, "distance": round(d, 3), "measured": {k: (round(v, 4) if v is not None else None) for k, v in m.items()}})
        if best is None or d < best[0]:
            best = (d, copy.deepcopy(c))
        if it == iterations - 1:
            break
        # --- adjust each knob toward its target (signs are known, steps are damped) ---
        if "sub" in target and m["sub"] > 0:
            err_db = float(np.clip(10 * math.log10(target["sub"] / m["sub"]), -6, 6))
            # the sub band is shared by kick, 808 and pads: tilt the whole low end on the master...
            master_fx = c["mix"].setdefault("master_fx", [])
            tilt = next((f for f in master_fx if f.get("id") == "fit_low_tilt"), None)
            if tilt is None:
                tilt = {"type": "eq", "id": "fit_low_tilt", "params": {"bands": [{"type": "lowshelf", "freq": 70.0, "gain_db": 0.0, "q": 0.7}]}}
                master_fx.insert(0, tilt)
            band = tilt["params"]["bands"][0]
            band["gain_db"] = float(np.clip(band["gain_db"] + err_db * 0.9, -12, 6))
            # ...and nudge the 808 itself a little
            gains["808"] = float(np.clip(gains.get("808", -6) + err_db * 0.3, -20, 2))
        if "high" in target and m["high"] > 0:
            gains["hat"] = float(np.clip(gains.get("hat", -15) + np.clip(10 * math.log10(target["high"] / m["high"]), -5, 5) * 0.8, -30, -2))
        hat = c["drums"].get("hat")
        if hat and "hat_density_16" in target:
            if m["hat_density_16"] < target["hat_density_16"] - 0.22 and hat.get("rate", 0.5) > 0.25:
                hat["rate"] = 0.25
            elif m["hat_density_16"] > target["hat_density_16"] + 0.22 and hat.get("rate", 0.5) < 0.5:
                hat["rate"] = 0.5
        if hat and "hat_roll_rate" in target:
            hat["roll_density"] = float(np.clip(hat.get("roll_density", 0.3) + (target["hat_roll_rate"] - m["hat_roll_rate"]) * 0.7, 0.0, 1.0))
        if target.get("bass_harm_db") is not None and m.get("bass_harm_db") is not None:
            cur = c["bass"].get("drive")
            if cur is None:
                from .synth import PRESETS
                cur = PRESETS[c["bass"]["preset"]]["params"].get("drive", 0.0)
            gap = target["bass_harm_db"] - m["bass_harm_db"]
            new_drive = float(np.clip(cur + gap / 25.0, 0.0, 1.0))
            c["bass"]["drive"] = new_drive
            fx = c["bass"].setdefault("fx", [])
            dist = next((f for f in fx if f.get("id") == "fit_dist"), None)
            if gap > 3 and new_drive >= 0.99:
                # the oscillator drive is maxed: add real distortion on top, keeping the sub clean
                if dist is None:
                    dist = {"type": "distortion", "id": "fit_dist", "params": {"drive_db": 8.0, "mode": "hard", "keep_sub_hz": 60.0, "tone_hz": 7000.0}}
                    fx.append(dist)
                else:
                    dist["params"]["drive_db"] = float(np.clip(dist["params"]["drive_db"] + gap * 0.8, 0, 36))
            elif dist is not None and gap < -3:
                dist["params"]["drive_db"] = float(np.clip(dist["params"]["drive_db"] + gap * 0.8, 0, 36))
        mel = c["melody"]
        if "mel_onsets" in target and m["mel_onsets"] > 0:
            mel["density"] = float(np.clip(mel.get("density", 0.6) * (target["mel_onsets"] / m["mel_onsets"]) ** 0.8, 0.12, 0.95))
            if m["mel_onsets"] > target["mel_onsets"] * 1.3:
                c["harmony"]["rhythm"] = "sustain"
        if "mel_centroid" in target and m["mel_centroid"] > 0:
            shift = int(round(12 * math.log2(target["mel_centroid"] / m["mel_centroid"]) / 2))  # half-steps, damped
            lo, hi = mel["register"]
            lo = int(np.clip(lo + shift, 52, 84))
            mel["register"] = [lo, lo + (hi - mel["register"][0])]
        if "lufs" in target:
            c["mix"]["target_lufs"] = float(np.clip(target["lufs"], -14, -5))
    if best is None:
        return {"character": character, "report": {"fitted": False, "reason": "could not measure the test renders"}}
    fitted = best[1]
    report = {"fitted": True, "iterations": len(history), "start_distance": history[0]["distance"] if history else None,
              "best_distance": round(best[0], 3), "target": {k: round(v, 4) for k, v in target.items()}, "history": history}
    fitted.setdefault("learned", {})["fit"] = {k: v for k, v in report.items() if k != "history"}
    return {"character": fitted, "report": report}
