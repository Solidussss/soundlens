from __future__ import annotations

"""SoundLens accuracy v2.

Runtime patches for the three measurements users notice immediately:
- BPM: multi-region consensus + beat-grid stability + half/double-time handling.
- Key: harmonic-only, multi-chroma, multi-window ensemble with honest confidence.
- Artist Match: transposition-aware chroma distance and library-size bias correction.

The public function signatures stay compatible with the existing app.
"""

import math
from typing import Dict, List, Tuple

import librosa
import numpy as np

import soundlens_pro as core
import compare_to_profile_pro as compare

EPS = 1e-9
NOTES = core.NOTES_SHARP
MAJOR = np.asarray(core.MAJOR_PROFILE, dtype=float)
MINOR = np.asarray(core.MINOR_PROFILE, dtype=float)


def _mono(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        # librosa typically returns channels x samples; fall back safely.
        axis = 0 if arr.shape[0] <= 8 else -1
        arr = np.mean(arr, axis=axis)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if arr.size:
        arr = arr - float(np.mean(arr))
    return arr


def _resample(y: np.ndarray, sr: int, target: int) -> Tuple[np.ndarray, int]:
    if sr == target:
        return y, sr
    return librosa.resample(y, orig_sr=sr, target_sr=target).astype(np.float32), target


def _fold_tempo(value: float) -> List[float]:
    if not np.isfinite(value) or value <= 0:
        return []
    values = {float(value)}
    values.add(float(value) * 0.5)
    values.add(float(value) * 2.0)
    out = []
    for tempo in values:
        while tempo < 55:
            tempo *= 2.0
        while tempo > 190:
            tempo /= 2.0
        if 55 <= tempo <= 190:
            out.append(float(tempo))
    return out


def _pulse_score(onset: np.ndarray, sr: int, hop: int, bpm: float) -> float:
    if onset.size < 12 or bpm <= 0:
        return 0.0
    x = np.asarray(onset, dtype=float)
    x = np.maximum(x - np.median(x), 0.0)
    if float(np.max(x)) <= EPS:
        return 0.0
    ac = librosa.autocorrelate(x)
    if ac.size < 4 or float(ac[0]) <= EPS:
        return 0.0
    ac = ac / (float(ac[0]) + EPS)
    lag = (60.0 * sr) / (hop * bpm)
    score = 0.0
    for mult, weight in ((1.0, 1.0), (2.0, 0.48), (4.0, 0.22), (0.5, 0.10)):
        idx = int(round(lag * mult))
        if 0 < idx < len(ac):
            score += weight * max(0.0, float(ac[idx]))
    return score


def _beat_grid_score(onset: np.ndarray, sr: int, hop: int, bpm: float) -> float:
    """How consistently strong onsets recur on the candidate pulse grid."""
    if onset.size < 16 or bpm <= 0:
        return 0.0
    period = (60.0 * sr) / (hop * bpm)
    if period < 1.5:
        return 0.0
    x = np.asarray(onset, dtype=float)
    denom = float(np.percentile(x, 95)) + EPS
    x = np.clip(x / denom, 0.0, 1.0)
    phases = max(1, int(round(period)))
    best = 0.0
    # Search a bounded number of phases; pulse position matters as much as periodicity.
    for phase in range(min(phases, 64)):
        positions = np.arange(phase, len(x), period)
        if len(positions) < 4:
            continue
        vals = []
        for p in positions:
            i = int(round(p))
            if 0 <= i < len(x):
                a, b = max(0, i - 1), min(len(x), i + 2)
                vals.append(float(np.max(x[a:b])))
        if vals:
            best = max(best, float(np.mean(vals)))
    return best


def detect_bpm_v2(y: np.ndarray, sr: int) -> float:
    try:
        audio = _mono(y)
        audio, work_sr = _resample(audio, int(sr), 16000)
        duration = len(audio) / max(work_sr, 1)
        if duration < 4.0:
            return 0.0

        # Percussive emphasis makes kick/snare/hat timing dominate over sustained melody.
        try:
            percussive = librosa.effects.percussive(audio, margin=3.0)
            if float(np.sqrt(np.mean(percussive * percussive))) < 1e-5:
                percussive = audio
        except Exception:
            percussive = audio

        hop = 256
        centers = (0.15, 0.35, 0.55, 0.75, 0.90) if duration >= 50 else ((0.22, 0.50, 0.78) if duration >= 20 else (0.50,))
        win_sec = min(22.0, duration)
        win = max(int(6 * work_sr), int(win_sec * work_sr))
        candidate_votes: List[float] = []
        region_data = []

        for ratio in centers:
            c = int(len(percussive) * ratio)
            st = max(0, min(c - win // 2, len(percussive) - win))
            piece = percussive[st:st + win]
            if len(piece) < work_sr * 4:
                continue
            onset = librosa.onset.onset_strength(y=piece, sr=work_sr, hop_length=hop, aggregate=np.median)
            onset = np.asarray(onset, dtype=float)
            if onset.size < 16 or float(np.max(onset)) <= EPS:
                continue
            region_data.append(onset)

            # Independent estimators create candidates; final selection is scored below.
            try:
                t = float(np.ravel(librosa.feature.tempo(onset_envelope=onset, sr=work_sr, hop_length=hop, aggregate=np.median))[0])
                candidate_votes.extend(_fold_tempo(t))
            except Exception:
                pass
            try:
                tempo, _beats = librosa.beat.beat_track(onset_envelope=onset, sr=work_sr, hop_length=hop, bpm=None)
                candidate_votes.extend(_fold_tempo(float(np.ravel(tempo)[0])))
            except Exception:
                pass

        if not region_data:
            return core.detect_bpm(y, sr) if core.detect_bpm is not detect_bpm_v2 else 0.0

        # Dense candidate grid plus direct-estimator neighborhoods prevents quantization misses.
        candidates = set(float(v) for v in range(55, 191))
        for vote in candidate_votes:
            for delta in np.arange(-2.0, 2.01, 0.5):
                v = vote + float(delta)
                if 55 <= v <= 190:
                    candidates.add(round(v, 2))

        vote_array = np.asarray(candidate_votes, dtype=float) if candidate_votes else np.asarray([], dtype=float)
        scored: List[Tuple[float, float]] = []
        for bpm in candidates:
            pulse_scores = [_pulse_score(onset, work_sr, hop, bpm) for onset in region_data]
            grid_scores = [_beat_grid_score(onset, work_sr, hop, bpm) for onset in region_data]
            pulse = float(np.median(pulse_scores))
            grid = float(np.median(grid_scores))
            consistency = 1.0 - min(1.0, float(np.std(pulse_scores)) / max(float(np.mean(pulse_scores)), 0.05))

            direct = 0.0
            if vote_array.size:
                # Include half/double relationships but reward an exact consensus most.
                distances = np.minimum.reduce([
                    np.abs(vote_array - bpm),
                    np.abs(vote_array * 2.0 - bpm) + 1.5,
                    np.abs(vote_array * 0.5 - bpm) + 1.5,
                ])
                direct = float(np.mean(np.clip(1.0 - distances / 6.0, 0.0, 1.0)))

            score = pulse * 0.50 + grid * 0.28 + consistency * 0.10 + direct * 0.12
            scored.append((score, bpm))

        scored.sort(reverse=True)
        if not scored or scored[0][0] <= 0:
            return 0.0
        best_score, best_bpm = scored[0]

        # Resolve classic half/double ambiguity by comparing grid stability directly.
        related = [item for item in scored[:30] if abs(item[1] - best_bpm * 2) < 2.0 or abs(item[1] * 2 - best_bpm) < 2.0]
        for alt_score, alt_bpm in related:
            # Only flip if the alternative has materially stronger evidence.
            if alt_score > best_score * 1.08:
                best_score, best_bpm = alt_score, alt_bpm

        return round(float(best_bpm), 1)
    except Exception:
        return 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float) - float(np.mean(a))
    b = np.asarray(b, dtype=float) - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    return float(np.dot(a, b) / denom)


def _key_scores(chroma: np.ndarray) -> Dict[Tuple[str, str], float]:
    vec = np.asarray(chroma, dtype=float)
    if vec.ndim == 2:
        # Robust aggregation reduces one loud note/808 from taking over the key.
        vec = np.median(vec, axis=1) * 0.35 + np.mean(vec, axis=1) * 0.65
    vec = np.maximum(vec, 0.0)
    if float(np.sum(vec)) <= EPS:
        return {}
    vec = vec / (float(np.sum(vec)) + EPS)
    out: Dict[Tuple[str, str], float] = {}
    for root, note in enumerate(NOTES):
        out[(note, "Major")] = _corr(vec, np.roll(MAJOR, root))
        out[(note, "Minor")] = _corr(vec, np.roll(MINOR, root))
    return out


def detect_key_v2(y: np.ndarray, sr: int) -> Tuple[str, str, str, float]:
    try:
        audio = _mono(y)
        audio, work_sr = _resample(audio, int(sr), 22050)
        duration = len(audio) / max(work_sr, 1)
        if duration < 4.0:
            return "Uncertain", "Uncertain", "Uncertain", 0.0

        # Remove percussion/808 transients before tonal estimation.
        try:
            harmonic = librosa.effects.harmonic(audio, margin=4.0)
            if float(np.sqrt(np.mean(harmonic * harmonic))) < 1e-5:
                harmonic = audio
        except Exception:
            harmonic = audio

        centers = (0.12, 0.30, 0.48, 0.66, 0.84) if duration >= 35 else ((0.20, 0.50, 0.80) if duration >= 16 else (0.50,))
        win_sec = min(14.0, duration)
        win = max(int(5 * work_sr), int(win_sec * work_sr))
        totals: Dict[Tuple[str, str], float] = {}
        wins: Dict[Tuple[str, str], int] = {}
        usable = 0

        for ratio in centers:
            c = int(len(harmonic) * ratio)
            st = max(0, min(c - win // 2, len(harmonic) - win))
            piece = harmonic[st:st + win]
            if len(piece) < work_sr * 4:
                continue

            estimators = []
            try:
                estimators.append((librosa.feature.chroma_cqt(y=piece, sr=work_sr, hop_length=512, bins_per_octave=36), 1.00))
            except Exception:
                pass
            try:
                estimators.append((librosa.feature.chroma_cens(y=piece, sr=work_sr, hop_length=512), 0.85))
            except Exception:
                pass
            try:
                estimators.append((librosa.feature.chroma_stft(y=piece, sr=work_sr, n_fft=4096, hop_length=512), 0.65))
            except Exception:
                pass
            if not estimators:
                continue
            usable += 1

            window_scores: Dict[Tuple[str, str], float] = {}
            total_weight = 0.0
            for chroma, weight in estimators:
                scores = _key_scores(chroma)
                if not scores:
                    continue
                vals = np.asarray(list(scores.values()), dtype=float)
                lo, hi = float(np.min(vals)), float(np.max(vals))
                span = max(hi - lo, EPS)
                for key_id, value in scores.items():
                    window_scores[key_id] = window_scores.get(key_id, 0.0) + ((value - lo) / span) * weight
                total_weight += weight
            if total_weight <= 0 or not window_scores:
                continue
            for key_id in list(window_scores):
                window_scores[key_id] /= total_weight

            ranked_window = sorted(window_scores.items(), key=lambda kv: kv[1], reverse=True)
            winner = ranked_window[0][0]
            wins[winner] = wins.get(winner, 0) + 1
            # Top candidates get graded support; this preserves genuine modulation ambiguity.
            for rank, (key_id, score) in enumerate(ranked_window[:5]):
                rank_weight = (1.0, 0.55, 0.30, 0.16, 0.08)[rank]
                totals[key_id] = totals.get(key_id, 0.0) + float(score) * rank_weight

        if not totals or usable == 0:
            return "Uncertain", "Uncertain", "Uncertain", 0.0

        ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best = ranked[0]
        second_id, second = ranked[1] if len(ranked) > 1 else (("Uncertain", "Uncertain"), 0.0)
        third = ranked[2][1] if len(ranked) > 2 else 0.0
        vote_ratio = wins.get(best_id, 0) / max(usable, 1)
        gap = (best - second) / max(best, EPS)
        spread = (best - third) / max(best, EPS)

        # Relative major/minor pairs are naturally ambiguous; lower confidence when they tie.
        note_idx = NOTES.index(best_id[0])
        if best_id[1] == "Major":
            relative = (NOTES[(note_idx + 9) % 12], "Minor")
        else:
            relative = (NOTES[(note_idx + 3) % 12], "Major")
        relative_score = totals.get(relative, 0.0)
        relative_gap = (best - relative_score) / max(best, EPS)

        confidence = 24.0 + vote_ratio * 34.0 + gap * 28.0 + spread * 10.0
        if second_id == relative and gap < 0.10:
            confidence = min(confidence, 52.0)
        if relative_gap < 0.06:
            confidence = min(confidence, 48.0)
        if vote_ratio < 0.40:
            confidence = min(confidence, 46.0)
        confidence = float(np.clip(confidence, 18.0, 92.0))

        note, mode = best_id
        return f"{note} {mode}", note, mode, round(confidence, 1)
    except Exception:
        return "Uncertain", "Uncertain", "Uncertain", 0.0


def _rotate_chroma_block(vector: List[float], shift: int) -> List[float]:
    """Rotate the 12 chroma mean/std pairs inside the embedding vector."""
    out = list(vector)
    start = 40
    end = min(len(out), 64)
    if end - start < 24:
        return out
    pairs = [out[start + i * 2:start + i * 2 + 2] for i in range(12)]
    pairs = pairs[-shift:] + pairs[:-shift] if shift else pairs
    flat = [v for pair in pairs for v in pair]
    out[start:start + 24] = flat
    return out


def vector_weights_v2(length: int) -> List[float]:
    weights: List[float] = []
    # Timbre and its variability are the strongest identity/style cues.
    weights.extend([1.55] * min(40, max(0, length - len(weights))))
    # Chroma is useful after transposition alignment, but should not dominate artist identity.
    weights.extend([0.55] * min(24, max(0, length - len(weights))))
    # Spectral contrast strongly captures arrangement/timbre density.
    weights.extend([1.30] * min(14, max(0, length - len(weights))))
    # Remaining texture/rhythm/loudness-shape values.
    weights.extend([1.15] * max(0, length - len(weights)))
    return weights[:length]


def standardized_distance_v2(song_vector, proto_vector, means, stdevs, weights):
    """Minimum standardized distance after circular chroma alignment.

    This makes Artist Match key-invariant while preserving timbre, spectral,
    dynamics and rhythm differences.
    """
    dims = min(len(song_vector), len(proto_vector), len(means), len(stdevs), len(weights))
    if dims <= 0:
        return None

    def distance(a, b):
        weighted = 0.0
        total = 0.0
        for i in range(dims):
            sd = stdevs[i] if float(stdevs[i]) > 1e-9 else 1.0
            za = (float(a[i]) - float(means[i])) / sd
            zb = (float(b[i]) - float(means[i])) / sd
            w = max(0.0, float(weights[i]))
            weighted += ((za - zb) ** 2) * w
            total += w
        return math.sqrt(weighted / total) if total > 0 else float("inf")

    best = distance(song_vector, proto_vector)
    if dims >= 64:
        for shift in range(1, 12):
            shifted = _rotate_chroma_block(list(proto_vector), shift)
            best = min(best, distance(song_vector, shifted))
    return best if math.isfinite(best) else None


_original_compare_library = compare.compare_against_track_library


def compare_against_track_library_v2(report_dict, profile_files, top_n):
    ranked, profiles, nearest = _original_compare_library(report_dict, profile_files, max(top_n, 12))
    if not ranked:
        return ranked, profiles, nearest

    # Correct profile-size opportunity bias. Artists with 60 prototypes should not
    # beat artists with 10 simply because they had six times as many lottery tickets.
    for item in ranked:
        components = item.get("score_components") or {}
        top10 = float(components.get("top10_count") or 0.0)
        top20 = float(components.get("top20_count") or 0.0)
        track_count = max(1.0, float(item.get("track_count") or 1.0))
        support10 = top10 / max(1.0, min(track_count, 10.0))
        support20 = top20 / max(1.0, min(track_count, 20.0))
        support_density = min(1.0, 0.65 * support10 + 0.35 * support20)

        raw = float(item.get("match_score") or 0.0)
        # Keep nearest-track quality primary; use normalized support as confidence evidence.
        adjusted = raw * 0.88 + (support_density * 100.0) * 0.12
        item["match_score"] = round(max(0.0, min(96.0, adjusted)), 2)
        components["profile_support_density"] = round(support_density * 100.0, 2)
        item["score_components"] = components

    ranked.sort(key=lambda x: float(x.get("match_score") or 0.0), reverse=True)
    # Recompute confidence/labels after the corrected ranking.
    for i, item in enumerate(ranked):
        score = float(item.get("match_score") or 0.0)
        second = float(ranked[1]["match_score"] if i == 0 and len(ranked) > 1 else ranked[0]["match_score"])
        item["confidence"] = compare.confidence_from_gap(score, second, int(item.get("compared_fields") or 0), int(item.get("track_count") or 0))
        item["confidence_percent"] = compare.confidence_percent(score, second, int(item.get("compared_fields") or 0), int(item.get("track_count") or 0))
        item["match_label"] = compare.label_for_score(score, item.get("confidence"))
    return ranked[:top_n], profiles, nearest


def install() -> None:
    # Core analyze_audio resolves these globals at call time, so patching the module
    # updates every existing analysis route without changing its API.
    core.detect_bpm = detect_bpm_v2
    core.detect_key = detect_key_v2

    # Artist Match internals also resolve these functions dynamically.
    compare.vector_weights = vector_weights_v2
    compare.standardized_distance = standardized_distance_v2
    compare.compare_against_track_library = compare_against_track_library_v2

    print("[soundlens] accuracy v2 loaded: BPM + key ensemble + Artist Match bias correction")
