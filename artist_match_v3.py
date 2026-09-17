from __future__ import annotations

"""SoundLens Artist Match v3.

Adds three things without replacing the existing public API:
1. Internal sub-style clusters per artist so one averaged artist identity does not blur eras/lanes.
2. Lightweight segment embeddings so several parts of the upload vote independently.
3. Cluster/segment diagnostics that can be used by the UI/AI and offline confusion tests.

The existing whole-song matcher remains the primary signal. These additions are bounded
and deliberately modest so they improve separation without making Railway analysis slow.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

import soundlens_pro as core
import compare_to_profile_pro as compare

EPS = 1e-9

_original_fingerprint = core.analyze_audio_fingerprint
_original_compare_library = compare.compare_against_track_library


def _segment_vector(piece: np.ndarray, sr: int) -> List[float]:
    """Build the same 91-ish feature ordering used by audio_embedding_vector_from_report."""
    y = np.asarray(piece, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.size < sr * 3:
        return []
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y))) + EPS
    y = (y / peak) * 0.95
    try:
        timbre = librosa.effects.preemphasis(y)
    except Exception:
        timbre = y

    mfcc = librosa.feature.mfcc(y=timbre, sr=sr, n_mfcc=20)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    onset = librosa.onset.onset_strength(y=y, sr=sr)

    out: List[float] = []
    for row in mfcc[:20]:
        out.extend([float(np.mean(row)), float(np.std(row))])
    for row in chroma[:12]:
        out.extend([float(np.mean(row)), float(np.std(row))])
    for row in contrast[:7]:
        out.extend([float(np.mean(row)), float(np.std(row))])

    ce = np.mean(chroma, axis=1)
    ce = ce / (float(np.sum(ce)) + EPS)
    chroma_entropy = -float(np.sum(ce * np.log(ce + EPS)))
    for value in (
        chroma_entropy,
        float(np.mean(centroid)), float(np.std(centroid)),
        float(np.mean(rolloff)), float(np.std(rolloff)),
        float(np.mean(flatness)), float(np.std(flatness)),
        float(np.mean(zcr)), float(np.std(zcr)),
        float(np.mean(rms)), float(np.std(rms)),
        float(np.mean(onset)), float(np.std(onset)),
    ):
        out.append(value if math.isfinite(value) else 0.0)
    return out


def analyze_audio_fingerprint_v3(y: np.ndarray, sr: int):
    fp = _original_fingerprint(y, sr)
    if not isinstance(fp, dict):
        return fp
    try:
        audio = np.asarray(y, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0 if audio.shape[0] <= 8 else -1)
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
            work_sr = target_sr
        else:
            work_sr = int(sr)

        duration = len(audio) / max(work_sr, 1)
        if duration < 12:
            return fp

        # Five windows across the song. 12s is long enough for stable timbre/rhythm
        # while remaining cheap enough for production analysis.
        window_sec = min(12.0, max(7.0, duration / 8.0))
        window = int(window_sec * work_sr)
        centers = (0.12, 0.31, 0.50, 0.69, 0.88) if duration >= 45 else (0.18, 0.40, 0.62, 0.84)
        segments = []
        for ratio in centers:
            center = int(len(audio) * ratio)
            start = max(0, min(center - window // 2, len(audio) - window))
            piece = audio[start:start + window]
            vec = _segment_vector(piece, work_sr)
            if vec:
                segments.append({
                    "start": round(start / work_sr, 2),
                    "end": round((start + len(piece)) / work_sr, 2),
                    "vector": vec,
                })
        if segments:
            fp["segment_embeddings"] = segments
            fp["segment_embedding_count"] = len(segments)
            fp["fingerprint_version"] = max(float(fp.get("fingerprint_version") or 0), 3.0)
    except Exception:
        pass
    return fp


def _euclidean(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return float("inf")
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)) / n)


def _kmeans(vectors: List[List[float]], k: int, iterations: int = 10) -> List[List[float]]:
    if not vectors:
        return []
    dims = min(len(v) for v in vectors if v)
    clean = [list(map(float, v[:dims])) for v in vectors if len(v) >= dims]
    if not clean:
        return []
    k = max(1, min(k, len(clean)))
    # Deterministic far-apart initialization.
    centroids = [clean[0]]
    while len(centroids) < k:
        candidate = max(clean, key=lambda v: min(_euclidean(v, c) for c in centroids))
        centroids.append(candidate[:])
    for _ in range(iterations):
        groups = [[] for _ in range(k)]
        for v in clean:
            idx = min(range(k), key=lambda i: _euclidean(v, centroids[i]))
            groups[idx].append(v)
        changed = False
        for i, group in enumerate(groups):
            if not group:
                continue
            new = [sum(v[d] for v in group) / len(group) for d in range(dims)]
            if _euclidean(new, centroids[i]) > 1e-6:
                changed = True
            centroids[i] = new
        if not changed:
            break
    return centroids


def _build_clusters(library: List[Dict[str, Any]]) -> Dict[str, List[List[float]]]:
    by_artist: Dict[str, List[List[float]]] = defaultdict(list)
    for item in library:
        vec = item.get("vector") or []
        if isinstance(vec, list) and vec:
            by_artist[str(item.get("artist") or "Unknown")].append(vec)
    clusters: Dict[str, List[List[float]]] = {}
    for artist, vectors in by_artist.items():
        n = len(vectors)
        k = 1 if n < 12 else (2 if n < 28 else (3 if n < 55 else 4))
        clusters[artist] = _kmeans(vectors, k)
    return clusters


def _distance_similarity(song_vector, target_vector, means, stdevs, weights) -> float:
    d = compare.standardized_distance(song_vector, target_vector, means, stdevs, weights)
    if d is None:
        return 0.0
    return float(compare.distance_to_similarity(d))


def _segment_votes(report_dict, library, means, stdevs, weights) -> Dict[str, Any]:
    segments = ((report_dict.get("fingerprint") or {}).get("segment_embeddings") or [])
    if not isinstance(segments, list) or not segments:
        return {"count": 0, "votes": {}, "winner": None, "consensus": 0.0}

    votes: Dict[str, float] = defaultdict(float)
    winners = []
    for seg in segments[:5]:
        vec = seg.get("vector") if isinstance(seg, dict) else None
        if not isinstance(vec, list) or not vec:
            continue
        nearest = []
        for item in library:
            proto = item.get("vector") or []
            if not proto:
                continue
            score = _distance_similarity(vec, proto, means, stdevs, weights)
            nearest.append((score, str(item.get("artist") or "Unknown")))
        nearest.sort(reverse=True)
        # Top 8 neighbors vote with rank decay so one exact-ish prototype does not dominate.
        local: Dict[str, float] = defaultdict(float)
        for rank, (score, artist) in enumerate(nearest[:8], start=1):
            local[artist] += (score / 100.0) / (rank ** 0.75)
        if local:
            winner = max(local, key=local.get)
            winners.append(winner)
            total = sum(local.values()) or 1.0
            for artist, value in local.items():
                votes[artist] += value / total

    count = len(winners)
    if not count:
        return {"count": 0, "votes": {}, "winner": None, "consensus": 0.0}
    normalized = {a: v / count for a, v in votes.items()}
    winner = max(normalized, key=normalized.get)
    consensus = winners.count(winner) / count
    return {
        "count": count,
        "votes": normalized,
        "winner": winner,
        "consensus": consensus,
        "segment_winners": winners,
    }


def compare_against_track_library_v3(report_dict, profile_files, top_n):
    # Start with the already-hardened v2 + contrastive + TuneBat matcher.
    ranked, profiles, nearest = _original_compare_library(report_dict, profile_files, max(top_n, 12))
    if not ranked:
        return ranked, profiles, nearest

    library, _ = compare.load_track_library(profile_files)
    song_vector = compare.audio_embedding_vector_from_report(report_dict)
    if not library or not song_vector:
        return ranked[:top_n], profiles, nearest

    dims = min(len(song_vector), min(len(item.get("vector") or []) for item in library if item.get("vector")))
    if dims <= 0:
        return ranked[:top_n], profiles, nearest
    song_vector = song_vector[:dims]
    means, stdevs = compare.library_stats(library, dims)
    weights = compare.vector_weights(dims)

    clusters = _build_clusters(library)
    cluster_scores: Dict[str, float] = {}
    cluster_ids: Dict[str, int] = {}
    for artist, centroids in clusters.items():
        if not centroids:
            continue
        scores = [_distance_similarity(song_vector, c[:dims], means, stdevs, weights) for c in centroids]
        best_idx = int(np.argmax(scores))
        cluster_scores[artist] = float(scores[best_idx])
        cluster_ids[artist] = best_idx + 1

    seg = _segment_votes(report_dict, library, means, stdevs, weights)
    seg_votes = seg.get("votes") or {}

    for item in ranked:
        artist = str(item.get("profile_name") or "Unknown")
        base = float(item.get("match_score") or 0.0)
        cluster = cluster_scores.get(artist)
        segment = float(seg_votes.get(artist, 0.0)) * 100.0 if seg.get("count") else None

        parts = [(base, 0.72)]
        if cluster is not None:
            parts.append((cluster, 0.18))
        if segment is not None:
            parts.append((segment, 0.10))
        adjusted = sum(v * w for v, w in parts) / sum(w for _, w in parts)

        components = item.get("score_components") or {}
        components["substyle_cluster"] = round(cluster, 2) if cluster is not None else None
        components["segment_consensus"] = round(segment, 2) if segment is not None else None
        item["score_components"] = components
        item["substyle_cluster"] = cluster_ids.get(artist)
        item["segment_vote"] = round(segment, 2) if segment is not None else None
        item["match_score"] = round(max(0.0, min(96.0, adjusted)), 2)

    ranked.sort(key=lambda x: float(x.get("match_score") or 0.0), reverse=True)
    for i, item in enumerate(ranked):
        score = float(item.get("match_score") or 0.0)
        competitor = float(ranked[1]["match_score"] if i == 0 and len(ranked) > 1 else ranked[0]["match_score"])
        item["confidence"] = compare.confidence_from_gap(score, competitor, int(item.get("compared_fields") or 0), int(item.get("track_count") or 0))
        item["confidence_percent"] = compare.confidence_percent(score, competitor, int(item.get("compared_fields") or 0), int(item.get("track_count") or 0))
        item["match_label"] = compare.label_for_score(score, item.get("confidence"))

    if ranked:
        ranked[0]["segment_analysis"] = {
            "segments": int(seg.get("count") or 0),
            "winner": seg.get("winner"),
            "consensus": round(float(seg.get("consensus") or 0.0) * 100.0, 1),
            "segment_winners": seg.get("segment_winners") or [],
        }
    return ranked[:top_n], profiles, nearest


def install() -> None:
    core.analyze_audio_fingerprint = analyze_audio_fingerprint_v3
    compare.compare_against_track_library = compare_against_track_library_v3
    print("[soundlens] Artist Match v3 loaded: substyle clusters + segment voting")
