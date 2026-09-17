from __future__ import annotations

"""Contrastive artist-profile weighting for SoundLens.

The old artist matcher mostly asked whether a song was close to an artist's
tracks. That can overvalue scene-common traits: if every artist in a library has
large 808s, that feature should contribute very little to identity.

This module estimates, per embedding dimension, how much artists differ from one
another relative to how much songs vary inside the same artist. Dimensions with
high between-artist / within-artist separation receive more identity weight;
scene-common or unstable dimensions receive less.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import compare_to_profile_pro as compare

EPS = 1e-9

_original_library_stats = compare.library_stats
_original_vector_weights = compare.vector_weights

# Exposed for diagnostics/tests. Values are recalculated for the active library.
LAST_CONTRASTIVE_WEIGHTS: List[float] = []
LAST_CONTRASTIVE_DIAGNOSTICS: Dict[str, Any] = {}


def _finite(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _variance(values: List[float], center: float | None = None) -> float:
    if not values:
        return 0.0
    c = _mean(values) if center is None else float(center)
    return sum((v - c) ** 2 for v in values) / max(1, len(values))


def _robust_dimension_stats(library: List[Dict[str, Any]], index: int) -> Tuple[float, float, float, int]:
    """Return global mean/std, discriminative weight, and artist coverage."""
    by_artist: Dict[str, List[float]] = defaultdict(list)
    all_values: List[float] = []

    for item in library:
        vector = item.get("vector") or []
        if index >= len(vector):
            continue
        value = _finite(vector[index])
        if value is None:
            continue
        artist = str(item.get("artist") or "Unknown")
        by_artist[artist].append(value)
        all_values.append(value)

    if not all_values:
        return 0.0, 1.0, 0.55, 0

    global_mean = _mean(all_values)
    global_var = _variance(all_values, global_mean)
    global_std = math.sqrt(max(global_var, EPS))

    artist_means: List[float] = []
    within_vars: List[float] = []
    reliable_artists = 0

    for values in by_artist.values():
        if not values:
            continue
        m = _mean(values)
        artist_means.append(m)
        if len(values) >= 2:
            within_vars.append(_variance(values, m))
            reliable_artists += 1

    if len(artist_means) < 2:
        return global_mean, global_std, 0.65, len(artist_means)

    between_var = _variance(artist_means)
    # Use the median-ish protection of a floor tied to global variance so tiny
    # within variance cannot explode a dimension's importance.
    within_var = _mean(within_vars) if within_vars else global_var
    within_floor = max(global_var * 0.08, EPS)
    separation = between_var / max(within_var, within_floor)

    # Convert Fisher-like separation into a bounded identity multiplier.
    # 1.0 ~= neutral; <1 scene-common/unstable; >1 artist-distinctive.
    weight = 0.50 + 0.72 * math.log1p(max(0.0, separation))
    coverage = len(artist_means)
    if coverage < 4:
        weight *= 0.85
    if reliable_artists < max(2, coverage // 3):
        weight *= 0.90
    weight = max(0.42, min(2.15, weight))

    return global_mean, global_std, weight, coverage


def contrastive_library_stats(library: List[Dict[str, Any]], dims: int):
    """Library normalization that suppresses scene-common dimensions.

    compare.standardized_distance divides by returned stdevs. We encode the
    contrastive identity multiplier by increasing the effective stdev for weak
    dimensions and decreasing it for dimensions that reliably separate artists.
    """
    global LAST_CONTRASTIVE_WEIGHTS, LAST_CONTRASTIVE_DIAGNOSTICS

    if not library or dims <= 0:
        LAST_CONTRASTIVE_WEIGHTS = []
        LAST_CONTRASTIVE_DIAGNOSTICS = {"dims": 0, "artists": 0}
        return [], []

    means: List[float] = []
    effective_stdevs: List[float] = []
    identity_weights: List[float] = []
    coverages: List[int] = []

    for i in range(dims):
        mean, std, identity_weight, coverage = _robust_dimension_stats(library, i)
        means.append(mean)
        identity_weights.append(identity_weight)
        coverages.append(coverage)
        # distance contribution is proportional to 1/std^2, so sqrt(weight)
        # yields roughly the requested multiplier without destabilizing scale.
        effective = max(std / math.sqrt(max(identity_weight, 0.05)), 1e-6)
        effective_stdevs.append(effective)

    LAST_CONTRASTIVE_WEIGHTS = identity_weights
    artists = len({str(item.get("artist") or "Unknown") for item in library})
    LAST_CONTRASTIVE_DIAGNOSTICS = {
        "dims": dims,
        "artists": artists,
        "mean_identity_weight": round(_mean(identity_weights), 4) if identity_weights else 0.0,
        "strong_dimensions": sum(1 for w in identity_weights if w >= 1.25),
        "suppressed_dimensions": sum(1 for w in identity_weights if w <= 0.70),
        "min_artist_coverage": min(coverages) if coverages else 0,
    }
    return means, effective_stdevs


def contrastive_vector_weights(length: int) -> List[float]:
    """Base semantic weights; contrastive weights are encoded by library_stats."""
    base = list(_original_vector_weights(length))
    if not base:
        return base

    # Keep key/chroma intentionally modest. Artist identity should be mostly
    # timbre, texture, rhythm and dynamics, not whether two songs share a key.
    for i in range(40, min(64, len(base))):
        base[i] = min(base[i], 0.50)

    # The final scalar block includes loudness/energy-adjacent descriptors.
    # They remain useful, but shared mastering/bass trends should not dominate.
    for i in range(78, len(base)):
        base[i] = min(base[i], 1.00)

    return base


def install() -> None:
    compare.library_stats = contrastive_library_stats
    compare.vector_weights = contrastive_vector_weights
    print("[soundlens] contrastive artist profiles loaded: scene-common traits suppressed")
