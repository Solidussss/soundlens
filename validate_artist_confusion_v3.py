from __future__ import annotations

"""Offline Artist Match v3 confusion validator.

Usage:
    python validate_artist_confusion_v3.py

Treats saved track prototypes as labeled examples. Each sampled prototype is
removed from the candidate pool before classification, so a track cannot match
itself. Produces overall/top-k accuracy and the artist pairs SoundLens confuses
most often. This is intentionally offline and never runs on customer requests.
"""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import compare_to_profile_pro as compare
import accuracy_v2
import contrastive_artist_profiles
import tunebat_catalog_prior

# Install the same scoring foundations production uses. Segment voting is omitted
# here because historical profile prototypes do not contain segment embeddings.
accuracy_v2.install()
contrastive_artist_profiles.install()
tunebat_catalog_prior.install()

PROFILE_DIR = Path(__file__).parent / "artist_profiles"
RANDOM_SEED = 1337
MAX_PER_ARTIST = 8
TOP_K = 3


def load_library():
    files = sorted(PROFILE_DIR.glob("*_profile.json"))
    return compare.load_track_library(files)


def classify(query: Dict[str, Any], pool: List[Dict[str, Any]], means, stdevs, weights, dims: int):
    q = (query.get("vector") or [])[:dims]
    nearest = []
    for candidate in pool:
        vec = (candidate.get("vector") or [])[:dims]
        d = compare.standardized_distance(q, vec, means, stdevs, weights)
        if d is None:
            continue
        score = compare.distance_to_similarity(d)
        nearest.append((float(score), str(candidate.get("artist") or "Unknown")))
    nearest.sort(reverse=True)

    votes = defaultdict(float)
    for rank, (score, artist) in enumerate(nearest[:24], start=1):
        votes[artist] += (score / 100.0) / (rank ** 0.82)
    ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return [artist for artist, _ in ranked[:TOP_K]]


def main():
    random.seed(RANDOM_SEED)
    library, _profiles = load_library()
    if not library:
        raise SystemExit("No artist prototypes found")

    valid = [item for item in library if isinstance(item.get("vector"), list) and item.get("vector")]
    dims = min(len(item["vector"]) for item in valid)
    means, stdevs = compare.library_stats(valid, dims)
    weights = compare.vector_weights(dims)

    by_artist = defaultdict(list)
    for item in valid:
        by_artist[str(item.get("artist") or "Unknown")].append(item)

    tests = []
    for artist, items in sorted(by_artist.items()):
        sample = list(items)
        random.shuffle(sample)
        tests.extend(sample[: min(MAX_PER_ARTIST, len(sample))])

    total = 0
    top1 = 0
    topk = 0
    confusions = Counter()
    per_artist = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})

    for query in tests:
        # Remove this exact prototype by identity/title/vector equality.
        pool = [item for item in valid if item is not query]
        ranked = classify(query, pool, means, stdevs, weights, dims)
        if not ranked:
            continue
        truth = str(query.get("artist") or "Unknown")
        pred = ranked[0]
        total += 1
        per_artist[truth]["n"] += 1
        if pred == truth:
            top1 += 1
            per_artist[truth]["top1"] += 1
        else:
            confusions[(truth, pred)] += 1
        if truth in ranked:
            topk += 1
            per_artist[truth]["top3"] += 1

    result = {
        "samples": total,
        "artists": len(by_artist),
        "top1_accuracy": round(top1 / max(total, 1) * 100.0, 2),
        "top3_accuracy": round(topk / max(total, 1) * 100.0, 2),
        "worst_confusions": [
            {"true_artist": pair[0], "predicted_artist": pair[1], "count": count}
            for pair, count in confusions.most_common(30)
        ],
        "per_artist": {
            artist: {
                "samples": values["n"],
                "top1_accuracy": round(values["top1"] / max(values["n"], 1) * 100.0, 2),
                "top3_accuracy": round(values["top3"] / max(values["n"], 1) * 100.0, 2),
            }
            for artist, values in sorted(per_artist.items())
        },
    }

    out = Path(__file__).parent / "artist_confusion_v3.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "samples": result["samples"],
        "artists": result["artists"],
        "top1_accuracy": result["top1_accuracy"],
        "top3_accuracy": result["top3_accuracy"],
        "worst_confusions": result["worst_confusions"][:10],
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
