from __future__ import annotations

from typing import Any

import soundlens_3d as event_model
import app as legacy
import hardened_app as hardened
import visual_app as visual

app = visual.app


# ---------------------------------------------------------------------------
# Event-list quality control v3.2
# ---------------------------------------------------------------------------
# Keep the visual timeline chronological, but maintain a separate ranked list
# of standout musical events. This prevents "first in the song" from being
# confused with "most important in the song" and removes near-duplicate events
# that carry essentially the same evidence.

_directional_fuse = event_model._fuse_candidates


def _event_signature(event: dict[str, Any]) -> tuple[str, ...]:
    kinds = tuple(sorted(str(k) for k in (event.get("evidence_types") or [])))
    direction = str(event.get("direction") or "")
    return kinds + ((direction,) if direction else ())


def _standout_score(event: dict[str, Any]) -> float:
    importance = float(event.get("importance") or 0.0)
    confidence = float(event.get("confidence") or 0.0)
    evidence_count = min(4, int(event.get("evidence_count") or 0))
    kinds = set(event.get("evidence_types") or [])

    score = (
        importance * 0.56
        + confidence * 0.28
        + (evidence_count / 4.0) * 0.10
        + (0.04 if "section" in kinds else 0.0)
        + (0.02 if {"bass", "energy"}.issubset(kinds) else 0.0)
    )
    return max(0.0, min(1.0, score))


def _quality_fuse_candidates(candidates, times, sections, duration):
    events = _directional_fuse(candidates, times, sections, duration)
    if not events:
        return events

    # Compute an explicit standout score without changing the existing detector's
    # confidence/importance values. Those remain useful on their own.
    for event in events:
        event["standout_score"] = round(_standout_score(event), 4)

    # Remove musically redundant events only when they are close in time AND have
    # effectively the same evidence signature. Different event families are kept.
    ranked = sorted(
        events,
        key=lambda e: (
            float(e.get("standout_score") or 0.0),
            float(e.get("importance") or 0.0),
            float(e.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    chosen: list[dict[str, Any]] = []
    redundancy_window = max(2.0, min(5.0, float(duration or 0.0) / 45.0))

    for event in ranked:
        sig = _event_signature(event)
        duplicate = False
        for old in chosen:
            if abs(float(event.get("time") or 0.0) - float(old.get("time") or 0.0)) > redundancy_window:
                continue
            old_sig = _event_signature(old)
            if sig == old_sig:
                duplicate = True
                break
            # Treat one evidence set that is almost entirely contained in another
            # as redundant only when their directions also agree.
            kinds = set(event.get("evidence_types") or [])
            old_kinds = set(old.get("evidence_types") or [])
            direction = str(event.get("direction") or "")
            old_direction = str(old.get("direction") or "")
            overlap = len(kinds & old_kinds) / max(1, min(len(kinds), len(old_kinds)))
            if overlap >= 0.85 and direction and direction == old_direction:
                duplicate = True
                break
        if not duplicate:
            chosen.append(event)

    # Do not force weak events just to reach a target count. A sparse song should
    # remain sparse. Cap only the strongest events, then return them in time order.
    chosen = chosen[: event_model.MAX_PINS]
    standout_order = sorted(
        chosen,
        key=lambda e: float(e.get("standout_score") or 0.0),
        reverse=True,
    )
    for rank, event in enumerate(standout_order, start=1):
        event["standout_rank"] = rank

    return sorted(chosen, key=lambda e: float(e.get("time") or 0.0))


event_model._fuse_candidates = _quality_fuse_candidates


# app.py imported build_visual_map directly during module import. Replace that
# reference too so /analyze uses the v3.2 event-quality model in production.
_original_build_visual_map = legacy.build_visual_map


def _quality_build_visual_map(audio_path, max_slices: int = 320):
    result = event_model.build_visual_map(audio_path, max_slices=max_slices)
    events = list(result.get("pins") or [])
    standouts = sorted(
        events,
        key=lambda e: (
            float(e.get("standout_score") or 0.0),
            float(e.get("importance") or 0.0),
        ),
        reverse=True,
    )
    result["version"] = "3.2"
    result["event_model"] = "section_aware_fusion_v3_2"
    result["standouts"] = standouts[:5]
    result["top_event"] = standouts[0] if standouts else None
    timeline = result.get("ai_timeline_summary")
    if isinstance(timeline, dict):
        timeline["source"] = "musical_event_model_v3_2"
        timeline["standouts"] = result["standouts"]
        timeline["guidance"] = (
            "Use the chronological events as the authoritative timeline and the "
            "standouts list when the producer asks for the biggest or most important moments."
        )
    return result


legacy.build_visual_map = _quality_build_visual_map
hardened.legacy.build_visual_map = _quality_build_visual_map
