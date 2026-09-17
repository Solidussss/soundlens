"""SoundLens process-wide production hooks.

Python imports ``sitecustomize`` automatically during interpreter startup when
this repository is on sys.path (the normal Railway/Uvicorn layout). Keep this
file dependency-light: it patches the 3D event model before ``app.py`` imports
``build_visual_map`` so every analysis path receives the same quality control.
"""
from __future__ import annotations

from typing import Any

import soundlens_3d as event_model


_base_fuse_candidates = event_model._fuse_candidates
_base_build_visual_map = event_model.build_visual_map


def _signature(event: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(k) for k in (event.get("evidence_types") or [])))


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
    events = _base_fuse_candidates(candidates, times, sections, duration)
    if not events:
        return events

    for event in events:
        event["standout_score"] = round(_standout_score(event), 4)

    ranked = sorted(
        events,
        key=lambda e: (
            float(e.get("standout_score") or 0.0),
            float(e.get("importance") or 0.0),
            float(e.get("confidence") or 0.0),
        ),
        reverse=True,
    )

    # Deduplicate only events that are both close in time and driven by nearly
    # the same measured evidence. Different musical phenomena remain separate.
    chosen: list[dict[str, Any]] = []
    redundancy_window = max(2.0, min(5.0, float(duration or 0.0) / 45.0))
    for event in ranked:
        kinds = set(event.get("evidence_types") or [])
        sig = _signature(event)
        duplicate = False
        for old in chosen:
            if abs(float(event.get("time") or 0.0) - float(old.get("time") or 0.0)) > redundancy_window:
                continue
            old_kinds = set(old.get("evidence_types") or [])
            old_sig = _signature(old)
            if sig == old_sig:
                duplicate = True
                break
            overlap = len(kinds & old_kinds) / max(1, min(len(kinds), len(old_kinds)))
            if overlap >= 0.90:
                duplicate = True
                break
        if not duplicate:
            chosen.append(event)
        if len(chosen) >= event_model.MAX_PINS:
            break

    standout_order = sorted(
        chosen,
        key=lambda e: float(e.get("standout_score") or 0.0),
        reverse=True,
    )
    for rank, event in enumerate(standout_order, start=1):
        event["standout_rank"] = rank

    # Pins stay chronological for the 3D timeline.
    return sorted(chosen, key=lambda e: float(e.get("time") or 0.0))


def _quality_build_visual_map(audio_path, max_slices: int = 320):
    result = _base_build_visual_map(audio_path, max_slices=max_slices)
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
            "Use chronological events for where things happen and ranked standouts "
            "for questions about the biggest or most important moments."
        )
    return result


event_model._fuse_candidates = _quality_fuse_candidates
event_model.build_visual_map = _quality_build_visual_map
print("[soundlens] event quality model v3.2 startup hook loaded")
