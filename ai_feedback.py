from __future__ import annotations

import json
import os
from typing import Any, Dict, List


AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _take_list(items, limit=5):
    if not isinstance(items, list):
        return []
    return [str(x)[:220] for x in items[:limit] if str(x).strip()]


def _top_artists(comparison: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    ranked = comparison.get("ranked_profiles") or comparison.get("matches") or []
    if not isinstance(ranked, list):
        return []

    output = []
    for item in ranked[:limit]:
        if not isinstance(item, dict):
            continue
        name = (
            item.get("artist")
            or item.get("profile_name")
            or item.get("name")
            or item.get("display_name")
            or item.get("profile")
            or "Unknown"
        )
        score = (
            item.get("score")
            or item.get("match_score")
            or item.get("final_score")
            or item.get("overall_score")
            or item.get("similarity")
        )
        output.append({"artist": name, "score": score})
    return output


def _compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    basic = report.get("basic", {}) or {}
    scores = report.get("scores", {}) or {}
    loudness = report.get("loudness", {}) or {}
    frequency = report.get("frequency", {}) or {}
    rhythm = report.get("rhythm", {}) or {}
    stem = report.get("stem_balance", {}) or {}
    comparison = report.get("artist_comparison", {}) or {}

    bands = frequency.get("band_percentages", {}) or {}
    sections = report.get("sections", []) or []

    return {
        "file": basic.get("file_name"),
        "basic": {
            "bpm": _safe_float(basic.get("bpm")),
            "key": basic.get("key"),
            "duration_seconds": _safe_float(basic.get("duration_seconds")),
            "sample_rate": basic.get("sample_rate"),
        },
        "scores": {
            "release": scores.get("release"),
            "mix": scores.get("mix"),
            "master": scores.get("master"),
            "arrangement": scores.get("arrangement"),
            "energy": _safe_float(scores.get("energy")),
            "bass_strength": _safe_float(scores.get("bass_strength")),
            "darkness": _safe_float(scores.get("darkness")),
            "brightness": _safe_float(scores.get("brightness")),
            "drum_bounce": _safe_float(scores.get("drum_bounce")),
            "vocal_space": _safe_float(scores.get("vocal_space")),
        },
        "loudness": {
            "peak_db": _safe_float(loudness.get("peak_db")),
            "rms_db": _safe_float(loudness.get("rms_db")),
            "dynamic_range_db": _safe_float(loudness.get("dynamic_range_db")),
            "clipping_detected": loudness.get("clipping_detected"),
            "clipping_samples": loudness.get("clipping_samples"),
            "headroom_db": _safe_float(loudness.get("headroom_db")),
        },
        "frequency": {
            "dominant_band": frequency.get("dominant_band"),
            "brightness_label": frequency.get("brightness_label"),
            "low_end_total_percent": _safe_float(frequency.get("low_end_total_percent")),
            "mid_total_percent": _safe_float(frequency.get("mid_total_percent")),
            "top_total_percent": _safe_float(frequency.get("top_total_percent")),
            "band_percentages": {
                key: round(_safe_float(value, 0) or 0, 2)
                for key, value in bands.items()
                if key in {"Sub", "Bass / 808", "Mud", "Low Mids", "Mids / Melody", "Harsh Zone", "Highs", "Air", "Vocal Range"}
            },
        },
        "rhythm": {
            "onset_density": _safe_float(rhythm.get("onset_density")),
            "drum_activity": rhythm.get("drum_activity"),
            "estimated_bars": rhythm.get("estimated_bars"),
        },
        "stem_balance": {
            "enabled": stem.get("enabled"),
            "status": stem.get("status"),
            "confidence": stem.get("confidence"),
            "vocal_to_beat_db": _safe_float(stem.get("vocal_to_beat_db")),
            "bass_to_vocal_db": _safe_float(stem.get("bass_to_vocal_db")),
            "bass_to_other_db": _safe_float(stem.get("bass_to_other_db")),
            "beat_vocal_balance_score": stem.get("beat_vocal_balance_score"),
            "melody_presence_score": stem.get("melody_presence_score"),
            "warnings": _take_list(stem.get("warnings"), 4),
        },
        "artist_comparison": {
            "top_artists": _top_artists(comparison, 5),
            "verdict": comparison.get("verdict"),
        },
        "sections": [
            {
                "name": section.get("name"),
                "start": _safe_float(section.get("start")),
                "end": _safe_float(section.get("end")),
                "energy_label": section.get("energy_label"),
                "avg_energy": _safe_float(section.get("avg_energy")),
            }
            for section in sections[:8]
            if isinstance(section, dict)
        ],
        "standard_feedback": {
            "top_problems": _take_list(report.get("top_problems"), 5),
            "next_steps": _take_list(report.get("next_steps"), 5),
            "artist_notes": _take_list(report.get("artist_notes"), 4),
            "producer_notes": _take_list(report.get("producer_notes"), 4),
            "master_notes": _take_list(report.get("master_notes"), 4),
        },
    }


def _fallback_feedback(report: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    scores = report.get("scores", {}) or {}
    basic = report.get("basic", {}) or {}
    comparison = report.get("artist_comparison", {}) or {}
    top_artist = None
    artists = _top_artists(comparison, 1)
    if artists:
        top_artist = artists[0].get("artist")

    top_problems = _take_list(report.get("top_problems"), 3)
    next_steps = _take_list(report.get("next_steps"), 3)

    release = scores.get("release", 0)
    mix = scores.get("mix", 0)
    master = scores.get("master", 0)

    return {
        "ai_enabled": False,
        "ai_error": reason,
        "top_problems": top_problems,
        "next_steps": next_steps,
        "ai_review": {
            "first_impression": f"SoundLens read this as a {release}/100 release with a {mix}/100 mix and {master}/100 master. The main value is in fixing the first technical issue before judging the song emotionally.",
            "biggest_strengths": [
                f"The track has a clear measurable profile around {round(_safe_float(basic.get('bpm'), 0) or 0)} BPM.",
                f"The closest artist lane is {top_artist}." if top_artist else "The artist profile comparison returned a usable direction.",
            ],
            "biggest_weaknesses": top_problems or ["No major technical red flag was isolated by the standard SoundLens engine."],
            "mix_advice": next_steps[0] if next_steps else "Use a reference track and make small balance changes rather than changing everything at once.",
            "artist_direction": f"The closest profile direction is {top_artist}, but use that as a lane check rather than a final identity." if top_artist else "Artist direction will be stronger when the comparison engine has a clear top match.",
            "priority_fix": next_steps[0] if next_steps else "Pick one reference track and match the loudness and low-end balance first.",
            "release_verdict": "Close enough to keep developing, but make the priority fix before treating it as release-ready.",
        },
    }


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty AI response.")

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def _clean_ai_result(data: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    review = data.get("ai_review") if isinstance(data.get("ai_review"), dict) else data

    cleaned_review = {
        "first_impression": str(review.get("first_impression", "")).strip()[:700],
        "biggest_strengths": _take_list(review.get("biggest_strengths"), 3),
        "biggest_weaknesses": _take_list(review.get("biggest_weaknesses"), 3),
        "mix_advice": str(review.get("mix_advice", "")).strip()[:700],
        "artist_direction": str(review.get("artist_direction", "")).strip()[:700],
        "priority_fix": str(review.get("priority_fix", "")).strip()[:500],
        "release_verdict": str(review.get("release_verdict", "")).strip()[:700],
    }

    fallback = _fallback_feedback(report)
    for key, value in cleaned_review.items():
        if not value:
            cleaned_review[key] = fallback["ai_review"].get(key)

    top_problems = _take_list(data.get("top_problems"), 5) or _take_list(review.get("top_problems"), 5) or _take_list(report.get("top_problems"), 5)
    next_steps = _take_list(data.get("next_steps"), 5) or _take_list(review.get("next_steps"), 5) or _take_list(report.get("next_steps"), 5)

    return {
        "ai_enabled": True,
        "model": AI_MODEL,
        "top_problems": top_problems,
        "next_steps": next_steps,
        "ai_review": cleaned_review,
    }


def generate_soundlens_ai_feedback(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates varied, report-grounded written feedback.
    The AI only sees the SoundLens JSON metrics, never the audio file.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _fallback_feedback(report, "OPENAI_API_KEY is not configured.")

    compact = _compact_report(report)

    system_prompt = """You are SoundLens AI, a concise producer/mix-engineer assistant for underground artists and producers.
You do not listen to audio. You only interpret the provided SoundLens report JSON.
Be specific, grounded, and useful.
Never invent facts, lyrics, instruments, artist names, or emotions that are not supported by the report.
Do not repeat the same wording across sections.
Do not over-compliment. If the score is weak, be direct but constructive.
Avoid generic lines like "keep working" or "great track."
Focus on what should be fixed first and why it matters.
Return JSON only."""

    user_prompt = f"""Create a SoundLens AI review from this report.

Rules:
- Mention actual metrics only when they support the advice.
- Choose the most important points; do not summarize every number.
- If artist comparison exists, use it as a style lane, not as a guarantee.
- Keep each field short and readable.
- top_problems and next_steps should be improved/rewritten versions of the report's current problems/fixes, not random new issues.
- Return exactly this JSON shape:

{{
  "top_problems": ["", "", ""],
  "next_steps": ["", "", ""],
  "ai_review": {{
    "first_impression": "",
    "biggest_strengths": ["", ""],
    "biggest_weaknesses": ["", ""],
    "mix_advice": "",
    "artist_direction": "",
    "priority_fix": "",
    "release_verdict": ""
  }}
}}

REPORT JSON:
{json.dumps(compact, ensure_ascii=False)}
"""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=AI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=900,
        )

        text = getattr(response, "output_text", "") or ""
        data = _extract_json(text)
        return _clean_ai_result(data, report)

    except Exception as error:
        return _fallback_feedback(report, str(error))
