
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
OPENAI_AUDIO_MODEL = os.getenv("OPENAI_AUDIO_MODEL", "").strip()
MAX_AUDIO_UPLOAD_MB = float(os.getenv("SOUNDLENS_AI_AUDIO_MAX_MB", "8"))


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _take_list(items, limit=5):
    if not isinstance(items, list):
        return []
    return [str(x)[:320] for x in items[:limit] if str(x).strip()]


def _top_artists(comparison: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    ranked = comparison.get("ranked_profiles") or comparison.get("matches") or []
    out = []
    if not isinstance(ranked, list):
        return out
    for item in ranked[:limit]:
        if not isinstance(item, dict):
            continue
        out.append({
            "artist": item.get("artist") or item.get("profile_name") or item.get("name") or item.get("profile") or "Unknown",
            "score": item.get("score") or item.get("match_score") or item.get("final_score") or item.get("overall_score") or item.get("similarity"),
        })
    return out


def _compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    basic = report.get("basic", {}) or {}
    scores = report.get("scores", {}) or {}
    loud = report.get("loudness", {}) or {}
    freq = report.get("frequency", {}) or {}
    rhythm = report.get("rhythm", {}) or {}
    stem = report.get("stem_balance", {}) or {}
    comp = report.get("artist_comparison", {}) or {}
    bands = freq.get("band_percentages", {}) or {}
    sections = report.get("sections", []) or []
    return {
        "file": basic.get("file_name"),
        "basic": {"bpm": _safe_float(basic.get("bpm")), "key": basic.get("key"), "duration_seconds": _safe_float(basic.get("duration_seconds"))},
        "scores": {k: scores.get(k) for k in ["release", "mix", "master", "arrangement", "energy", "bass_strength", "darkness", "brightness", "drum_bounce", "vocal_space"]},
        "loudness": {"peak_db": _safe_float(loud.get("peak_db")), "rms_db": _safe_float(loud.get("rms_db")), "dynamic_range_db": _safe_float(loud.get("dynamic_range_db")), "clipping_detected": loud.get("clipping_detected"), "clipping_samples": loud.get("clipping_samples"), "headroom_db": _safe_float(loud.get("headroom_db"))},
        "frequency": {"dominant_band": freq.get("dominant_band"), "brightness_label": freq.get("brightness_label"), "low_end_total_percent": _safe_float(freq.get("low_end_total_percent")), "mid_total_percent": _safe_float(freq.get("mid_total_percent")), "top_total_percent": _safe_float(freq.get("top_total_percent")), "band_percentages": {k: round(_safe_float(v,0) or 0,2) for k,v in bands.items()}},
        "rhythm": {"onset_density": _safe_float(rhythm.get("onset_density")), "drum_activity": rhythm.get("drum_activity"), "estimated_bars": rhythm.get("estimated_bars")},
        "stem_balance": {"enabled": stem.get("enabled"), "status": stem.get("status"), "confidence": stem.get("confidence"), "vocal_to_beat_db": _safe_float(stem.get("vocal_to_beat_db")), "bass_to_vocal_db": _safe_float(stem.get("bass_to_vocal_db")), "bass_to_other_db": _safe_float(stem.get("bass_to_other_db")), "beat_vocal_balance_score": stem.get("beat_vocal_balance_score"), "melody_presence_score": stem.get("melody_presence_score"), "warnings": _take_list(stem.get("warnings"), 4)},
        "artist_comparison": {"top_artists": _top_artists(comp, 5), "verdict": comp.get("verdict")},
        "sections": [{"name": s.get("name"), "start": _safe_float(s.get("start")), "end": _safe_float(s.get("end")), "energy_label": s.get("energy_label"), "avg_energy": _safe_float(s.get("avg_energy"))} for s in sections[:10] if isinstance(s, dict)],
        "standard_feedback": {"top_problems": _take_list(report.get("top_problems"), 5), "next_steps": _take_list(report.get("next_steps"), 5), "artist_notes": _take_list(report.get("artist_notes"), 4), "producer_notes": _take_list(report.get("producer_notes"), 4), "master_notes": _take_list(report.get("master_notes"), 4)},
    }


def extract_deep_audio_summary(audio_path: str | Path | None) -> Dict[str, Any]:
    if not audio_path:
        return {"enabled": False, "reason": "No audio path supplied."}
    try:
        import librosa
        import numpy as np
        path = Path(audio_path)
        if not path.exists():
            return {"enabled": False, "reason": "Audio file not found."}
        y, sr = librosa.load(path, mono=True, sr=22050)
        if y.size == 0:
            return {"enabled": False, "reason": "Audio loaded empty."}
        max_seconds = 120
        if len(y) > sr * max_seconds:
            start = max(0, (len(y)//2) - (sr*max_seconds//2))
            y = y[start:start + sr*max_seconds]
        hop=512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=hop)[0]
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop)
        duration = float(len(y)/sr)
        def seg(a,b):
            if len(times)==0: return {}
            idx = np.where((times >= duration*a) & (times <= duration*b))[0]
            if idx.size == 0: return {}
            oi = idx[idx < len(onset)]
            return {"range_sec":[round(duration*a,2), round(duration*b,2)], "avg_rms":round(float(np.mean(rms[idx])),6), "peak_rms":round(float(np.max(rms[idx])),6), "brightness_hz":round(float(np.mean(centroid[idx])),2), "rolloff_hz":round(float(np.mean(rolloff[idx])),2), "zero_crossing":round(float(np.mean(zcr[idx])),6), "onset_strength":round(float(np.mean(onset[oi])),4) if oi.size else None}
        loudest = int(np.argmax(rms)) if len(rms) else 0
        quietest = int(np.argmin(rms)) if len(rms) else 0
        return {"enabled": True, "duration_analyzed_sec": round(duration,2), "overall": {"avg_rms": round(float(np.mean(rms)),6), "rms_variation": round(float(np.std(rms)),6), "avg_brightness_hz": round(float(np.mean(centroid)),2), "avg_rolloff_hz": round(float(np.mean(rolloff)),2), "avg_zero_crossing": round(float(np.mean(zcr)),6), "avg_onset_strength": round(float(np.mean(onset)),4) if len(onset) else None}, "segments": {"intro": seg(0,.2), "early_middle": seg(.2,.45), "middle": seg(.45,.7), "ending": seg(.7,1)}, "moments": {"loudest_sec": round(float(times[min(loudest,len(times)-1)]),2) if len(times) else 0, "quietest_sec": round(float(times[min(quietest,len(times)-1)]),2) if len(times) else 0}}
    except Exception as e:
        return {"enabled": False, "reason": str(e)}


def _audio_input(audio_path: str | Path | None) -> Optional[Dict[str, str]]:
    if not audio_path or not OPENAI_AUDIO_MODEL:
        return None
    p = Path(audio_path)
    if not p.exists() or p.stat().st_size > MAX_AUDIO_UPLOAD_MB*1024*1024:
        return None
    ext = p.suffix.lower().lstrip(".")
    if ext not in {"wav","mp3","m4a","aac","ogg","flac"}:
        return None
    try:
        return {"format": ext, "data": base64.b64encode(p.read_bytes()).decode("utf-8")}
    except Exception:
        return None


def _fallback(report: Dict[str, Any], reason: str="") -> Dict[str, Any]:
    scores = report.get("scores",{}) or {}
    problems = _take_list(report.get("top_problems"),3)
    steps = _take_list(report.get("next_steps"),3)
    return {"ai_enabled": False, "ai_error": reason, "top_problems": problems, "next_steps": steps, "ai_review": {"first_impression": f"SoundLens read this as a {scores.get('release',0)}/100 release. AI could not fully generate, so this fallback is based on the standard analysis.", "biggest_strengths": ["The track produced a complete SoundLens report."], "biggest_weaknesses": problems or ["No single major issue was isolated."], "mix_advice": steps[0] if steps else "Compare against a reference and fix one balance issue first.", "artist_direction": "Use the Artist Match page as the style lane.", "priority_fix": steps[0] if steps else "Fix the most audible balance issue first.", "release_verdict": "Re-upload after the main fix and compare the score."}, "ai_fixes": [{"title":"Priority Fix", "why_it_matters": problems[0] if problems else "This is the safest first improvement.", "what_to_do": steps[0] if steps else "Make one reference-based mix change.", "how_to_check":"Re-upload and compare the new report."}]}


def _json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end+1]
    return json.loads(text)


def _clean(data: Dict[str,Any], report: Dict[str,Any]) -> Dict[str,Any]:
    fb = _fallback(report)
    r = data.get("ai_review") if isinstance(data.get("ai_review"), dict) else {}
    def s(k, lim=900): return str(r.get(k) or fb["ai_review"].get(k) or "").strip()[:lim]
    fixes=[]
    for f in (data.get("ai_fixes") or [])[:5]:
        if isinstance(f, dict):
            fixes.append({"title":str(f.get("title","")).strip()[:120], "why_it_matters":str(f.get("why_it_matters","")).strip()[:800], "what_to_do":str(f.get("what_to_do","")).strip()[:800], "how_to_check":str(f.get("how_to_check","")).strip()[:600]})
    return {"ai_enabled": True, "model": AI_MODEL, "audio_model": OPENAI_AUDIO_MODEL or None, "top_problems": _take_list(data.get("top_problems"),5) or fb["top_problems"], "next_steps": _take_list(data.get("next_steps"),5) or fb["next_steps"], "ai_review": {"first_impression":s("first_impression"), "biggest_strengths":_take_list(r.get("biggest_strengths"),3) or fb["ai_review"]["biggest_strengths"], "biggest_weaknesses":_take_list(r.get("biggest_weaknesses"),3) or fb["ai_review"]["biggest_weaknesses"], "mix_advice":s("mix_advice"), "artist_direction":s("artist_direction"), "priority_fix":s("priority_fix",700), "release_verdict":s("release_verdict")}, "ai_fixes": fixes or fb["ai_fixes"]}


def generate_soundlens_ai_feedback(report: Dict[str, Any], audio_path: str | Path | None = None) -> Dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return _fallback(report, "OPENAI_API_KEY is not configured.")
    compact = _compact_report(report)
    audio_summary = extract_deep_audio_summary(audio_path)
    audio = _audio_input(audio_path)
    system = """You are SoundLens AI, a producer and mix-engineer assistant for underground rap/trap/rage artists. Be specific, critical, useful, and grounded. Use the SoundLens report, deep audio timeline summary, and direct audio attachment if present. If no direct audio is attached, do not claim you personally listened; say what the analysis indicates. Never invent lyrics or instruments. Return JSON only."""
    prompt = f"""Create a premium AI-only Fixes page and review. Do not just repeat scores. Explain what is happening, why it matters, what to do, and how to check it. Use segment timing when useful.
Return exactly this JSON shape:
{{"top_problems":["","",""],"next_steps":["","",""],"ai_review":{{"first_impression":"","biggest_strengths":["",""],"biggest_weaknesses":["",""],"mix_advice":"","artist_direction":"","priority_fix":"","release_verdict":""}},"ai_fixes":[{{"title":"","why_it_matters":"","what_to_do":"","how_to_check":""}}]}}
REPORT: {json.dumps(compact, ensure_ascii=False)}
DEEP_AUDIO_TIMELINE: {json.dumps(audio_summary, ensure_ascii=False)}"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        model = OPENAI_AUDIO_MODEL or AI_MODEL
        if audio:
            user_content = [{"type":"input_text","text":prompt},{"type":"input_audio","input_audio":audio}]
            resp = client.responses.create(model=model, input=[{"role":"system","content":system},{"role":"user","content":user_content}], max_output_tokens=1400)
        else:
            resp = client.responses.create(model=model, input=[{"role":"system","content":system},{"role":"user","content":prompt}], max_output_tokens=1400)
        return _clean(_json(getattr(resp, "output_text", "") or ""), report)
    except Exception as e:
        return _fallback(report, str(e))
