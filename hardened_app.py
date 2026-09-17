from __future__ import annotations

import contextvars
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import stripe
from fastapi import File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import app as legacy
import soundlens_pro as slp

# Re-export the existing FastAPI application. Railway runs this module instead
# of app.py so production hardening can stay small, reviewable and reversible.
app = legacy.app


# ---------------------------------------------------------------------------
# Route replacement helpers
# ---------------------------------------------------------------------------

def _remove_route(path: str, method: str) -> None:
    method = method.upper()
    kept = []
    for route in app.router.routes:
        methods = getattr(route, "methods", set()) or set()
        if getattr(route, "path", None) == path and method in methods:
            continue
        kept.append(route)
    app.router.routes = kept


def _dt(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Analyzer crash protection
# ---------------------------------------------------------------------------

_original_estimate_arrangement = slp.estimate_arrangement


def _fallback_arrangement(y, sr: int, duration: float, rhythm):
    """Conservative energy-based fallback for unusual/very short audio.

    This runs only when the richer novelty detector fails. It intentionally uses
    generic labels rather than inventing verse/chorus structure.
    """
    if duration <= 0:
        return []
    if duration < 12:
        return [slp.ArrangementSection(
            name="Full Track",
            start=0.0,
            end=float(duration),
            avg_energy=slp.section_energy(y, sr, 0.0, duration),
            energy_label="Medium",
        )]

    bar_seconds = max(float(getattr(rhythm, "seconds_per_bar", 0.0) or 0.0), 1.5)
    target = max(12.0, min(28.0, bar_seconds * 8.0))
    section_count = max(2, min(7, int(round(duration / target))))
    boundaries = np.linspace(0.0, float(duration), section_count + 1)
    energies = [
        slp.section_energy(y, sr, float(a), float(b))
        for a, b in zip(boundaries[:-1], boundaries[1:])
    ]
    avg = float(np.mean(energies)) + slp.EPSILON
    peak = float(np.max(energies)) + slp.EPSILON

    sections = []
    for idx, (start, end, energy) in enumerate(zip(boundaries[:-1], boundaries[1:], energies)):
        rel_avg = energy / avg
        rel_peak = energy / peak
        if rel_avg >= 1.10 or rel_peak >= 0.86:
            level = "High"
        elif rel_avg <= 0.82 or rel_peak <= 0.52:
            level = "Low"
        else:
            level = "Medium"

        if idx == 0 and (end - start) <= min(24.0, duration * 0.30):
            name = "Intro"
        elif idx == section_count - 1 and level == "Low":
            name = "Outro"
        elif level == "High":
            name = "Peak Section"
        elif level == "Low":
            name = "Breakdown / Low Energy"
        else:
            name = "Main Section"

        sections.append(slp.ArrangementSection(
            name=name,
            start=float(start),
            end=float(end),
            avg_energy=float(energy),
            energy_label=level,
        ))
    return sections


def _safe_estimate_arrangement(y, sr: int, duration: float, rhythm):
    try:
        return _original_estimate_arrangement(y, sr, duration, rhythm)
    except ValueError as error:
        # Production logs showed librosa novelty arrays occasionally returning
        # different frame counts (for example 87 vs 43), which previously killed
        # the entire analysis. Preserve the report with a conservative fallback.
        if "broadcast" not in str(error).lower() and "shapes" not in str(error).lower():
            raise
        print(f"[HARDENING] arrangement novelty mismatch recovered: {error}")
        return _fallback_arrangement(y, sr, duration, rhythm)


# analyze_audio resolves estimate_arrangement from its module globals at runtime.
slp.estimate_arrangement = _safe_estimate_arrangement


# ---------------------------------------------------------------------------
# 2. Upload isolation, validation and cleanup
# ---------------------------------------------------------------------------

_ALLOWED_AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif"}
_MAX_UPLOAD_BYTES = int(os.getenv("SOUNDLENS_MAX_UPLOAD_MB", "80")) * 1024 * 1024
_current_upload: contextvars.ContextVar[Path | None] = contextvars.ContextVar("soundlens_upload", default=None)


def _safe_save_upload(file: UploadFile) -> Path:
    original = Path(file.filename or "uploaded_audio.wav")
    suffix = original.suffix.lower()
    if suffix not in _ALLOWED_AUDIO:
        raise HTTPException(status_code=415, detail="Upload a WAV, MP3, FLAC, M4A, AAC, OGG or AIFF audio file.")

    target = legacy.UPLOADS_DIR / f"audio_{secrets.token_hex(16)}{suffix}"
    total = 0
    try:
        with target.open("wb") as output:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio file is too large. Maximum upload size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                output.write(chunk)
        if total <= 44:
            raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")
        _current_upload.set(target)
        return target
    except Exception:
        target.unlink(missing_ok=True)
        raise


legacy.save_upload = _safe_save_upload


def _cleanup_current_upload() -> None:
    path = _current_upload.get()
    if path:
        try:
            path.unlink(missing_ok=True)
        except Exception as error:
            print(f"[HARDENING] upload cleanup failed: {type(error).__name__}: {error}")
        finally:
            _current_upload.set(None)


# Replace analyze so failures have real HTTP status codes and temp uploads are
# always deleted after the measured analysis has finished.
_remove_route("/analyze", "POST")


@app.post("/analyze")
def hardened_analyze(
    stems: bool = None,
    report_title: str | None = None,
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    try:
        result = legacy.analyze(
            stems=stems,
            report_title=report_title,
            file=file,
            authorization=authorization,
        )
        if isinstance(result, dict) and result.get("error"):
            message = str(result.get("error") or "Analysis failed.")
            status = 422 if "6 minutes" in message.lower() or "audio" in message.lower() else 500
            return JSONResponse(status_code=status, content=result)
        return result
    finally:
        _cleanup_current_upload()


_remove_route("/compare-profile", "POST")


@app.post("/compare-profile")
def hardened_compare_profile(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    try:
        result = legacy.compare_profile(file=file, authorization=authorization)
        if isinstance(result, dict) and str(result.get("verdict", "")).lower().startswith("profile comparison failed"):
            return JSONResponse(status_code=500, content=result)
        return result
    finally:
        _cleanup_current_upload()


# ---------------------------------------------------------------------------
# 3. Stable email verification links + resend throttling
# ---------------------------------------------------------------------------

_VERIFICATION_RESEND_SECONDS = int(os.getenv("SOUNDLENS_VERIFICATION_RESEND_SECONDS", "60"))


def _verification_can_send(user: dict) -> tuple[bool, int]:
    last = _dt(user.get("verification_last_sent_at"))
    if not last:
        return True, 0
    elapsed = (datetime.now(timezone.utc) - last).total_seconds()
    remaining = max(0, int(_VERIFICATION_RESEND_SECONDS - elapsed))
    return elapsed >= _VERIFICATION_RESEND_SECONDS, remaining


def _ensure_verification_token(user: dict) -> str:
    token = str(user.get("email_verification_token") or "").strip()
    if not token:
        token = secrets.token_urlsafe(32)
        user["email_verification_token"] = token
    return token


_remove_route("/auth/resend-verification", "POST")


@app.post("/auth/resend-verification")
def hardened_resend_verification(payload: legacy.ResendVerificationPayload):
    db = legacy.load_users_db()
    email = legacy.validate_public_signup_email(payload.email)
    user = legacy.find_user_by_email(db, email)
    if not user:
        # Avoid turning this endpoint into an account-enumeration oracle.
        return {"ok": True, "message": "If that account still needs verification, an email will be sent."}
    if user.get("email_verified"):
        return {"ok": True, "message": "Email is already verified."}

    allowed, remaining = _verification_can_send(user)
    if not allowed:
        return {"ok": True, "message": f"A verification email was just sent. Try again in {remaining} seconds."}

    _ensure_verification_token(user)
    user["verification_last_sent_at"] = legacy.now_iso()
    legacy.save_users_db(db)
    sent = legacy.send_signup_verification_email(user)
    legacy.track_event("verification_resent", user, {"sent": bool(sent)})
    if not sent:
        raise HTTPException(status_code=503, detail="Verification email delivery is temporarily unavailable.")
    return {"ok": True, "message": "Verification email sent."}


# Replace login only to stop invalidating an earlier verification email every
# time an unverified user retries their password.
_remove_route("/auth/login", "POST")


@app.post("/auth/login")
def hardened_login(payload: legacy.AuthPayload):
    db = legacy.load_users_db()
    email = legacy.normalize_email(payload.email)
    user = legacy.find_user_by_email(db, email)
    if not user:
        legacy.track_event("login_failed", None, {"email": email, "reason": "unknown_email"})
        raise HTTPException(status_code=401, detail="Email or password is wrong.")

    expected = legacy.hash_password(payload.password, user.get("password_salt", ""))
    if not secrets.compare_digest(str(expected), str(user.get("password_hash") or "")):
        legacy.track_event("login_failed", user, {"email": email, "reason": "wrong_password"})
        raise HTTPException(status_code=401, detail="Email or password is wrong.")

    if not bool(user.get("email_verified")):
        _ensure_verification_token(user)
        allowed, remaining = _verification_can_send(user)
        sent = False
        if allowed:
            user["verification_last_sent_at"] = legacy.now_iso()
            legacy.save_users_db(db)
            sent = legacy.send_signup_verification_email(user)
        legacy.track_event("legacy_verification_requested", user, {
            "email": email,
            "sent": bool(sent),
            "throttled": not allowed,
            "plan": user.get("plan", "free"),
        })
        if allowed and not sent:
            raise HTTPException(status_code=503, detail="Your password is correct, but verification email delivery is unavailable.")
        detail = "Verify your email before signing in."
        if not allowed and remaining:
            detail += f" A verification email was sent recently; retry sending in {remaining} seconds if needed."
        raise HTTPException(status_code=403, detail=detail)

    token = secrets.token_urlsafe(32)
    db.setdefault("tokens", {})[token] = user["id"]
    user["last_login_at"] = legacy.now_iso()
    user["last_active_at"] = user["last_login_at"]
    legacy.save_users_db(db)
    legacy.track_event("login_success", user, {})
    return {"token": token, "user": legacy.public_user(user)}


# ---------------------------------------------------------------------------
# 4. Safer Stripe checkout confirmation + authenticated webhook fallback
# ---------------------------------------------------------------------------


def _stripe_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    method = getattr(obj, "to_dict_recursive", None)
    if callable(method):
        return method()
    try:
        return dict(obj)
    except Exception:
        return {}


_remove_route("/stripe/confirm-checkout", "GET")


@app.get("/stripe/confirm-checkout")
def hardened_confirm_checkout(session_id: str, authorization: str | None = Header(default=None)):
    if not legacy.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured yet.")
    db, user = legacy.get_current_user(authorization)

    try:
        raw = stripe.checkout.Session.retrieve(session_id)
        session = _stripe_dict(raw)
    except Exception as error:
        print(f"[STRIPE] checkout confirmation retrieve failed: {type(error).__name__}: {error}")
        raise HTTPException(status_code=400, detail="Could not verify this Stripe checkout.")

    try:
        metadata = _stripe_dict(session.get("metadata") or {})
        session_user_id = str(session.get("client_reference_id") or metadata.get("user_id") or "")
        if session_user_id != str(user.get("id") or ""):
            raise HTTPException(status_code=403, detail="This checkout does not belong to your SoundLens account.")

        if str(session.get("status") or "").lower() not in {"complete", ""}:
            raise HTTPException(status_code=409, detail="Stripe checkout is not complete yet.")
        if str(session.get("payment_status") or "").lower() not in {"paid", "no_payment_required"}:
            raise HTTPException(status_code=409, detail="Stripe payment is not complete yet.")

        plan = str(metadata.get("plan") or "pro").lower().strip()
        if plan not in {"pro", "studio"}:
            plan = "pro"

        user["plan"] = plan
        user["upgraded_at"] = legacy.now_iso()
        user["stripe_customer_id"] = session.get("customer") or user.get("stripe_customer_id")
        user["stripe_subscription_id"] = session.get("subscription") or user.get("stripe_subscription_id")
        user.pop("admin_pro_granted", None)
        user.pop("admin_studio_granted", None)
        user["studio_revoked_by_admin"] = False
        legacy.save_users_db(db)
        try:
            legacy.track_event("stripe_checkout_confirmed", user, {"plan": plan, "session_id": session_id})
        except Exception as event_error:
            print(f"[STRIPE] non-fatal event logging error: {type(event_error).__name__}: {event_error}")
        return {"ok": True, "user": legacy.public_user(user)}
    except HTTPException:
        raise
    except Exception as error:
        print(f"[STRIPE] checkout confirmation failed: {type(error).__name__}: {error}")
        raise HTTPException(status_code=500, detail="Stripe checkout was verified, but SoundLens could not activate the plan. Please contact support.")


_remove_route("/stripe/webhook", "POST")


@app.post("/stripe/webhook")
async def hardened_stripe_webhook(request: Request):
    if not legacy.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured yet.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    try:
        if legacy.STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(payload, signature, legacy.STRIPE_WEBHOOK_SECRET)
            event = _stripe_dict(event)
        else:
            # Until a webhook signing secret is configured, never trust the posted
            # JSON directly. Use only its event ID, then retrieve the authentic
            # event from Stripe using the server-side secret key.
            posted = json.loads(payload.decode("utf-8"))
            event_id = str(posted.get("id") or "").strip()
            if not event_id.startswith("evt_"):
                raise ValueError("Missing Stripe event id")
            event = _stripe_dict(stripe.Event.retrieve(event_id))
    except Exception as error:
        print(f"[STRIPE] webhook verification failed: {type(error).__name__}: {error}")
        raise HTTPException(status_code=400, detail="Webhook verification failed.")

    event_type = str(event.get("type") or "")
    obj = _stripe_dict((_stripe_dict(event.get("data") or {})).get("object") or {})

    if event_type == "checkout.session.completed":
        metadata = _stripe_dict(obj.get("metadata") or {})
        email = obj.get("customer_email") or metadata.get("email")
        plan = str(metadata.get("plan") or "pro").lower()
        if email:
            legacy.set_user_plan_by_email(email, plan if plan in {"pro", "studio"} else "pro", obj.get("customer"), obj.get("subscription"))

    elif event_type in {"customer.subscription.updated", "customer.subscription.deleted", "customer.subscription.paused"}:
        customer_id = obj.get("customer")
        subscription_id = obj.get("id")
        metadata = _stripe_dict(obj.get("metadata") or {})
        status = str(obj.get("status") or "").lower()
        paid_active = status in {"active", "trialing"}
        db = legacy.load_users_db()
        changed = False
        for user in db.get("users", {}).values():
            if user.get("stripe_customer_id") == customer_id or (subscription_id and user.get("stripe_subscription_id") == subscription_id):
                if paid_active:
                    plan = str(metadata.get("plan") or user.get("plan") or "pro").lower()
                    if plan == "studio" and user.get("studio_revoked_by_admin"):
                        user["plan"] = "pro"
                    else:
                        user["plan"] = plan if plan in {"pro", "studio"} else "pro"
                    user["upgraded_at"] = legacy.now_iso()
                else:
                    user["plan"] = "free"
                    user["downgraded_at"] = legacy.now_iso()
                changed = True
        if changed:
            legacy.save_users_db(db)

    return {"received": True}


# ---------------------------------------------------------------------------
# 5. Recover deferred AI reviews after process restarts
# ---------------------------------------------------------------------------

_ai_recovery_lock = threading.Lock()
_ai_recovering: set[str] = set()


def _persist_ai_result(report_file: Path, result: dict) -> None:
    payload = legacy.read_json_file(report_file, {})
    if not isinstance(payload, dict):
        return
    report = payload.get("report")
    if isinstance(report, dict):
        report["top_problems"] = result.get("top_problems", report.get("top_problems", []))
        report["next_steps"] = result.get("next_steps", report.get("next_steps", []))
        report["ai_suggested_direction"] = result.get("suggested_direction", [])
        report["ai_review"] = result.get("ai_review", {})
        report["ai_enabled"] = bool(result.get("ai_enabled", False))
        report["ai_model"] = result.get("model")
    payload["ai_feedback"] = result
    payload["ai_status"] = "ready" if result else "failed"
    payload["ai_updated_at"] = legacy.now_iso()
    legacy.write_json_file(report_file, payload)


def _recover_ai(report_id: str, report_file: Path) -> None:
    try:
        payload = legacy.read_json_file(report_file, {})
        report = payload.get("report") if isinstance(payload, dict) else None
        if not isinstance(report, dict):
            return
        result = legacy._generate_ai_feedback_after_analysis(report)
        if result:
            _persist_ai_result(report_file, result)
            with legacy.ASYNC_AI_LOCK:
                legacy.ASYNC_AI_RESULTS[report_id] = result
                if len(legacy.ASYNC_AI_RESULTS) > 100:
                    for key in list(legacy.ASYNC_AI_RESULTS)[:-100]:
                        legacy.ASYNC_AI_RESULTS.pop(key, None)
        else:
            payload["ai_status"] = "failed"
            payload["ai_updated_at"] = legacy.now_iso()
            legacy.write_json_file(report_file, payload)
    except Exception as error:
        print(f"[HARDENING] AI recovery failed report={report_id}: {type(error).__name__}: {error}")
    finally:
        with _ai_recovery_lock:
            _ai_recovering.discard(report_id)


_remove_route("/analysis-ai/{report_id}", "GET")


@app.get("/analysis-ai/{report_id}")
def hardened_analysis_ai_status(report_id: str, authorization: str | None = Header(default=None)):
    _, user = legacy.get_current_user(authorization)
    report_id = str(report_id)

    with legacy.ASYNC_AI_LOCK:
        cached = legacy.ASYNC_AI_RESULTS.get(report_id)
    if isinstance(cached, dict):
        return {"ok": True, "status": "ready", "ai_feedback": cached}

    report_file = legacy.SAVED_REPORTS_DIR / user["id"] / f"{report_id}.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report not found.")

    payload = legacy.read_json_file(report_file, {})
    existing = payload.get("ai_feedback") if isinstance(payload, dict) else None
    if isinstance(existing, dict) and (existing.get("ai_enabled") or existing.get("ai_review")):
        return {"ok": True, "status": "ready", "ai_feedback": existing}

    with _ai_recovery_lock:
        if report_id not in _ai_recovering:
            _ai_recovering.add(report_id)
            threading.Thread(target=_recover_ai, args=(report_id, report_file), daemon=True).start()

    return {"ok": True, "status": "pending", "ai_feedback": None}


# Lightweight health endpoint for Railway and future uptime checks.
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "SoundLens",
        "data_dir": bool(legacy.DATA_DIR.exists()),
        "stripe": bool(legacy.STRIPE_SECRET_KEY),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
    }
