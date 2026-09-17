from __future__ import annotations

import secrets
import uuid

from fastapi import HTTPException

import app as legacy
import hardened_app as hardened

app = hardened.app

# Replace only the legacy signup route. hardened_app already replaced login,
# resend verification, analysis, Stripe and deferred AI routes.
hardened._remove_route("/auth/signup", "POST")


@app.post("/auth/signup")
def hardened_signup(payload: legacy.AuthPayload):
    db = legacy.load_users_db()
    email = legacy.validate_public_signup_email(payload.email)
    password = str(payload.password or "")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    existing = legacy.find_user_by_email(db, email)
    if existing:
        if existing.get("email_verified"):
            raise HTTPException(status_code=400, detail="An account with this email already exists.")

        # Keep the existing verification token valid. Repeated signup attempts must
        # never make a verification email already sitting in the inbox go stale.
        hardened._ensure_verification_token(existing)
        allowed, remaining = hardened._verification_can_send(existing)
        sent = False
        if allowed:
            existing["verification_last_sent_at"] = legacy.now_iso()
            legacy.save_users_db(db)
            sent = legacy.send_signup_verification_email(existing)
            legacy.track_event("verification_resent_via_signup", existing, {"sent": bool(sent)})
            if not sent:
                raise HTTPException(status_code=503, detail="Verification email delivery is temporarily unavailable.")

        message = f"Verification email sent to {email}. Confirm it before signing in."
        if not allowed:
            message = f"A verification email was sent recently. Try again in {remaining} seconds if you need another one."
        return {"ok": True, "verification_required": True, "email": email, "message": message}

    user_id = str(uuid.uuid4())
    salt = str(uuid.uuid4())
    user = {
        "id": user_id,
        "email": email,
        "display_name": (payload.display_name or email.split("@")[0]).strip()[:80],
        "password_salt": salt,
        "password_hash": legacy.hash_password(password, salt),
        "plan": "free",
        "usage": {"date": legacy.utc_today(), "count": 0},
        "total_uploads": 0,
        "created_at": legacy.now_iso(),
        "email_verified": False,
        "email_verification_token": secrets.token_urlsafe(32),
        "email_verified_at": None,
        "verification_last_sent_at": legacy.now_iso(),
    }
    db.setdefault("users", {})[user_id] = user
    legacy.save_users_db(db)

    sent = legacy.send_signup_verification_email(user)
    if not sent:
        # Preserve the old safety behavior: if account verification cannot be
        # delivered, do not strand a half-created account.
        db = legacy.load_users_db()
        db.get("users", {}).pop(user_id, None)
        legacy.save_users_db(db)
        legacy.track_event("signup_email_failed", None, {"email": email})
        raise HTTPException(status_code=503, detail="SoundLens could not send the verification email. No account was created.")

    legacy.track_event("signup_pending_verification", user, {"email": email})
    return {
        "ok": True,
        "verification_required": True,
        "email": email,
        "message": f"Verification email sent to {email}. Confirm it before signing in.",
    }
