from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import html as html_lib
import json
import os
import shutil
import traceback
import uuid
import smtplib
import secrets
import tempfile
import zipfile
from email.message import EmailMessage
import stripe

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from soundlens_pro import analyze_audio, render_report
from compare_to_profile_pro import compare_audio_to_profiles
from ai_feedback import generate_soundlens_ai_feedback

app = FastAPI()

app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

PROFILES_DIR = Path("artist_profiles")
STEMS_DIR = Path("stems")
STEMS_DIR.mkdir(exist_ok=True)

# Local launch database.
# This is good for local testing and early MVP work.
# For production, move this to Postgres/Supabase/Firebase and use real Stripe webhooks.
# Persistent application data.
# On Railway, mount a Volume at /data. You may override this with
# SOUNDLENS_DATA_DIR. Code remains in the project directory while user data
# lives outside the Git checkout, so a deploy cannot replace it.
_default_data_dir = "/data" if Path("/data").exists() else "."
DATA_DIR = Path(os.getenv("SOUNDLENS_DATA_DIR", _default_data_dir)).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

USERS_DB_PATH = DATA_DIR / "soundlens_users.json"
SAVED_REPORTS_DIR = DATA_DIR / "saved_reports"
SAVED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_EVENTS_PATH = DATA_DIR / "soundlens_admin_events.json"
FEEDBACK_PATH = DATA_DIR / "soundlens_feedback.json"
CONTACT_MESSAGES_PATH = DATA_DIR / "soundlens_contact_messages.json"
BACKUPS_DIR = DATA_DIR / "soundlens_backups"
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)


def migrate_legacy_data_once() -> None:
    """Copy legacy project-root data into the persistent data directory.

    This is copy-only and runs only when the persistent destination does not
    already exist. It never overwrites production data.
    """
    if DATA_DIR.resolve() == Path(".").resolve():
        return

    legacy_files = [
        "soundlens_users.json",
        "soundlens_admin_events.json",
        "soundlens_feedback.json",
        "soundlens_contact_messages.json",
    ]
    for filename in legacy_files:
        source = Path(filename)
        destination = DATA_DIR / filename
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)

    legacy_reports = Path("saved_reports")
    if legacy_reports.exists() and not any(SAVED_REPORTS_DIR.iterdir()):
        shutil.copytree(legacy_reports, SAVED_REPORTS_DIR, dirs_exist_ok=True)


migrate_legacy_data_once()
PASSWORD_RESET_EXPIRY_MINUTES = int(os.getenv("SOUNDLENS_PASSWORD_RESET_EXPIRY_MINUTES", "30"))

SOUNDLENS_NOTIFY_EMAIL = os.getenv("SOUNDLENS_NOTIFY_EMAIL", "soudlensmail@gmail.com")
SOUNDLENS_ADMIN_EMAILS = {
    email.strip().lower()
    for email in os.getenv("SOUNDLENS_ADMIN_EMAILS", "jaydenflynn9@gmail.com").split(",")
    if email.strip()
}

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USERNAME or SOUNDLENS_NOTIFY_EMAIL)


FREE_DAILY_UPLOAD_LIMIT = int(os.getenv("SOUNDLENS_FREE_DAILY_UPLOAD_LIMIT", "3"))
PRO_PRICE_CAD = os.getenv("SOUNDLENS_PRO_PRICE_CAD", "7.99")

USE_DEMUCS_BY_DEFAULT = os.getenv("SOUNDLENS_USE_DEMUCS", "0") == "1"
COMPARE_USE_STEMS = os.getenv("SOUNDLENS_COMPARE_USE_STEMS", "0") == "1"

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID", "price_1TlO4U2UFBtBMGztgrQDYTvB")
STRIPE_STUDIO_PRICE_ID = os.getenv("STRIPE_STUDIO_PRICE_ID", "price_1TlO7o2UFBtBMGzt9SmiPpMq")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
SOUNDLENS_PUBLIC_URL = os.getenv("SOUNDLENS_PUBLIC_URL", "https://www.soundlensapp.com").rstrip("/")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY



class AuthPayload(BaseModel):
    email: str
    password: str
    display_name: str | None = None


class FeedbackPayload(BaseModel):
    rating: int | None = None
    accuracy: str | None = None
    message: str
    email: str | None = None
    report_id: str | None = None
    page: str | None = None


class EventPayload(BaseModel):
    event: str
    page: str | None = None
    details: dict | None = None


class ResendVerificationPayload(BaseModel):
    email: str


class AdminLifetimeAccountPayload(BaseModel):
    email: str
    password: str
    display_name: str | None = None


class AdminEmailPayload(BaseModel):
    email: str


class AdminPasswordPayload(BaseModel):
    email: str
    password: str


class PasswordResetRequestPayload(BaseModel):
    email: str


class PasswordResetConfirmPayload(BaseModel):
    token: str
    password: str


class ChangePasswordPayload(BaseModel):
    current_password: str
    new_password: str


class ContactPayload(BaseModel):
    name: str | None = None
    email: str
    subject: str | None = None
    message: str


class DeleteAccountPayload(BaseModel):
    password: str



def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(value: str | None):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def create_data_backup() -> Path:
    """Create a copy-only ZIP of current user/admin data. Never mutates source data."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = BACKUPS_DIR / f"soundlens_data_backup_{stamp}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [USERS_DB_PATH, ADMIN_EVENTS_PATH, FEEDBACK_PATH, CONTACT_MESSAGES_PATH]:
            if path.exists():
                archive.write(path, arcname=path.name)
        if SAVED_REPORTS_DIR.exists():
            for path in SAVED_REPORTS_DIR.rglob("*"):
                if path.is_file():
                    archive.write(path, arcname=str(path))
    return target


def read_json_file(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def write_json_file(path: Path, data) -> None:
    """Atomically write JSON so an interrupted deploy/write cannot blank data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def append_json_list(path: Path, item: dict) -> None:
    data = read_json_file(path, [])
    if not isinstance(data, list):
        data = []
    data.append(item)
    write_json_file(path, data)


def track_event(event_type: str, user: dict | None = None, details: dict | None = None) -> None:
    details = details or {}
    safe_user = None
    if user:
        safe_user = {
            "id": user.get("id"),
            "email": user.get("email"),
            "display_name": user.get("display_name"),
            "plan": user.get("plan", "free"),
        }

    append_json_list(ADMIN_EVENTS_PATH, {
        "id": str(uuid.uuid4()),
        "created_at": now_iso(),
        "event": event_type,
        "user": safe_user,
        "details": details,
    })


def send_notification_email(subject: str, body: str) -> bool:
    if not SMTP_HOST or not SMTP_USERNAME or not SMTP_PASSWORD or not SOUNDLENS_NOTIFY_EMAIL:
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM_EMAIL
        msg["To"] = SOUNDLENS_NOTIFY_EMAIL
        msg.set_content(body)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        return True
    except Exception as error:
        track_event("email_failed", None, {"subject": subject, "error": str(error)})
        return False



def send_notification_email_to(
    to_email: str | None,
    subject: str,
    body: str,
    html_body: str | None = None,
) -> bool:
    if not to_email:
        return False
    if not SMTP_HOST or not SMTP_USERNAME or not SMTP_PASSWORD:
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM_EMAIL
        msg["To"] = to_email
        msg.set_content(body)
        if html_body:
            msg.add_alternative(html_body, subtype="html")

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        return True
    except Exception as error:
        track_event("email_failed", None, {"to": to_email, "subject": subject, "error": str(error)})
        return False


def build_branded_email(
    title: str,
    message: str,
    button_text: str | None = None,
    button_url: str | None = None,
    note: str | None = None,
) -> str:
    logo_url = f"{SOUNDLENS_PUBLIC_URL}/static/soundlens_email_logo.png"
    safe_title = html_lib.escape(str(title or "SoundLens"))
    safe_message = html_lib.escape(str(message or "")).replace("\n", "<br>")
    safe_note = html_lib.escape(str(note or "")).replace("\n", "<br>") if note else ""

    button_html = ""
    if button_text and button_url:
        button_html = f"""
        <tr><td align="center" style="padding:8px 32px 30px;">
          <a href="{html_lib.escape(str(button_url), quote=True)}"
             style="display:inline-block;background:#6366f1;color:#fff;text-decoration:none;
                    font-family:Arial,Helvetica,sans-serif;font-size:16px;font-weight:700;
                    padding:14px 24px;border-radius:12px;">
            {html_lib.escape(str(button_text))}
          </a>
        </td></tr>"""

    note_html = f"""
        <tr><td style="padding:0 32px 28px;color:#9ca3af;font-family:Arial,Helvetica,sans-serif;
                      font-size:13px;line-height:20px;text-align:center;">{safe_note}</td></tr>
    """ if safe_note else ""

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title></head>
<body style="margin:0;padding:0;background:#0b0d13;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#0b0d13;">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"
       style="max-width:620px;background:#151926;border:1px solid #2d3348;border-radius:20px;overflow:hidden;">
<tr><td align="center" style="background:#000;padding:18px 20px 10px;">
<img src="{logo_url}" alt="SoundLens" width="420"
     style="display:block;width:100%;max-width:420px;height:auto;border:0;">
</td></tr>
<tr><td style="padding:34px 32px 14px;color:#fff;font-family:Arial,Helvetica,sans-serif;
               font-size:30px;font-weight:800;line-height:38px;text-align:center;">{safe_title}</td></tr>
<tr><td style="padding:0 32px 26px;color:#cbd5e1;font-family:Arial,Helvetica,sans-serif;
               font-size:16px;line-height:25px;text-align:center;">{safe_message}</td></tr>
{button_html}
{note_html}
<tr><td style="padding:24px 28px;background:#0f1117;border-top:1px solid #2d3348;
               color:#9ca3af;font-family:Arial,Helvetica,sans-serif;font-size:12px;
               line-height:19px;text-align:center;">
<strong style="color:#fff;">SoundLens</strong><br>
Music analysis built for artists and producers.<br>
<a href="{SOUNDLENS_PUBLIC_URL}" style="color:#8b8ffb;text-decoration:none;">soundlensapp.com</a>
<br><br>© 2026 SoundLens
</td></tr>
</table>
</td></tr></table>
</body></html>"""


def build_verification_link(token: str) -> str:
    return f"{SOUNDLENS_PUBLIC_URL}/auth/verify-email?token={token}"


def send_signup_verification_email(user: dict) -> bool:
    token = user.get("email_verification_token")
    if not token:
        return False

    verify_link = build_verification_link(token)
    subject = "Confirm your SoundLens email"
    body = f"""Welcome to SoundLens.

Confirm your email to finish setting up your account:

{verify_link}

If you did not create a SoundLens account, you can ignore this email.

SoundLens
soundlensapp.com
"""
    html_body = build_branded_email(
        title="Confirm your email",
        message="Verify your email address to finish setting up your SoundLens account.",
        button_text="Confirm Email",
        button_url=verify_link,
        note="If you did not create this account, you can safely ignore this email.",
    )
    return send_notification_email_to(user.get("email"), subject, body, html_body)


def is_admin_user(user: dict) -> bool:
    return normalize_email(user.get("email")) in SOUNDLENS_ADMIN_EMAILS


def get_admin_user(authorization: str | None) -> tuple[dict, dict]:
    db, user = get_current_user(authorization)
    if not is_admin_user(user):
        raise HTTPException(status_code=403, detail="Admin access only.")
    return db, user



def load_users_db() -> dict:
    if not USERS_DB_PATH.exists():
        return {"users": {}, "tokens": {}}

    try:
        data = json.loads(USERS_DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {"users": {}, "tokens": {}}

    data.setdefault("users", {})
    data.setdefault("tokens", {})
    return data


def save_users_db(data: dict) -> None:
    write_json_file(USERS_DB_PATH, data)


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def public_user(user: dict) -> dict:
    plan = user.get("plan", "free")
    usage = user.setdefault("usage", {})
    today = utc_today()
    if usage.get("date") != today:
        usage["date"] = today
        usage["count"] = 0

    return {
        "id": user.get("id"),
        "email": user.get("email"),
        "display_name": user.get("display_name") or user.get("email"),
        "plan": plan,
        "is_pro": plan in {"pro", "studio", "lifetime"},
        "daily_limit": None if plan in {"pro", "studio", "lifetime"} else FREE_DAILY_UPLOAD_LIMIT,
        "uploads_today": int(usage.get("count", 0) or 0),
        "uploads_remaining": None if plan in {"pro", "studio", "lifetime"} else max(0, FREE_DAILY_UPLOAD_LIMIT - int(usage.get("count", 0) or 0)),
        "created_at": user.get("created_at"),
        "email_verified": bool(user.get("email_verified")),
    }


def get_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None

    authorization = authorization.strip()
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()

    return authorization


def get_current_user(authorization: str | None) -> tuple[dict, dict]:
    db = load_users_db()
    token = get_bearer_token(authorization)
    user_id = db.get("tokens", {}).get(token)

    if not token or not user_id or user_id not in db.get("users", {}):
        raise HTTPException(status_code=401, detail="Please log in to use SoundLens.")

    return db, db["users"][user_id]


def check_and_increment_upload(user: dict) -> None:
    plan = user.get("plan", "free")
    usage = user.setdefault("usage", {})
    today = utc_today()

    if usage.get("date") != today:
        usage["date"] = today
        usage["count"] = 0

    if plan not in {"pro", "studio", "lifetime"} and int(usage.get("count", 0) or 0) >= FREE_DAILY_UPLOAD_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Free limit reached. You get {FREE_DAILY_UPLOAD_LIMIT} uploads per day. Upgrade to Pro for unlimited uploads.",
        )

    usage["count"] = int(usage.get("count", 0) or 0) + 1


def save_report_for_user(user: dict, report_dict: dict, text_report: str, original_filename: str) -> dict:
    user_id = user["id"]
    report_id = str(uuid.uuid4())
    user_dir = SAVED_REPORTS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "id": report_id,
        "user_id": user_id,
        "created_at": now_iso(),
        "original_filename": original_filename,
        "title": report_dict.get("basic", {}).get("file_name") or original_filename,
        "release_score": report_dict.get("scores", {}).get("release"),
        "mix_score": report_dict.get("scores", {}).get("mix"),
        "energy_score": report_dict.get("scores", {}).get("energy"),
        "arrangement_score": report_dict.get("scores", {}).get("arrangement"),
        "report": report_dict,
        "text_report": text_report,
    }

    (user_dir / f"{report_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def list_reports_for_user(user: dict) -> list[dict]:
    user_dir = SAVED_REPORTS_DIR / user["id"]
    if not user_dir.exists():
        return []

    reports = []
    for path in sorted(user_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        report_body = data.get("report", {}) or {}
        comparison = report_body.get("artist_comparison", {}) or {}
        closest_artist = None

        ranked = comparison.get("ranked_profiles") if isinstance(comparison, dict) else None
        if isinstance(ranked, list) and ranked:
            closest_artist = ranked[0].get("profile_name") or ranked[0].get("artist")

        if not closest_artist:
            closest_artist = report_body.get("closest_artist") or report_body.get("closest_style")

        reports.append({
            "id": data.get("id"),
            "created_at": data.get("created_at"),
            "title": data.get("title"),
            "original_filename": data.get("original_filename"),
            "release_score": data.get("release_score"),
            "mix_score": data.get("mix_score"),
            "energy_score": data.get("energy_score"),
            "arrangement_score": data.get("arrangement_score"),
            "closest_artist": closest_artist,
        })

    return reports


@app.post("/auth/signup")
def signup(payload: AuthPayload):
    db = load_users_db()
    email = normalize_email(payload.email)

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email.")

    if not payload.password or len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    existing = next((u for u in db["users"].values() if normalize_email(u.get("email")) == email), None)
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    user_id = str(uuid.uuid4())
    salt = str(uuid.uuid4())
    token = str(uuid.uuid4())

    user = {
        "id": user_id,
        "email": email,
        "display_name": payload.display_name or email.split("@")[0],
        "password_salt": salt,
        "password_hash": hash_password(payload.password, salt),
        "plan": "free",
        "usage": {"date": utc_today(), "count": 0},
        "created_at": now_iso(),
        "email_verified": False,
        "email_verification_token": str(uuid.uuid4()),
        "email_verified_at": None,
    }

    db["users"][user_id] = user
    db["tokens"][token] = user_id
    save_users_db(db)
    track_event("signup_success", user, {"email": email})
    send_signup_verification_email(user)

    return {"token": token, "user": public_user(user)}


@app.post("/auth/login")
def login(payload: AuthPayload):
    db = load_users_db()
    email = normalize_email(payload.email)

    user = next((u for u in db["users"].values() if normalize_email(u.get("email")) == email), None)
    if not user:
        track_event("login_failed", None, {"email": email, "reason": "unknown_email"})
        raise HTTPException(status_code=401, detail="Email or password is wrong.")

    expected = hash_password(payload.password, user.get("password_salt", ""))
    if expected != user.get("password_hash"):
        track_event("login_failed", user, {"email": email, "reason": "wrong_password"})
        raise HTTPException(status_code=401, detail="Email or password is wrong.")

    token = str(uuid.uuid4())
    db["tokens"][token] = user["id"]
    user["last_login_at"] = now_iso()
    user["last_active_at"] = user["last_login_at"]
    save_users_db(db)
    track_event("login_success", user, {})

    return {"token": token, "user": public_user(user)}


@app.get("/auth/me")
def me(authorization: str | None = Header(default=None)):
    db, user = get_current_user(authorization)
    # Save in case usage reset was applied.
    save_users_db(db)
    return {"user": public_user(user)}


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    db = load_users_db()
    token = get_bearer_token(authorization)
    if token and token in db.get("tokens", {}):
        user_id = db["tokens"].get(token)
        user = db.get("users", {}).get(user_id)
        if user:
            track_event("logout", user, {})
        del db["tokens"][token]
        save_users_db(db)
    return {"ok": True}


@app.post("/auth/forgot-password")
def forgot_password(payload: PasswordResetRequestPayload):
    db = load_users_db()
    email = normalize_email(payload.email)
    user = find_user_by_email(db, email)
    # Always return the same response to avoid exposing which emails are registered.
    generic = {"ok": True, "message": "If an account exists for that email, a reset link has been sent."}
    if not user:
        track_event("password_reset_requested", None, {"email": email, "account_found": False})
        return generic
    token = secrets.token_urlsafe(32)
    user["password_reset_token"] = token
    user["password_reset_expires_at"] = (datetime.now(timezone.utc) + timedelta(minutes=PASSWORD_RESET_EXPIRY_MINUTES)).isoformat()
    save_users_db(db)
    link = f"{SOUNDLENS_PUBLIC_URL}/?reset_token={token}"
    reset_subject = "Reset your SoundLens password"
    reset_body = f"""A password reset was requested for your SoundLens account.

Reset your password here:
{link}

This link expires in {PASSWORD_RESET_EXPIRY_MINUTES} minutes. If you did not request this, you can ignore this email.

SoundLens
soundlensapp.com
"""
    reset_html = build_branded_email(
        title="Reset your password",
        message="We received a request to reset the password for your SoundLens account.",
        button_text="Reset Password",
        button_url=link,
        note=f"This link expires in {PASSWORD_RESET_EXPIRY_MINUTES} minutes. If you did not request this, you can ignore this email.",
    )
    sent = send_notification_email_to(user.get("email"), reset_subject, reset_body, reset_html)
    track_event("password_reset_requested", user, {"sent": sent})
    return generic


@app.post("/auth/reset-password")
def reset_password(payload: PasswordResetConfirmPayload):
    if not payload.password or len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    token = str(payload.token or "").strip()
    db = load_users_db()
    user = next((u for u in db.get("users", {}).values() if u.get("password_reset_token") == token), None)
    if not user:
        raise HTTPException(status_code=400, detail="This reset link is invalid or expired.")
    expires = parse_iso(user.get("password_reset_expires_at"))
    if not expires or expires < datetime.now(timezone.utc):
        user["password_reset_token"] = None
        user["password_reset_expires_at"] = None
        save_users_db(db)
        raise HTTPException(status_code=400, detail="This reset link is invalid or expired.")
    salt = str(uuid.uuid4())
    user["password_salt"] = salt
    user["password_hash"] = hash_password(payload.password, salt)
    user["password_reset_token"] = None
    user["password_reset_expires_at"] = None
    user["password_changed_at"] = now_iso()
    # Invalidate every existing session for this account.
    db["tokens"] = {k:v for k,v in db.get("tokens", {}).items() if v != user.get("id")}
    save_users_db(db)
    track_event("password_reset_completed", user, {})
    return {"ok": True, "message": "Password updated. Log in with your new password."}


@app.post("/auth/change-password")
def change_password(payload: ChangePasswordPayload, authorization: str | None = Header(default=None)):
    db, user = get_current_user(authorization)
    current_password = str(payload.current_password or "")
    new_password = str(payload.new_password or "")

    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
    if current_password == new_password:
        raise HTTPException(status_code=400, detail="Choose a new password that is different from your current password.")

    expected = hash_password(current_password, user.get("password_salt", ""))
    if expected != user.get("password_hash"):
        track_event("password_change_failed", user, {"reason": "wrong_current_password"})
        raise HTTPException(status_code=401, detail="Current password is incorrect.")

    salt = str(uuid.uuid4())
    user["password_salt"] = salt
    user["password_hash"] = hash_password(new_password, salt)
    user["password_reset_token"] = None
    user["password_reset_expires_at"] = None
    user["password_changed_at"] = now_iso()
    user.pop("password_changed_by_admin", None)

    # Keep this browser signed in, but invalidate every other session for the account.
    current_token = get_bearer_token(authorization)
    user_id = user.get("id")
    db["tokens"] = {
        token: uid for token, uid in db.get("tokens", {}).items()
        if uid != user_id or token == current_token
    }
    save_users_db(db)
    track_event("password_changed", user, {})
    return {"ok": True, "message": "Password changed successfully.", "user": public_user(user)}


@app.delete("/account")
def delete_account(payload: DeleteAccountPayload, authorization: str | None = Header(default=None)):
    db, user = get_current_user(authorization)
    expected = hash_password(payload.password, user.get("password_salt", ""))
    if expected != user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Password is incorrect.")
    user_id = user.get("id")
    email = user.get("email")
    # Backup first, then remove only this account's records.
    create_data_backup()
    user_dir = SAVED_REPORTS_DIR / str(user_id)
    if user_dir.exists():
        shutil.rmtree(user_dir)
    db["tokens"] = {k:v for k,v in db.get("tokens", {}).items() if v != user_id}
    db.get("users", {}).pop(user_id, None)
    save_users_db(db)
    track_event("account_deleted", None, {"user_id": user_id, "email": email})
    return {"ok": True, "message": "Your SoundLens account and saved reports were deleted."}


@app.post("/contact")
def contact(payload: ContactPayload, authorization: str | None = Header(default=None)):
    user = None
    try:
        _, user = get_current_user(authorization)
    except Exception:
        pass
    email = normalize_email(payload.email)
    message = str(payload.message or "").strip()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email.")
    if len(message) < 10:
        raise HTTPException(status_code=400, detail="Please include a little more detail in your message.")
    item = {"id": str(uuid.uuid4()), "created_at": now_iso(), "name": str(payload.name or "").strip()[:120], "email": email, "subject": str(payload.subject or "General question").strip()[:160], "message": message[:5000], "user_id": user.get("id") if user else None}
    append_json_list(CONTACT_MESSAGES_PATH, item)
    sent = send_notification_email("New SoundLens contact message", f"Name: {item['name']}\nEmail: {email}\nSubject: {item['subject']}\n\n{item['message']}")
    track_event("contact_submitted", user, {"email": email, "subject": item["subject"], "notification_sent": sent})
    return {"ok": True, "message": "Your message was sent. SoundLens will reply by email."}


@app.get("/billing/config")
def billing_config():
    return {
        "free": {
            "name": "Free",
            "daily_upload_limit": FREE_DAILY_UPLOAD_LIMIT,
            "features": ["3 uploads per day", "Basic artist match", "Basic analysis", "Basic fixes"],
        },
        "pro": {
            "name": "Pro",
            "price_cad": PRO_PRICE_CAD,
            "features": ["Unlimited uploads", "Full artist match", "Saved reports", "Upload history", "Future AI features"],
            "stripe_enabled": bool(os.getenv("STRIPE_SECRET_KEY")),
        },
    }


@app.post("/billing/upgrade-demo")
def upgrade_demo(authorization: str | None = Header(default=None)):
    db, user = get_current_user(authorization)
    user["plan"] = "pro"
    user["upgraded_at"] = now_iso()
    save_users_db(db)
    return {"user": public_user(user), "message": "Local demo upgrade enabled. Replace this with Stripe before public launch."}


@app.get("/reports")
def my_reports(authorization: str | None = Header(default=None)):
    db, user = get_current_user(authorization)
    return {"reports": list_reports_for_user(user), "user": public_user(user)}


@app.get("/reports/{report_id}")
def get_report(report_id: str, authorization: str | None = Header(default=None)):
    db, user = get_current_user(authorization)
    path = SAVED_REPORTS_DIR / user["id"] / f"{report_id}.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")

    return json.loads(path.read_text(encoding="utf-8"))


@app.delete("/reports/{report_id}")
def delete_report(report_id: str, authorization: str | None = Header(default=None)):
    db, user = get_current_user(authorization)
    path = SAVED_REPORTS_DIR / user["id"] / f"{report_id}.json"

    if path.exists():
        path.unlink()
        track_event("report_deleted", user, {"report_id": report_id})

    return {"ok": True}



def find_user_by_email(db: dict, email: str) -> dict | None:
    clean_email = normalize_email(email)
    return next((u for u in db.get("users", {}).values() if normalize_email(u.get("email")) == clean_email), None)


def set_user_plan_by_email(email: str, plan: str, stripe_customer_id: str | None = None, stripe_subscription_id: str | None = None) -> bool:
    db = load_users_db()
    user = find_user_by_email(db, email)
    if not user:
        return False
    user["plan"] = plan
    if plan in {"pro", "studio"}:
        user["upgraded_at"] = now_iso()
    user["stripe_customer_id"] = stripe_customer_id or user.get("stripe_customer_id")
    user["stripe_subscription_id"] = stripe_subscription_id or user.get("stripe_subscription_id")
    save_users_db(db)
    return True


@app.post("/stripe/create-checkout-session")
def create_checkout_session(payload: dict, authorization: str | None = Header(default=None)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe is not configured yet.")

    db, user = get_current_user(authorization)
    requested_plan = str(payload.get("plan", "pro")).lower().strip()

    if requested_plan == "studio":
        price_id = STRIPE_STUDIO_PRICE_ID
        plan_name = "studio"
    else:
        price_id = STRIPE_PRO_PRICE_ID
        plan_name = "pro"

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=user.get("email"),
            client_reference_id=user.get("id"),
            success_url=f"{SOUNDLENS_PUBLIC_URL}/?checkout=success&plan={plan_name}",
            cancel_url=f"{SOUNDLENS_PUBLIC_URL}/?checkout=cancelled",
            metadata={"user_id": user.get("id"), "email": user.get("email"), "plan": plan_name},
            subscription_data={"metadata": {"user_id": user.get("id"), "email": user.get("email"), "plan": plan_name}},
        )
        return {"checkout_url": session.url}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Stripe checkout failed: {error}")


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe is not configured yet.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature")

    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(payload, signature, STRIPE_WEBHOOK_SECRET)
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Webhook verification failed: {error}")
    else:
        try:
            event = json.loads(payload.decode("utf-8"))
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Webhook parse failed: {error}")

    event_type = event.get("type")
    obj = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        email = obj.get("customer_email") or obj.get("metadata", {}).get("email")
        plan = obj.get("metadata", {}).get("plan", "pro")
        if email:
            set_user_plan_by_email(email, plan, obj.get("customer"), obj.get("subscription"))

    elif event_type in {"customer.subscription.deleted", "customer.subscription.paused"}:
        customer_id = obj.get("customer")
        db = load_users_db()
        changed = False
        for user in db.get("users", {}).values():
            if user.get("stripe_customer_id") == customer_id:
                user["plan"] = "free"
                user["downgraded_at"] = now_iso()
                changed = True
        if changed:
            save_users_db(db)

    return {"received": True}


@app.get("/stripe/success")
def stripe_success():
    return RedirectResponse(url="/?checkout=success")


@app.get("/stripe/cancel")
def stripe_cancel():
    return RedirectResponse(url="/?checkout=cancelled")




@app.get("/auth/verify-email")
async def verify_email(token: str):
    db = load_users_db()
    token = str(token or "").strip()

    if not token:
        raise HTTPException(status_code=400, detail="Missing verification token.")

    for user in db.get("users", {}).values():
        if user.get("email_verification_token") == token:
            user["email_verified"] = True
            user["email_verified_at"] = now_iso()
            user["email_verification_token"] = None
            save_users_db(db)
            track_event("email_verified", user, {"source": "verification_link"})
            return RedirectResponse(url=f"{SOUNDLENS_PUBLIC_URL}/?verified=1", status_code=303)

    raise HTTPException(status_code=400, detail="Invalid or expired verification link.")


@app.post("/auth/resend-verification")
async def resend_verification(payload: ResendVerificationPayload):
    db = load_users_db()
    email = normalize_email(payload.email)
    user = None

    for candidate in db.get("users", {}).values():
        if normalize_email(candidate.get("email")) == email:
            user = candidate
            break

    if not user:
        raise HTTPException(status_code=404, detail="Account not found.")

    if user.get("email_verified"):
        return {"ok": True, "message": "Email is already verified."}

    user["email_verification_token"] = str(uuid.uuid4())
    save_users_db(db)

    sent = send_signup_verification_email(user)
    track_event("verification_resent", user, {"sent": sent})

    if not sent:
        return {"ok": False, "message": "Verification link was created, but SMTP is not configured yet."}

    return {"ok": True, "message": "Verification email sent."}


@app.post("/feedback")
async def submit_feedback(payload: FeedbackPayload, authorization: str | None = Header(default=None)):
    user = None
    try:
        _, user = get_current_user(authorization)
    except Exception:
        user = None

    message = str(payload.message or "").strip()
    if len(message) < 2:
        raise HTTPException(status_code=400, detail="Please enter feedback before submitting.")

    item = {
        "id": str(uuid.uuid4()),
        "created_at": now_iso(),
        "rating": payload.rating,
        "accuracy": payload.accuracy,
        "message": message,
        "email": normalize_email(payload.email) if payload.email else (user.get("email") if user else None),
        "report_id": payload.report_id,
        "page": payload.page,
        "user": {
            "id": user.get("id"),
            "email": user.get("email"),
            "display_name": user.get("display_name"),
            "plan": user.get("plan", "free"),
        } if user else None,
    }

    append_json_list(FEEDBACK_PATH, item)
    track_event("feedback_submitted", user, {"rating": payload.rating, "accuracy": payload.accuracy, "page": payload.page})

    send_notification_email(
        "New SoundLens feedback",
        f"New feedback\n\nRating: {payload.rating}\nAccuracy: {payload.accuracy}\nEmail: {item.get('email')}\nPage: {payload.page}\nReport ID: {payload.report_id}\n\nMessage:\n{message}"
    )

    return {"ok": True, "message": "Feedback sent. Thank you."}


@app.post("/track-event")
async def track_client_event(payload: EventPayload, authorization: str | None = Header(default=None)):
    user = None
    try:
        _, user = get_current_user(authorization)
    except Exception:
        user = None

    event_name = str(payload.event or "").strip()[:80]
    if not event_name:
        raise HTTPException(status_code=400, detail="Missing event.")

    track_event(event_name, user, {"page": payload.page, "details": payload.details or {}})
    return {"ok": True}


@app.get("/admin/stats")
async def admin_stats(authorization: str | None = Header(default=None)):
    db, admin = get_admin_user(authorization)
    users = list((db.get("users") or {}).values())
    events = read_json_file(ADMIN_EVENTS_PATH, [])
    feedback = read_json_file(FEEDBACK_PATH, [])
    if not isinstance(events, list): events = []
    if not isinstance(feedback, list): feedback = []

    today = utc_today()
    pro_users = [u for u in users if u.get("plan") in {"pro", "studio"}]
    lifetime_users = [u for u in users if u.get("plan") == "lifetime"]
    uploads_today = 0
    reports_saved = 0
    active_today_ids = set()
    last_active_by_id = {}

    for event in events:
        uid = (event.get("user") or {}).get("id")
        created = str(event.get("created_at") or "")
        if uid and created:
            if created > last_active_by_id.get(uid, ""):
                last_active_by_id[uid] = created
            if created.startswith(today):
                active_today_ids.add(uid)

    user_rows = []
    for user in users:
        usage = user.get("usage", {}) or {}
        if usage.get("date") == today:
            uploads_today += int(usage.get("count", 0) or 0)
        user_dir = SAVED_REPORTS_DIR / str(user.get("id"))
        report_count = len(list(user_dir.glob("*.json"))) if user_dir.exists() else 0
        reports_saved += report_count
        row = public_user(user)
        row.update({
            "last_login_at": user.get("last_login_at"),
            "last_active": last_active_by_id.get(user.get("id")) or user.get("last_active_at") or user.get("last_login_at"),
            "total_uploads": int(user.get("total_uploads", 0) or 0),
            "reports_saved": report_count,
        })
        user_rows.append(row)

    user_rows.sort(key=lambda u: u.get("last_active") or u.get("created_at") or "", reverse=True)
    latest_feedback = sorted(feedback, key=lambda f: f.get("created_at", ""), reverse=True)[:20]
    latest_events = sorted(events, key=lambda e: e.get("created_at", ""), reverse=True)[:250]

    page_views = [e for e in events if e.get("event") == "page_view"]
    click_events = [e for e in events if e.get("event") == "ui_click"]
    completed = len([e for e in events if e.get("event") == "analysis_completed"])
    started = len([e for e in events if e.get("event") == "analysis_started"])
    returning_ids = set()
    seen_days = {}
    for event in events:
        uid = (event.get("user") or {}).get("id")
        day = str(event.get("created_at") or "")[:10]
        if uid and day:
            seen_days.setdefault(uid, set()).add(day)
    returning_ids = {uid for uid, days in seen_days.items() if len(days) > 1}
    page_counts = {}
    click_counts = {}
    for event in page_views:
        details = event.get("details") or {}
        page = details.get("page") or (details.get("details") or {}).get("page") or "unknown"
        page_counts[page] = page_counts.get(page, 0) + 1
    for event in click_events:
        detail = ((event.get("details") or {}).get("details") or {})
        label = detail.get("label") or detail.get("id") or detail.get("tag") or "unknown"
        click_counts[label] = click_counts.get(label, 0) + 1
    top_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_clicks = sorted(click_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Real click heatmap points. Coordinates are normalized to the active
    # SoundLens page, allowing desktop/mobile data to be viewed together.
    # No screenshots, passwords, form values, or uploaded audio are stored.
    heatmap_points = []
    for event in reversed(click_events):
        detail = ((event.get("details") or {}).get("details") or {})
        try:
            x_percent = float(detail.get("x_percent"))
            y_percent = float(detail.get("y_percent"))
        except (TypeError, ValueError):
            continue

        if not (0 <= x_percent <= 100 and 0 <= y_percent <= 100):
            continue

        heatmap_points.append({
            "x": round(x_percent, 3),
            "y": round(y_percent, 3),
            "page": str(detail.get("page") or "unknown")[:80],
            "device": str(detail.get("device") or "unknown")[:20],
            "target": str(
                detail.get("label")
                or detail.get("id")
                or detail.get("tag")
                or "Click"
            )[:80],
            "created_at": event.get("created_at"),
        })

        # Keep the admin payload responsive even after heavy usage.
        if len(heatmap_points) >= 5000:
            break

    heatmap_points.reverse()

    heatmap_page_counts = {}
    for point in heatmap_points:
        page = point["page"]
        heatmap_page_counts[page] = heatmap_page_counts.get(page, 0) + 1

    return {
        "ok": True,
        "admin": public_user(admin),
        "stats": {
            "total_users": len(users),
            "pro_users": len(pro_users),
            "lifetime_users": len(lifetime_users),
            "free_users": len([u for u in users if u.get("plan", "free") == "free"]),
            "active_today": len(active_today_ids),
            "uploads_today": uploads_today,
            "reports_saved": reports_saved,
            "feedback_count": len(feedback),
            "events_tracked": len(events),
            "returning_users": len(returning_ids),
            "page_views": len(page_views),
            "clicks_tracked": len(click_events),
            "analysis_completion_rate": round((completed / started * 100), 1) if started else 0,
        },
        "users": user_rows,
        "latest_feedback": latest_feedback,
        "latest_events": latest_events,
        "top_pages": [{"name": name, "count": count} for name, count in top_pages],
        "top_clicks": [{"name": name, "count": count} for name, count in top_clicks],
        "heatmap_points": heatmap_points,
        "heatmap_pages": [
            {"name": name, "count": count}
            for name, count in sorted(
                heatmap_page_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
    }


@app.get("/admin/export-data")
def admin_export_data(authorization: str | None = Header(default=None)):
    _, admin = get_admin_user(authorization)
    backup = create_data_backup()
    track_event("admin_data_exported", admin, {"filename": backup.name})
    return FileResponse(backup, filename=backup.name, media_type="application/zip")


@app.post("/admin/lifetime-account")
def create_lifetime_account(payload: AdminLifetimeAccountPayload, authorization: str | None = Header(default=None)):
    db, admin = get_admin_user(authorization)
    email = normalize_email(payload.email)
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email.")
    if not payload.password or len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Temporary password must be at least 6 characters.")
    if find_user_by_email(db, email):
        raise HTTPException(status_code=400, detail="That account already exists. Use Grant Existing User Lifetime instead.")

    user_id = str(uuid.uuid4())
    salt = str(uuid.uuid4())
    user = {
        "id": user_id,
        "email": email,
        "display_name": payload.display_name or email.split("@")[0],
        "password_salt": salt,
        "password_hash": hash_password(payload.password, salt),
        "plan": "lifetime",
        "usage": {"date": utc_today(), "count": 0},
        "total_uploads": 0,
        "created_at": now_iso(),
        "email_verified": True,
        "email_verification_token": None,
        "email_verified_at": now_iso(),
        "lifetime_granted_at": now_iso(),
        "lifetime_granted_by": admin.get("email"),
    }
    db["users"][user_id] = user
    save_users_db(db)
    track_event("lifetime_account_created", user, {"admin_email": admin.get("email")})
    return {"ok": True, "message": f"Lifetime account created for {email}.", "user": public_user(user)}


@app.post("/admin/grant-lifetime")
def grant_lifetime(payload: AdminEmailPayload, authorization: str | None = Header(default=None)):
    db, admin = get_admin_user(authorization)
    user = find_user_by_email(db, payload.email)
    if not user:
        raise HTTPException(status_code=404, detail="Account not found.")
    user["plan"] = "lifetime"
    user["lifetime_granted_at"] = now_iso()
    user["lifetime_granted_by"] = admin.get("email")
    save_users_db(db)
    track_event("lifetime_access_granted", user, {"admin_email": admin.get("email")})
    return {"ok": True, "message": f"Lifetime access granted to {user.get('email')}.", "user": public_user(user)}



@app.post("/admin/pro-account")
def create_pro_account(payload: AdminLifetimeAccountPayload, authorization: str | None = Header(default=None)):
    db, admin = get_admin_user(authorization)
    email = normalize_email(payload.email)
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email.")
    if not payload.password or len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Temporary password must be at least 6 characters.")
    if find_user_by_email(db, email):
        raise HTTPException(status_code=400, detail="That account already exists. Use Grant Pro instead.")

    user_id = str(uuid.uuid4())
    salt = str(uuid.uuid4())
    user = {
        "id": user_id,
        "email": email,
        "display_name": payload.display_name or email.split("@")[0],
        "password_salt": salt,
        "password_hash": hash_password(payload.password, salt),
        "plan": "pro",
        "usage": {"date": utc_today(), "count": 0},
        "total_uploads": 0,
        "created_at": now_iso(),
        "email_verified": True,
        "email_verification_token": None,
        "email_verified_at": now_iso(),
        "admin_pro_granted": True,
        "admin_pro_granted_at": now_iso(),
        "admin_pro_granted_by": admin.get("email"),
        "account_restored_by_admin": True,
    }
    db.setdefault("users", {})[user_id] = user
    save_users_db(db)
    track_event("pro_account_created", user, {"admin_email": admin.get("email"), "restored": True})
    return {"ok": True, "message": f"Pro account created/restored for {email}.", "user": public_user(user)}

@app.post("/admin/grant-pro")
def grant_pro(payload: AdminEmailPayload, authorization: str | None = Header(default=None)):
    db, admin = get_admin_user(authorization)
    email = normalize_email(payload.email)
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email.")

    user = find_user_by_email(db, email)
    created = False
    if not user:
        # Recovery path: allow an admin to restore Pro using only the customer's email.
        # We cannot recover their old password, so create a secure random unusable
        # credential; the customer can use the existing Forgot Password flow to set one.
        user_id = str(uuid.uuid4())
        salt = str(uuid.uuid4())
        recovery_secret = str(uuid.uuid4()) + str(uuid.uuid4())
        user = {
            "id": user_id,
            "email": email,
            "display_name": email.split("@")[0],
            "password_salt": salt,
            "password_hash": hash_password(recovery_secret, salt),
            "plan": "pro",
            "usage": {"date": utc_today(), "count": 0},
            "total_uploads": 0,
            "created_at": now_iso(),
            "email_verified": True,
            "email_verification_token": None,
            "email_verified_at": now_iso(),
            "account_restored_by_admin": True,
        }
        db.setdefault("users", {})[user_id] = user
        created = True
    elif user.get("plan") == "lifetime":
        raise HTTPException(status_code=400, detail="This account already has Lifetime access.")

    user["plan"] = "pro"
    user["admin_pro_granted"] = True
    user["admin_pro_granted_at"] = now_iso()
    user["admin_pro_granted_by"] = admin.get("email")
    save_users_db(db)
    track_event("pro_account_restored" if created else "pro_access_granted", user, {"admin_email": admin.get("email")})

    if created:
        return {"ok": True, "message": f"Pro restored for {email}. The user can use Forgot Password to set a new password.", "user": public_user(user)}
    return {"ok": True, "message": f"Pro access granted to {email}.", "user": public_user(user)}


@app.post("/admin/set-password")
def admin_set_password(payload: AdminPasswordPayload, authorization: str | None = Header(default=None)):
    db, admin = get_admin_user(authorization)
    email = normalize_email(payload.email)
    password = str(payload.password or "")

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Temporary password must be at least 8 characters.")

    user = find_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="Account not found. Grant Pro first if this is a recovered customer account.")

    salt = str(uuid.uuid4())
    user["password_salt"] = salt
    user["password_hash"] = hash_password(password, salt)
    user["password_reset_token"] = None
    user["password_reset_expires_at"] = None
    user["password_changed_at"] = now_iso()
    user["password_changed_by_admin"] = admin.get("email")

    # Sign the account out everywhere so only the new password can be used.
    user_id = user.get("id")
    db["tokens"] = {k: v for k, v in db.get("tokens", {}).items() if v != user_id}
    save_users_db(db)
    track_event("admin_password_changed", user, {"admin_email": admin.get("email")})

    return {
        "ok": True,
        "message": f"Password updated for {email}. Their plan remains {user.get('plan', 'free').title()}.",
        "user": public_user(user),
    }


@app.post("/admin/remove-pro")
def remove_pro(payload: AdminEmailPayload, authorization: str | None = Header(default=None)):
    db, admin = get_admin_user(authorization)
    user = find_user_by_email(db, payload.email)
    if not user:
        raise HTTPException(status_code=404, detail="Account not found.")
    if user.get("plan") == "lifetime":
        raise HTTPException(status_code=400, detail="This account has Lifetime access. Pro removal does not change Lifetime accounts.")
    if user.get("plan") != "pro":
        raise HTTPException(status_code=400, detail="This account is not currently on Pro.")

    # Do not silently override an active Stripe-backed subscription. Manual Pro
    # can be removed here; paid Pro must be cancelled through Stripe first.
    if user.get("stripe_subscription_id") and not user.get("admin_pro_granted"):
        raise HTTPException(status_code=400, detail="This Pro account is tied to Stripe. Cancel the Stripe subscription before removing Pro here.")

    if user.get("stripe_subscription_id") and user.get("admin_pro_granted"):
        user.pop("admin_pro_granted", None)
        user.pop("admin_pro_granted_at", None)
        user.pop("admin_pro_granted_by", None)
        save_users_db(db)
        track_event("admin_pro_grant_removed", user, {"admin_email": admin.get("email"), "stripe_still_active": True})
        return {"ok": True, "message": f"Manual Pro grant removed for {user.get('email')}, but the Stripe subscription still keeps Pro active.", "user": public_user(user)}

    user["plan"] = "free"
    user["admin_pro_granted"] = False
    user["admin_pro_removed_at"] = now_iso()
    user["admin_pro_removed_by"] = admin.get("email")
    user.pop("admin_pro_granted_at", None)
    user.pop("admin_pro_granted_by", None)
    save_users_db(db)
    track_event("pro_access_removed", user, {"admin_email": admin.get("email")})
    return {"ok": True, "message": f"Pro access removed from {user.get('email')}. Account is now Free.", "user": public_user(user)}


@app.get("/")
def home():
    return FileResponse("index.html")


def save_upload(file: UploadFile) -> Path:
    safe_name = Path(file.filename or "uploaded_audio.wav").name
    file_path = UPLOADS_DIR / safe_name

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path


def generate_ai_feedback(report_dict):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return {
            "top_problems": report_dict.get("top_problems", []),
            "next_steps": report_dict.get("next_steps", []),
            "suggested_direction": [
                "AI feedback is off because OPENAI_API_KEY is not set."
            ],
        }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = f"""
You are SoundLens, a producer-focused music analysis assistant.

Use ONLY the analysis data below. Do not invent facts.
Music is subjective, so do not say the song is bad.

Return ONLY valid JSON. No markdown. No explanation.

Use this exact JSON shape:
{{
  "top_problems": [
    "short problem 1",
    "short problem 2",
    "short problem 3",
    "short problem 4",
    "short problem 5"
  ],
  "next_steps": [
    "practical fix 1",
    "practical fix 2",
    "practical fix 3",
    "practical fix 4"
  ],
  "suggested_direction": [
    "style direction 1",
    "style direction 2",
    "style direction 3"
  ]
}}

Focus on underground rap, rage, trap, 808s, vocals, melody, bounce, clipping, mix space, and stem balance.
If stem_balance exists, prioritize vocal-to-beat balance, bass masking, and melody presence.

Analysis data:
{json.dumps(report_dict, indent=2)}
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.7,
        )

        return json.loads(response.output_text)

    except Exception as error:
        print(f"AI feedback failed: {error}")

        return {
            "top_problems": report_dict.get("top_problems", []),
            "next_steps": report_dict.get("next_steps", []),
            "suggested_direction": [
                "AI feedback failed. Using standard SoundLens feedback."
            ],
        }


@app.post("/analyze")
def analyze(stems: bool = None, file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    try:
        db, user = get_current_user(authorization)
        check_and_increment_upload(user)
        user["last_active_at"] = now_iso()
        user["total_uploads"] = int(user.get("total_uploads", 0) or 0) + 1
        save_users_db(db)
        track_event("analysis_started", user, {"filename": file.filename})

        file_path = save_upload(file)

        report = analyze_audio(
            file_path,
            use_stems=USE_DEMUCS_BY_DEFAULT if stems is None else stems,
            demucs_output_dir=STEMS_DIR,
        )

        report_dict = asdict(report)

        artist_comparison = {}
        try:
            artist_comparison = compare_audio_to_profiles(
                audio_file=file_path,
                profiles_folder=str(PROFILES_DIR),
                top_n=10,
                include_report=False,
                use_stems=COMPARE_USE_STEMS,
                demucs_output_dir=str(STEMS_DIR),
            )
            if "style_suggestions" not in artist_comparison:
                artist_comparison["style_suggestions"] = []
            report_dict["artist_comparison"] = artist_comparison
        except Exception as compare_error:
            print("COMPARE ERROR INSIDE ANALYZE:")
            traceback.print_exc()
            artist_comparison = {
                "error": str(compare_error),
                "verdict": f"Profile comparison failed: {compare_error}",
                "ranked_profiles": [],
                "style_suggestions": [],
            }
            report_dict["artist_comparison"] = artist_comparison

        ai_feedback = generate_soundlens_ai_feedback(report_dict)

        report_dict["top_problems"] = ai_feedback.get(
            "top_problems",
            report_dict.get("top_problems", []),
        )

        report_dict["next_steps"] = ai_feedback.get(
            "next_steps",
            report_dict.get("next_steps", []),
        )

        report_dict["ai_suggested_direction"] = ai_feedback.get(
            "suggested_direction",
            [],
        )

        report_dict["ai_review"] = ai_feedback.get("ai_review", {})
        report_dict["ai_enabled"] = bool(ai_feedback.get("ai_enabled", False))
        report_dict["ai_model"] = ai_feedback.get("model")

        text_report = render_report(report)
        track_event("analysis_completed", user, {"filename": file.filename, "release_score": report_dict.get("scores", {}).get("release")})
        saved_report = save_report_for_user(
            user=user,
            report_dict=report_dict,
            text_report=text_report,
            original_filename=file.filename or file_path.name,
        )

        return {
            "report": report_dict,
            "text_report": text_report,
            "ai_feedback": ai_feedback,
            "artist_comparison": artist_comparison,
            "saved_report": {
                "id": saved_report["id"],
                "created_at": saved_report["created_at"],
                "title": saved_report["title"],
            },
            "user": public_user(user),
        }

    except Exception as error:
        print("ANALYZE ERROR:")
        traceback.print_exc()
        try:
            track_event("analysis_failed", user if "user" in locals() else None, {"error": str(error), "filename": getattr(file, "filename", None)})
        except Exception:
            pass

        return {
            "error": str(error),
            "report": None,
            "text_report": "",
            "ai_feedback": {},
        }


@app.post("/compare-profile")
def compare_profile(file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    try:
        db, user = get_current_user(authorization)
        file_path = save_upload(file)

        result = compare_audio_to_profiles(
            audio_file=file_path,
            profiles_folder=str(PROFILES_DIR),
            top_n=10,
            include_report=False,
            use_stems=COMPARE_USE_STEMS,
            demucs_output_dir=str(STEMS_DIR),
        )

        if "style_suggestions" not in result:
            result["style_suggestions"] = []

        return result

    except Exception as error:
        print("COMPARE ERROR:")
        traceback.print_exc()

        return {
            "verdict": f"Profile comparison failed: {error}",
            "ranked_profiles": [],
            "style_suggestions": [],
        }