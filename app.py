from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import traceback
import uuid
import smtplib
from email.message import EmailMessage
import stripe

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from soundlens_pro import analyze_audio, render_report
from compare_to_profile_pro import compare_audio_to_profiles

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
USERS_DB_PATH = Path("soundlens_users.json")
SAVED_REPORTS_DIR = Path("saved_reports")
SAVED_REPORTS_DIR.mkdir(exist_ok=True)

ADMIN_EVENTS_PATH = Path("soundlens_admin_events.json")
FEEDBACK_PATH = Path("soundlens_feedback.json")

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



def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_file(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def write_json_file(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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



def send_notification_email_to(to_email: str | None, subject: str, body: str) -> bool:
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

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        return True
    except Exception as error:
        track_event("email_failed", None, {"to": to_email, "subject": subject, "error": str(error)})
        return False


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
@soundlensapp_
"""
    return send_notification_email_to(user.get("email"), subject, body)


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
    USERS_DB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
        "is_pro": plan in {"pro", "studio"},
        "daily_limit": None if plan in {"pro", "studio"} else FREE_DAILY_UPLOAD_LIMIT,
        "uploads_today": int(usage.get("count", 0) or 0),
        "uploads_remaining": None if plan in {"pro", "studio"} else max(0, FREE_DAILY_UPLOAD_LIMIT - int(usage.get("count", 0) or 0)),
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

    if plan not in {"pro", "studio"} and int(usage.get("count", 0) or 0) >= FREE_DAILY_UPLOAD_LIMIT:
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
        "master_score": report_dict.get("scores", {}).get("master"),
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
            "master_score": data.get("master_score"),
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

    return {"token": token, "user": public_user(user)}


@app.post("/auth/login")
def login(payload: AuthPayload):
    db = load_users_db()
    email = normalize_email(payload.email)

    user = next((u for u in db["users"].values() if normalize_email(u.get("email")) == email), None)
    if not user:
        raise HTTPException(status_code=401, detail="Email or password is wrong.")

    expected = hash_password(payload.password, user.get("password_salt", ""))
    if expected != user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Email or password is wrong.")

    token = str(uuid.uuid4())
    db["tokens"][token] = user["id"]
    save_users_db(db)

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
        del db["tokens"][token]
        save_users_db(db)
    return {"ok": True}


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
    if not isinstance(events, list):
        events = []
    if not isinstance(feedback, list):
        feedback = []

    today = utc_today()
    pro_users = [u for u in users if u.get("plan") in {"pro", "studio"}]
    uploads_today = 0
    for u in users:
        usage = u.get("usage", {}) or {}
        if usage.get("date") == today:
            uploads_today += int(usage.get("count", 0) or 0)

    event_counts = {}
    for event in events:
        name = event.get("event", "unknown")
        event_counts[name] = event_counts.get(name, 0) + 1

    latest_users = sorted(users, key=lambda u: u.get("created_at", ""), reverse=True)[:10]
    latest_feedback = sorted(feedback, key=lambda f: f.get("created_at", ""), reverse=True)[:10]
    latest_events = sorted(events, key=lambda e: e.get("created_at", ""), reverse=True)[:25]

    return {
        "ok": True,
        "admin": public_user(admin),
        "stats": {
            "total_users": len(users),
            "pro_users": len(pro_users),
            "free_users": max(0, len(users) - len(pro_users)),
            "uploads_today": uploads_today,
            "feedback_count": len(feedback),
            "events_tracked": len(events),
            "event_counts": event_counts,
        },
        "latest_users": [public_user(u) for u in latest_users],
        "latest_feedback": latest_feedback,
        "latest_events": latest_events,
    }


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
        save_users_db(db)

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

        ai_feedback = generate_ai_feedback(report_dict)

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

        text_report = render_report(report)
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