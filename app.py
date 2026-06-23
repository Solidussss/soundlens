from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import traceback
import uuid

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

FREE_DAILY_UPLOAD_LIMIT = int(os.getenv("SOUNDLENS_FREE_DAILY_UPLOAD_LIMIT", "3"))
PRO_PRICE_CAD = os.getenv("SOUNDLENS_PRO_PRICE_CAD", "7.99")

USE_DEMUCS_BY_DEFAULT = os.getenv("SOUNDLENS_USE_DEMUCS", "0") == "1"
COMPARE_USE_STEMS = os.getenv("SOUNDLENS_COMPARE_USE_STEMS", "0") == "1"


class AuthPayload(BaseModel):
    email: str
    password: str
    display_name: str | None = None


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        "is_pro": plan == "pro",
        "daily_limit": None if plan == "pro" else FREE_DAILY_UPLOAD_LIMIT,
        "uploads_today": int(usage.get("count", 0) or 0),
        "uploads_remaining": None if plan == "pro" else max(0, FREE_DAILY_UPLOAD_LIMIT - int(usage.get("count", 0) or 0)),
        "created_at": user.get("created_at"),
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

    if plan != "pro" and int(usage.get("count", 0) or 0) >= FREE_DAILY_UPLOAD_LIMIT:
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