from __future__ import annotations

import secrets
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

import app as legacy
import hardened_app as hardened

app = hardened.app


# ---------------------------------------------------------------------------
# One-time production session rotation
# ---------------------------------------------------------------------------

_SESSION_ROTATION_MARKER = legacy.DATA_DIR / ".sessions_rotated_20260917"


def _rotate_legacy_sessions_once() -> None:
    users_path = legacy.USERS_DB_PATH
    if _SESSION_ROTATION_MARKER.exists():
        return
    if not users_path.exists():
        print("[security] persistent user database not mounted yet; session rotation deferred")
        return

    db = legacy.read_json_file(users_path, {})
    if not isinstance(db, dict):
        raise RuntimeError("SoundLens user database is not a JSON object")

    tokens = db.get("tokens")
    token_count = len(tokens) if isinstance(tokens, dict) else 0

    legacy.BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = legacy.BACKUPS_DIR / f"soundlens_users_before_session_rotation_{stamp}.json"
    shutil.copy2(users_path, backup)

    db["tokens"] = {}
    legacy.write_json_file(users_path, db)
    verify = legacy.read_json_file(users_path, {})
    remaining = len((verify.get("tokens") or {})) if isinstance(verify, dict) else -1
    if remaining != 0:
        raise RuntimeError("SoundLens session rotation verification failed")

    _SESSION_ROTATION_MARKER.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    print(f"[security] rotated {token_count} legacy sessions; backup created; remaining=0")


_rotate_legacy_sessions_once()


# Remove dead legacy comparison endpoint.
hardened._remove_route("/compare-profile", "POST")


# ---------------------------------------------------------------------------
# Stable signup verification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3D event-model UI wiring
# ---------------------------------------------------------------------------
# Keep index.html itself stable while replacing only the small runtime functions
# that need to understand visual_map v3. This can be removed once index.html is
# split into normal JS modules.

_INDEX_PATH = Path(__file__).with_name("index.html")


def _patch_3d_frontend(html: str) -> str:
    old_build = """function buildPins(){const colors={energy:0xffffff,bass:0x8174ff,transient:0xffcf72,stereo:0x58d7ff,clip:0xff647c,resonance:0xff79db,brightness:0xa8efff};(visual.pins||[]).forEach((pin,idx)=>{const p=pointForPin(pin);const sphere=new THREE.Mesh(new THREE.SphereGeometry(.19,20,20),new THREE.MeshBasicMaterial({color:colors[pin.kind]||0xffffff}));sphere.position.copy(p);sphere.userData={pin,index:idx};pinGroup.add(sphere);pinMeshes.push(sphere);const lineGeo=new THREE.BufferGeometry().setFromPoints([p,new THREE.Vector3(p.x,p.y+1.8,p.z)]);const line=new THREE.Line(lineGeo,new THREE.LineBasicMaterial({color:colors[pin.kind]||0xffffff,transparent:true,opacity:.34}));pinGroup.add(line);});}"""
    new_build = """function pinMatchesMode(pin,mode){if(mode==='full')return true;const kinds=new Set([pin.kind,...(Array.isArray(pin.evidence_types)?pin.evidence_types:[])]);if(mode==='bass')return kinds.has('bass');if(mode==='stereo')return kinds.has('stereo');if(mode==='transient')return kinds.has('transient');return true}
function applyPinMode(mode){pinGroup.children.forEach(o=>{if(o.userData?.pin)o.visible=pinMatchesMode(o.userData.pin,mode)});if(selectedPin&&!pinMatchesMode(selectedPin,mode)){selectedPin=null;document.getElementById('pinCard').classList.remove('on');document.getElementById('aiState').textContent='track context loaded'}}
function buildPins(){const colors={energy:0xffffff,bass:0x8174ff,transient:0xffcf72,stereo:0x58d7ff,clip:0xff647c,resonance:0xff79db,brightness:0xa8efff};(visual.pins||[]).forEach((pin,idx)=>{const p=pointForPin(pin);const sphere=new THREE.Mesh(new THREE.SphereGeometry(.19,20,20),new THREE.MeshBasicMaterial({color:colors[pin.kind]||0xffffff}));sphere.position.copy(p);sphere.userData={pin,index:idx};pinGroup.add(sphere);pinMeshes.push(sphere);const lineGeo=new THREE.BufferGeometry().setFromPoints([p,new THREE.Vector3(p.x,p.y+1.8,p.z)]);const line=new THREE.Line(lineGeo,new THREE.LineBasicMaterial({color:colors[pin.kind]||0xffffff,transparent:true,opacity:.34}));line.userData={pin,index:idx};pinGroup.add(line);});applyPinMode(currentMode)}"""

    old_mode = """function setMode(mode){currentMode=mode;document.querySelectorAll('.mode').forEach(b=>b.classList.toggle('on',b.dataset.mode===mode));sculpture.children.forEach(m=>{if(!m.material||!m.userData.band)return;const b=m.userData.band;let op=b==='skin'?.075:.55;if(mode==='bass')op=['sub','bass'].includes(b)?.85:(b==='skin'?.025:.055);if(mode==='stereo')op=b==='skin'?.28:.16;if(mode==='transient')op=b==='skin'?.04:.31;m.material.opacity=op;});}"""
    new_mode = """function setMode(mode){currentMode=mode;document.querySelectorAll('.mode').forEach(b=>b.classList.toggle('on',b.dataset.mode===mode));sculpture.children.forEach(m=>{if(!m.material||!m.userData.band)return;const b=m.userData.band;let op=b==='skin'?.075:.55;if(mode==='bass')op=['sub','bass'].includes(b)?.85:(b==='skin'?.025:.055);if(mode==='stereo')op=b==='skin'?.28:.16;if(mode==='transient')op=b==='skin'?.04:.31;m.material.opacity=op;});applyPinMode(mode);}"""

    old_show = """function showPin(pin){selectedPin=pin;document.getElementById('pinKind').textContent=pin.kind;document.getElementById('pinTitle').textContent=pin.title;document.getElementById('pinDetail').textContent=pin.detail;document.getElementById('pinTime').textContent=`${fmt(pin.time)} · click play to hear this point`;document.getElementById('pinCard').classList.add('on');const p=pointForPin(pin);focusCamera(p);audio.currentTime=Math.min(audio.duration||pin.time,pin.time);updatePlayhead();}"""
    new_show = """function showPin(pin){selectedPin=pin;const conf=Number(pin.confidence);const confText=Number.isFinite(conf)?` · ${Math.round(conf*100)}% confidence`:'';const section=pin.section_label?` · ${pin.section_label}`:'';document.getElementById('pinKind').textContent=`${pin.kind}${section}${confText}`;document.getElementById('pinTitle').textContent=pin.title;const types=Array.isArray(pin.evidence_types)?pin.evidence_types:[];const evidence=types.length?` Evidence: ${types.join(' + ')}.`:'';document.getElementById('pinDetail').textContent=`${pin.detail||''}${evidence}`;document.getElementById('pinTime').textContent=`${fmt(pin.time)} · ${pin.evidence_count||types.length||1} signal${(pin.evidence_count||types.length||1)===1?'':'s'} agree · click play to hear this point`;document.getElementById('pinCard').classList.add('on');document.getElementById('aiState').textContent=`focused on ${pin.title||pin.kind}`;const p=pointForPin(pin);focusCamera(p);const dur=Number.isFinite(audio.duration)?audio.duration:Number(visual?.duration||0);audio.currentTime=Math.min(dur||pin.time,pin.time);updatePlayhead();}"""

    old_audio = """const audio=document.getElementById('audio');
function updatePlayhead(){if(!playhead||!audio.duration)return;const progress=Math.max(0,Math.min(1,audio.currentTime/audio.duration));playhead.position.x=-trackLength/2+progress*trackLength;document.getElementById('scrub').value=Math.round(progress*1000);document.getElementById('time').textContent=`${fmt(audio.currentTime)} / ${fmt(audio.duration)}`}
audio.addEventListener('timeupdate',updatePlayhead);audio.addEventListener('play',()=>document.getElementById('play').textContent='❚❚');audio.addEventListener('pause',()=>document.getElementById('play').textContent='▶');"""
    new_audio = """const audio=document.getElementById('audio');
function authoritativeDuration(){const native=Number(audio.duration);if(Number.isFinite(native)&&native>0)return native;const mapped=Number(visual?.duration);return Number.isFinite(mapped)&&mapped>0?mapped:0}
function updatePlayhead(){const dur=authoritativeDuration();if(!dur)return;const progress=Math.max(0,Math.min(1,Number(audio.currentTime||0)/dur));if(playhead)playhead.position.x=-trackLength/2+progress*trackLength;document.getElementById('scrub').value=Math.round(progress*1000);document.getElementById('time').textContent=`${fmt(audio.currentTime||0)} / ${fmt(dur)}`}
['loadedmetadata','durationchange','canplay','timeupdate'].forEach(evt=>audio.addEventListener(evt,updatePlayhead));audio.addEventListener('play',()=>document.getElementById('play').textContent='❚❚');audio.addEventListener('pause',()=>document.getElementById('play').textContent='▶');"""

    old_analyze_piece = """audio.src=audioURL;audio.load();buildSculpture();setMode('full');const basic="""
    new_analyze_piece = """audio.src=audioURL;audio.load();buildSculpture();setMode('full');updatePlayhead();const basic="""

    replacements = [
        (old_build, new_build, "pins"),
        (old_mode, new_mode, "mode"),
        (old_show, new_show, "pin card"),
        (old_audio, new_audio, "audio timeline"),
        (old_analyze_piece, new_analyze_piece, "initial timeline"),
    ]
    for old, new, label in replacements:
        if old not in html:
            print(f"[3d-ui] patch target missing: {label}")
            continue
        html = html.replace(old, new, 1)
    return html


hardened._remove_route("/", "GET")


@app.get("/", response_class=HTMLResponse)
def production_index():
    html = _INDEX_PATH.read_text(encoding="utf-8")
    return HTMLResponse(_patch_3d_frontend(html))
