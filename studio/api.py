"""FastAPI router for SoundLens Studio (mounted by app.py at ``/studio-api``; page at ``/studio``).

Authentication reuses SoundLens's existing bearer tokens via the ``get_user``
callable passed in by app.py, so accounts, verification and plans are shared.
For local development only, ``SOUNDLENS_STUDIO_DEV_AUTH=1`` lets requests without
a token act as a local "dev" user.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import SAMPLE_RATE, analysis, characters, composer, dsp, providers, synth
from .actions import ActionError, apply_actions
from .orchestrator import interpret
from .project import beats_to_seconds, total_beats, track_by_ref
from .render import Renderer, song_samples, stem_key
from .store import JobQueue, ProjectStore, safe_id

STUDIO_HTML = Path(__file__).resolve().parents[1] / "studio.html"
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
FREE_DAILY_GENERATIONS = int(os.getenv("SOUNDLENS_STUDIO_FREE_DAILY", "20"))


class GeneratePayload(BaseModel):
    character_id: str
    prompt: str = Field(default="", max_length=500)
    title: str | None = Field(default=None, max_length=80)
    bpm: float | None = Field(default=None, ge=40, le=240)
    key: str | None = None
    scale: str | None = None
    duration_s: float = Field(default=60, ge=10, le=300)
    energy: float = Field(default=0.7, ge=0, le=1)
    mood: str | None = None
    vocals: dict | None = None
    instruments: dict[str, str] | None = None
    seed: int | None = None


class EditPayload(BaseModel):
    actions: list[dict]
    note: str | None = Field(default=None, max_length=200)


class CommandPayload(BaseModel):
    text: str = Field(max_length=500)
    context: dict | None = None
    dry_run: bool = False


class ExportPayload(BaseModel):
    format: str = "wav"  # wav | mp3 | stems


def create_router(get_user: Callable[[str | None], tuple[dict, dict]], data_dir: Path) -> APIRouter:
    root = Path(data_dir) / "studio"
    store = ProjectStore(root)
    jobs = JobQueue(root)
    custom_chars_root = root / "characters"

    def owner_chars(owner: str) -> Path:
        return custom_chars_root / safe_id(owner)
    router = APIRouter()

    def renderer(owner: str, pid: str) -> Renderer:
        return Renderer(store.cache, store.assets_dir(owner, pid))

    def owner_of(authorization: str | None) -> tuple[str, dict]:
        if not authorization and os.getenv("SOUNDLENS_STUDIO_DEV_AUTH") == "1":
            return "dev", {"id": "dev", "plan": "studio"}
        _, user = get_user(authorization)
        return safe_id(str(user["id"]).replace("-", "_")[:80]), user

    def load(owner: str, pid: str) -> dict:
        try:
            return store.load(owner, safe_id(pid))
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "Project not found.")

    def character_for(project: dict) -> dict | None:
        try:
            return characters.get(project.get("character_id", ""), owner_chars(project.get("owner", "")) if project.get("owner") else [])
        except ValueError:
            return None

    def view(owner: str, project: dict) -> dict:
        h = store.history(owner, project["id"])
        revs = [e["rev"] for e in h["log"]]
        i = revs.index(h["head"])
        return {"project": project, "revision": h["head"], "can_undo": i > 0, "can_redo": i < len(revs) - 1,
                "history": [{"rev": e["rev"], "note": e["note"]} for e in h["log"][-30:]],
                "stems": {t["id"]: stem_key(project, t) for t in project["tracks"]},
                "duration_s": song_samples(project) / SAMPLE_RATE, "total_beats": total_beats(project)}

    def check_quota(owner: str, user: dict) -> None:
        if user.get("plan") in ("pro", "studio", "lifetime"):
            return
        usage_path = root / "usage" / f"{owner}.json"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
        if usage.get("date") != today:
            usage = {"date": today, "count": 0}
        if usage["count"] >= FREE_DAILY_GENERATIONS:
            raise HTTPException(429, f"Free accounts can generate {FREE_DAILY_GENERATIONS} songs per day.")
        usage["count"] += 1
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        usage_path.write_text(json.dumps(usage))

    # -- page + catalog ----------------------------------------------------------
    @router.get("/studio", include_in_schema=False)
    @router.get("/Studio", include_in_schema=False)
    def studio_page():
        if not STUDIO_HTML.exists():
            raise HTTPException(404, "studio.html missing")
        return HTMLResponse(STUDIO_HTML.read_text(encoding="utf-8"))

    @router.get("/studio-api/characters")
    def list_characters(authorization: str | None = Header(default=None)):
        try:
            owner, _ = owner_of(authorization)
        except HTTPException:
            owner = None
        chars = characters.load_all(owner_chars(owner) if owner else [])
        return {"characters": [characters.public_view(c, owner_view=bool(c.get("custom"))) for c in chars.values()]}

    @router.get("/studio-api/catalog")
    def catalog():
        return {
            "presets": {k: {"label": v["label"], "engine": v["engine"]} for k, v in synth.PRESETS.items()},
            "effects": {k: {"params": {p: list(r) for p, r in v["params"].items()}, "enums": v.get("enums", {})} for k, v in dsp.EFFECTS.items()},
            "scales": sorted(__import__("studio.theory", fromlist=["SCALES"]).SCALES),
            "providers": providers.status(),
        }

    # -- characters learned from songs ------------------------------------------
    @router.get("/studio-api/avatars/options")
    def avatar_options():
        from . import avatars
        return avatars.options()

    @router.post("/studio-api/avatars/render")
    def avatar_render(payload: dict):
        from . import avatars
        look = payload.get("look") or avatars.random_look(payload.get("seed") or uuid.uuid4().hex[:8], payload.get("mood"))
        try:
            look = avatars.validate_look(look)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"look": look, "svg": avatars.render_svg(look, 256)}

    @router.post("/studio-api/characters/learn")
    async def learn_character(files: list[UploadFile] = File(...), name: str = Form(...), tagline: str = Form(""),
                              look: str = Form("{}"), rights: str = Form(""), authorization: str | None = Header(default=None)):
        """Upload songs → learn a character. Audio is analysed then deleted; only statistics are kept."""
        owner, _ = owner_of(authorization)
        from . import avatars, learn
        if rights != "yes":
            raise HTTPException(400, "Confirm you're allowed to use these songs for analysis.")
        name = name.strip()[:40]
        if not name:
            raise HTTPException(400, "Give the character a name.")
        if not 1 <= len(files) <= 20:
            raise HTTPException(400, "Upload between 1 and 20 songs (5–10 is a good amount).")
        try:
            look_d = avatars.validate_look(json.loads(look or "{}") or avatars.random_look(uuid.uuid4().hex[:8]))
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(400, f"Bad look: {e}")
        work_dir = root / "learning" / f"{owner}_{uuid.uuid4().hex[:8]}"
        work_dir.mkdir(parents=True)
        saved = []
        for f in files:
            suffix = Path(f.filename or "song.wav").suffix.lower()
            if suffix not in (".wav", ".mp3", ".flac", ".aif", ".aiff", ".m4a", ".ogg"):
                shutil.rmtree(work_dir, ignore_errors=True)
                raise HTTPException(400, f"{f.filename}: upload WAV, MP3, FLAC, AIFF, M4A or OGG.")
            dest = work_dir / f"{len(saved):02d}{suffix}"
            size = 0
            with dest.open("wb") as fh:
                while chunk := await f.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        shutil.rmtree(work_dir, ignore_errors=True)
                        raise HTTPException(413, f"{f.filename} is larger than 80 MB.")
                    fh.write(chunk)
            saved.append((dest, Path(f.filename or dest.name).name[:120]))

        def work(progress):
            try:
                results = []
                for i, (path, orig) in enumerate(saved):
                    progress(0.05 + 0.45 * i / len(saved), f"listening to song {i + 1} of {len(saved)}")
                    try:
                        r = learn.analyze_track(path)
                    except Exception as e:  # one bad file shouldn't sink the rest
                        r = {"file": orig, "error": f"{type(e).__name__}"}
                    r["file"] = orig
                    results.append(r)
                progress(0.5, "building the character")
                style = learn.aggregate(results)
                cid = f"c_{uuid.uuid4().hex[:10]}"
                char = learn.character_from_style(style, cid, name, {"look": look_d, "palette": list(avatars.BG[look_d["bg"]])},
                                                  tagline.strip()[:100])
                # calibration: generate test beats and tune the sound until they measure like the uploads
                steps = iter(range(100))
                fit = learn.fit_character(char, learn.targets_from([t for t in results if "error" not in t]), iterations=4, seconds=28,
                                          progress=lambda m: progress(0.55 + 0.4 * min(next(steps), 4) / 4, m), work_dir=work_dir / "fit")
                char = fit["character"]
                char["custom"] = True
                char["owner"] = owner
                d = owner_chars(owner)
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{cid}.json").write_text(json.dumps(char, indent=2))
                summary = characters.learned_summary(char)
                summary["fit"] = {k: fit["report"].get(k) for k in ("start_distance", "best_distance")}
                return {"character_id": cid, "summary": summary}
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)  # never keep the uploaded songs

        return jobs.submit(owner, "learn_character", work, {"name": name, "songs": len(saved)})

    @router.post("/studio-api/characters/{cid}")
    def update_character(cid: str, payload: dict, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        from . import avatars
        path = owner_chars(owner) / f"{safe_id(cid)}.json"
        if not path.exists():
            raise HTTPException(404, "Character not found.")
        c = json.loads(path.read_text())
        if "name" in payload:
            n = str(payload["name"]).strip()[:40]
            if not n:
                raise HTTPException(400, "Name can't be empty.")
            c["name"] = n
        if "tagline" in payload:
            c["tagline"] = str(payload["tagline"]).strip()[:100]
        if "look" in payload:
            try:
                c["visual"]["look"] = avatars.validate_look(payload["look"])
            except ValueError as e:
                raise HTTPException(400, str(e))
        characters.validate_character(c)
        path.write_text(json.dumps(c, indent=2))
        return characters.public_view(c, owner_view=True)

    @router.delete("/studio-api/characters/{cid}")
    def delete_character(cid: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        path = owner_chars(owner) / f"{safe_id(cid)}.json"
        if not path.exists():
            raise HTTPException(404, "Character not found.")
        trash = root / "trash" / f"char_{owner}_{safe_id(cid)}_{int(time.time())}.json"
        trash.parent.mkdir(parents=True, exist_ok=True)
        path.replace(trash)
        return {"ok": True}

    # -- jobs --------------------------------------------------------------------
    @router.get("/studio-api/jobs/{job_id}")
    def job_status(job_id: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        try:
            job = jobs.get(safe_id(job_id))
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "Job not found.")
        if job["owner"] != owner:
            raise HTTPException(404, "Job not found.")
        return job

    # -- projects ----------------------------------------------------------------
    @router.get("/studio-api/projects")
    def list_projects(authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        return {"projects": store.list(owner)}

    @router.post("/studio-api/projects")
    def create_project(payload: GeneratePayload, authorization: str | None = Header(default=None)):
        owner, user = owner_of(authorization)
        try:
            character = characters.get(payload.character_id, owner_chars(owner))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if payload.scale and payload.scale not in __import__("studio.theory", fromlist=["SCALES"]).SCALES:
            raise HTTPException(400, "Unknown scale.")
        for role, pre in (payload.instruments or {}).items():
            if pre not in synth.PRESETS:
                raise HTTPException(400, f"Unknown instrument {pre}.")
        check_quota(owner, user)
        params = payload.model_dump(exclude_none=True)

        def work(progress):
            progress(0.05, "composing")
            t0 = time.time()
            project = composer.generate_project(character, params, seed=params.get("seed"))
            project["owner"] = owner
            store.create(owner, project)
            r = renderer(owner, project["id"])
            for i, t in enumerate(project["tracks"]):
                progress(0.1 + 0.75 * i / max(1, len(project["tracks"])), f"rendering {t['name']}")
                r.stem(project, t)
            progress(0.88, "mastering")
            calib = r.calibrate_master(project)
            project.setdefault("generation_notes", {})["master_calibration"] = calib
            project["generation_notes"]["seconds_to_generate"] = round(time.time() - t0, 2)
            store.commit(owner, project["id"], project, "mastered")
            store.prune_cache()
            return {"project_id": project["id"]}

        return jobs.submit(owner, "generate", work, {"character": character["id"]})

    @router.get("/studio-api/projects/{pid}")
    def get_project(pid: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        return view(owner, load(owner, pid))

    @router.delete("/studio-api/projects/{pid}")
    def delete_project(pid: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        load(owner, pid)
        store.delete(owner, pid)
        return {"ok": True}

    def commit_edit(owner: str, pid: str, actions: list[dict], note: str) -> dict:
        with store.lock(pid):
            project = load(owner, pid)
            rev = store.history(owner, pid)["head"]
            try:
                new, info = apply_actions(project, actions, character_for(project), revision=rev)
            except ActionError as e:
                raise HTTPException(400, str(e))
            store.commit(owner, pid, new, note, actions)
            out = view(owner, new)
            out["diff"] = info["diff"]
            return out

    @router.post("/studio-api/projects/{pid}/edit")
    def edit(pid: str, payload: EditPayload, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        note = payload.note or ", ".join(a.get("op", "?") for a in payload.actions)[:200]
        return commit_edit(owner, pid, payload.actions, note)

    @router.post("/studio-api/projects/{pid}/command")
    def command(pid: str, payload: CommandPayload, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        project = load(owner, pid)
        result = interpret(project, payload.text, payload.context or {}, character=character_for(project))
        response: dict[str, Any] = {"interpretation": dict(result)}
        if result["actions"] and not payload.dry_run:
            response.update(commit_edit(owner, pid, result["actions"], payload.text))
        return response

    @router.post("/studio-api/projects/{pid}/undo")
    def undo(pid: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        load(owner, pid)
        with store.lock(pid):
            try:
                return view(owner, store.move_head(owner, pid, -1))
            except ValueError as e:
                raise HTTPException(400, str(e))

    @router.post("/studio-api/projects/{pid}/redo")
    def redo(pid: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        load(owner, pid)
        with store.lock(pid):
            try:
                return view(owner, store.move_head(owner, pid, 1))
            except ValueError as e:
                raise HTTPException(400, str(e))

    @router.get("/studio-api/projects/{pid}/stems/{track_id}")
    def stem(pid: str, track_id: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        project = load(owner, pid)
        try:
            track = next(t for t in project["tracks"] if t["id"] == track_id)
        except StopIteration:
            raise HTTPException(404, "Track not found.")
        try:
            path, scale = renderer(owner, pid).stem_wav(project, track)
        except FileNotFoundError:
            raise HTTPException(410, "This track's audio file is missing.")
        return FileResponse(path, media_type="audio/wav", headers={"X-Stem-Scale": str(scale), "Cache-Control": "private, max-age=31536000, immutable"})

    @router.get("/studio-api/projects/{pid}/mix")
    def exact_mix(pid: str, authorization: str | None = Header(default=None)):
        """The exact server mix (master bus included), cached by everything that affects it."""
        owner, _ = owner_of(authorization)
        project = load(owner, pid)
        import hashlib
        import soundfile as sf
        mix_state = {"stems": {t["id"]: stem_key(project, t) for t in project["tracks"]},
                     "mixer": [(t["id"], t["gain_db"], t["pan"], t["mute"], t["solo"]) for t in project["tracks"]],
                     "master": project.get("master")}
        key = hashlib.sha256(json.dumps(mix_state, sort_keys=True).encode()).hexdigest()[:24]
        path = store.cache / f"mix_{key}.wav"
        if not path.exists():
            y = np.clip(renderer(owner, pid).mix(project), -1.0, 1.0)
            tmp = path.with_suffix(".tmp.wav")
            sf.write(str(tmp), y.T, SAMPLE_RATE, subtype="PCM_16")
            tmp.replace(path)
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @router.post("/studio-api/projects/{pid}/export")
    def export(pid: str, payload: ExportPayload, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        project = load(owner, pid)
        fmt = payload.format
        if fmt not in ("wav", "mp3", "stems"):
            raise HTTPException(400, "Format must be wav, mp3 or stems.")

        def work(progress):
            r = renderer(owner, pid)
            base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (project.get("title") or "soundlens"))[:40]
            name = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"
            d = store.exports_dir(owner, pid)
            progress(0.2, "rendering")
            if fmt == "wav":
                path = r.export_wav(project, d / f"{name}.wav")
                integrity = analysis.export_integrity(path, SAMPLE_RATE, 2, song_samples(project) / SAMPLE_RATE)
            elif fmt == "mp3":
                path = r.export_mp3(project, d / f"{name}.mp3")
                integrity = {"ok": path.exists() and path.stat().st_size > 1000, "bytes": path.stat().st_size}
            else:
                path = r.export_stems_zip(project, d / f"{name}_stems.zip")
                integrity = {"ok": path.stat().st_size > 1000, "bytes": path.stat().st_size}
            if not integrity.get("ok"):
                raise RuntimeError(f"export failed its integrity check: {integrity}")
            return {"file": path.name, "integrity": integrity}

        return jobs.submit(owner, "export", work, {"project_id": pid, "format": fmt})

    @router.get("/studio-api/projects/{pid}/exports/{name}")
    def download_export(pid: str, name: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        load(owner, pid)
        d = store.exports_dir(owner, pid)
        path = (d / Path(name).name)
        if not path.exists() or path.parent.resolve() != d.resolve():
            raise HTTPException(404, "Export not found.")
        return FileResponse(path, filename=path.name)

    @router.post("/studio-api/projects/{pid}/qa")
    def qa(pid: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        project = load(owner, pid)

        def work(progress):
            r = renderer(owner, pid)
            y = r.mix(project)
            rep = analysis.summarize(y, SAMPLE_RATE, project["params"]["bpm"], project["params"]["key"], project["params"]["scale"])
            timing = {}
            for role in ("kick", "snare", "clap"):
                try:
                    t = track_by_ref(project, role)
                except ValueError:
                    continue
                if t["kind"] == "midi":
                    from .project import track_notes_absolute
                    timing[role] = analysis.timing_alignment(r.stem(project, t), [n["t"] for n in track_notes_absolute(t)], project["params"]["bpm"])
            rep["timing_alignment"] = timing
            intentional = any(f["type"] == "clipper" for f in project.get("master", {}).get("fx", []))
            rep["notes"] = (["Clipping is part of this character's sound (master clipper)."] if intentional else []) + \
                ["Tempo estimates on half-time trap are unreliable; the timing check above measures the drum stems directly."]
            return rep

        return jobs.submit(owner, "qa", work, {"project_id": pid})

    # -- audio in: uploads, separation, transcription, generative providers --------------
    @router.post("/studio-api/projects/{pid}/upload-audio")
    async def upload_audio(pid: str, file: UploadFile = File(...), role: str = Form("other"), name: str = Form(""),
                           start_beat: float = Form(0.0), authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        project = load(owner, pid)
        import soundfile as sf
        assets = store.assets_dir(owner, pid)
        tmp = assets / f"upload_{uuid.uuid4().hex[:10]}.tmp"
        size = 0
        with tmp.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    fh.close(); tmp.unlink(missing_ok=True)
                    raise HTTPException(413, "File is larger than 80 MB.")
                fh.write(chunk)
        try:
            audio, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise HTTPException(400, "Upload WAV, FLAC or AIFF audio.")
        dest = assets / f"{uuid.uuid4().hex[:12]}.wav"
        sf.write(str(dest), audio, sr, subtype="FLOAT")
        tmp.unlink(missing_ok=True)
        seconds = audio.shape[0] / sr
        length = seconds / (60.0 / project["params"]["bpm"])
        action = {"op": "add_audio_track", "role": role, "name": name or Path(file.filename or "Audio").stem[:40], "uri": dest.name,
                  "length": length, "start": start_beat, "origin_bpm": project["params"]["bpm"],
                  "generator": {"engine": "upload", "filename": Path(file.filename or "").name[:120]}}
        return commit_edit(owner, pid, [action], f"upload {action['name']}")

    @router.post("/studio-api/projects/{pid}/tracks/{track_id}/transcribe")
    def transcribe(pid: str, track_id: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        project = load(owner, pid)
        bp = providers.BasicPitchProvider()
        ok, why = bp.available()
        if not ok:
            raise HTTPException(503, why)
        track = next((t for t in project["tracks"] if t["id"] == track_id), None)
        if not track:
            raise HTTPException(404, "Track not found.")

        def work(progress):
            import soundfile as sf
            r = renderer(owner, pid)
            y = r.stem(project, track)
            tmp = store.assets_dir(owner, pid) / f"transcribe_{uuid.uuid4().hex[:8]}.wav"
            sf.write(str(tmp), (y / max(1.0, float(np.max(np.abs(y))))).T, SAMPLE_RATE)
            notes = bp.transcribe(tmp, project["params"]["bpm"])
            tmp.unlink(missing_ok=True)
            if not notes:
                raise RuntimeError("no notes were detected")
            is_bass = track["role"] in ("808", "bass")
            new = composer.make_track(f"{track['name']} (transcribed)", "808" if is_bass else "melody",
                                      synth.preset("808_clean" if is_bass else "piano"), notes, total_beats(project),
                                      seed=1, gain_db=-12, color="#9fa6bb",
                                      generator={"engine": "transcribed", "from_track": track_id, "warning": "automatic transcription; check the notes"})
            new["mute"] = True
            with store.lock(pid):
                cur = load(owner, pid)
                cur["tracks"].append(new)
                store.commit(owner, pid, cur, f"transcribe {track['name']}")
            return {"track_id": new["id"], "notes": len(notes)}

        return jobs.submit(owner, "transcribe", work, {"project_id": pid})

    @router.post("/studio-api/projects/{pid}/tracks/{track_id}/separate")
    def separate(pid: str, track_id: str, authorization: str | None = Header(default=None)):
        owner, _ = owner_of(authorization)
        project = load(owner, pid)
        dm = providers.DemucsProvider()
        ok, why = dm.available()
        if not ok:
            raise HTTPException(503, why)
        track = next((t for t in project["tracks"] if t["id"] == track_id and t["kind"] == "audio"), None)
        if not track:
            raise HTTPException(404, "Pick an audio track to separate.")

        def work(progress):
            import soundfile as sf
            r = renderer(owner, pid)
            assets = store.assets_dir(owner, pid)
            src = assets / f"sep_{uuid.uuid4().hex[:8]}.wav"
            sf.write(str(src), r.stem(project, track).T, SAMPLE_RATE, subtype="FLOAT")
            progress(0.1, "separating (this takes about 1.5× the song length on CPU)")
            stems = dm.separate(src, assets / "separated")
            actions = []
            for name, path in stems.items():
                dest = assets / f"{uuid.uuid4().hex[:12]}.wav"
                shutil.copy(path, dest)
                actions.append({"op": "add_audio_track", "role": {"vocals": "lead_vocal", "bass": "808"}.get(name, "other"),
                                "name": f"{track['name']} – {name}", "uri": dest.name, "length": total_beats(project),
                                "generator": {"engine": "demucs", "from_track": track_id, "warning": "separated audio: expect bleed"}})
            actions.append({"op": "set_mixer", "track": track_id, "mute": True})
            with store.lock(pid):
                cur = load(owner, pid)
                new, _ = apply_actions(cur, actions, character_for(cur))
                store.commit(owner, pid, new, f"separate {track['name']}", actions)
            return {"stems": sorted(stems)}

        return jobs.submit(owner, "separate", work, {"project_id": pid})

    @router.post("/studio-api/projects/{pid}/generate-layer")
    def generate_layer(pid: str, payload: dict, authorization: str | None = Header(default=None)):
        """ACE-Step lego: generate a new audio layer (e.g. vocals) in the context of the current mix."""
        owner, _ = owner_of(authorization)
        project = load(owner, pid)
        ace = providers.AceStepProvider()
        ok, why = ace.available()
        if not ok:
            raise HTTPException(503, f"Generative layers are unavailable: {why}")
        role = str(payload.get("role", "lead_vocal"))
        caption = str(payload.get("caption", ""))[:500]
        lyrics = str(payload.get("lyrics", ""))[:4000]
        c = character_for(project) or {}
        tags = ", ".join((c.get("generation_hints") or {}).get("caption_tags", []))

        def work(progress):
            import soundfile as sf
            r = renderer(owner, pid)
            assets = store.assets_dir(owner, pid)
            ctx_path = assets / f"ctx_{uuid.uuid4().hex[:8]}.wav"
            sf.write(str(ctx_path), np.clip(r.mix(project), -1, 1).T, SAMPLE_RATE, subtype="PCM_16")
            dest = assets / f"{uuid.uuid4().hex[:12]}.wav"
            key = f"{project['params']['key']} {'Minor' if 'minor' in project['params']['scale'] or project['params']['scale'] in ('dorian', 'phrygian') else 'Major'}"
            ace.lego(ctx_path, role, f"{caption}, {tags}".strip(", "), lyrics, project["params"]["bpm"], key, dest, progress)
            info = sf.info(str(dest))
            action = {"op": "add_audio_track", "role": role, "name": role.replace("_", " ").title(), "uri": dest.name,
                      "length": info.duration / (60.0 / project["params"]["bpm"]),
                      "generator": {"engine": "acestep", "task": "lego", "caption": caption}}
            with store.lock(pid):
                cur = load(owner, pid)
                new, _ = apply_actions(cur, [action], character_for(cur))
                store.commit(owner, pid, new, f"generate {role}", [action])
            return {"ok": True}

        return jobs.submit(owner, "generate_layer", work, {"project_id": pid, "role": role})

    _ = beats_to_seconds
    return router
