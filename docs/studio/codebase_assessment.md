# SoundLens codebase assessment (for the Studio feature)

Inspected 2026-11-13/14 on branch `claude/youthful-cori-0uanur`. Repositories: `Solidussss/soundlens`
(this repo, the web analyzer) and `Solidussss/soundlens-haze` (branch `soundlens-0.2.0`, read-only: the
desktop app, the Haze plugin, and the account server that hosts Beat Battles).

## What exists

| Area | What it is | Where |
|---|---|---|
| Frontend | One 135 KB `index.html`: vanilla JS, no build step, three.js via importmap (3D "Galaxy" event map), dark glass design tokens (`--bg #05060a`, `--purple #7275ff`, …) | `index.html` |
| Backend | FastAPI. One app built in four layers: `visual_app` → `production_app` → `hardened_app` → `app`, where each layer replaces routes (`_remove_route`). The Procfile now points at `visual_app:app` (it said `app:app`, which skipped the hardening layers) | `app.py`, `hardened_app.py`, `production_app.py`, `visual_app.py`, `sitecustomize.py` |
| Storage | JSON files with atomic writes in `SOUNDLENS_DATA_DIR` (Railway volume at `/data`): users + tokens, admin events, feedback, saved reports, reference library, backups | `app.py` top |
| Auth | Email + password, email verification (Resend/SMTP), bearer tokens stored in the users JSON, `localStorage['soundlens_auth_token']` in the browser; plans free/pro/studio/lifetime | `app.py` `/auth/*` |
| Payments | Stripe checkout, confirm and webhook | `/stripe/*` |
| Audio analysis | `soundlens_pro.analyze_audio` (librosa: tempo, key, loudness, sections, etc.), optional Demucs 4-stem split via CLI, artist-profile comparison (40 profiles), 3D event map, tempo/key accuracy layer, Tunebat catalogue prior, AI feedback (OpenAI) | `soundlens_pro.py`, `compare_to_profile_pro.py`, `soundlens_3d.py`, `accuracy_v2.py`, `artist_match_v3.py`, `ai_feedback.py` |
| Samples / music library | No sample library. There are reference *analysis* assets: artist profiles, Tunebat catalogue JSON, and test audio in `uploads/`, `tracks/`, `stems/` | repo root |
| Beat Battle | **Not in this repo.** It lives in `soundlens-haze/license-server/src/battles/` (plain Node `http`, SQLite `battles.db`, media on a `/data` volume) and is served at beatbattle.soundlensapp.com. Entries are submitted with `PUT /api/battles/c/:slug/entry` (raw WAV/MP3 body, validated from its bytes) and require a "rights" confirmation. **Separate account system** (cookie `sl_session`), not shared with this app's users | haze repo |
| Desktop | "SoundLens" Electron app that controls FL Studio from text ("make the drums brighter") through an intent parser, plus the Haze JUCE plugin | haze repo |
| Deployment | Railway project `soundlens` runs `license-server` (Beat Battle, beatbattle.soundlensapp.com). www.soundlensapp.com is still served from a separate project. **The Python analyzer is not deployed.** Full map and the move plan: `docs/INFRASTRUCTURE.md` | Railway |
| Tests | `tests/` holds 9 tests (accuracy layer, event model). All pass before and after this change | `tests/` |

## Reuse, modify, build

**Reused as is:** FastAPI app and middleware, `get_current_user` (the studio uses the same tokens and accounts), `DATA_DIR` and its atomic-write approach, the design tokens, OpenAI usage pattern (optional LLM interpreter), the Demucs CLI approach (wrapped as a provider), and librosa analysis.

**Modified:** `app.py` gets 9 lines at the end. They mount `studio.api` inside a `try`, so a Studio failure can never stop the analyzer. `.gitignore` gets `experiments/output/`.

**Built new:** everything under `studio/` plus `studio.html` (the `/studio` page). See `README.md`.

**Later integration points:**
- the analyzer can run on a Studio mix (call `analyze_audio` on an export);
- Beat Battle submission (cross-service; see `plan.md`);
- the haze intent parser's vocabulary could merge with `studio/orchestrator.py`.

## Problems found during inspection (not changed; outside this task's scope)

1. **`app.mount("/static", StaticFiles(directory="."))` serves the whole repository.** I confirmed locally that `/static/app.py` returns the server source with status 200. Locally, where `DATA_DIR` defaults to `.`, the same mount would serve `soundlens_users.json` (password hashes and live tokens). In production the data sits on `/data`, so it is not exposed, but source code, artist profiles and the audio in the repo are. **Recommended fix:** serve a dedicated `public/` folder instead.
2. **Passwords are hashed with one salted SHA-256** (`hash_password`). That is too fast to resist offline cracking if the users file leaks. Move to argon2id, bcrypt or scrypt and rehash on next login.
3. **Commercial recordings are committed** in `uploads/`, `tracks/` and `stems/` (about 1.5 GB). `.gitignore` excludes them, but they were added before it. `__pycache__/*.pyc` files are committed as well. Remove them from git history only after you decide; it is destructive.
4. CORS is `allow_origins=["*"]` with credentials enabled. It is harmless while auth uses bearer headers, but it is worth tightening.
5. The owner's email is hard-coded in `public_user()`.
