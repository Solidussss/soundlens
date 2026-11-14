# SoundLens analyzer + Studio (`Solidussss/soundlens`)

The Python web app: audio analyzer (Lab, Artist Match, Galaxy, Artist Database) plus SoundLens Studio
(AI music creation at `/studio`). **Not deployed yet**; see `docs/INFRASTRUCTURE.md` for what runs where.

## Run locally

```bash
pip install -r requirements.txt
SOUNDLENS_STUDIO_DEV_AUTH=1 uvicorn visual_app:app --port 8000   # same entry point as the Procfile
python -m pytest tests studio/tests -q                           # 40 tests
```

## Layout

| Path | What |
|---|---|
| `visual_app.py` → `production_app.py` → `hardened_app.py` → `app.py` | **One app, four layers.** Each module imports the previous one and replaces some routes. `visual_app:app` is the entry point (Procfile) |
| `sitecustomize.py` | patches the analysis modules at interpreter start (accuracy layer, contrastive profiles, catalogue prior) |
| `soundlens_pro.py`, `compare_to_profile_pro.py`, `soundlens_3d.py`, `accuracy_v2.py`, `artist_match_v3.py`, `contrastive_artist_profiles.py`, `tunebat_catalog_prior.py`, `ai_feedback.py` | analyzer modules |
| `index.html` | analyzer frontend (single file) |
| `studio/`, `studio.html` | SoundLens Studio. See `docs/studio/README.md` |
| `artist_profiles/` | artist profiles the analyzer compares against (runtime data) |
| `artist_images.json` | artist pictures for the Artist Database |
| `data/catalog/` | TuneBat metadata catalogue (read-only, runtime) |
| `data/reports/` | analysis reports used to *build* artist profiles (offline input) |
| `scripts/` | offline tools: `batch_analyze.py`, `build_artist_profile.py`, `validate_artist_match.py`, `validate_artist_confusion_v3.py`, `rotate_sessions_once.py`; run from the repo root, e.g. `python scripts/build_artist_profile.py` |
| `experiments/` | Studio proof-of-concept experiment |
| `tests/`, `studio/tests/` | tests |
| `docs/` | `INFRASTRUCTURE.md`, `ops/` (deploy/data-safety notes), `studio/` |
| `assets/` | marketing images not served by the app |

## Not in git (local only)

`uploads/`, `tracks/`, `stems/` (test audio, including commercial songs), `saved_reports/`, `soundlens_reports/`,
`test_reports/` (generated). User data lives in `SOUNDLENS_DATA_DIR` (`/data` on Railway).
These folders were tracked until the cleanup commit. To get them back after pulling:

```bash
git checkout 9c0b76e -- uploads tracks stems saved_reports && git reset -q uploads tracks stems saved_reports
```
