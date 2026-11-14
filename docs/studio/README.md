# SoundLens Studio

Pick a character, describe the song, and get a multitrack project where every instrument is its own editable track (notes + sound + effects). Change any part by clicking or by typing; everything else stays exactly as it was.

**Status:** working prototype. It is not linked from the main SoundLens nav yet. Open `/studio` directly.

## Run it on your Mac

```bash
pip install -r requirements.txt
# optional, for "Transcribe to MIDI" (Basic Pitch on ONNX, no TensorFlow needed):
pip install --no-deps basic-pitch && pip install pretty-midi mir-eval "resampy<0.4.3" scikit-learn onnxruntime
SOUNDLENS_STUDIO_DEV_AUTH=1 uvicorn visual_app:app --port 8000   # dev auth: no login needed locally
open http://localhost:8000/studio
```

Without `SOUNDLENS_STUDIO_DEV_AUTH=1`, it uses your normal SoundLens login: log in on `/`, then open `/studio`.

```bash
python -m pytest studio/tests -q          # 31 tests, ~20 s
python experiments/poc_phase2.py          # the measured PoC (~3 min) → experiments/output/
```

## Environment variables

| Variable | Purpose |
|---|---|
| `SOUNDLENS_STUDIO_DEV_AUTH=1` | local dev only: requests without a token act as user "dev". **Never set in production** |
| `ACESTEP_API_URL` | URL of a running ACE-Step 1.5 API server; turns on AI vocal layers |
| `ACESTEP_BASE_MODEL` / `ACESTEP_TURBO_MODEL` | model names (defaults `acestep-v15-base` / `acestep-v15-turbo`) |
| `SOUNDLENS_STUDIO_LLM=1` (+ `OPENAI_API_KEY`) | optional LLM fallback for commands the rules don't understand |
| `SOUNDLENS_STUDIO_FREE_DAILY` | free-plan generations per day (default 20) |
| `SOUNDLENS_STUDIO_CACHE_GB` | stem cache cap before LRU pruning (default 5) |

## Code map

| File | Role |
|---|---|
| `studio/composer.py` | structured generation: sections → harmony → kick → 808 (follows kick) → hats/snare/perc → chords → melody (motifs) → extras |
| `studio/characters/*.json`, `studio/characters.py` | character profiles (data); add one by adding a file |
| `studio/synth.py` | 33 instrument presets on 10 engines (kick, snare, clap, hat, perc, 808, saw/supersaw, FM, additive keys, texture) |
| `studio/dsp.py` | gain, EQ, filter, distortion (oversampled, sub-preserving), saturation, clipper, compressor, look-ahead limiter, reverb, delay, stereo, bitcrush |
| `studio/render.py` | deterministic stems, content-hash cache, mix, master calibration, WAV/MP3/stems+MIDI export |
| `studio/project.py` | project document (tracks, clips, notes, fx, sections) and validation |
| `studio/actions.py` | 23 validated edit operations + per-edit diff of changed vs untouched tracks |
| `studio/orchestrator.py` | natural language → actions (rules; optional LLM), with honest refusals |
| `studio/analysis.py` | LUFS, true peak, clipping, silence, BPM, key, stereo, 808 distortion profile, drum timing, null tests, export integrity |
| `studio/references.py` | reference collections with rights records and permitted-use enforcement |
| `studio/providers/` | `acestep` (REST client, untested here), `demucs`, `basic_pitch` |
| `studio/store.py` | projects, snapshot undo/redo, jobs, cache pruning |
| `studio/api.py` | REST API at `/studio-api/*` and the `/studio` page |
| `studio.html` | the editor |

## Docs
- `codebase_assessment.md`: what SoundLens has, what was reused, and problems found
- `architecture.md`: decisions (hybrid architecture, Base vs Turbo, actions, characters)
- `research_generation_models.md`, `research_separation_transcription.md`: model research with sources and verification status
- `poc_results.md`: measured results, failures, listening rubric; audio in `poc/`
- `costs.md`: development vs production cost estimates
- `plan.md`: status by phase, next steps, risks
