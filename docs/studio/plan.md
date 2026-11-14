# Development plan, status and risks

Legend: ✅ working and tested · 🟡 built but not runnable here (needs a GPU or downloads) · ⬜ not started

## Phase 1: Investigation ✅
- Codebase: `codebase_assessment.md`
- Models: `research_generation_models.md`, `research_separation_transcription.md`
- Architecture: `architecture.md`
- Costs: `costs.md`

## Phase 2: Audio proof of concept ✅
Run `experiments/poc_phase2.py`. Results are in `poc_results.md`.
- Generate ✅
- Layers by design ✅
- 808 replacement ✅
- Melody instrument change ✅
- Section regeneration ✅
- Preservation null tests ✅
- Export ✅
- Transcription accuracy measured ✅
- **Next:** run the same script against the ACE-Step provider on a GPU (step G1 below).

## Phase 3: Characters ✅ (feature level)
- `studio/characters/*.json`: 5 original characters, validated schema, SVG emblems.
- `studio/references.py`: collections with rights records and permitted-use enforcement, feature profiles, and suggested settings.
- Consistency measured: 15/15 identification.
- ⬜ Reference collections with *your own or licensed* audio.
- ⬜ Retrieval-assisted conditioning.
- ⬜ LoRA (needs training rights and a GPU).

## Phase 4: Multitrack editor ✅ (first version)
`studio.html` at `/studio`:
- character picker, prompt, advanced settings, job progress;
- synced Web Audio playback of real stems, with mute/solo/volume/pan;
- waveforms and section ruler;
- section looping;
- clip move / trim / split / duplicate / delete;
- piano roll (add/move/resize/delete, velocity, quantize);
- instrument swap and per-track fx chain (add/bypass/remove);
- 808 drive/decay/glide;
- add part, upload audio, transcribe;
- undo/redo (also ⌘Z / ⇧⌘Z);
- export WAV/MP3/stems+MIDI;
- QA panel;
- "Live preview" vs "Exact master" monitoring.

⬜ Not done yet:
- effect parameter knobs (effects can be added and bypassed, but parameters are set by commands and presets);
- reordering tracks;
- automation;
- a "re-master to target" button;
- touch gestures for clips on phones;
- a track-count limit / virtualisation for very long songs.

## Phase 5: AI instructions ✅ (rules) / 🟡 (LLM)
- The rule interpreter handles the spec's examples plus compound requests ("louder and more distorted"). It refuses impossible ones (vocals that don't exist, guitar) with a reason.
- The optional LLM interpreter goes through the same validation.
- ⬜ Evaluate it on a set of about 100 real user phrasings.

## Phase 6: SoundLens integration
- ✅ Same accounts and tokens as the analyzer, same data volume, and a guarded mount in `app.py`.
- ⬜ Link `/studio` from the main nav in `index.html`. Deliberately not done, so the existing UI is unchanged until you decide.
- ⬜ **Beat Battle submission:**
  1. Battles run on the haze license-server with a **separate account system**.
  2. Options:
     - (a) account linking, like the existing plugin-link flow, then server-to-server `PUT /api/battles/c/:slug/entry` with the user's battle session;
     - (b) "export, then upload" in the battle page (works today).
  3. **Policy decision needed:** battles are producer competitions, and the entry form requires confirming "only the challenge sample and sounds you have the right to use". Decide whether AI-generated entries are allowed, labelled, or in a separate category before wiring (a).
- ⬜ Send a Studio export to the analyzer ("analyze my mix").

## Phase 7: Testing and optimisation
- ✅ 31 automated tests (`python -m pytest studio/tests -q`, about 20 s) plus a headless browser run.
- ⬜ Listening evaluation (rubric in `poc_results.md`).
- ⬜ GPU latency and cost measurement.
- ⬜ Load test.

## Next concrete steps (in order)

**G1 — ACE-Step on a GPU (needs your approval for paid compute):**
1. Rent one 24 GB GPU (L4/A10) for a day.
2. Run `uv run acestep-api` with `acestep-v15-turbo` and `acestep-v15-base`.
3. Set `ACESTEP_API_URL`.
4. Verify the response shapes in `studio/providers/acestep.py` (marked untested).
5. Measure:
   - text2music latency and quality on the five character captions;
   - `lego` vocals over a procedural VANTA beat (does it stay on tempo and key?);
   - `repaint` of one section: does the rest stay intact? Null-test outside the window;
   - Base vs Turbo repaint A/B;
   - **808 study:** ask ACE-Step for "blown-out distorted 808" vs procedural + DSP; compare THD/crest and listen.

**G2 — Demucs on the procedural songs.** `pip install demucs==4.1.0` on a machine with Hugging Face access. The procedural stems give ground truth, so this measures *true* leakage and SDR per stem for trap material (impossible with real songs).

**G3 — Envelope-aware 808 drive** to match the reference crest (3–8 dB), and a "re-master" button.

**G4 — Security fixes from the assessment:** the `/static` exposure and password hashing. Separate change, needs your go-ahead.

**G5 — Nav link + Beat Battle flow** once the policy is decided.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| ACE-Step response format differs from the docs (untested) | vocal layer fails | defensive parsing and logged payloads; fix in G1 |
| Neural models can't reproduce blown-out 808s | core style gap | procedural 808 + DSP path measured to cover the range; LoRA later |
| Separation and weight licences unclear (Demucs, RoFormer) | legal exposure for Architecture A | keep separation for user uploads, get legal review, prefer structured generation |
| Training data provenance for character LoRAs | legal / reputational | rights records enforced in code; train only on owned or licensed audio |
| Procedural sound quality vs Suno-class output | users may find it plain | neural layers via providers; better synthesis; sample instruments with clear licences |
| Master-bus coupling surprises users | "I changed the 808 and the snare got quieter" | "Exact master" monitor; explain in UI; re-master button |
| In-process jobs and a local stem cache | doesn't scale past one instance | queue + GPU worker + object storage before public launch |
| Beat Battle fairness with AI entries | community trust | policy decision first; label AI-assisted entries |
| Voice identity misuse | legal / ethical | only original or consented voices; no artist sound-alikes in character prompts |
