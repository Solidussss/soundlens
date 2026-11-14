# Studio architecture and decisions

## ADR-1: Hybrid. Structured generation is the source of truth; neural models render or replace layers

**Context.** The product promise is "change anything". A single stereo file cannot deliver that:

- **Architecture A** (generate a mix, then separate it) yields at most about 4–6 stems, with bleed. Our measurement on SoundLens's own Demucs stems is below.
- Separated stems carry no note data. A separated 808 is audio only: you cannot change its notes, swap its instrument cleanly, or move a melody.
- No open model with a usable licence generates a full multitrack song natively. ACE-Step 1.5 `lego` adds one track per call (12 classes, no 808 class) on its Base model. Everything else is either research-only or vocal/accompaniment only (see `research_generation_models.md`).

**Decision.**
1. **Structure layer (always present, CPU).** `studio/composer.py` produces sections, harmony and **per-part note data**. `studio/synth.py` and `studio/dsp.py` render each part as its own track. Every track is independent by construction. Notes are real MIDI. Rendering is deterministic, so untouched tracks are provably unchanged after any edit.
2. **Neural layers (GPU, optional, via `studio/providers`).**
   - **Vocals:** ACE-Step `lego` with the `vocals` / `backing_vocals` classes, conditioned on the instrumental and the lyrics. This is the only realistic vocal path today.
   - **Sounds the synths can't make** (guitar, strings, real-sounding keys): ACE-Step `lego` for that class, in context.
   - **Whole-song "polish" or alternative takes:** ACE-Step `cover` / `repaint` on the procedural mix, kept as an *audio* track next to the editable parts, never replacing them.
3. **Architecture A stays available** for audio users bring in (uploads, ACE-Step full songs): Demucs separation, then optional Basic Pitch transcription *as suggestions*.
4. **Provider-agnostic.** Each provider reports `available()` with a reason. The UI shows that reason instead of a fake button.

**Consequences.**
- (+) Note-level editing, exact instrument swaps, section-level regeneration and bit-exact preservation work today on CPU.
- (−) The procedural engine's sound quality and musical range are below a neural model's. It cannot sing, and it has no acoustic instruments. That gap is exactly what the ACE-Step layers are for.
- (−) Neural layers come back as audio, so they lose note-level editing. They get DSP, clip edits, repaint and (lossy) transcription.

## ADR-2: Base vs Turbo (ACE-Step 1.5)

`lego`, `extract` and `complete` require Base (verified in `acestep/constants.py`). Turbo (8 steps, no CFG) handles text2music, repaint and cover. Use **Turbo for full-song generation and repaint**, **Base for lego/extract** and as the **LoRA training target**. Nobody has published a Base vs Turbo repaint-quality comparison, so A/B-test it on our material before fixing the choice.

## ADR-3: Edits are validated actions; the language model never touches audio

Every UI control and every natural-language command becomes a list of `studio/actions.py` operations (23 ops). The flow:
1. Ops are applied to a copy.
2. The whole document is validated.
3. The result is committed as a new history snapshot, which gives undo/redo for every op.

`diff_tracks` compares stem cache keys, so each edit reports exactly which tracks will sound different. The interpreter order is:
1. deterministic DSP (gain, EQ, distortion…);
2. structured note edits (instrument swap, transpose, clear section);
3. regeneration with a new seed, scoped to a part and/or section.

An optional LLM interpreter (`SOUNDLENS_STUDIO_LLM=1`) emits the same JSON. It is validated identically and never reaches audio directly.

## ADR-4: Character identity is data plus measured targets

Characters are JSON (`studio/characters/*.json`). Each one defines tempo range, scales, kit, 808 model and fx, melodic tendencies, chord progressions, arrangement, mix bus, loudness target, vocal delivery descriptors, and caption tags for neural providers. Adding a character means adding a file.

Reference recordings attach as **collections with an explicit rights record**: holder, licence, and permitted uses ∈ {analysis, conditioning, training}. Code enforces the uses (`studio/references.py`). Altering a recording never changes its rights status.

Methods to keep a consistent sonic identity, in increasing cost and risk:

| Method | What it captures | Status here |
|---|---|---|
| Feature extraction → engine settings | loudness, 808 distortion, spectral balance, clipping | **implemented** (`references.suggest_character_settings`) |
| Reference-audio conditioning (ACE-Step `cover` / reference audio) | timbre, mix colour, per generation | provider plumbing only; needs GPU + "conditioning" rights |
| Retrieval-assisted (pick the nearest licensed reference per request) | per-song variety within an identity | not built; cheap once collections exist |
| LoRA / LoKr on ACE-Step Base | strongest consistent identity, including vocals | needs ≥16 GB GPU, 10–100 songs **with training rights** |
| Training specialised models | full control | not justified at this stage |

Measured: with feature-based identities alone, a nearest-centroid classifier identifies the character of 15/15 generations (3 seeds × 5 characters) from six audio features. Between-character feature distance is 2.9× the within-character distance.

## Data flow

```
browser (studio.html) ──REST──▶ studio/api.py ──▶ actions / orchestrator ──▶ store (JSON snapshots)
        ▲  16-bit stem WAVs                         │
        └──────────── render.Renderer ◀──────────────┘  stems cached by content hash (LRU-pruned)
                         │  synth + dsp (CPU)          providers: acestep (GPU REST), demucs, basic_pitch
```

Jobs (generate, export, QA, separate, transcribe, generate-layer) run on an in-process thread pool with persisted state. **Before multi-instance production**, move them to a real queue (Redis/RQ or similar) plus a GPU worker; the job interface does not change.
