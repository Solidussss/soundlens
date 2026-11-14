# Research: separation, transcription, reference projects, editor libraries (2026-11-13)

Same verification caveats as `research_generation_models.md`: GitHub/PyPI/npm read
directly; huggingface.co, arxiv.org, mvsep.com unreachable → snippets.

## Source separation

* **Demucs**: facebookresearch/demucs archived 2025-01-01; adefossez/demucs is the
  maintained fork ("not actively working on Demucs"). **demucs 4.1.0 on PyPI
  (2026-07-11)** loads safetensors weights from Hugging Face (`adefossez/HTDemucs`,
  `HTDemucs-ft`), falling back to `dl.fbaipublicfiles.com` `.th` pickles. Code MIT;
  README: *"The license of the weights is not clear."* → **commercial use of weights
  legally unclear.** htdemucs multisong SDR: vocals 8.24, bass 11.76, drums 10.88,
  other 5.74. CPU ≈1.5× track duration; `_ft` ≈4× slower. 6-stem piano "doesn't work well".
* **Newer** (ZFTurbo Music-Source-Separation-Training, MIT code, no weight licences):

| Model | Stems | SDR dB | Set |
|---|---|---|---|
| BS-RoFormer (viperx) | vocals/other | vocals 10.87 | multisong |
| Mel-Band RoFormer (Kim) | vocals/other | vocals 10.98 | multisong |
| MDX23C | vocals/other | vocals 10.17 | multisong |
| BS-RoFormer MUSDB-only | 4 | avg 9.65 | MUSDB |
| SCNet Large | 4 | avg 9.70 | MUSDB |
| **SCNet XL IHF** | 4 | **avg 10.08** | MUSDB |

* **python-audio-separator** (MIT, v0.48.0 2026-10-29, maintained): UVR MDX/VR/Demucs/RoFormer in pip form, SHA-checked downloads, CPU/CUDA/CoreML. No per-model weight licences.
* **Drum sub-separation**: DrumSep MDX23C kick 16.66 / snare 11.53 / **hh 4.04** dB (no licence stated); LarsNet weights **CC BY-NC**. Hi-hats separate poorly everywhere — matters for trap hat rolls.
* **Meta SAM Audio** (Dec 2025): text/visual/time-span prompted isolation; gated HF checkpoints; "SAM License" (royalty-free, use restrictions, indemnity). No SDR vs RoFormer published.
* **Banquet/query-bandit**: MIT code, query-based stems, weights on Zenodo without licence.
* **Distorted 808s (practitioner knowledge, no primary benchmark)**: 4-stem models put the sub in "bass" and the clipped harmonics + click in "drums"/"other" → thin bass stem and an 808 ghost elsewhere. MUSDB contains almost no trap, so SDR does not predict trap behaviour. *Our own measurement on SoundLens's existing htdemucs stems is in `poc_results.md`.*
* **Licensing headline: no high-quality separation model has a clean, explicit commercial weight licence.**

## Transcription

| Tool | Licence | Notes |
|---|---|---|
| **Spotify Basic Pitch 0.4.0** | Apache-2.0 | bundles ONNX/TFLite/CoreML; polyphonic, pitch bends; "best on one instrument at a time"; last release 2024-08; pip deps target Py≤3.11 (we installed `--no-deps` + onnxruntime on 3.13 and it works) |
| MuScriptor (Jul 2026) | MIT code / **CC BY-NC weights** | best multi-instrument F1 reported; non-commercial |
| YourMT3+ | **GPL-3.0** | copyleft |
| Google MT3 | Apache-2.0 | dormant, weak on real mixes |
| Transkun V2 | MIT | piano only, excellent |
| ADTOF | **CC BY-NC-SA** | drums; non-commercial |

Recommendation: Basic Pitch for melodic stems; pYIN/onset tracking for 808s; onset detection per drum stem. **Never treat recovered MIDI as ground truth** (our measured F1 is in `poc_results.md`).

## Reference projects

* **Remiqora** — https://github.com/inikolax/remiqora — MIT, very active (last commit 2026-11-12), pre-release v0.6.0. Wraps ACE-Step 1.5 (`acestep-api` :8001; cover/repaint/extract/add-part, LoRA), YuE2, SheetSage2, MuScriptor (audio.cpp), Demucs, Stable Audio 3, Qwen3.5 assistant. Vue 3 + TS front end with a **custom Web Audio engine (`src/audio/`)**: timeline, mixing, 8 effects, MIDI parser/synth, WAV/MP3 encoders, non-destructive trim/split/duplicate/fades, SoundTouch stretch, 30-step undo, DAWproject/Reaper export. FastAPI reverse proxy + SQLite. **Best architectural reference**; Vue-specific so lift ideas/algorithms, not components. Its NC components (YuE2, MuScriptor) must not be copied into a commercial product.
* **StemForge** — https://github.com/tsondo/StemForge — LICENSE currently Apache-2.0 (one listing says PolyForm Noncommercial → **check LICENSE git history before reuse**). Demucs + BS-RoFormer with auto model choice, BasicPitch, faster-whisper + pYIN vocal melody, ACE-Step via Ace-Step-Wrangler, Stable Audio Open, RVC. FastAPI + vanilla JS + wavesurfer. Useful reference for the separation→MIDI pipeline.

## Browser editor building blocks

| Library | Licence | Fit |
|---|---|---|
| waveform-playlist 16 | MIT | best off-the-shelf multitrack clip editor (React/Tone.js or Lit) |
| wavesurfer.js v8 | BSD-3 | waveform rendering only |
| Tone.js | MIT | transport/effects |
| peaks.js | LGPL-3.0 | poor fit |
| openDAW | **AGPL-3.0** or paid | avoid for SaaS unless licensed |

## Instrument replacement when only audio exists

Transcribe → MIDI → re-synthesize is the most reliable/editable route; DDSP (Apache, monophonic, ageing); RAVE (**NC**); ACE-Step repaint/cover/lego as generative re-performance on GPU.
