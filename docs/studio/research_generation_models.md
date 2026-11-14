# Research: generative music back ends (as of 2026-11-13)

Collected with web tools from this cloud session. **Verification status matters**:
huggingface.co, arxiv.org and stability.ai did not resolve from the research
sandbox, so facts from those hosts come from search snippets and are marked
**(snippet)**; secondary sites are **(secondary)**; anything unverified is
**unverified**. GitHub pages were read directly. Re-verify licences with
counsel before production.

## Comparison table

| Model | Version / date | Licence code / weights | Commercial? | Vocals | Track-level | Edit / repaint | Conditioning | Min VRAM / Mac | Latency | Integration |
|---|---|---|---|---|---|---|---|---|---|---|
| **ACE-Step 1.5** (2B, XL 4B DiT + 0.6/1.7/4B LM) | repo v0.1.8, 2026-05-18; XL 2026-04-02 | MIT / MIT (snippet) | Yes (card says outputs commercial OK) | Yes, 50+ languages | **Generative**: `lego` adds 1 of 12 tracks, `extract` isolates 1, `complete` adds accompaniment — **Base models only** | `repaint` (time range), `cover`, Retake, Flow-Edit (v0.1.8) | caption, tagged lyrics, BPM 30–300, key, time sig 2/3/4/6, 10–600 s, reference audio | ≤4 GB tier (DiT only, INT8, offload); MPS listed, MLX scripts | vendor: <2 s/song A100, <10 s RTX 3090; 30 s turbo 3.2 s | **Low** – async REST server |
| Stable Audio 3.0 (Small-Music/SFX 433M, Medium 1.4B, Large 2.7B API) | May 2026 (secondary) | MIT / Stability Community Licence | Yes under $1M revenue | **No** | No native stems in open repo | audio-to-audio, inpainting, continuation | text, audio, length | Small on CPU; Medium CUDA+FA2 | Medium 1.31 s for 380 s (H200) | Low–medium |
| Stable Audio 2.5 API | API only | proprietary | API terms | No | No | audio-to-audio (snippet) | text, audio | n/a | n/a | Low |
| YuE2 3B | Sep 2026 | Apache-2.0 / CC BY-NC 4.0 + creator permission | **No for companies** without licence | Yes | No | score-level | lyrics, score | 24 GB | — | Med–high |
| HeartMuLa-oss-3B | Feb 2026 | Apache-2.0 / Apache-2.0 | Yes | Yes | No | No (covers via MuLaCover) | lyrics, tags | n/s | RTF≈1 | Medium |
| DiffRhythm 1.2 / 2 | 2025 | Apache-2.0 / Apache-2.0 | Yes | Yes | No | v1.2 editing/continuation | LRC lyrics, text/ref | 8 GB chunked | — | Medium |
| SongGeneration / LeVo v2-large | Mar 2026 (secondary) | **unclear custom Tencent** | **unverified** | Yes | vocal + accompaniment split | No | lyrics, text, prompt audio | ~10 GB | — | Medium |
| SongBloom 2B | 2025 | Apache-2.0 (secondary, conflicting) | unverified | Yes | No | No | lyrics + required 10 s ref | n/v | — | Medium |
| MusicGen / AudioCraft | legacy | MIT / **CC-BY-NC** | **No** | No | MusicGen-Stem research | melody cond. | text, melody | ~16 GB | — | Low |
| DiffSinger + OpenUtau | active | Apache-2.0 / per-voicebank | code yes | SVS | n/a | note-level | score, phonemes, pitch | light | fast | High |
| SoulX-Singer | SVC 2026-03-16 | Apache-2.0 / Apache-2.0 | Yes | SVS + SVC (zh/en/yue) | n/a | score edits | MIDI/F0 + lyrics | n/s | — | Medium |
| RVC WebUI | active | MIT (base trained on VCTK) | Yes | voice conversion | n/a | n/a | audio | CPU OK | ~170 ms | Medium |
| Seed-VC | **archived 2025-11** | **GPL-3.0** | copyleft risk | SVC | n/a | n/a | audio + ref | MPS | ~300 ms | Medium |

## ACE-Step 1.5 details (https://github.com/ace-step/ACE-Step-1.5)

* Releases v0.1.0 (2026-02-16) → **v0.1.8 (2026-05-18)**. v0.1.5 "fixed the Lego task" → Lego was buggy until March 2026. Pin a tag.
* Variants: `acestep-v15-base` (50 steps, CFG, **all 7 tasks**), `-sft` and `-turbo` (8 steps, no CFG) support only `text2music, repaint, cover, cover-nofsq`. XL (4B) variants likewise. Weights ≈4.7 GB (2B) / ≈9 GB (4B) bf16.
* Tasks:
  * `repaint` – regenerate `repainting_start..repainting_end` seconds, rest kept. Tutorial range 3–90 s.
  * `lego` (Base) – "Generate the {TRACK} track based on the audio context:" with `src_audio`.
  * `extract` (Base) – "Extract the {TRACK} track from the audio:". **Generative re-synthesis, not mask separation** (inference from architecture) → expect artefacts; benchmark against a separator.
  * `complete` (Base) – add accompaniment to a single track; caption required.
  * `cover` – follows source structure; `audio_cover_strength` 0–1.
  * Track names: `vocals, backing_vocals, drums, bass, guitar, keyboard, percussion, strings, synth, fx, brass, woodwinds`. **No "808" class** — an 808 would be `bass`/`drums`; untested.
* Conditioning: `bpm` 30–300, `key_scale` ("C Major", "Am"), `time_signature` 2/3/4/6, `audio_duration` 10–600 s, lyrics with `[Verse]/[Chorus]` and `[Instrumental]`, 51 languages, batch ≤8.
* REST API (`uv run acestep-api`, port 8001): `POST /release_task` → `task_id` (multipart `src_audio`/`ctx_audio` or server path `src_audio_path`; `model` picks DiT); `POST /query_result {task_id_list}` → status 0 running / 1 done / 2 failed, file URLs; `GET /v1/audio?path=`; plus `/format_input`, `/v1/models`, `/v1/init`, `/health`, `/v1/training/start`, `/v1/training/start_lokr`.
* Hardware: CUDA recommended; MPS/ROCm/XPU/CPU listed; **CPU "slow, testing only"; no published Mac/CPU timings.**
* LoRA: audio + `.lyrics.txt` + captions; take BPM/key from an external analyser (LM "will produce hallucinations"); ~100 songs → 500 epochs, 10–20 songs → 800 epochs; 16 GB VRAM min. Community trap/rap LoRAs exist (licence/quality unverified).
* **Base vs Turbo**: Base is *required* for lego/extract/complete. Turbo supports repaint/cover at 8 steps (≈4–8× cheaper per call). Repaint quality Base vs Turbo is undocumented → A/B test on our material. Recommendation: Turbo for generate/repaint/cover; Base for lego/extract/complete and as the LoRA training target.

## Vocals

No open model was found that is explicitly rap-capable **and** note-timing-controllable. Practical route: ACE-Step lyric-conditioned vocals (+ `lego` for `vocals`/`backing_vocals` over a beat, `repaint` for phrases), optionally RVC (MIT) for a consistent, *original, consented* voice identity. DiffSinger/SoulX-Singer give note-level control for sung (not rapped) parts.

## Native multitrack generation

Real and shippable: ACE-Step `lego`/`extract`/`complete` (12 classes, one track per call), LeVo vocal/accompaniment (licence unclear). Research only: MusicGen-Stem (bass/drums/other, likely NC), Stemphonic (no weights found).

## Unverified

HF licence texts, arXiv contents, Stability rate card (one tracker reports SA 2.5/3.0 at 8 credits ≈ $0.08/call after 2026-10-23), SongGeneration/SongBloom licence files, MusicGen-Stem weight licence, ACE-Step Mac/CPU speed.
