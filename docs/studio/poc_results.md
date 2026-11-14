# Phase 2 proof of concept: measured results

Source: `experiments/poc_phase2.py`, run 2026-11-14 on a 4-vCPU, no-GPU cloud container. Raw output is in
`poc/results.json`. Listen to `poc/01_original.mp3` … `poc/06_no_drums_first_hook.mp3` (160 kbps).

**I could not listen to the audio.** Every quality statement below is a measurement. The listening evaluation
still has to be done by you; a rubric is at the end.

## 1. Generate a short instrumental, with layers by design

VANTA, prompt "dark aggressive rage beat", seed 11, 36.9 s.

The prompt was read as "dark → minor-family scale" and "aggressive → upper tempo range". The result is 165 BPM, F phrygian.

| Track | Instrument | Notes |
|---|---|---|
| Kick | Hard distorted kick | 69 |
| Snare | Tight snare | 24 |
| Clap | Snappy clap | 26 |
| Hi-Hats | Crisp hats | 268 |
| 808 | Blown-out 808 + DSP distortion | 97 (follows the kick, with slides) |
| Chords | Warm pad | 72 |
| Melody | Rage synth | 183 |
| Atmosphere | Dark drone | 7 |

- **Speed:** composing took 0.004 s; rendering all 8 stems took **11.1 s on CPU**.
- **Loudness:** the master calibrated to −5.8 LUFS against a target of −6. Reference mixes in the repo measure −6.1 to −11.9 LUFS.
- **Timing:** the kick stem's detected onsets match the note grid **100 %** (median error 1.5 ms). Snare and clap match 88 %; the misses are flams inside clap bursts, not timing errors.
- **Export integrity:** the WAV was read back and checked for sample rate, channel count, duration (±0.05 s) and finite samples. Pass.
- **Stems ZIP:** 8 WAVs, 8 MIDI files and `project.json`.

## 2–4. Change one layer, keep the rest

For every edit we:
- null-tested **each untouched stem** against its pre-edit render;
- checked that the pre-master mix difference equals exactly the changed tracks' own difference.

| Edit | How it was requested | Tracks re-rendered | Untouched stems bit-identical | Mix difference explained by changed tracks | Time |
|---|---|---|---|---|---|
| Replace 808 (blown → clean, distortion removed) | actions | 1 of 8 | **7/7 yes** | residual −318 dB (float noise) | 0.14 s |
| "make the 808 a lot more distorted" | natural language → `update_effect` +10 dB drive, `set_instrument_params` drive +0.25 | 1 of 8 | **7/7 yes** | −315 dB | 0.65 s |
| "change the melody to a bell" | NL → `set_instrument bell_bright` (same notes) | 1 of 8 | **7/7 yes** | −308 dB | 1.6 s |
| Regenerate the melody in Verse 1 only | `regenerate_part` with a section scope | 1 of 8 | **7/7 yes** | −303 dB | 1.7 s |
| "take the drums out of the first hook" | NL → `clear_section` on drums | 4 of 8 | **4/4 yes** | −313 dB | 0.8 s |

**Section regeneration, inside the regenerated track.**
- Before the section, the audio is identical. Dry render: bit-identical. With the FFT reverb: max difference 1e-16, which is −318 dB.
- Audible differences (above −60 dB) are confined to the section plus about 1.25 s of reverb/delay tail after it.
- A first run showed differences across the *whole* track. Cause: per-note randomness was keyed by note index, so inserting notes shifted every later note. It is now keyed by note content, and a test guards it.

**Master bus coupling: the failure mode to know about.** VANTA's master has a clipper and a limiter, so changing the 808 changes how the *other* tracks get clipped in the final mix. After compensating for master gain, the post-master difference deviates from "only the 808 changed" by −8.7 to −9.0 dB relative to the change itself. Practical consequences:
- the stems stay independent;
- the *mastered* mix is a non-linear blend, like in any real session;
- the "Exact master" monitor lets users hear it.

Replacing a loud blown 808 with a clean one also dropped integrated loudness from −5.8 to −11.3 LUFS, because master gain is kept as the user set it. The UI should offer "re-master to target" (`Renderer.calibrate_master`) as a button. It is not wired yet.

## 5–6. Play and export

Browser test (headless Chromium, `studio.html` against the real server):
- generated a project;
- loaded 8 stems into Web Audio and played them in sync;
- applied 3 natural-language edits, an undo, the exact-master playback and QA;
- at phone width (390 px): no console errors, no horizontal overflow.

Export WAV/MP3/stems runs as a background job with an integrity check.

## Transcription vs ground truth (Basic Pitch 0.4.0, ONNX)

This is the experiment the project needed: we *have* the true notes, so we can measure what audio-to-MIDI actually recovers.

| Stem | Ref notes | Detected | Precision | Recall | **F1 (onset + pitch)** | F1 ignoring octave |
|---|---|---|---|---|---|---|
| Rage synth melody | 183 | 200 | 0.78 | 0.85 | **0.81** | 0.81 |
| 808, clean | 97 | 39 | 0.39 | 0.16 | **0.22** | 0.22 |
| 808, blown-out | 97 | 61 | 0.05 | 0.03 | **0.04** | 0.05 |
| Warm pad chords (slow attack) | 72 | 119 | 0.03 | 0.06 | **0.04** | 0.04 |

**Conclusion.**
- Transcription is usable as a *suggestion* for clean monophonic leads.
- It is **unusable for distorted 808s** and slow-attack pads.
- This confirms the rule "never treat recovered MIDI as ground truth", and it strongly favours the structured architecture, where the notes exist by design.

**Gotcha:** passing `minimum_frequency` to basic-pitch 0.4.0 made it return nothing.

## Architecture A data: the Demucs stems already in the repo (6 songs, analysis only, anonymised)

| Ref | Bass THD % | Bass harmonic ratio dB | Crest dB (active) | Sum of stems vs original mix |
|---|---|---|---|---|
| ref_1 | 26.6 | −17.1 | 7.9 | −8.9 dB (poor) |
| ref_2 | 21.6 | −22.5 | 8.0 | +2.8 dB (stems don't match this upload) |
| ref_3 | 8.9 | −28.9 | 7.2 | +3.0 dB (mismatch) |
| ref_4 | 20.6 | −21.2 | 3.8 | +2.8 dB (mismatch) |
| ref_5 | 27.4 | −12.3 | 5.5 | −40.0 dB (good) |
| ref_6 | 25.0 | −16.2 | 2.7 | −33.2 dB (good) |

- Where the stems match the upload, the four Demucs stems sum back to the mix within −33 to −40 dB.
- Three folders evidently came from a different file version or offset.
- The separated bass stems show **no flat-topped clipping at all** (flat-top ratio about 1e-5), even on tracks whose masters clip thousands of times. This supports the practitioner claim that separation moves the clipped harmonics out of the "bass" stem. The distortion that defines the sound is spread across other stems, which is why "separate, then edit the 808" can't fully capture an underground 808.

## Underground 808 study: clean render + DSP vs reference distortion

Our 808 renders on the same 32-beat pattern:

| Render | THD % | Harmonic ratio dB | Crest dB |
|---|---|---|---|
| 808_clean | 5.5 | −33.6 | 4.9 |
| 808_warm | 24.0 | −15.9 | 1.4 |
| 808_distorted | 33.4 | −12.4 | 0.9 |
| 808_blown | 34.7 | −11.6 | 0.8 |
| clean + DSP hard distortion (sub kept) | 55.9 | −7.4 | 4.9 |
| clean + DSP clipper +12 dB | 30.6 | −14.1 | 1.0 |
| warm + tube + clipper | 41.0 | −9.1 | 0.2 |

- **Spread:** a clean render plus DSP covers THD from 5 % to 56 % and harmonic ratio from −34 to −7 dB. That spans and exceeds the reference range (THD 9–27 %, ratio −29 to −12 dB), with the parameters explicit and adjustable after generation.
- **Mismatch:** our distorted presets are far more compressed (crest about 1 dB) than the reference bass stems (3–8 dB). The references keep their transient punch and decay. Our tanh drive flattens the whole envelope. **Action:** add envelope-aware drive (drive that follows the amplitude envelope) and calibrate presets to the reference crest.
- **What this does *not* show:** that DSP beats a neural model generating a distorted mix directly. That needs ACE-Step on a GPU (not available here) and a listening test. The protocol is in `plan.md`.

## Character consistency (5 characters × 3 seeds)

- **Features:** sub share, high share, 808 harmonic ratio, 808 THD, hats per bar, BPM. LUFS is excluded because the calibration target sets it.
- **Distances:** mean within-character distance 1.28, between-character 3.71.
- **Identification:** leave-one-out nearest-centroid accuracy **15/15**.
- **Loudness after calibration:**
  - VANTA −5.9
  - Pulse Runner −9.3 (target −9)
  - Mercury Halo −9.2
  - Lowtide −11.3
  - Static Saint −11.2 to −11.6 (target −10.5; the limiter caps it)

## Known failures and limits (honest list)

- **Tempo estimation:** librosa's estimator reports 215 BPM (a 3/2 relation) on half-time trap at 165 BPM. Use the stem timing check instead.
- **Key estimation:** the Krumhansl profile estimator often reports the relative or parent key for modal songs (phrygian, dorian). QA reports "same pitch collection" rather than pass/fail.
- **Sound range:** the procedural engine has no vocals and no acoustic instruments. Requests for them are refused with an explanation, not faked.
- **Time signature:** only 4/4.
- **Long renders:** a 3-minute song is about 45–60 s to render on 4 vCPU (scales linearly with length). Edits re-render only the changed tracks.
- **Untested providers:** the ACE-Step client and the Demucs provider can't run here (no GPU; Hugging Face, GitHub and fbaipublicfiles downloads are blocked in this sandbox). They are implemented against documented APIs and marked untested.

## Listening rubric (to do on a Mac with headphones and phone speakers)

For each MP3 pair, score 1–5:
1. Does it sound like the character?
2. Is the 808 the right kind of dirty?
3. Did anything you *didn't* change sound different? (Expected: no, except master-bus interaction.)
4. Would you post this as a sketch?

Record the scores in `poc/listening.md`.
