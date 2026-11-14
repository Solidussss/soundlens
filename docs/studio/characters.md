# Characters learned from songs

## What it does

1. In the Studio, click **+ New character**. Give it a name and a look, add 5–10 songs by one artist, tick the rights box, then click **Learn this character**.
2. **Listening** (about 10 s per song on a laptop CPU). For each song, `studio/learn.py` measures:
   - tempo and half-time vs full-time feel;
   - one-bar grooves on a 16-step grid: the low-end groove (kick + 808 attacks), snares/claps, hi-hat density, 32nd and triplet rolls;
   - 808: register, note lengths, slides and octave jumps, distortion (harmonics relative to its own fundamental);
   - key and mode, and the common 4-bar chord moves;
   - melodic texture: density, register, sustained vs plucked;
   - section lengths, loudness, clipping, stereo, brightness.
3. **Learning.** Grooves are clustered (bars within 2 hits of each other) and kept as majority-vote patterns, so stray detections don't become "the groove". Everything else is a median across songs. Tempos are folded to the artist's consensus, so one half/double-time misread doesn't move the character.
4. **Calibration.** The character generates short test beats, measures them with the same analyzer, and adjusts its own knobs until its output measures like the uploads. The knobs are 808 level, drive and register, hat level, rate and rolls, melody density and register, and loudness target.
5. **Cleanup.** The uploaded songs are deleted. Only the statistics and groove patterns are stored, on the character (`<data>/studio/characters/<owner>/`). Characters are private to the account that made them.

## What it does not do

- It does not copy melodies, samples, vocals or any audio. Generated songs are new notes played by the built-in instruments, following the learned *style*.
- It does not imitate a voice. There are no vocals yet; when ACE-Step vocals arrive they use original voices only.
- It does not use an artist's name or likeness. Avatars are original cartoon rappers built from parts (`studio/avatars.py`). The UI never shows which artist a character learned from; users name their own characters.

**Why:** a cartoon of a real rapper, or their name, used to sell a product runs into right-of-publicity and trademark law, and implies they endorse SoundLens. Musical style (tempo, drum habits, 808 treatment) isn't owned by anyone, but the recordings are. That is why the songs are analyzed and then deleted, never stored or used to train a model. Training a neural model (e.g. an ACE-Step LoRA) on an artist's songs needs a licence, and the reference-collection rights record (`studio/references.py`) enforces that.

## How accurate is the listening? (measured)

**Ground truth** (`experiments/learn_ground_truth.py`): 10 rendered songs whose exact patterns are known, analyzed from the plain stereo mix, like an upload.

| Measure | Result |
|---|---|
| Tempo exactly right (±0.2 %) | 8 / 10 (the 2 misses are half/double-time reads) |
| Half- vs full-time feel | 9 / 10 |
| Snare/clap position agreement (F1) | 0.52 |
| 808 / low-end groove agreement (F1) | 0.40 |

**Reading:** the learner captures *tendencies* (tempo, feel, where the low end and backbeat sit, roll and hat density) reliably enough to steer generation. It is not a beat transcriber. Separating stems with Demucs first (`pip install demucs` on your Mac) is expected to improve the drum and 808 numbers; this hasn't been measured yet because the sandbox can't download the weights.

**Style match** (`experiments/style_match.py`): see the results section below.
