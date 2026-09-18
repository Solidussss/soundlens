# SoundLens Visual MIDI Lab

Experimental visual melody generator for SoundLens.

## v0.1
- FL-style piano roll preview
- Heart, star, butterfly and wave silhouettes
- Key + scale constraints
- BPM, bar length and note-density controls
- Visual vs musical balance control
- Chord-weighted note selection
- In-browser synth preview
- Standard MIDI file export

## Run
```bash
cd visual-midi
npm start
```

Then open `http://localhost:3000`.

This is intentionally isolated from the main SoundLens app so it can later become its own service/repository if the concept proves strong.
