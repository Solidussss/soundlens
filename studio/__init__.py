"""SoundLens Studio: structured AI music creation and multitrack editing.

The package is deliberately independent from the existing analyzer
(``app.py``/``soundlens_pro.py``). ``app.py`` mounts :mod:`studio.api` when it
is importable; nothing else in SoundLens depends on this package.

Layers (each importable and testable without the web server):

* :mod:`studio.theory`      keys, scales, chords
* :mod:`studio.project`     the editable project document (tracks, clips, notes, fx)
* :mod:`studio.synth`       DSP instruments that render note data to audio
* :mod:`studio.dsp`         real effects (EQ, filters, distortion, compression, ...)
* :mod:`studio.render`      deterministic per-track rendering + mixdown + export
* :mod:`studio.characters`  character profiles (musical identities)
* :mod:`studio.composer`    structured generation (sections, harmony, parts)
* :mod:`studio.actions`     validated, reversible edit operations
* :mod:`studio.orchestrator` natural language -> actions
* :mod:`studio.analysis`    measurable quality checks
* :mod:`studio.providers`   interchangeable generative back ends
"""

SAMPLE_RATE = 44100
