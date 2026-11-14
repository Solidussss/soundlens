"""Source separation (Architecture A: generate/upload a mix, then split it).

Uses the ``demucs`` CLI the SoundLens analyzer already relies on
(``soundlens_pro.run_demucs``). demucs 4.1 downloads weights from Hugging Face
(``adefossez/HTDemucs``), falling back to ``dl.fbaipublicfiles.com``. The weight
licence is described by its author as "not clear" — get legal review before
using separated output commercially.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .base import Provider, ProviderUnavailable

STEM_ROLES = {"vocals": "lead_vocal", "drums": "other", "bass": "808", "other": "other", "piano": "piano", "guitar": "other"}


class DemucsProvider(Provider):
    name = "demucs"
    capabilities = {"separate_4", "separate_6"}
    cost_note = "CPU ≈1.5× song length for htdemucs (author's figure); GPU much faster"

    def available(self) -> tuple[bool, str]:
        try:
            import demucs  # noqa: F401
        except ImportError:
            return False, "pip install demucs (weights download from Hugging Face on first use)"
        return True, "installed (weights fetched on first run)"

    def separate(self, audio: Path, out_dir: Path, model: str = "htdemucs") -> dict[str, Path]:
        self.require()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(out_dir), str(audio)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise ProviderUnavailable(f"demucs failed: {proc.stderr[-500:]}")
        folder = out_dir / model / audio.stem
        stems = {p.stem: p for p in folder.glob("*.wav")}
        if not stems:
            raise ProviderUnavailable("demucs produced no stems")
        return stems


def existing_stems(folder: Path) -> dict[str, Path]:
    """Load stems already produced by the analyzer (``stems/htdemucs/<song>/``)."""
    return {p.stem: p for p in Path(folder).glob("*.wav")}


_ = shutil
