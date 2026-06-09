"""Audio prosody extractor: openSMILE eGeMAPSv02 (88 affective-speech functionals) + an
energy/silence probe. Weak-but-orthogonal on Track 1 (keep for fusion); noise on Track 2.

NOTE: ~85% of clips are near-silent (participants wore headphones), so most of the signal here is
in the rare voiced clips. Requires `opensmile` (pip) and ffmpeg on PATH.
"""
from __future__ import annotations
import numpy as np
from .base import BaseExtractor
from ..datasets import video_path


class AudioExtractor(BaseExtractor):
    name = "audio"

    def __init__(self, workers: int = 8, cache_dir=None):
        super().__init__(s=0, workers=workers, cache_dir=cache_dir)

    def init_worker(self):
        import opensmile
        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                     feature_level=opensmile.FeatureLevel.Functionals)

    def extract_clip(self, track, participant, video) -> dict:
        import tempfile, os, subprocess, soundfile as sf
        mp4 = video_path(track, participant, video)
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "a.wav")
            subprocess.run(["ffmpeg", "-y", "-i", str(mp4), "-ac", "1", "-ar", "16000", wav],
                           capture_output=True)
            if not os.path.exists(wav):
                return {"n_det": 0}
            feats = self.smile.process_file(wav).iloc[0].to_dict()
            x, sr = sf.read(wav)
            x = np.asarray(x, float)
            rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
            db = 20 * np.log10(rms)
            frame = sr // 50
            energies = np.array([np.sqrt(np.mean(x[i:i + frame] ** 2) + 1e-12)
                                 for i in range(0, max(len(x) - frame, 1), frame)])
            sil = float(np.mean(energies < 10 ** (-50 / 20))) if energies.size else 1.0
        feats.update(rms_db=db, silent_frac=sil, voiced_frac=1 - sil,
                     energy_std=float(energies.std()) if energies.size else 0.0,
                     energy_max=float(energies.max()) if energies.size else 0.0)
        feats["n_det"] = 1
        return feats
