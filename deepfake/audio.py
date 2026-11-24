"""Shared audio handling. Every clip, real or fake, goes through the same steps,
so a detector cannot tell them apart by format, loudness or leading silence.
"""
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16_000
MAX_SECONDS = 4  # same window for every clip, so length is never a clue


def prepare(wave, sr):
    """Mono, 16 kHz, trimmed silence, at most 4 s, same loudness."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != SR:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=SR)
    wave, _ = librosa.effects.trim(wave, top_db=30)
    wave = wave[: SR * MAX_SECONDS]
    rms = np.sqrt(np.mean(wave**2)) or 1.0
    return np.clip(wave * (0.05 / rms), -1, 1).astype(np.float32)


def save(path, wave):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, SR, subtype="PCM_16")


def load(path):
    wave, sr = sf.read(path, dtype="float32")
    return prepare(wave, sr)


def phone_call(wave):
    """Simulate a WhatsApp voice note / phone call: 8 kHz, Opus at 16 kbit/s.

    Scam voice messages usually arrive this way, and compression removes many
    of the high-frequency traces detectors rely on.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src, enc, dst = (Path(tmp) / n for n in ("in.wav", "x.ogg", "out.wav"))
        save(src, wave)
        run = lambda *a: subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *a], check=True)  # noqa: E731
        run("-i", src, "-ar", "8000", "-ac", "1", "-c:a", "libopus", "-b:a", "16k", enc)
        run("-i", enc, "-ar", str(SR), dst)
        wave, sr = sf.read(dst, dtype="float32")
    return prepare(wave, sr)
