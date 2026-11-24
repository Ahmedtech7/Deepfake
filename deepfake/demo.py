"""Local demo: upload or record a clip and see whether it sounds real or fake,
next to the English-trained detector for comparison. Runs offline.

    python -m deepfake.demo        # then open http://127.0.0.1:7860

The detector is trained on the training sentences only, clean and phone-quality
(results/emb_*.npz must exist). Example clips in demo/examples come from the
unseen test sentences and stay local.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from transformers import pipeline

from . import audio, evaluate, features, zero_shot

WEB = Path(__file__).parent / "web"
EXAMPLES = Path("demo/examples")
LABELS = {"حقيقي": "real", "مستنسخ": "clone", "آلي": "tts"}


def decode(data):
    """Any browser or phone format (webm, ogg, m4a, mp3, wav) -> prepared 16 kHz clip."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in", Path(tmp) / "out.wav"
        src.write_bytes(data)
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", str(audio.SR), dst])
        if r.returncode:
            raise HTTPException(400, "تعذّر قراءة الملف الصوتي")
        wave, sr = sf.read(dst, dtype="float32")
    return audio.prepare(wave, sr)


def build():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids, src, X = evaluate.load("clean")
    _, _, X_p = evaluate.load("phone")
    train = ~evaluate.split_sentences(ids)  # test sentences stay unseen, so the examples are honest
    ours = evaluate.train(np.vstack([X[train], X_p[train]]), np.concatenate([src[train], src[train]]))
    extractor, model = features.load_model(device)
    english = pipeline("audio-classification", model=zero_shot.MODEL)
    app = FastAPI()

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/examples")
    def examples():
        return [{"name": p.name, "kind": LABELS.get(p.stem.split("_")[0], "real")}
                for p in sorted(EXAMPLES.glob("*.wav"))]

    @app.get("/examples/{name}")
    def example(name: str):
        path = EXAMPLES / Path(name).name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/check")
    async def check(file: UploadFile):
        wave = decode(await file.read())
        if len(wave) < audio.SR // 2:
            raise HTTPException(400, "المقطع قصير، سجّل ثانيتين على الأقل")
        return {
            "ours": float(evaluate.score(ours, features.vector(extractor, model, wave, device)[None])[0]),
            "english": float(zero_shot.fake_score(english, wave)),
        }

    return app


if __name__ == "__main__":
    uvicorn.run(build(), host="127.0.0.1", port=7860)
