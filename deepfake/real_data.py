"""Download real Arabic speech: Google FLEURS, Arabic (ar_eg), CC BY 4.0.

Keeps one recording per sentence, so the same text never sits in both the
training and the test split, and writes data/real/*.wav plus data/real.tsv
(id, speaker gender, text). The texts are reused to generate the fakes.

    python -m deepfake.real_data
"""
import csv
import io
from pathlib import Path

import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download

from . import audio

OUT = Path("data/real")
SPLITS = ("validation", "test")


def main():
    rows, seen = [], set()
    for split in SPLITS:
        path = hf_hub_download("google/fleurs", f"parquet-data/ar_eg/{split}-00000-of-00001.parquet", repo_type="dataset")
        for r in pd.read_parquet(path).itertuples():
            if r.id in seen:
                continue
            seen.add(r.id)
            wave, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
            clip_id = f"fleurs_{r.id}"
            audio.save(OUT / f"{clip_id}.wav", audio.prepare(wave, sr))
            rows.append({"id": clip_id, "gender": r.gender, "text": r.raw_transcription})
    with open("data/real.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "gender", "text"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} real clips in {OUT}")


if __name__ == "__main__":
    main()
