"""Fake set 1: Meta MMS-TTS Arabic (facebook/mms-tts-ara), a standard
text-to-speech model with one synthetic voice. Reads the same sentences as
the real clips.

    python -m deepfake.fake_mms
"""
import csv
from pathlib import Path

import torch
from transformers import AutoTokenizer, VitsModel

from . import audio

OUT = Path("data/fake_mms")


def main():
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ara")
    model = VitsModel.from_pretrained("facebook/mms-tts-ara").eval()
    rows = list(csv.DictReader(open("data/real.tsv", encoding="utf-8"), delimiter="\t"))
    for i, r in enumerate(rows):
        out = OUT / f"mms_{r['id']}.wav"
        if out.exists():
            continue
        torch.manual_seed(i)
        with torch.no_grad():
            wave = model(**tok(r["text"], return_tensors="pt")).waveform[0].numpy()
        audio.save(out, audio.prepare(wave, model.config.sampling_rate))
    print(f"{len(rows)} MMS clips in {OUT}")


if __name__ == "__main__":
    main()
