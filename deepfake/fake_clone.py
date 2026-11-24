"""Fake set 2: voice-cloning deepfakes with Chatterbox Multilingual
(Resemble AI, MIT license). For each sentence it copies the voice of a
different real speaker of the same gender from a short reference clip,
which is how scam voice clones are made.

Runs in its own environment (.venv-tts, Python 3.11) because Chatterbox pins
older libraries:
    .venv-tts/bin/python deepfake/fake_clone.py
"""
import csv
import random
from pathlib import Path

import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

OUT = Path("data/fake_clone")
CLIPS = 200
WORDS = 10


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    rows = list(csv.DictReader(open("data/real.tsv", encoding="utf-8"), delimiter="\t"))
    rng = random.Random(0)
    # shortcut: 200 sentences, first 10 words each (~1 min per clip on an M3 laptop);
    # generate all 427 in full on a GPU for a bigger test set.
    targets = rng.sample(rows, CLIPS)
    by_gender = {}
    for r in rows:
        by_gender.setdefault(r["gender"], []).append(r["id"])
    OUT.mkdir(parents=True, exist_ok=True)
    for r in targets:
        out = OUT / f"clone_{r['id']}.wav"
        if out.exists():
            continue
        # Reference voice: another speaker's clip, never the clip being faked.
        ref = rng.choice([i for i in by_gender[r["gender"]] if i != r["id"]])
        text = " ".join(r["text"].split()[:WORDS])
        wave = model.generate(text, language_id="ar", audio_prompt_path=f"data/real/{ref}.wav")
        sf.write(out, wave.squeeze().cpu().numpy(), model.sr)
        print(out.name, "<-", ref, flush=True)


if __name__ == "__main__":
    main()
