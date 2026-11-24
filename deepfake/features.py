"""Embed every clip with XLS-R (facebook/wav2vec2-xls-r-300m), a speech model
pre-trained on 128 languages including Arabic. A clip becomes the mean of one
hidden layer's frames (1024 numbers). Early-to-middle layers keep the
low-level acoustic traces that synthetic speech leaves behind.

Embeddings are cached per condition in results/emb_<condition>.npz.

    python -m deepfake.features            # clean + phone
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from . import audio

MODEL = "facebook/wav2vec2-xls-r-300m"
LAYER = 6
SETS = {"real": "data/real", "mms": "data/fake_mms", "clone": "data/fake_clone"}
CONDITIONS = {"clean": lambda w: w, "phone": audio.phone_call}


def clips():
    for source, folder in SETS.items():
        for path in sorted(Path(folder).glob("*.wav")):
            yield source, path


def load_model(device):
    extractor = AutoFeatureExtractor.from_pretrained(MODEL)
    return extractor, Wav2Vec2Model.from_pretrained(MODEL).to(device).eval()


def vector(extractor, model, wave, device):
    """One prepared clip -> its 1024-number embedding."""
    inputs = extractor(wave, sampling_rate=audio.SR, return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = model(**inputs, output_hidden_states=True).hidden_states[LAYER]
    return hidden.mean(dim=1)[0].cpu().numpy()


def embed(condition, device):
    extractor, model = load_model(device)
    ids, sources, vecs = [], [], []
    for source, path in clips():
        vecs.append(vector(extractor, model, CONDITIONS[condition](audio.load(path)), device))
        ids.append(path.stem)
        sources.append(source)
    out = Path(f"results/emb_{condition}.npz")
    np.savez(out, ids=np.array(ids), sources=np.array(sources), X=np.stack(vecs))
    print(f"{len(ids)} clips -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = p.parse_args(argv)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for c in args.conditions:
        embed(c, device)


if __name__ == "__main__":
    main()
