"""Baseline: a popular English-trained deepfake detector from Hugging Face
(MelodyMachine/Deepfake-audio-detection-V2), applied unchanged to the same
Arabic test clips. Writes results/zero_shot.json for evaluate.py.

    python -m deepfake.zero_shot
"""
import json
from pathlib import Path

import numpy as np
from transformers import pipeline

from . import audio, evaluate, features

MODEL = "MelodyMachine/Deepfake-audio-detection-V2"


def fake_score(clf, wave):
    """Probability of the model's 'fake' label."""
    out = clf({"raw": wave, "sampling_rate": audio.SR}, top_k=None)
    return next(o["score"] for o in out if o["label"].lower() in {"fake", "spoof", "deepfake"})


def main():
    clf = pipeline("audio-classification", model=MODEL)
    items = list(features.clips())
    ids = np.array([p.stem for _, p in items])
    test = evaluate.split_sentences(ids)
    results = {}
    for condition, transform in features.CONDITIONS.items():
        labels, scores = [], []
        for (source, path), is_test in zip(items, test):
            if not is_test:
                continue
            labels.append(int(source != "real"))
            scores.append(fake_score(clf, transform(audio.load(path))))
        results[f"zero_shot_{condition}"] = evaluate.metrics(np.array(labels), np.array(scores))
        print(condition, results[f"zero_shot_{condition}"])
    Path("results/zero_shot.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
