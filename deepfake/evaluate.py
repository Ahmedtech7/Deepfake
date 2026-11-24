"""Run every experiment and write results/RESULTS.md.

    python -m deepfake.evaluate

Metrics (fake = positive class):
  EER  equal error rate: the threshold where real clips flagged as fake and
       fakes passed as real happen equally often. Lower is better; 50% is guessing.
  AUC  area under the ROC curve. 1.0 is perfect; 0.5 is guessing.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_SHARE = 0.4
SEED = 0


def eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def metrics(labels, scores):
    return {"eer": eer(labels, scores), "auc": float(roc_auc_score(labels, scores)), "n": int(len(labels))}


def sentence_of(clip_id):
    """mms_fleurs_12 / clone_fleurs_12 / fleurs_12 -> fleurs_12 (the shared text)."""
    return clip_id[clip_id.index("fleurs_"):]


def split_sentences(ids):
    """Split by sentence, so no text is in both train and test."""
    sentences = sorted({sentence_of(i) for i in ids})
    rng = np.random.default_rng(SEED)
    test = set(rng.choice(sentences, size=int(len(sentences) * TEST_SHARE), replace=False))
    return np.array([sentence_of(i) in test for i in ids])


def load(condition):
    d = np.load(f"results/emb_{condition}.npz")
    return d["ids"], d["sources"], d["X"]


def train(X, sources):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
    return clf.fit(X, (sources != "real").astype(int))


def score(clf, X):
    return clf.predict_proba(X)[:, 1]


def subset(mask_sources, sources, keep):
    return mask_sources & np.isin(sources, keep)


def run_experiments():
    ids, src, X = load("clean")
    _, src_p, X_p = load("phone")
    test = split_sentences(ids)
    train_mask = ~test
    results = {}

    def evaluate(name, clf, Xe, mask, fakes):
        m = subset(mask, src, ["real", *fakes])
        results[name] = metrics((src[m] != "real").astype(int), score(clf, Xe[m]))

    # 1. Seen generators: train on real + MMS + clone, test on new sentences.
    both = train(X[subset(train_mask, src, ["real", "mms", "clone"])], src[subset(train_mask, src, ["real", "mms", "clone"])])
    evaluate("seen_generators_clean", both, X, test, ["mms", "clone"])
    evaluate("seen_generators_phone", both, X_p, test, ["mms", "clone"])

    # 2. Unseen generator: train only on MMS fakes, test on voice clones it never saw.
    mms_only = train(X[subset(train_mask, src, ["real", "mms"])], src[subset(train_mask, src, ["real", "mms"])])
    evaluate("unseen_clone_clean", mms_only, X, test, ["clone"])

    # 3. Phone-call training: add compressed copies of the training clips.
    m = subset(train_mask, src, ["real", "mms", "clone"])
    aug = train(np.vstack([X[m], X_p[m]]), np.concatenate([src[m], src[m]]))
    evaluate("phone_trained_phone", aug, X_p, test, ["mms", "clone"])
    evaluate("phone_trained_clean", aug, X, test, ["mms", "clone"])
    return results


def zero_shot(results):
    """English-trained detector from Hugging Face, applied as-is to Arabic."""
    path = Path("results/zero_shot.json")
    if path.exists():
        results.update(json.loads(path.read_text()))


def write(results):
    names = {
        "zero_shot_clean": "Off-the-shelf English-trained detector, clean audio",
        "zero_shot_phone": "Off-the-shelf English-trained detector, phone-quality audio",
        "seen_generators_clean": "Our detector, clean audio",
        "seen_generators_phone": "Our detector, phone-quality audio (trained on clean)",
        "phone_trained_phone": "Our detector trained with phone audio, phone-quality audio",
        "phone_trained_clean": "Our detector trained with phone audio, clean audio",
        "unseen_clone_clean": "Our detector trained without voice clones, tested on voice clones",
    }
    lines = ["# Results", "", "Test set: sentences never used in training. Fake = MMS-TTS + Chatterbox voice clones unless noted.", "",
             "| Experiment | EER (lower is better) | AUC | Test clips |", "|---|---|---|---|"]
    for key, label in names.items():
        if key in results:
            r = results[key]
            lines.append(f"| {label} | {r['eer']:.1%} | {r['auc']:.3f} | {r['n']} |")
    Path("results/RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("results/metrics.json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines))


def main():
    results = run_experiments()
    zero_shot(results)
    write(results)


if __name__ == "__main__":
    main()
