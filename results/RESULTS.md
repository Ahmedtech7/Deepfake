# Results

Test set: sentences never used in training. Fake = MMS-TTS + Chatterbox voice clones unless noted.

| Experiment | EER (lower is better) | AUC | Test clips |
|---|---|---|---|
| Off-the-shelf English-trained detector, clean audio | 60.5% | 0.430 | 415 |
| Off-the-shelf English-trained detector, phone-quality audio | 58.6% | 0.490 | 415 |
| Our detector, clean audio | 0.0% | 1.000 | 415 |
| Our detector, phone-quality audio (trained on clean) | 0.0% | 1.000 | 415 |
| Our detector trained with phone audio, phone-quality audio | 0.0% | 1.000 | 415 |
| Our detector trained with phone audio, clean audio | 0.0% | 1.000 | 415 |
| Our detector trained without voice clones, tested on voice clones | 24.1% | 0.850 | 245 |
