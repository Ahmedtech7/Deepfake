# Deepfake: Arabic Voice Detector

Detects AI-generated Arabic speech, including voice clones delivered as
WhatsApp-quality audio, and measures how an off-the-shelf English-trained
detector does on the same clips.

## Data

- **Real:** 427 Arabic clips from Google FLEURS (ar_eg, CC BY 4.0), one per sentence.
- **Fake, text-to-speech:** the same sentences read by Meta MMS-TTS (Arabic).
- **Fake, voice clones:** 200 sentences spoken in a real speaker's voice by
  Chatterbox Multilingual (each clone uses a different speaker's clip as reference).

Every clip goes through the same steps (16 kHz mono, trimmed, 4 s, same
loudness), so format, length and volume give nothing away. The train/test split
is by sentence: no text appears in both.

## Method

XLS-R 300m (pre-trained on 128 languages) turns each clip into a 1024-number
embedding (layer 6, mean-pooled); a logistic regression classifies it.
"Phone quality" is 8 kHz Opus at 16 kbit/s, as in WhatsApp voice notes.

## Results (unseen test sentences)

| Experiment | EER | AUC |
|---|---|---|
| English-trained detector (MelodyMachine V2), clean | 60.5% | 0.43 |
| English-trained detector, phone quality | 58.6% | 0.49 |
| Ours, clean | 0.0% | 1.00 |
| Ours, phone quality (trained on clean audio) | 0.0% | 1.00 |
| Ours, trained without voice clones, tested on voice clones | 24.1% | 0.85 |

The English detector caught 87% of the voice clones but none of the Arabic
text-to-speech clips, so its overall ranking is no better than chance.

**Limits.** All real clips come from one dataset and the fakes from two
generators, so this test is easier than real life. A generator the detector
has never seen lowers accuracy (last row). Shuffling the training labels drops
the detector to chance, which rules out leakage in the evaluation.

## Run

Needs Python 3.11+ and `ffmpeg`. `pip install -r requirements.txt`

```bash
python -m deepfake.real_data      # real clips
python -m deepfake.fake_mms       # text-to-speech fakes
python deepfake/fake_clone.py     # voice clones (separate env: chatterbox-tts)
python -m deepfake.features       # embeddings, clean + phone
python -m deepfake.zero_shot      # English-trained baseline
python -m deepfake.evaluate       # results/RESULTS.md
python -m deepfake.demo           # local demo page at http://127.0.0.1:7860
pytest
```

## Responsible use

Generated audio stays local and is not published. Voice clones were made only
from openly licensed recordings, for detection research. MMS-TTS is licensed
CC BY-NC 4.0, so this project is non-commercial.
