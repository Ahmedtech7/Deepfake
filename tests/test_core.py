import numpy as np
import pytest

from deepfake import audio, evaluate


def tone(seconds, sr=audio.SR, freq=220):
    t = np.arange(int(seconds * sr)) / sr
    return 0.3 * np.sin(2 * np.pi * freq * t)


def test_prepare_resamples_caps_and_levels():
    out = audio.prepare(tone(10, sr=24_000), 24_000)
    assert out.dtype == np.float32
    assert len(out) == audio.SR * audio.MAX_SECONDS
    assert np.sqrt(np.mean(out**2)) == pytest.approx(0.05, rel=0.05)


def test_prepare_trims_leading_silence():
    wave = np.concatenate([np.zeros(audio.SR), tone(1)])
    assert len(audio.prepare(wave, audio.SR)) < 1.2 * audio.SR


def test_phone_call_keeps_rate_and_roughly_the_length():
    out = audio.phone_call(audio.prepare(tone(2), audio.SR))
    assert abs(len(out) - 2 * audio.SR) < 0.1 * audio.SR


def test_phone_call_removes_high_frequencies():
    # 8 kHz sampling keeps nothing above 4 kHz.
    high = audio.prepare(tone(2, freq=6000), audio.SR)
    out = audio.phone_call(high)
    assert np.abs(np.fft.rfft(out)).max() < 0.1 * np.abs(np.fft.rfft(high)).max()


def test_eer_extremes():
    labels = np.array([0, 0, 1, 1])
    assert evaluate.eer(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert evaluate.eer(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0


def test_split_keeps_each_sentence_on_one_side():
    ids = np.array([f"{p}fleurs_{i}" for i in range(50) for p in ("", "mms_", "clone_")])
    test = evaluate.split_sentences(ids)
    side = {}
    for clip, is_test in zip(ids, test):
        assert side.setdefault(evaluate.sentence_of(clip), is_test) == is_test
    assert 0.3 < test.mean() < 0.5
