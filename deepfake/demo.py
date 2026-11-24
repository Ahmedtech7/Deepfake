"""Local demo page: upload or record a clip, see whether it sounds real or fake,
next to the English-trained detector for comparison. Runs offline.

    python -m deepfake.demo        # then open http://127.0.0.1:7860

The detector is trained on the training sentences only, clean and phone-quality
(results/emb_*.npz must exist). Example clips come from the unseen test sentences.
"""
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from transformers import pipeline

from . import audio, evaluate, features, zero_shot


def build():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids, src, X = evaluate.load("clean")
    _, _, X_p = evaluate.load("phone")
    train = ~evaluate.split_sentences(ids)  # test sentences stay unseen, so the examples are honest
    ours = evaluate.train(np.vstack([X[train], X_p[train]]), np.concatenate([src[train], src[train]]))
    extractor, model = features.load_model(device)
    english = pipeline("audio-classification", model=zero_shot.MODEL)

    def check(clip):
        if clip is None:
            return "ارفع مقطع أو سجّل صوتك أولاً", ""
        sr, wave = clip
        wave = wave.astype(np.float32)
        if np.abs(wave).max() > 1:  # Gradio gives int16 samples
            wave /= 32768
        wave = audio.prepare(wave, sr)
        if len(wave) < audio.SR // 2:
            return "المقطع قصير، سجّل ثانيتين على الأقل", ""
        p_ours = evaluate.score(ours, features.vector(extractor, model, wave, device)[None])[0]
        p_en = zero_shot.fake_score(english, wave)
        verdict = lambda p: f"{'🔴 مزيّف' if p >= 0.5 else '🟢 حقيقي'} — احتمال التزييف {p:.0%}"  # noqa: E731
        return verdict(p_ours), verdict(p_en)

    examples = sorted(str(p) for p in Path("demo/examples").glob("*.wav"))
    with gr.Blocks(title="كاشف الأصوات المزيفة بالعربي") as page:
        gr.Markdown("# كاشف الأصوات المزيفة بالعربي\nارفع مقطع صوتي أو سجّل بالمايك، والكاشف يقول هل الصوت حقيقي أو مولّد بالذكاء الاصطناعي.")
        clip = gr.Audio(sources=["upload", "microphone"], type="numpy", label="المقطع الصوتي")
        btn = gr.Button("افحص", variant="primary")
        with gr.Row():
            out_ours = gr.Textbox(label="كاشفنا (متدرّب على العربي)")
            out_en = gr.Textbox(label="كاشف إنجليزي جاهز (للمقارنة)")
        if examples:
            gr.Examples(examples, inputs=clip, label="مقاطع جاهزة (حقيقي / مستنسخ)")
        gr.Markdown("مشروع بحثي: متدرّب على تسجيلات محدودة وأداتي تزييف، وممكن يغلط على أصوات من أدوات ما شافها.")
        btn.click(check, clip, [out_ours, out_en])
    return page


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", css=".gradio-container{direction:rtl}")
