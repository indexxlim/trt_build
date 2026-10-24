"""공통 입력(data/inputs.npz)과 HF 기준 결과(data/reference.json)를 만든다.

conda env `trt` (transformers 4.24, av, datasets)에서 한 번만 실행:
    CUDA_VISIBLE_DEVICES=0 python benchmark/prepare_inputs.py
"""

import io
import itertools
import json
import os
import sys

import av
import numpy as np
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, INPUTS, REFERENCE, VARIANT  # noqa: E402

KOREAN_NEWS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "korean_news.mp4")


def decode_audio(path, sr=16000):
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sr)
    buf = io.BytesIO()
    with av.open(path, metadata_errors="ignore") as c:
        for frame in itertools.chain(c.decode(audio=0), [None]):
            for f in resampler.resample(frame):
                buf.write(f.to_ndarray())
    return np.frombuffer(buf.getbuffer(), dtype=np.int16).astype(np.float32) / 32768.0


def main():
    processor = WhisperProcessor.from_pretrained(VARIANT)
    tok = processor.tokenizer
    model = WhisperForConditionalGeneration.from_pretrained(VARIANT).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = {"en": np.asarray(ds[0]["audio"]["array"], dtype=np.float32), "ko": decode_audio(KOREAN_NEWS)[: 30 * 16000]}

    arrays, reference = {}, {}
    for s, lang in [("en", "en"), ("ko", "ko")]:
        feats = processor(audio[s], sampling_rate=16000, return_tensors="pt").input_features
        forced = processor.get_decoder_prompt_ids(language=lang, task="transcribe", no_timestamps=True)
        prompt = [model.config.decoder_start_token_id] + [t for _, t in forced]
        with torch.no_grad():
            ids = model.generate(feats.to(device), forced_decoder_ids=forced, max_length=448, num_beams=1)[0]
        text = tok.decode(ids, skip_special_tokens=True)
        arrays[f"{s}_features"] = feats.numpy().astype(np.float32)
        arrays[f"{s}_prompt"] = np.array(prompt, dtype=np.int64)
        reference[s] = {"text": text, "n_tokens": int(len(ids) - len(prompt)), "audio_sec": len(audio[s]) / 16000}
        print(s, reference[s])

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(INPUTS, **arrays)
    with open(REFERENCE, "w", encoding="utf-8") as f:
        json.dump(reference, f, ensure_ascii=False, indent=2)
    print("saved", INPUTS, REFERENCE)


if __name__ == "__main__":
    main()
