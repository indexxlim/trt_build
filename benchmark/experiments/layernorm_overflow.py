"""E2. large-v2 인코더 LayerNorm 입력이 FP16 범위(65504)를 넘는지 측정 (conda env `trt`, CPU로 충분).

opset 12 export는 LayerNorm을 ReduceMean / Sub / Pow / ReduceMean ... 으로 풀어서 내보내고, TRT FP16에서는
(x - mean)^2 를 FP16으로 계산한다. 이 값이 65504를 넘으면 inf -> NaN.

    CUDA_VISIBLE_DEVICES= python benchmark/experiments/layernorm_overflow.py
"""
import _paths  # noqa: F401
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

torch.set_num_threads(16)
V = "openai/whisper-large-v2"
model = WhisperForConditionalGeneration.from_pretrained(V).eval()
proc = WhisperProcessor.from_pretrained(V)
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
x = proc(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_features

stats = []


def hook(name):
    def f(mod, inp, out):
        h = inp[0].float()
        d = h - h.mean(-1, keepdim=True)
        stats.append((name, h.abs().max().item(), (d ** 2).max().item(), (d ** 2).sum(-1).max().item()))
    return f


for n, m in model.model.encoder.named_modules():
    if isinstance(m, torch.nn.LayerNorm):
        m.register_forward_hook(hook(n))
with torch.no_grad():
    model.model.encoder(x)

print(f"{'layernorm':32s} {'max|x|':>8s} {'max (x-mean)^2':>15s} {'sum_d (x-mean)^2':>17s}")
for s in sorted(stats, key=lambda s: -s[3])[:8]:
    print(f"{s[0]:32s} {s[1]:8.1f} {s[2]:15.0f} {s[3]:17.0f}")
print("FP16 max = 65504; overflow layers (square / sum):",
      sum(s[2] > 65504 for s in stats), "/", sum(s[3] > 65504 for s in stats), "of", len(stats))
