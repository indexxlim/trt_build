"""E5. torch.compile(backend="torch_tensorrt")를 인코더만 / 인코더+디코더에 걸었을 때 컴파일 시간과 재컴파일 (whisper-tiny).

Torch-TensorRT 1.4.0은 정적 shape만 지원해서 KV cache 길이가 바뀌는 디코더는 step마다 재컴파일된다.

    CUDA_VISIBLE_DEVICES=0 python benchmark/experiments/torchtrt_decoder_recompile.py
"""
import os, sys, time, logging
import _paths  # noqa: F401
import torch, torch_tensorrt, torch._dynamo as dynamo
from transformers import WhisperForConditionalGeneration
from common import load_inputs
V = os.environ.get("V", "openai/whisper-tiny")
x = load_inputs()["en"]; feats = torch.from_numpy(x["features"]).cuda(); prompt = x["prompt"]
forced = [(i, t) for i, t in enumerate(prompt[1:], 1)]
opts = {"enabled_precisions": {torch.float, torch.half}}

def run(tag, model):
    dynamo.reset()
    t0 = time.time()
    with torch.no_grad(): ids = model.generate(feats, forced_decoder_ids=forced, max_length=448)[0]
    t1 = time.time()
    with torch.no_grad(): model.generate(feats, forced_decoder_ids=forced, max_length=448)
    torch.cuda.synchronize(); t2 = time.time()
    print(f"{tag}: first {t1-t0:.1f}s | second {1000*(t2-t1):.0f}ms | {len(ids)} ids | frames compiled: {dynamo.utils.counters['stats'].get('unique_graphs')}")

m = WhisperForConditionalGeneration.from_pretrained(V).eval().cuda()
run("eager", m)
m.model.encoder = torch.compile(m.model.encoder, backend="torch_tensorrt", options=opts)
run("encoder compiled", m)
m.model.decoder = torch.compile(m.model.decoder, backend="torch_tensorrt", options=opts)
run("encoder+decoder compiled", m)
