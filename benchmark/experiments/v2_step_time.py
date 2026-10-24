"""E12. v2 디코더 1 step(CUDA graph replay)의 GPU 시간을 프로파일러 없이 CUDA event로 측정 (large-v2 FP16, conda env `trt`).

nsys --cuda-graph-trace=node 는 graph 시간을 약 17% 부풀리므로(6.30 vs 5.38ms/step) 이 값으로 보정한다.

    CUDA_VISIBLE_DEVICES=0 python benchmark/experiments/v2_step_time.py
"""
import os, sys
import _paths  # noqa: F401
import torch
from transformers import WhisperConfig
from Whisper import trt_kv
from common import load_inputs
cfg = WhisperConfig.from_pretrained("openai/whisper-large-v2")
E = _paths.LARGE_V2_DIR + "/tensorrt-kv-opset17"
r = trt_kv.WhisperTRTKVGraph(f"{E}/encoder-crosskv-fp16-bs1.engine", f"{E}/decoder-static-fp16-bs1-O3.engine", cfg)
x = load_inputs()["ko"]
feats = torch.from_numpy(x["features"]).cuda()
with torch.no_grad(), torch.cuda.stream(r.stream):
    ck, cv = r.encode(feats); r.cross_k.copy_(ck); r.cross_v.copy_(cv); r._reset(x["prompt"])
    for _ in range(20): r.graph.replay()
    r._reset(x["prompt"])
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); n = 126
    for _ in range(n): r.graph.replay()
    e.record(); e.synchronize()
    print(f"v2 graph replay (= decoder 1 step) GPU time: {s.elapsed_time(e)/n:.3f} ms/step (126 steps 연속, 프로파일러 없음)")
    s.record()
    for _ in range(20): r.encode(feats)
    e.record(); e.synchronize()
    print(f"v2 encoder: {s.elapsed_time(e)/20:.2f} ms")
