"""E13. TensorRT-LLM vs 자체 구현 v1/v2: 디코더 토큰당 수치 비교 (표준 라이브러리만, 프로파일 CSV 필요).

profiling/<tag>/ 의 trace CSV에서 디코더 구간(analyze.DECODER_MARK 이후)의 커널 종류별 시간, 호출 수를 토큰당으로 내고,
토큰당 읽는 가중치 바이트 / GEMM 시간 = 실효 메모리 대역폭을 계산한다.
v2는 nsys 노드 기록이 시간을 부풀리므로, E12(v2_step_time.py)의 실측 step 시간으로 비율 환산한 값을 함께 보라.

    python benchmark/experiments/head_to_head.py
"""
import collections
import csv
import os
import re
import sys

import _paths

sys.path.insert(0, os.path.join(_paths.BENCH_DIR, "profiling"))
import analyze as A  # noqa: E402

N = A.N_TOKENS
# large-v2 디코더 GEMM 가중치: 층당 self q,k,v,o + cross q,o + fc1, fc2 (cross K/V projection은 인코더 쪽 1회)
D, FF, L, V = 1280, 5120, 32, 51865
PARAMS_LAYERS = L * (6 * D * D + 2 * D * FF)
PARAMS_HEAD = V * D
BYTES = {
    "trtllm-fp16": (PARAMS_LAYERS + PARAMS_HEAD) * 2,
    "trtllm-int8": PARAMS_LAYERS * 1 + PARAMS_HEAD * 2,  # weight-only INT8, lm_head는 FP16
    "nndf-port-kv-fp16": (PARAMS_LAYERS + PARAMS_HEAD) * 2,
    "nndf-port-kv-graph-fp16": (PARAMS_LAYERS + PARAMS_HEAD) * 4,  # FP32 GEMV로 돌기 때문
}

print(f"decoder GEMM params: layers {PARAMS_LAYERS / 1e6:.0f}M + proj_out {PARAMS_HEAD / 1e6:.0f}M")
for tag, nbytes in BYTES.items():
    d = os.path.join(_paths.BENCH_DIR, "profiling", tag)
    pre = "nsys" if os.path.exists(f"{d}/nsys_cuda_gpu_trace.csv") else "torchprof"
    rows = list(csv.DictReader(open(f"{d}/{pre}_cuda_gpu_trace.csv", encoding="utf-8")))
    idx = next(i for i, r in enumerate(rows) if re.search(A.DECODER_MARK[tag], r["Name"]))
    agg = collections.defaultdict(lambda: [0.0, 0])
    for r in rows[idx:]:
        c = A.classify(r["Name"])
        agg[c][0] += int(r["Duration (ns)"]) / 1e6
        agg[c][1] += 0 if r["Name"].startswith("[CUDA mem") else 1
    dec = A.decompose(tag)
    gemm = agg["GEMM/GEMV"][0] / N
    print(f"== {tag}")
    print(f"  E2E {dec['e2e']:.0f} | enc {dec['enc']:.1f} | 실제 디코더 {(dec['e2e'] - dec['enc']) / N:.2f} ms/tok"
          f" | 디코더 GPU {dec['dec_gpu'] / N:.2f} | 대기 {dec['dec_idle'] / N:.2f} ms/tok")
    print(f"  GEMM {gemm:.2f} ms/tok x{agg['GEMM/GEMV'][1] / N:.0f} | weights {nbytes / 1e9:.2f} GB/tok"
          f" -> {nbytes / (gemm / 1e3) / 1e12:.2f} TB/s")
    total = sum(v[0] for v in agg.values())
    for c in ["attention", "TRT myelin (fused)", "elementwise/copy/etc"]:
        print(f"  {c:22s} {agg[c][0] / N:.2f} ms/tok x{agg[c][1] / N:.1f} ({100 * agg[c][0] / total:.0f}%)")
    print(f"  GEMM share {100 * agg['GEMM/GEMV'][0] / total:.0f}% | kernels/tok {sum(v[1] for v in agg.values()) / N:.0f}")
