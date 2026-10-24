"""profiling/<tag>/ 의 nsys CSV + cProfile 결과를 요약해 PROFILE.md 로 만든다 (표준 라이브러리만 사용).

    python benchmark/profiling/analyze.py

주의: nsys는 커널 launch마다 기록 비용이 붙어 wall 시간이 부풀려진다 (launch 수가 많을수록 더).
그래서 판단은 GPU 커널 시간(정확), 커널 수, GPU 유휴 비율, CUDA API 호출 수 위주로 한다.
"""

import csv
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = ["hf-fp16", "torchtrt-fp16", "nndf-nonkv-fp16", "nndf-port-kv-fp16", "nndf-port-kv-graph-fp16",
        "ct2-float16", "ct2-int8", "trtllm-fp16", "trtllm-int8"]
N_TOKENS = 126  # 한국어 샘플 생성 토큰 수 (EOT 포함)
RESULT_FILE = {
    "hf-fp16": "hf_transformers-fp16", "torchtrt-fp16": "torch_compile_torch_tensorrt_enc-fp16",
    "nndf-nonkv-fp16": "nndf_trt_nonkv-fp16", "nndf-port-kv-fp16": "nndf_port_trt_kv-fp16", "nndf-port-kv-graph-fp16": "nndf_port_trt_kv_graph-fp16",
    "ct2-float16": "ctranslate2-float16", "ct2-int8": "ctranslate2-int8_float16",
    "trtllm-fp16": "tensorrt_llm-fp16", "trtllm-int8": "tensorrt_llm-int8",
}


def bench_wall(tag):
    """프로파일러 없이 잰 KO E2E p50 (benchmark/results)."""
    import json

    p = os.path.join(os.path.dirname(HERE), "results", RESULT_FILE[tag] + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)["samples"]["ko"]["e2e"]["p50"]

# 커널 이름 -> 분류 (위에서부터 먼저 맞는 것)
CATEGORIES = [
    ("memcpy/memset", r"^\[CUDA mem"),
    ("attention", r"mha|fmha|flash|attention|mmha|multihead|xqa"),
    ("softmax", r"softmax"),
    ("norm", r"norm"),
    ("GEMM/GEMV", r"gemm|gemv|cutlass|xmma|cublas|matmul|sm80_|sm90_|ampere_|wgmma|splitk|weightonly|weight_only"),
    ("TRT myelin (fused)", r"__myl|myelin"),
    ("decode/sampling", r"topk|sampling|argmax|beam|penalt|logits|banbad|banrepeat"),
    ("elementwise/copy/etc", r"."),
]


def classify(name):
    for cat, pat in CATEGORIES:
        if re.search(pat, name, re.I):
            return cat
    return "other"


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def short(name, n=70):
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= n else name[: n - 1] + "…"


def analyze_nsys(d):
    """nsys CSV가 있으면 nsys, 없으면 torch profiler(torchprof_*) CSV를 쓴다."""
    for tool in ("nsys", "torchprof"):
        nvtx = [r for r in read_csv(os.path.join(d, f"{tool}_nvtx_pushpop_trace.csv")) if r["Name"].startswith("e2e:")]
        trace = read_csv(os.path.join(d, f"{tool}_cuda_gpu_trace.csv"))
        api = read_csv(os.path.join(d, f"{tool}_cuda_api_sum.csv"))
        if nvtx and trace:
            break
    else:
        return None
    t0, t1 = int(nvtx[0]["Start (ns)"]), int(nvtx[0]["End (ns)"])
    wall = (t1 - t0) / 1e6

    ivs, by_cat, by_kernel = [], defaultdict(float), defaultdict(lambda: [0.0, 0])
    n_kernels = 0
    for r in trace:
        s, dur = int(r["Start (ns)"]), int(r["Duration (ns)"])
        if s < t0 or s > t1:
            continue
        name = r["Name"]
        ms = dur / 1e6
        cat = classify(name)
        by_cat[cat] += ms
        by_kernel[name][0] += ms
        by_kernel[name][1] += 1
        if not name.startswith("[CUDA mem"):
            n_kernels += 1
        ivs.append((s, s + dur))

    # GPU busy = 구간 합집합
    ivs.sort()
    busy, cur_s, cur_e = 0, None, None
    for s, e in ivs:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    busy /= 1e6
    kernel_sum = sum(v[0] for v in by_kernel.values())

    api_rows = sorted(api, key=lambda r: -float(r["Total Time (ns)"]))[:6]  # noqa
    return {
        "tool": tool, "wall": wall, "busy": busy, "idle_pct": 100 * (1 - busy / wall) if wall else 0,
        "kernel_sum": kernel_sum, "n_kernels": n_kernels,
        "by_cat": sorted(by_cat.items(), key=lambda kv: -kv[1]),
        "top_kernels": sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:8],
        "api": [(r["Name"], float(r["Total Time (ns)"]) / 1e6, int(r["Num Calls"])) for r in api_rows],
    }


def cprofile_top(d, n=12):
    """cprofile.txt 의 tottime 상위 함수 (Python 쪽 병목)."""
    p = os.path.join(d, "cprofile.txt")
    if not os.path.exists(p):
        return None, []
    text = open(p, encoding="utf-8").read()
    wall = re.search(r"wall ([\d.]+) ms", text)
    section = text.split("## sort by tottime")[1].split("## sort by cumtime")[0]
    rows = []
    for line in section.splitlines():
        m = re.match(r"\s*([\d/]+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+(.*)", line)
        if m:
            ncalls, tottime, cumtime, fn = m.groups()
            fn = re.sub(r".*site-packages/", "", fn)
            fn = re.sub(r".*/trt_build/", "", fn)
            rows.append((fn, float(tottime) * 1000, ncalls))
    return (float(wall.group(1)) if wall else None), rows[:n]


# ---------------------------------------------------------------------------------------------
# TensorRT-LLM FP16 대비 차이 분해
# 인코더/디코더 경계 = 디코더 첫 step에만 나오는 커널이 처음 등장하는 위치 (None이면 벤치 인코더 시간 사용)
# 검증: 경계 앞 GPU 시간 ≈ 벤치에서 따로 잰 인코더 p50 (HF 48.6 vs 47.5ms, v1 16.8 vs 17.2ms 등)
# ---------------------------------------------------------------------------------------------
BASELINE = "trtllm-fp16"
DECODER_MARK = {
    "hf-fp16": r"gemv", "torchtrt-fp16": r"gemv",
    "nndf-nonkv-fp16": None,                       # 디코더 전용 커널이 따로 없음 -> 벤치 인코더 시간
    "nndf-port-kv-fp16": r"tilesize32x32",          # M=1 GEMM
    "nndf-port-kv-graph-fp16": r"gemvx_f32",        # 경계 앞에 cross K/V 복사 + cache 초기화가 포함됨
    "ct2-float16": r"gemv", "ct2-int8": r"gemmSN|gemv",
    "trtllm-fp16": r"paged_kv|cudaCoreGemm", "trtllm-int8": r"paged_kv",  # INT8은 인코더도 weight-only GEMM
}
DEC_CATS = ["GEMM/GEMV", "attention", "TRT myelin (fused)", "norm", "softmax", "elementwise/copy/etc",
            "decode/sampling", "memcpy/memset"]


def bench_encoder(tag):
    import json

    p = os.path.join(os.path.dirname(HERE), "results", RESULT_FILE[tag] + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        enc = json.load(f)["samples"]["ko"].get("encoder")
    return enc["p50"] if enc else None


def decompose(tag):
    d = os.path.join(HERE, tag)
    ns = analyze_nsys(d)
    e2e = bench_wall(tag)
    if not ns or not e2e:
        return None
    rows = read_csv(os.path.join(d, f"{ns['tool']}_cuda_gpu_trace.csv"))
    pat = DECODER_MARK.get(tag)
    if pat:
        idx = next((i for i, r in enumerate(rows) if re.search(pat, r["Name"])), 0)
        enc_gpu = sum(int(r["Duration (ns)"]) for r in rows[:idx]) / 1e6
        dec_rows = rows[idx:]
    else:
        enc_gpu = bench_encoder(tag)
        t, idx = 0.0, 0
        for idx, r in enumerate(rows):
            t += int(r["Duration (ns)"]) / 1e6
            if t >= enc_gpu:
                break
        dec_rows = rows[idx + 1:]
    by_cat = defaultdict(float)
    n_k = 0
    for r in dec_rows:
        by_cat[classify(r["Name"])] += int(r["Duration (ns)"]) / 1e6
        n_k += 0 if r["Name"].startswith("[CUDA mem") else 1
    dec_gpu = max(0.0, ns["busy"] - enc_gpu)
    dec_wall = e2e - enc_gpu
    return {
        "e2e": e2e, "enc": enc_gpu, "dec_wall": dec_wall, "dec_gpu": dec_gpu,
        "dec_idle": max(0.0, dec_wall - dec_gpu), "k_per_tok": n_k / N_TOKENS,
        "cat_per_tok": {c: by_cat.get(c, 0.0) / N_TOKENS for c in DEC_CATS},
    }


def gap_section():
    base = decompose(BASELINE)
    if not base:
        return []
    rows = {t: decompose(t) for t in TAGS}
    lines = [
        "## TensorRT-LLM FP16 대비 차이 분해 (한국어 30초, 126토큰)", "",
        "E2E = 인코더 + 디코더(GPU가 일한 시간) + 디코더(GPU가 CPU를 기다린 시간). "
        "Δ는 TensorRT-LLM FP16과의 차이(ms). 디코더 GPU 시간은 프로파일러 아래에서 잰 값이라 "
        "실제 E2E보다 약간 클 수 있다(이때 대기 = 0).", "",
        "| 백엔드 | 실제 E2E | ΔE2E | 인코더 (Δ) | 디코더 GPU (Δ) | 디코더 대기 (Δ) | 디코더 GPU ms/토큰 | 커널/토큰 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in TAGS:
        r = rows.get(t)
        if not r:
            continue
        d = lambda k: r[k] - base[k]  # noqa: E731
        lines.append(
            f"| {t} | {r['e2e']:.0f} | {r['e2e'] - base['e2e']:+.0f} | {r['enc']:.0f} ({d('enc'):+.0f}) | "
            f"{r['dec_gpu']:.0f} ({d('dec_gpu'):+.0f}) | {r['dec_idle']:.0f} ({d('dec_idle'):+.0f}) | "
            f"{r['dec_gpu'] / N_TOKENS:.2f} | {r['k_per_tok']:.0f} |")
    lines += ["", "디코더 GPU 시간을 토큰당(ms)으로, 커널 종류별로 나눈 값:", "",
              "| 백엔드 | " + " | ".join(DEC_CATS) + " |", "|---|" + "---|" * len(DEC_CATS)]
    for t in TAGS:
        r = rows.get(t)
        if r:
            lines.append(f"| {t} | " + " | ".join(f"{r['cat_per_tok'][c]:.2f}" for c in DEC_CATS) + " |")
    return lines + [""]


def main():
    out = ["# 프로파일링 결과 (한국어 샘플 E2E 1회, 126토큰)", "",
           "생성: `benchmark/profiling/analyze.py`. 원본: `benchmark/profiling/<tag>/` "
           "(`nsys.nsys-rep`는 Nsight Systems GUI로 열 수 있음, `cprofile.txt`).", "",
           "- **nsys wall**: nsys 아래에서 잰 E2E 시간. 커널 launch마다 기록 비용이 붙어 실제보다 길다 (launch가 많을수록 더 부풀려짐).",
           "- **GPU busy**: 이 구간에서 GPU 커널/복사가 실제로 돈 시간 (구간 합집합). **유휴%** = GPU가 CPU를 기다린 비율.",
           "- **실제 E2E**: 프로파일러 없이 잰 KO E2E p50 (`results/`, 다른 날/다른 GPU 부하에서 측정). "
           "**실제 유휴 % ≈ 1 − GPU busy / 실제 E2E** (GPU 부하 차이로 대략적인 값).",
           "- **커널/토큰**: 토큰 하나 만드는 데 띄운 GPU 커널 수.", ""]

    summary = ["| 백엔드 | 실제 E2E (ms) | GPU busy (ms) | 실제 유휴 % | nsys wall (ms) | 커널 수 | 커널/토큰 | 가장 큰 GPU 시간 | cProfile 상위 Python 함수 (자체 시간) |",
               "|---|---|---|---|---|---|---|---|---|"]
    details = []
    for tag in TAGS:
        d = os.path.join(HERE, tag)
        if not os.path.isdir(d):
            continue
        ns = analyze_nsys(d)
        cp_wall, cp_rows = cprofile_top(d)
        top_py = "<br>".join(f"`{short(fn, 55)}` {ms:.0f}ms" for fn, ms, _ in cp_rows[:3]) or "-"
        if ns:
            cat, ms = ns["by_cat"][0]
            real = bench_wall(tag)
            real_s = f"{real:.0f}" if real else "-"
            real_idle = f"{max(0.0, 100 * (1 - ns['busy'] / real)):.0f}%" if real else "-"
            summary.append(
                f"| {tag} | {real_s} | {ns['busy']:.0f} | {real_idle} | {ns['wall']:.0f}{'' if ns['tool'] == 'nsys' else ' (torch profiler)'} | {ns['n_kernels']:,} | "
                f"{ns['n_kernels'] / N_TOKENS:.0f} | {cat} {ms:.0f}ms ({100 * ms / ns['kernel_sum']:.0f}%) | {top_py} |")
        else:
            summary.append(f"| {tag} | - | nsys 없음 | - | - | - | - | - | {top_py} |")

        details += [f"## {tag}", ""]
        if ns:
            details += [f"[{ns['tool']}] wall {ns['wall']:.1f} ms · GPU busy {ns['busy']:.1f} ms · 유휴 {ns['idle_pct']:.0f}% · "
                        f"커널 {ns['n_kernels']:,}개 ({ns['n_kernels'] / N_TOKENS:.0f}/토큰) · 커널 시간 합 {ns['kernel_sum']:.1f} ms", "",
                        "| GPU 시간 분류 | ms | % |", "|---|---|---|"]
            details += [f"| {c} | {m:.1f} | {100 * m / ns['kernel_sum']:.0f}% |" for c, m in ns["by_cat"]]
            details += ["", "| 상위 커널 | ms | 호출 수 |", "|---|---|---|"]
            details += [f"| `{short(k, 90)}` | {v[0]:.1f} | {v[1]:,} |" for k, v in ns["top_kernels"]]
            details += ["", "| CUDA API (CPU 쪽) | ms | 호출 수 |", "|---|---|---|"]
            details += [f"| `{n}` | {m:.1f} | {c:,} |" for n, m, c in ns["api"]]
        if cp_rows:
            details += ["", f"cProfile wall {cp_wall:.1f} ms (cProfile 오버헤드 포함)", "",
                        "| Python 함수 (tottime 상위) | 자체 ms | 호출 수 |", "|---|---|---|"]
            details += [f"| `{short(fn, 100)}` | {ms:.1f} | {nc} |" for fn, ms, nc in cp_rows]
        details.append("")

    text = "\n".join(out + summary + [""] + gap_section() + details)
    with open(os.path.join(HERE, "PROFILE.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
