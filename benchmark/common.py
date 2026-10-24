"""Whisper 추론 백엔드 비교용 공통 유틸 (python 3.8+, numpy만 필수).

모든 백엔드는 data/inputs.npz 의 같은 log-mel 입력(HF WhisperProcessor로 만든 (1, 80, 3000))과
같은 프롬프트(<|startoftranscript|><|lang|><|transcribe|><|notimestamps|>)로 greedy decoding 한다.
정답 비교 기준은 data/reference.json (HF transformers FP32 generate 결과).
"""

import datetime
import json
import os
import subprocess
import time

import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BENCH_DIR, "data")
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
INPUTS = os.path.join(DATA_DIR, "inputs.npz")
REFERENCE = os.path.join(DATA_DIR, "reference.json")

VARIANT = "openai/whisper-large-v2"
SAMPLES = ("en", "ko")  # en: librispeech dummy[0] (5.9s), ko: korean_news.mp4 앞 30초

WARMUP = 2
ITERS = 10


def load_inputs():
    """{'en': {'features': (1,80,3000) float32, 'prompt': [int]}, 'ko': {...}}"""
    d = np.load(INPUTS)
    return {s: {"features": d[f"{s}_features"], "prompt": d[f"{s}_prompt"].tolist()} for s in SAMPLES}


def load_reference():
    with open(REFERENCE, encoding="utf-8") as f:
        return json.load(f)


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _profile_once(fn, label):
    """PROFILE=nsys|cprofile 이고 PROFILE_TARGET == label 일 때 fn() 1회만 프로파일링한다.

    nsys     : `nsys profile --capture-range=cudaProfilerApi` 로 실행하면 이 구간만 기록 (NVTX range = label)
    cprofile : PROFILE_OUT/cprofile.txt 에 Python 함수별 시간 (cumtime / tottime 상위)
    """
    mode = os.environ.get("PROFILE")
    out_dir = os.environ.get("PROFILE_OUT", os.path.join(BENCH_DIR, "profiling", "out"))
    os.makedirs(out_dir, exist_ok=True)
    _sync()
    if mode == "nsys":
        import torch

        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push(label)
        fn()
        _sync()
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
    elif mode == "torchprof":
        # nsys 2023.2가 CUDA 13 앱(TensorRT-LLM 1.2)의 GPU trace를 못 잡아서 torch profiler(Kineto/CUPTI)로 대신 기록.
        # 같은 프로세스의 모든 CUDA 커널(TRT-LLM C++ 포함)이 잡힌다. analyze.py가 읽는 nsys CSV 형식으로 저장.
        import csv

        import torch
        from torch.profiler import ProfilerActivity, profile, record_function

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function(label):
                fn()
                _sync()
        prof.export_chrome_trace(os.path.join(out_dir, "torchprof_trace.json"))
        kernels, rng = [], None
        for e in prof.events():
            if e.name == label and e.device_type == torch.autograd.DeviceType.CPU:
                rng = (int(e.time_range.start * 1000), int(e.time_range.end * 1000))
            elif e.device_type == torch.autograd.DeviceType.CUDA:
                name = e.name
                if name.lower().startswith("memcpy") or name.lower().startswith("memset"):
                    name = f"[CUDA {name}]"
                kernels.append((int(e.time_range.start * 1000), int(e.time_range.elapsed_us() * 1000), name))
        with open(os.path.join(out_dir, "torchprof_cuda_gpu_trace.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Start (ns)", "Duration (ns)", "Name"])
            w.writerows(sorted(kernels))
        with open(os.path.join(out_dir, "torchprof_nvtx_pushpop_trace.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Start (ns)", "End (ns)", "Name"])
            w.writerow([rng[0], rng[1], label])
        with open(os.path.join(out_dir, "torchprof_cuda_api_sum.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Total Time (ns)", "Num Calls", "Name"])
            for a in prof.key_averages():
                if a.key.startswith("cu") and a.device_type == torch.autograd.DeviceType.CPU:
                    w.writerow([int(a.cpu_time_total * 1000), a.count, a.key])
    elif mode == "cprofile":
        import cProfile
        import io
        import pstats

        prof = cProfile.Profile()
        t0 = time.perf_counter()
        prof.enable()
        fn()
        _sync()
        prof.disable()
        wall = (time.perf_counter() - t0) * 1000
        buf = io.StringIO()
        buf.write(f"# {label}: wall {wall:.1f} ms (cProfile 오버헤드 포함)\n\n## sort by tottime (함수 자체 시간)\n")
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(30)
        buf.write("\n## sort by cumtime (하위 호출 포함)\n")
        pstats.Stats(prof, stream=buf).sort_stats("cumtime").print_stats(40)
        with open(os.path.join(out_dir, "cprofile.txt"), "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
        prof.dump_stats(os.path.join(out_dir, "cprofile.pstats"))
    print(f"[profile] {mode} {label} -> {out_dir}")


def bench(fn, warmup=WARMUP, iters=ITERS, label=None):
    """fn()을 반복 실행해 wall-clock ms의 p50 / p99 / min을 돌려준다 (매 회 GPU sync).

    label: 'e2e:ko' 같은 측정 이름. PROFILE_TARGET과 같으면 warmup 뒤 1회를 프로파일링하고 반복 측정은 생략한다.
    """
    for _ in range(warmup):
        fn()
    _sync()
    if label is not None and os.environ.get("PROFILE") and os.environ.get("PROFILE_TARGET") == label:
        _profile_once(fn, label)
        return {"p50": float("nan"), "p99": float("nan"), "min": float("nan")}
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50": float(np.percentile(times, 50)),
        "p99": float(np.percentile(times, 99)),
        "min": float(times[0]),
    }


def cer(hyp, ref):
    """문자 단위 edit distance / len(ref) (공백 제거)."""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    prev = list(range(len(r) + 1))
    for i, ch in enumerate(h, 1):
        cur = [i] + [0] * len(r)
        for j, cr in enumerate(r, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ch != cr))
        prev = cur
    return prev[-1] / max(len(r), 1)


def gpu_snapshot():
    """측정 당시 GPU 사용 상태 (다른 프로세스와 GPU를 공유하면 수치가 흔들리므로 같이 기록)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", visible, "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        out = f"n/a ({e})"
    return {"gpu_index": visible, "nvidia_smi": out}


def save_result(name, precision, per_sample, extra=None):
    """per_sample: {'en': {'text', 'n_tokens', 'e2e': bench(), 'encoder': bench() | None}, 'ko': ...}"""
    if os.environ.get("PROFILE"):
        print(f"[profile] {name} {precision}: 프로파일링 실행이라 results/ 저장 생략")
        return None
    ref = load_reference()
    for s, r in per_sample.items():
        r["match_hf"] = r["text"] == ref[s]["text"]
        r["cer_vs_hf"] = cer(r["text"], ref[s]["text"])
    result = {
        "name": name,
        "precision": precision,
        "variant": VARIANT,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        **gpu_snapshot(),
        "samples": per_sample,
        **(extra or {}),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}-{precision}.json".replace(" ", "_"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    for s, r in per_sample.items():
        enc = f"enc {r['encoder']['p50']:.1f}ms | " if r.get("encoder") else ""
        print(f"[{name} {precision}] {s}: {enc}E2E p50 {r['e2e']['p50']:.1f}ms p99 {r['e2e']['p99']:.1f}ms"
              f" | {r['n_tokens']} tok | match_hf={r['match_hf']} cer={r['cer_vs_hf']:.4f}")
    print("saved", path)
    return result
