"""results/*.json -> 비교표 (markdown). 어느 env에서나 실행 가능 (표준 라이브러리만 사용).

    python benchmark/compare.py
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# 표에 나오는 순서와 이름
ORDER = [
    ("hf_transformers", "HF transformers generate (cache)", "기준선"),
    ("torch_compile_torch_tensorrt_enc", 'torch.compile(encoder, backend="torch_tensorrt") + HF decoder', "레거시"),
    ("nndf_trt_nonkv", "NNDF TensorRT (non-KV, opset 12)", "레거시"),
    ("nndf_port_trt_kv", "NNDF 흐름 자체 구현 TensorRT v1 (KV cache, opset 17)", "레거시"),
    ("nndf_port_trt_kv_graph", "NNDF 흐름 자체 구현 TensorRT v2 (+정적 shape, GPU 루프, CUDA graph)", "레거시"),
    ("ctranslate2", "CTranslate2", "현행"),
    ("tensorrt_llm", "TensorRT-LLM (C++ runtime)", "현행"),
]


def main():
    results = {}
    for p in glob.glob(os.path.join(HERE, "results", "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        results[(r["name"], r["precision"])] = r

    hf = results.get(("hf_transformers", "fp16"))
    rows = []
    for name, label, group in ORDER:
        for (n, prec), r in sorted(results.items()):
            if n != name:
                continue
            en, ko = r["samples"]["en"], r["samples"]["ko"]
            enc = en.get("encoder")
            speed = ""
            if hf:
                speed = (f"{hf['samples']['en']['e2e']['p50'] / en['e2e']['p50']:.1f}x / "
                         f"{hf['samples']['ko']['e2e']['p50'] / ko['e2e']['p50']:.1f}x")
            ok = "✅" if en["match_hf"] and ko["match_hf"] else f"❌ CER {max(en['cer_vs_hf'], ko['cer_vs_hf']):.3f}"
            rows.append([group, label, prec, f"{enc['p50']:.1f}" if enc else "-",
                         f"{en['e2e']['p50']:.1f}", f"{ko['e2e']['p50']:.1f}", speed, ok, r["time"][5:16]])

    header = ["구분", "백엔드", "정밀도", "인코더 p50 (ms)", "E2E EN p50 (ms)", "E2E KO p50 (ms)",
              "HF FP16 대비 (EN / KO)", "텍스트 == HF", "측정 시각"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    print("\n".join(lines))
    gpus = sorted({r["nvidia_smi"] for r in results.values()})
    print("\nGPU (측정 시점 nvidia-smi):", *gpus, sep="\n  ")


if __name__ == "__main__":
    main()
