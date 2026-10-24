"""NNDF 레거시 경로 + HF 기준선 벤치 (conda env `trt`).

backends
  hf       : HF transformers generate (use_cache=True)                       -- 기준선
  torchtrt : torch.compile(encoder, backend="torch_tensorrt") + HF FP16 decoder (Torch-TensorRT 1.4.0)
  nndf   : TensorRT OSS demo/HuggingFace(NNDF) 기반 non-KV 엔진 (opset 12)  -- Whisper/trt.py
  trtkv  : NNDF 레거시를 opset 17 + KV cache로 포팅한 엔진                   -- Whisper/trt_kv.py
  trtkv2 : trtkv + 정적 shape + GPU greedy 루프 + CUDA graph (같은 TRT 8.6)  -- Whisper/trt_kv.py

엔진은 notebooks/whisper2.ipynb(trtkv)와 v1 노트북(nndf)이 만든 파일을 재사용한다.
trtkv 엔진이 없으면 여기서 export + build 한다.

    CUDA_VISIBLE_DEVICES=0 python benchmark/nndf_legacy/bench.py --backend all --precision fp16
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # trt_build
sys.path[:0] = [os.path.dirname(HERE), ROOT, os.path.join(ROOT, "archived_models")]

import torch  # noqa: E402

from common import VARIANT, bench, load_inputs, save_result  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "notebooks", "models", VARIANT)


def run_hf(inputs, fp16):
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(VARIANT).eval().cuda()
    dtype = torch.float16 if fp16 else torch.float32
    model.to(dtype)
    per_sample = {}
    for s, x in inputs.items():
        feats = torch.from_numpy(x["features"]).cuda().to(dtype)
        prompt = x["prompt"]
        forced = [(i, t) for i, t in enumerate(prompt[1:], 1)]

        def gen():
            with torch.no_grad():
                return model.generate(feats, forced_decoder_ids=forced, max_length=448, num_beams=1, use_cache=True)[0]

        ids = gen().tolist()
        enc = bench(lambda: model.model.encoder(feats))
        per_sample[s] = {"text": decode(ids), "n_tokens": len(ids) - len(prompt), "encoder": enc, "e2e": bench(gen, label=f"e2e:{s}")}
    return per_sample


def run_torchtrt(inputs, fp16):
    """torch.compile(backend="torch_tensorrt") — Torch-TensorRT 1.4.0 (당시 이름은 "tensorrt"가 아니라 "torch_tensorrt").

    - 인코더만 컴파일. 디코더는 KV cache 길이가 매 step 바뀌어 정적 shape만 지원하는 1.4.0에서는 step마다 재컴파일됨
      (whisper-tiny 측정: 첫 실행 1165s, 두 번째 실행 78.8s, 그래프 171개) -> large-v2에서는 사용 불가로 제외.
    - 1.4.0은 .half() 모델에서 dtype assert가 나므로 인코더는 FP32 모델 + enabled_precisions={fp32, fp16}로 TRT 내부 FP16.
    - 미지원 op(aten.mm/bmm, softmax 구성 op, var_mean, erf 등)는 PyTorch로 fallback (README 참고).
    """
    import time

    import torch_tensorrt  # noqa: F401  dynamo backend 등록
    from transformers import WhisperForConditionalGeneration

    # large-v2(32층)는 미지원 op로 잘게 쪼개진 그래프를 다시 합치는 torch.fx fuser(dfs_find_cycle)가
    # 기본 재귀 한도(1000)에서 RecursionError -> 인코더 전체가 PyTorch FP32로 fallback 된다. 한도를 올려 우회.
    sys.setrecursionlimit(100000)
    model = WhisperForConditionalGeneration.from_pretrained(VARIANT).eval().cuda()
    dtype = torch.float16 if fp16 else torch.float32
    model.model.decoder.to(dtype)
    model.proj_out.to(dtype)
    precisions = {torch.float, torch.half} if fp16 else {torch.float}
    compiled = torch.compile(model.model.encoder, backend="torch_tensorrt", options={"enabled_precisions": precisions})

    class Encoder(torch.nn.Module):
        def __init__(self, inner, compiled_fn):
            super().__init__()
            self.inner = inner
            self.compiled_fn = compiled_fn

        def forward(self, input_features, **kwargs):
            out = self.compiled_fn(input_features.float(), **kwargs)
            out.last_hidden_state = out.last_hidden_state.to(dtype)
            return out

    model.model.encoder = Encoder(model.model.encoder, compiled)
    per_sample, compile_sec = {}, None
    for s, x in inputs.items():
        feats = torch.from_numpy(x["features"]).cuda().to(dtype)
        prompt = x["prompt"]
        forced = [(i, t) for i, t in enumerate(prompt[1:], 1)]

        def gen():
            with torch.no_grad():
                return model.generate(feats, forced_decoder_ids=forced, max_length=448, num_beams=1, use_cache=True)[0]

        t0 = time.time()
        ids = gen().tolist()
        compile_sec = compile_sec or time.time() - t0
        enc = bench(lambda: model.model.encoder(feats))
        per_sample[s] = {"text": decode(ids), "n_tokens": len(ids) - len(prompt), "encoder": enc, "e2e": bench(gen, label=f"e2e:{s}")}
    per_sample["_compile_sec"] = compile_sec
    return per_sample


def run_nndf(inputs, fp16):
    """v1 노트북과 같은 NNDF non-KV 엔진 + HF greedy_search."""
    from transformers import WhisperConfig
    from transformers.generation_logits_process import (
        ForceTokensLogitsProcessor, LogitsProcessorList, SuppressTokensAtBeginLogitsProcessor,
        SuppressTokensLogitsProcessor,
    )
    from transformers.generation_stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

    from NNDF.networks import NetworkMetadata, Precision
    from Whisper.export import WhisperDecoderTRTEngine, WhisperEncoderTRTEngine
    from Whisper.trt import WhisperTRTDecoder, WhisperTRTEncoder
    from Whisper.WhisperModelConfig import WhisperMetadata

    tag = "-fp16" if fp16 else ""
    engine_dir = os.path.join(MODEL_DIR, "tensorrt")
    enc_path = os.path.join(engine_dir, f"Whisper-large-v2{tag}-encoder.onnx-bs1.engine")
    dec_path = os.path.join(engine_dir, f"Whisper-large-v2{tag}-decoder-with-lm-head.onnx-bs1.engine")
    for p in (enc_path, dec_path):
        if not os.path.exists(p):
            sys.exit(f"missing {p}\n -> v1 노트북(git 74f9c1c의 notebooks/whisper2.ipynb)으로 NNDF 엔진을 먼저 빌드")

    meta = NetworkMetadata(variant=VARIANT, precision=Precision(fp16=fp16), other=WhisperMetadata(kv_cache=False))
    config = WhisperConfig.from_pretrained(VARIANT, use_cache=False)
    encoder = WhisperTRTEncoder(WhisperEncoderTRTEngine(enc_path, meta), meta, config, batch_size=1)
    decoder = WhisperTRTDecoder(WhisperDecoderTRTEngine(dec_path, meta), meta, config, batch_size=1, num_beams=1)

    start = torch.full((1, 1), config.decoder_start_token_id, dtype=torch.int32, device="cuda")
    stopping = StoppingCriteriaList([MaxLengthCriteria(448)])
    per_sample = {}
    for s, x in inputs.items():
        feats = torch.from_numpy(x["features"]).cuda()
        prompt = x["prompt"]
        processors = LogitsProcessorList([
            SuppressTokensLogitsProcessor(config.suppress_tokens),
            SuppressTokensAtBeginLogitsProcessor(config.begin_suppress_tokens, len(prompt)),
            ForceTokensLogitsProcessor([(i, t) for i, t in enumerate(prompt[1:], 1)]),
        ])

        def gen():
            with torch.no_grad():
                hidden = encoder(input_features=feats)
                decoder.set_encoder_hidden_states_for_inference_cycle(hidden)
                return decoder.greedy_search(input_ids=start, encoder_hidden_states=hidden,
                                             stopping_criteria=stopping, logits_processor=processors, use_cache=False)[0]

        ids = gen().tolist()
        enc = bench(lambda: encoder(input_features=feats))
        per_sample[s] = {"text": decode(ids), "n_tokens": len(ids) - len(prompt), "encoder": enc, "e2e": bench(gen, label=f"e2e:{s}")}
    encoder.release()
    decoder.release()
    return per_sample


def run_trtkv(inputs, fp16):
    from transformers import WhisperConfig

    from Whisper import trt_kv

    config = WhisperConfig.from_pretrained(VARIANT)
    onnx_dir = os.path.join(MODEL_DIR, f"onnx-kv-opset{trt_kv.ONNX_OPSET}")
    enc_onnx = os.path.join(onnx_dir, "encoder", "encoder-crosskv.onnx")
    dec_onnx = os.path.join(onnx_dir, "decoder", "decoder-kv.onnx")
    if not (os.path.exists(enc_onnx) and os.path.exists(dec_onnx)):
        from transformers import WhisperForConditionalGeneration

        trt_kv.export_onnx(WhisperForConditionalGeneration.from_pretrained(VARIANT), enc_onnx, dec_onnx)
    enc_path, dec_path = trt_kv.build_whisper_engines(
        config, enc_onnx, dec_onnx, os.path.join(MODEL_DIR, f"tensorrt-kv-opset{trt_kv.ONNX_OPSET}"), fp16=fp16)

    runner = trt_kv.WhisperTRTKV(enc_path, dec_path, config)
    per_sample = {}
    for s, x in inputs.items():
        feats = torch.from_numpy(x["features"]).cuda()
        prompt = x["prompt"]
        ids = runner.generate(feats, prompt)
        enc = bench(lambda: runner.encode(feats))
        per_sample[s] = {"text": decode(ids), "n_tokens": len(ids) - len(prompt), "encoder": enc,
                         "e2e": bench(lambda: runner.generate(feats, prompt), label=f"e2e:{s}")}
    return per_sample


def run_trtkv2(inputs, fp16):
    """trtkv v2: 정적 shape 디코더 + GPU greedy 루프 + CUDA graph (Whisper/trt_kv.py WhisperTRTKVGraph)."""
    from transformers import WhisperConfig

    from Whisper import trt_kv

    config = WhisperConfig.from_pretrained(VARIANT)
    onnx_dir = os.path.join(MODEL_DIR, f"onnx-kv-opset{trt_kv.ONNX_OPSET}")
    enc_onnx = os.path.join(onnx_dir, "encoder", "encoder-crosskv.onnx")
    dec_onnx_dyn = os.path.join(onnx_dir, "decoder", "decoder-kv.onnx")
    dec_onnx = os.path.join(onnx_dir, "decoder-static", "decoder-static.onnx")
    if not (os.path.exists(enc_onnx) and os.path.exists(dec_onnx)):
        from transformers import WhisperForConditionalGeneration

        model = WhisperForConditionalGeneration.from_pretrained(VARIANT)
        if not os.path.exists(enc_onnx):
            trt_kv.export_onnx(model, enc_onnx, dec_onnx_dyn)
        trt_kv.export_decoder_static_onnx(model, dec_onnx)
    engine_dir = os.path.join(MODEL_DIR, f"tensorrt-kv-opset{trt_kv.ONNX_OPSET}")
    enc_path, _ = trt_kv.build_whisper_engines(config, enc_onnx, dec_onnx_dyn, engine_dir, fp16=fp16)
    dec_path = trt_kv.build_decoder_static_engine(config, dec_onnx, engine_dir, fp16=fp16,
                                                  opt_level=int(os.environ.get("TRT_OPT_LEVEL", "3")))

    runner = trt_kv.WhisperTRTKVGraph(enc_path, dec_path, config)
    per_sample = {}
    for s, x in inputs.items():
        feats = torch.from_numpy(x["features"]).cuda()
        prompt = x["prompt"]
        ids = runner.generate(feats, prompt)
        enc = bench(lambda: runner.encode(feats))
        per_sample[s] = {"text": decode(ids), "n_tokens": len(ids) - len(prompt), "encoder": enc,
                         "e2e": bench(lambda: runner.generate(feats, prompt), label=f"e2e:{s}")}
    return per_sample


_tokenizer = None


def decode(ids):
    global _tokenizer
    if _tokenizer is None:
        from transformers import WhisperTokenizer

        _tokenizer = WhisperTokenizer.from_pretrained(VARIANT)
    return _tokenizer.decode(ids, skip_special_tokens=True)


BACKENDS = {"hf": ("hf_transformers", run_hf), "torchtrt": ("torch_compile_torch_tensorrt_enc", run_torchtrt), "nndf": ("nndf_trt_nonkv", run_nndf), "trtkv": ("nndf_port_trt_kv", run_trtkv), "trtkv2": ("nndf_port_trt_kv_graph", run_trtkv2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS) + ["all"], default="all")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    inputs = load_inputs()
    for key in (BACKENDS if args.backend == "all" else [args.backend]):
        name, fn = BACKENDS[key]
        per_sample = fn(inputs, args.precision == "fp16")
        extra = {"env": "conda trt (py3.8, torch 2.0.1, TRT 8.6.1, transformers 4.24, torch-tensorrt 1.4.0)"}
        if "_compile_sec" in per_sample:
            extra["first_run_incl_compile_sec"] = per_sample.pop("_compile_sec")
        save_result(name, args.precision, per_sample, extra=extra)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
