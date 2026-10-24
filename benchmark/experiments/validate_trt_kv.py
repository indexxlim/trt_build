"""E3/E9. trt_kv v1(동적) 엔진과 v2(정적 shape + CUDA graph) 엔진의 정확도 검증 (conda env `trt`).

1) v2 torch 모듈(WhisperDecoderStepStatic)의 step별 logits가 HF와 같은지
2) ONNX export + TRT 빌드 (FP32/FP16)
3) graph 없이 / CUDA graph로 생성한 토큰이 HF generate와 같은지 + 시간

    V=openai/whisper-tiny CUDA_VISIBLE_DEVICES=0 python benchmark/experiments/validate_trt_kv.py
"""
import os, sys, time
import _paths  # noqa: F401
import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from Whisper import trt_kv
from common import load_inputs
V = os.environ.get("V", "openai/whisper-tiny"); tag = V.split("/")[-1]
out = os.path.join(_paths.OUT_DIR, "trt_kv_validate", tag)
m = WhisperForConditionalGeneration.from_pretrained(V).eval()
tok = WhisperTokenizer.from_pretrained(V)
cfg = m.config
inp = load_inputs()
x = torch.from_numpy(inp["ko"]["features"]); prompt = inp["ko"]["prompt"]

# 1) torch equivalence: static step vs HF logits
with torch.no_grad():
    ck, cv = trt_kv.WhisperEncoderCrossKV(m)(x)
    ids = torch.tensor([prompt + [50364, 1234]])
    ref = m(encoder_outputs=(m.model.encoder(x).last_hidden_state,), decoder_input_ids=ids).logits[0]
    st = trt_kv.WhisperDecoderStepStatic(m)
    L, H, D, S = cfg.decoder_layers, cfg.decoder_attention_heads, cfg.d_model // cfg.decoder_attention_heads, cfg.max_target_positions
    sk = torch.zeros(L, 1, H, S, D); sv = torch.zeros_like(sk)
    err = 0
    for p in range(ids.shape[1]):
        lg, nk, nv = st(ids[:, p:p+1], torch.tensor([p]), sk, sv, ck, cv)
        sk[:, :, :, p] = nk[:, :, :, 0]; sv[:, :, :, p] = nv[:, :, :, 0]
        err = max(err, (lg[0] - ref[p]).abs().max().item())
    print(f"static torch step vs HF logits: max|diff| {err:.2e}")

# 2) export/build
enc_onnx = f"{out}/onnx/encoder/encoder-crosskv.onnx"; dec_onnx = f"{out}/onnx/decoder-static/decoder-static.onnx"
if not os.path.exists(enc_onnx):
    trt_kv.export_onnx(m, enc_onnx, f"{out}/onnx/decoder/decoder-kv.onnx")
if not os.path.exists(dec_onnx):
    trt_kv.export_decoder_static_onnx(m, dec_onnx)
m.cuda()
hf = tok.decode(m.generate(x.cuda(), forced_decoder_ids=[(i, t) for i, t in enumerate(prompt[1:], 1)], max_length=448)[0], skip_special_tokens=True)
for fp16 in (False, True):
    enc_e, _ = trt_kv.build_whisper_engines(cfg, enc_onnx, f"{out}/onnx/decoder/decoder-kv.onnx", f"{out}/engines", fp16=fp16)
    dec_e = trt_kv.build_decoder_static_engine(cfg, dec_onnx, f"{out}/engines", fp16=fp16)
    for use_graph in (False, True):
        r = trt_kv.WhisperTRTKVGraph(enc_e, dec_e, cfg, use_graph=use_graph)
        xs = x.cuda()
        toks = r.generate(xs, prompt); txt = tok.decode(toks, skip_special_tokens=True)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5): r.generate(xs, prompt)
        torch.cuda.synchronize()
        print(f"{'fp16' if fp16 else 'fp32'} graph={use_graph}: match_hf={txt == hf} | {len(toks)} toks | {(time.time()-t0)/5*1000:.1f} ms")
        if txt != hf: print("  HF :", hf[:80]); print("  TRT:", txt[:80])
        del r
