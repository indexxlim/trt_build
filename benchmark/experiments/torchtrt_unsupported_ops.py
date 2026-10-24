"""E4. Torch-TensorRT 1.4.0 dynamo backend가 Whisper 인코더 / 디코더에서 TRT로 못 바꾸는 aten op 목록 (conda env `trt`).

partition()을 verbose=True로 감싸 "Unsupported or Excluded Nodes"를 출력하게 한다. 결과는 out/torchtrt_ops.log

    CUDA_VISIBLE_DEVICES=0 python benchmark/experiments/torchtrt_unsupported_ops.py 2> benchmark/experiments/out/torchtrt_ops.log
"""
import _paths  # noqa: F401

import os, sys, logging, io, re, collections
import torch, torch_tensorrt, torch._dynamo as dynamo
import torch_tensorrt.dynamo.backend.backends as _bk
_orig = _bk.partition
def _verbose_partition(gm, *a, **kw):
    kw["verbose"] = True
    return _orig(gm, *a, **kw)
_bk.partition = _verbose_partition
from transformers import WhisperForConditionalGeneration
V = os.environ.get("V", "openai/whisper-tiny")
buf = io.StringIO(); h = logging.StreamHandler(buf); h.setLevel(logging.DEBUG)
for name in ["torch_tensorrt", "torch_tensorrt.dynamo", "torch_tensorrt.fx"]:
    lg = logging.getLogger(name); lg.setLevel(logging.DEBUG); lg.addHandler(h)
m = WhisperForConditionalGeneration.from_pretrained(V).eval().cuda()
opts = {"enabled_precisions": {torch.float, torch.half}, "debug": True}
feats = torch.randn(1, 80, 3000, device="cuda")
with torch.no_grad():
    enc = torch.compile(m.model.encoder, backend="torch_tensorrt", options=opts)
    hs = enc(feats).last_hidden_state
    print("=== ENCODER LOG ==="); print(buf.getvalue()[-6000:]); buf.seek(0); buf.truncate()
    dynamo.reset()
    dec = torch.compile(m.model.decoder, backend="torch_tensorrt", options=opts)
    out = dec(input_ids=torch.tensor([[50258, 50259]], device="cuda"), encoder_hidden_states=hs, use_cache=True)
    print("=== DECODER LOG ==="); print(buf.getvalue()[-8000:])
