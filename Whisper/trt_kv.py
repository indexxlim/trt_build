#
# Whisper TensorRT inference with KV cache (opset 17).
#
# NNDF 기반 Whisper/export.py + Whisper/trt.py는 KV cache 경로가 BART/T5 코드와 섞여 있어 동작하지 않으므로
# Whisper 전용으로 단순하게 다시 구성한다.
#
#   encoder engine : input_features (B, 80, 3000)
#                    -> cross_k, cross_v (L, B, H, 1500, D)   # 디코더 모든 층의 cross-attention K/V를 한 번만 계산
#   decoder engine : input_ids (B, T), self_k, self_v (L, B, H, P, D), cross_k, cross_v
#                    -> logits (B, V) [마지막 토큰], present_k, present_v (L, B, H, P+T, D)
#
# opset 17로 export하면 LayerNorm이 ONNX LayerNormalization -> TensorRT INormalizationLayer로 들어가고,
# INormalizationLayer는 FP16 모드에서도 정규화 계산을 FP32로 수행하므로 large 모델의 FP16 overflow가 생기지 않는다.
#

import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt

ONNX_OPSET = 17

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


# --------------------------------------------------------------------------------------
# Torch modules for export
# --------------------------------------------------------------------------------------
def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # (B, S, C) -> (B, H, S, D)
    bsz, seq_len, _ = x.shape
    return x.view(bsz, seq_len, num_heads, -1).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # (B, H, S, D) -> (B, S, C)
    bsz, _, seq_len, _ = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)


class WhisperEncoderCrossKV(nn.Module):
    """Whisper encoder + 디코더 각 층의 cross-attention K/V projection."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
        self.layers = model.model.decoder.layers
        self.num_heads = model.config.decoder_attention_heads

    def forward(self, input_features):
        hidden = self.encoder(input_features).last_hidden_state
        cross_k = torch.stack([_split_heads(l.encoder_attn.k_proj(hidden), self.num_heads) for l in self.layers])
        cross_v = torch.stack([_split_heads(l.encoder_attn.v_proj(hidden), self.num_heads) for l in self.layers])
        return cross_k, cross_v


class WhisperDecoderStep(nn.Module):
    """KV cache를 입력/출력으로 받는 Whisper decoder 1 step (+ proj_out). HF WhisperDecoder와 같은 연산."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.num_heads = model.config.decoder_attention_heads

    def _attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-1, -2))
        if mask is not None:
            scores = scores + mask
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def forward(self, input_ids, self_k, self_v, cross_k, cross_v):
        dec = self.decoder
        past_len = self_k.shape[3]
        tgt_len = input_ids.shape[1]

        x = dec.embed_tokens(input_ids) + dec.embed_positions.weight[past_len : past_len + tgt_len]

        # causal mask (tgt_len, past_len + tgt_len): query i는 key j <= past_len + i 까지만 본다
        q_pos = torch.arange(tgt_len, device=x.device).unsqueeze(1) + past_len
        k_pos = torch.arange(past_len + tgt_len, device=x.device).unsqueeze(0)
        mask = (k_pos > q_pos).to(x.dtype) * -10000.0

        present_k, present_v = [], []
        for i, layer in enumerate(dec.layers):
            # self-attention
            residual = x
            h = layer.self_attn_layer_norm(x)
            attn = layer.self_attn
            q = _split_heads(attn.q_proj(h) * attn.scaling, self.num_heads)
            k = torch.cat([self_k[i], _split_heads(attn.k_proj(h), self.num_heads)], dim=2)
            v = torch.cat([self_v[i], _split_heads(attn.v_proj(h), self.num_heads)], dim=2)
            present_k.append(k)
            present_v.append(v)
            x = residual + attn.out_proj(_merge_heads(self._attention(q, k, v, mask)))

            # cross-attention (K/V는 인코더 엔진에서 미리 계산)
            residual = x
            h = layer.encoder_attn_layer_norm(x)
            attn = layer.encoder_attn
            q = _split_heads(attn.q_proj(h) * attn.scaling, self.num_heads)
            x = residual + attn.out_proj(_merge_heads(self._attention(q, cross_k[i], cross_v[i])))

            # FFN
            residual = x
            h = layer.final_layer_norm(x)
            x = residual + layer.fc2(layer.activation_fn(layer.fc1(h)))

        x = dec.layer_norm(x)
        logits = self.proj_out(x[:, -1, :])
        return logits, torch.stack(present_k), torch.stack(present_v)


# --------------------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------------------
def export_onnx(model, encoder_path: str, decoder_path: str, opset: int = ONNX_OPSET):
    """CPU / FP32 모델을 opset 17 ONNX 두 개(encoder+crossKV, decoder step)로 export."""
    cfg = model.config
    num_layers, num_heads = cfg.decoder_layers, cfg.decoder_attention_heads
    head_dim = cfg.d_model // num_heads
    model = model.float().cpu().eval()

    for p in (encoder_path, decoder_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    with torch.no_grad():
        features = torch.randn(1, cfg.num_mel_bins, 3000)
        enc = WhisperEncoderCrossKV(model)
        cross_k, cross_v = enc(features)
        torch.onnx.export(
            enc, (features,), encoder_path,
            opset_version=opset,
            input_names=["input_features"],
            output_names=["cross_k", "cross_v"],
            dynamic_axes={"input_features": {0: "batch"}, "cross_k": {1: "batch"}, "cross_v": {1: "batch"}},
        )

        dec = WhisperDecoderStep(model)
        input_ids = torch.tensor([[cfg.decoder_start_token_id] * 3])
        self_kv = torch.randn(num_layers, 1, num_heads, 5, head_dim)
        kv_axes = {1: "batch", 3: "past_len"}
        torch.onnx.export(
            dec, (input_ids, self_kv, self_kv, cross_k, cross_v), decoder_path,
            opset_version=opset,
            input_names=["input_ids", "self_k", "self_v", "cross_k", "cross_v"],
            output_names=["logits", "present_k", "present_v"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "tgt_len"},
                "self_k": kv_axes, "self_v": kv_axes,
                "cross_k": {1: "batch"}, "cross_v": {1: "batch"},
                "logits": {0: "batch"},
                "present_k": {1: "batch", 3: "total_len"}, "present_v": {1: "batch", 3: "total_len"},
            },
        )
    return encoder_path, decoder_path


# --------------------------------------------------------------------------------------
# TensorRT engine build
# --------------------------------------------------------------------------------------
def build_engine(
    onnx_path: str,
    engine_path: str,
    shapes: Dict[str, Tuple[tuple, tuple, tuple]],
    fp16: bool = False,
    half_io: Tuple[str, ...] = (),
    workspace_gb: int = 4,
    opt_level: int = None,
):
    """ONNX -> TensorRT engine.

    shapes   : {input_name: (min, opt, max)} 최적화 프로파일
    half_io  : FP16 빌드에서 FP16으로 바꿀 입출력 이름 (KV cache 등 큰 텐서의 FP32<->FP16 변환 복사를 없앤다)
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(onnx_path):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {onnx_path}\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if opt_level is not None:
        config.builder_optimization_level = opt_level  # TRT 8.6: 0~5 (기본 3)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        tensors = [network.get_input(i) for i in range(network.num_inputs)]
        tensors += [network.get_output(i) for i in range(network.num_outputs)]
        for t in tensors:
            if t.name in half_io and t.dtype == trt.float32:
                t.dtype = trt.float16

    profile = builder.create_optimization_profile()
    for name, (mn, opt, mx) in shapes.items():
        profile.set_shape(name, mn, opt, mx)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed: {onnx_path}")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return engine_path


def build_whisper_engines(
    config,
    encoder_onnx: str,
    decoder_onnx: str,
    engine_dir: str,
    fp16: bool,
    batch_size: int = 1,
    max_prompt_len: int = 16,
    force: bool = False,
):
    num_layers, num_heads = config.decoder_layers, config.decoder_attention_heads
    head_dim = config.d_model // num_heads
    src_len = config.max_source_positions
    max_tgt = config.max_target_positions  # 448
    tag = "fp16" if fp16 else "fp32"
    enc_path = os.path.join(engine_dir, f"encoder-crosskv-{tag}-bs{batch_size}.engine")
    dec_path = os.path.join(engine_dir, f"decoder-kv-{tag}-bs{batch_size}.engine")

    b = batch_size
    cross = (num_layers, b, num_heads, src_len, head_dim)
    if force or not os.path.exists(enc_path):
        feats = (b, config.num_mel_bins, 3000)
        build_engine(encoder_onnx, enc_path, {"input_features": (feats, feats, feats)},
                     fp16=fp16, half_io=("cross_k", "cross_v"))
    if force or not os.path.exists(dec_path):
        kv = lambda n: (num_layers, b, num_heads, n, head_dim)
        build_engine(
            decoder_onnx, dec_path,
            {
                "input_ids": ((b, 1), (b, 1), (b, max_prompt_len)),
                "self_k": (kv(0), kv(64), kv(max_tgt - 1)),
                "self_v": (kv(0), kv(64), kv(max_tgt - 1)),
                "cross_k": (cross, cross, cross),
                "cross_v": (cross, cross, cross),
            },
            fp16=fp16, half_io=("self_k", "self_v", "cross_k", "cross_v", "present_k", "present_v"),
        )
    return enc_path, dec_path


# --------------------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------------------
class TRTModule:
    """TensorRT 8.6 engine runner (torch tensor in / out, execute_async_v3)."""

    def __init__(self, engine_path: str):
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.dtypes = {n: _TRT_TO_TORCH[self.engine.get_tensor_dtype(n)] for n in names}
        # 길이 0 텐서(첫 step의 빈 KV cache)는 data_ptr()이 0이라 TRT가 거부하므로 더미 주소를 넘긴다
        self._dummy = torch.empty(256, dtype=torch.uint8, device="cuda")

    def _address(self, t: torch.Tensor) -> int:
        return t.data_ptr() if t.numel() > 0 else self._dummy.data_ptr()

    def __call__(self, **inputs: torch.Tensor) -> List[torch.Tensor]:
        stream = torch.cuda.current_stream()
        keep = []
        for name in self.input_names:
            t = inputs[name]
            if t.dtype != self.dtypes[name] or not t.is_contiguous() or not t.is_cuda:
                t = t.to(device="cuda", dtype=self.dtypes[name]).contiguous()
            keep.append(t)
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, self._address(t))
        outputs = []
        for name in self.output_names:
            out = torch.empty(tuple(self.context.get_tensor_shape(name)), dtype=self.dtypes[name], device="cuda")
            self.context.set_tensor_address(name, self._address(out))
            outputs.append(out)
        self.context.execute_async_v3(stream.cuda_stream)
        return outputs


class WhisperTRTKV:
    """인코더 1회 + KV cache greedy decoding. HF generate(num_beams=1)와 같은 logits 처리."""

    def __init__(self, encoder_engine: str, decoder_engine: str, config):
        self.encoder = TRTModule(encoder_engine)
        self.decoder = TRTModule(decoder_engine)
        self.config = config
        num_heads = config.decoder_attention_heads
        self.empty_kv_shape = (config.decoder_layers, 1, num_heads, 0, config.d_model // num_heads)
        self.kv_dtype = self.decoder.dtypes["self_k"]
        self.suppress = torch.tensor(config.suppress_tokens or [], dtype=torch.long, device="cuda")
        self.begin_suppress = torch.tensor(config.begin_suppress_tokens or [], dtype=torch.long, device="cuda")

    def encode(self, input_features):
        return self.encoder(input_features=input_features)

    @torch.no_grad()
    def generate(self, input_features, prompt_ids: List[int], max_length: int = None) -> List[int]:
        """prompt_ids: [<|startoftranscript|>, <|ko|>, <|transcribe|>, <|notimestamps|>] 처럼 강제 토큰 전체."""
        max_length = max_length or self.config.max_length
        eos = self.config.eos_token_id
        cross_k, cross_v = self.encode(input_features)

        tokens = list(prompt_ids)
        input_ids = torch.tensor([tokens], dtype=torch.int32, device="cuda")
        self_k = torch.empty(self.empty_kv_shape, dtype=self.kv_dtype, device="cuda")
        self_v = torch.empty(self.empty_kv_shape, dtype=self.kv_dtype, device="cuda")

        while len(tokens) < max_length:
            logits, self_k, self_v = self.decoder(
                input_ids=input_ids, self_k=self_k, self_v=self_v, cross_k=cross_k, cross_v=cross_v
            )
            logits = logits.float()
            if len(self.suppress):
                logits[:, self.suppress] = -float("inf")
            if len(tokens) == len(prompt_ids) and len(self.begin_suppress):
                logits[:, self.begin_suppress] = -float("inf")
            next_token = logits.argmax(-1)
            tok = int(next_token.item())
            tokens.append(tok)
            if tok == eos:
                break
            input_ids = next_token.view(1, 1).to(torch.int32)
        return tokens


# ======================================================================================
# v2: 정적 shape + GPU greedy 루프 + CUDA graph
#
# 프로파일링(benchmark/profiling)에서 v1의 병목은 TRT/opset 한계가 아니라 실행 방식이었다:
#   - KV 길이가 매 step 바뀌는 동적 shape -> set_input_shape + Myelin shape 재계산 (step당 ~3.7ms CPU)
#   - 매 step .item() 동기화 -> CPU 준비와 GPU 실행이 직렬화
#   - 위 두 가지 때문에 CUDA graph 불가
# v2는 self K/V cache를 고정 길이(max_target_positions=448) 버퍼로 두고 현재 위치(pos)를 입력으로 받는다.
# 엔진은 새 토큰 1개의 K/V만 출력하고, 버퍼 갱신 / suppress / 강제 프롬프트 / argmax / EOT 판정까지
# 전부 GPU 텐서 연산이라 한 step(TRT enqueue + 후처리)을 CUDA graph로 캡처해 replay 한다.
# ======================================================================================
class WhisperDecoderStepStatic(nn.Module):
    """정적 shape decoder 1 step: 토큰 1개, 고정 길이 self K/V cache + pos 마스크."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.num_heads = model.config.decoder_attention_heads

    def forward(self, input_ids, pos, self_k, self_v, cross_k, cross_v):
        # input_ids (B, 1), pos (1,), self_k/v (L, B, H, S, D) = 위치 0..pos-1 까지 유효
        dec = self.decoder
        cache_len = self_k.shape[3]
        x = dec.embed_tokens(input_ids) + dec.embed_positions.weight.index_select(0, pos).unsqueeze(0)

        # 캐시 위치 j >= pos 는 아직 비어 있음 -> 마스크, 새 토큰(자기 자신)은 항상 유효
        empty = torch.arange(cache_len, device=x.device) >= pos
        mask = torch.cat([empty.to(x.dtype) * -10000.0, torch.zeros(1, dtype=x.dtype, device=x.device)])

        new_k, new_v = [], []
        for i, layer in enumerate(dec.layers):
            residual = x
            h = layer.self_attn_layer_norm(x)
            attn = layer.self_attn
            q = _split_heads(attn.q_proj(h) * attn.scaling, self.num_heads)
            k1 = _split_heads(attn.k_proj(h), self.num_heads)
            v1 = _split_heads(attn.v_proj(h), self.num_heads)
            new_k.append(k1)
            new_v.append(v1)
            k = torch.cat([self_k[i], k1], dim=2)
            v = torch.cat([self_v[i], v1], dim=2)
            scores = torch.matmul(q, k.transpose(-1, -2)) + mask
            x = residual + attn.out_proj(_merge_heads(torch.matmul(torch.softmax(scores, dim=-1), v)))

            residual = x
            h = layer.encoder_attn_layer_norm(x)
            attn = layer.encoder_attn
            q = _split_heads(attn.q_proj(h) * attn.scaling, self.num_heads)
            scores = torch.matmul(q, cross_k[i].transpose(-1, -2))
            x = residual + attn.out_proj(_merge_heads(torch.matmul(torch.softmax(scores, dim=-1), cross_v[i])))

            residual = x
            h = layer.final_layer_norm(x)
            x = residual + layer.fc2(layer.activation_fn(layer.fc1(h)))

        x = dec.layer_norm(x)
        logits = self.proj_out(x[:, -1, :])
        return logits, torch.stack(new_k), torch.stack(new_v)


def export_decoder_static_onnx(model, decoder_path: str, opset: int = ONNX_OPSET):
    cfg = model.config
    num_layers, num_heads = cfg.decoder_layers, cfg.decoder_attention_heads
    head_dim, src_len, cache_len = cfg.d_model // num_heads, cfg.max_source_positions, cfg.max_target_positions
    model = model.float().cpu().eval()
    os.makedirs(os.path.dirname(decoder_path), exist_ok=True)
    with torch.no_grad():
        args = (
            torch.tensor([[cfg.decoder_start_token_id]]),
            torch.tensor([3]),
            torch.randn(num_layers, 1, num_heads, cache_len, head_dim),
            torch.randn(num_layers, 1, num_heads, cache_len, head_dim),
            torch.randn(num_layers, 1, num_heads, src_len, head_dim),
            torch.randn(num_layers, 1, num_heads, src_len, head_dim),
        )
        torch.onnx.export(
            WhisperDecoderStepStatic(model), args, decoder_path, opset_version=opset,
            input_names=["input_ids", "pos", "self_k", "self_v", "cross_k", "cross_v"],
            output_names=["logits", "new_k", "new_v"],
        )
    return decoder_path


def build_decoder_static_engine(config, decoder_onnx: str, engine_dir: str, fp16: bool, opt_level: int = 3,
                                force: bool = False):
    num_layers, num_heads = config.decoder_layers, config.decoder_attention_heads
    head_dim = config.d_model // num_heads
    tag = "fp16" if fp16 else "fp32"
    path = os.path.join(engine_dir, f"decoder-static-{tag}-bs1-O{opt_level}.engine")
    if force or not os.path.exists(path):
        cache = (num_layers, 1, num_heads, config.max_target_positions, head_dim)
        cross = (num_layers, 1, num_heads, config.max_source_positions, head_dim)
        fixed = lambda shape: (shape, shape, shape)  # noqa: E731
        build_engine(
            decoder_onnx, path,
            {"input_ids": fixed((1, 1)), "pos": fixed((1,)), "self_k": fixed(cache), "self_v": fixed(cache),
             "cross_k": fixed(cross), "cross_v": fixed(cross)},
            fp16=fp16, half_io=("self_k", "self_v", "cross_k", "cross_v", "new_k", "new_v"), opt_level=opt_level,
        )
    return path


class WhisperTRTKVGraph:
    """v2 runtime: 정적 shape 디코더 + GPU greedy 루프를 CUDA graph로 replay (step마다 CPU-GPU 동기화 없음)."""

    PIPELINE = 2  # EOT 확인을 몇 step 늦게 할지 (그동안 GPU는 계속 앞서 간다)

    def __init__(self, encoder_engine: str, decoder_engine: str, config, use_graph: bool = True):
        self.config = config
        self.encoder = TRTModule(encoder_engine)
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(decoder_engine, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        dt = lambda n: _TRT_TO_TORCH[self.engine.get_tensor_dtype(n)]  # noqa: E731

        L, H = config.decoder_layers, config.decoder_attention_heads
        D, S, V = config.d_model // H, config.max_target_positions, config.vocab_size
        self.max_len = S
        dev = "cuda"
        # 고정 버퍼 (주소가 바뀌면 graph가 깨지므로 전부 미리 할당하고 in-place로만 갱신)
        self.ids = torch.zeros(1, 1, dtype=dt("input_ids"), device=dev)
        self.pos = torch.zeros(1, dtype=dt("pos"), device=dev)
        self.self_k = torch.zeros(L, 1, H, S, D, dtype=dt("self_k"), device=dev)
        self.self_v = torch.zeros_like(self.self_k)
        self.cross_k = torch.zeros(L, 1, H, config.max_source_positions, D, dtype=dt("cross_k"), device=dev)
        self.cross_v = torch.zeros_like(self.cross_k)
        self.logits = torch.empty(1, V, dtype=dt("logits"), device=dev)
        self.new_k = torch.empty(L, 1, H, 1, D, dtype=dt("new_k"), device=dev)
        self.new_v = torch.empty_like(self.new_k)
        self.tokens = torch.zeros(S + 1, dtype=torch.int32, device=dev)   # 생성 결과 (위치별 토큰)
        self.prompt = torch.zeros(S + 1, dtype=torch.int32, device=dev)   # 강제 프롬프트 토큰
        self.prompt_len = torch.zeros(1, dtype=torch.int32, device=dev)
        self.done = torch.zeros(1, dtype=torch.int32, device=dev)
        self.eos = config.eos_token_id

        # suppress_tokens: 항상 금지, begin_suppress_tokens: 첫 생성 토큰에서만 금지.
        # -inf를 쓰면 첫 토큰이 아닌 step에서 0 * -inf = NaN이 되고 argmax가 NaN을 고르므로 유한한 큰 음수를 쓴다.
        neg = -1e9
        self.bias = torch.zeros(V, dtype=torch.float32, device=dev)
        self.bias[torch.tensor(config.suppress_tokens or [], dtype=torch.long)] = neg
        self.begin_bias = torch.zeros(V, dtype=torch.float32, device=dev)
        self.begin_bias[torch.tensor(config.begin_suppress_tokens or [], dtype=torch.long)] = neg

        for name, t in [("input_ids", self.ids), ("pos", self.pos), ("self_k", self.self_k), ("self_v", self.self_v),
                        ("cross_k", self.cross_k), ("cross_v", self.cross_v), ("logits", self.logits),
                        ("new_k", self.new_k), ("new_v", self.new_v)]:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        self.stream = torch.cuda.Stream()
        self.done_host = torch.zeros(self.PIPELINE + 1, dtype=torch.int32).pin_memory()
        self.events = [torch.cuda.Event() for _ in range(self.PIPELINE + 1)]
        self.graph = None
        if use_graph:
            self._capture()

    def _step(self):
        """한 step: TRT 디코더 -> cache 갱신 -> logits 후처리 -> 다음 토큰. 전부 self.stream 위 GPU 연산."""
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        p = self.pos.long()
        self.self_k.index_copy_(3, p, self.new_k)
        self.self_v.index_copy_(3, p, self.new_v)
        nxt_pos = p + 1
        first_gen = (nxt_pos == self.prompt_len.long()).float()  # 이번 출력이 첫 생성 토큰
        scores = self.logits.float() + self.bias + first_gen * self.begin_bias
        pred = scores.argmax(-1).to(torch.int32)
        forced = nxt_pos < self.prompt_len.long()
        nxt = torch.where(forced, self.prompt.index_select(0, nxt_pos), pred)
        # EOT 이후 토큰은 기록하지 않음 (done이면 기존 값 유지)
        keep = self.done.bool()
        self.tokens.index_copy_(0, nxt_pos, torch.where(keep, self.tokens.index_select(0, nxt_pos), nxt))
        self.done.copy_((keep | ((nxt == self.eos) & ~forced)).to(torch.int32))
        self.ids.copy_(nxt.view(1, 1))
        self.pos.copy_(nxt_pos.to(self.pos.dtype))

    def _capture(self):
        with torch.cuda.stream(self.stream):
            self._reset([self.config.decoder_start_token_id])
            for _ in range(3):  # warmup (TRT는 캡처 전에 한 번 이상 enqueue 필요)
                self._step()
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self._step()

    def _reset(self, prompt_ids):
        n = len(prompt_ids)
        self.prompt.zero_()
        self.prompt[:n].copy_(torch.tensor(prompt_ids, dtype=torch.int32), non_blocking=False)
        self.prompt_len.fill_(n)
        self.tokens.zero_()
        self.tokens[:n].copy_(self.prompt[:n])
        self.ids.fill_(prompt_ids[0])
        self.pos.zero_()
        self.done.zero_()
        self.self_k.zero_()
        self.self_v.zero_()

    def encode(self, input_features):
        return self.encoder(input_features=input_features)

    @torch.no_grad()
    def generate(self, input_features, prompt_ids: List[int], max_length: int = None) -> List[int]:
        max_length = min(max_length or self.config.max_length, self.max_len)
        with torch.cuda.stream(self.stream):
            cross_k, cross_v = self.encode(input_features)
            self.cross_k.copy_(cross_k)
            self.cross_v.copy_(cross_v)
            self._reset(prompt_ids)
            n_steps = max_length - 1
            k = 0
            for k in range(n_steps):
                if self.graph is not None:
                    self.graph.replay()
                else:
                    self._step()
                slot = k % len(self.events)
                self.done_host[slot : slot + 1].copy_(self.done, non_blocking=True)
                self.events[slot].record()
                if k >= self.PIPELINE:  # PIPELINE step 전 결과만 확인 -> GPU를 막지 않음
                    old = (k - self.PIPELINE) % len(self.events)
                    self.events[old].synchronize()
                    if self.done_host[old]:
                        break
            self.stream.synchronize()
        toks = self.tokens[: k + 2].tolist()
        out = []
        for i, t in enumerate(toks):
            out.append(t)
            if t == self.eos and i >= len(prompt_ids):
                break
        return out
