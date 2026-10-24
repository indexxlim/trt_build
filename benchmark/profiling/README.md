# 백엔드별 프로파일링: TensorRT-LLM과 왜 차이 나는가

whisper-large-v2 FP16, 한국어 샘플(30초, 126토큰)의 E2E 1회를 warmup 2회 뒤에 프로파일링했다. 2026-10-24에 빈 GPU 0(다른 작업 없음)에서 백엔드 하나씩 순서대로 돌렸다. 수치 원본은 [PROFILE.md](PROFILE.md)에 있고 `analyze.py`가 만든다.

```bash
GPU=0 bash benchmark/profiling/run_profile.sh   # 백엔드별 cProfile + nsys (TRT-LLM은 torch profiler)
python benchmark/profiling/analyze.py            # -> PROFILE.md
```

| 도구 | 보는 것 |
|---|---|
| **nsys** 2023.2.3 (`-t cuda,nvtx,osrt --cuda-graph-trace=node`) | GPU 커널별 시간, CUDA API(launch, sync, memcpy). CUDA graph 안의 커널도 노드 단위로 펼쳐 기록 |
| **torch profiler** (Kineto/CUPTI) | TensorRT-LLM 전용. nsys 2023.2가 CUDA 13 앱의 GPU trace를 못 잡아서 대신 썼다 |
| **cProfile** | Python 함수별 자체 시간(tottime) |

## 1. TensorRT-LLM FP16 대비 차이 분해

E2E를 세 부분으로 나눴다.
- **인코더**
- **디코더 GPU**: 디코더 구간에서 GPU 커널이 실제로 돈 시간
- **디코더 대기**: 디코더 구간에서 GPU가 CPU를 기다린 시간 = 디코더 wall − 디코더 GPU

Δ는 TensorRT-LLM FP16과의 차이다. E2E는 프로파일러 없이 잰 값(`results/`, 10-23 빈 GPU 2), 나머지는 프로파일에서 잰 값이다.

| 백엔드 | E2E (ms) | ΔE2E | 인코더 (Δ) | 디코더 GPU (Δ) | 디코더 대기 (Δ) | 디코더 GPU/토큰 | 커널/토큰 | **차이의 주원인** |
|---|---|---|---|---|---|---|---|---|
| **TensorRT-LLM FP16** | **548** | 0 | 23 | 446 | 80 | **3.54ms** | 418 | 기준 |
| TensorRT-LLM INT8 | 580 | +32 | 26 (+3) | 473 (+27) | 81 (+2) | 3.75ms | 397 | 디코더 GPU (INT8 weight-only GEMV가 FP16보다 약간 느림) |
| 자체 구현 v2 (CUDA graph) | 724 | +176 | 17 (−6)³ | 678 (+232)³ | 21 (−59)³ | 5.38ms³ | 672 | **디코더 GPU**: FP32 GEMV, 매 step cross K 전치 |
| CTranslate2 FP16 | 802 | +254 | 63 (+40) | 565 (+119) | 175 (+95) | 4.48ms | 897 | 디코더 GPU(unfused) + 대기(커널 launch) + 인코더 |
| 자체 구현 v1 (KV) | 1044 | +495 | 17 (−6) | 815 (+369) | 212 (+133) | 6.47ms | 746 | 디코더 GPU(32x32 GEMM, `mha_v2`) + 대기(enqueue, `.item()`) |
| NNDF non-KV | 1092 | +544 | 42 (+19) | 1084 (+638) | 0 (−80) | 8.61ms | 444 | **디코더 GPU**: KV cache 없이 매 step 다시 계산 |
| CTranslate2 INT8 | 1126 | +578 | 72 (+49) | 908 (+462) | 146 (+66) | 7.21ms | 1206 | 디코더 GPU: INT8 GEMM과 (역)양자화 |
| HF transformers | 2712 | +2164 | 49 (+26) | 711 (+266) | 1953 (+1873) | 5.65ms | 836 | **디코더 대기**: Python op dispatch |
| `torch_tensorrt` 인코더 | 2911 | +2363 | 226 (+203) | 660 (+215) | 2025 (+1945) | 5.24ms | 836 | 디코더 대기(HF와 같음) + 인코더 FP32 |

### 디코더 GPU 시간을 커널 종류별로 (ms/토큰)

| 백엔드 | GEMM/GEMV | attention | TRT Myelin (fused) | norm | softmax | elementwise/copy | 합 |
|---|---|---|---|---|---|---|---|
| **TensorRT-LLM FP16** | **1.59** | **1.11** | 0.68 | - | 0.04 | 0.01 | 3.54 |
| TensorRT-LLM INT8 | 1.78 | 1.22 | 0.60 | - | 0.04 | 0.01 | 3.75 |
| 자체 구현 v2 | 3.46 | 1.50 | 1.67 | - | - | 0.67 | 6.30 |
| CTranslate2 FP16 | 2.69 | (GEMM에 포함) | - | 0.41 | 0.30 | 1.06 | 4.48 |
| 자체 구현 v1 | 2.66 | 2.18 | 1.61 | - | - | 0.01 | 6.47 |
| NNDF non-KV | 5.19 | 2.53 | 0.78 | - | - | 0.05 | 8.61 |
| CTranslate2 INT8 | 4.81 | (GEMM에 포함) | - | 0.47 | 0.30 | 1.60 | 7.21 |
| HF transformers | 3.62 | (bmm, GEMM에 포함) | - | 0.67 | 0.39 | 0.94 | 5.65 |

- **기준선(하한)**: large-v2 디코더 가중치는 FP16으로 약 1.6GB이고, batch 1에서는 토큰마다 이걸 전부 한 번씩 읽는다. A100(약 2TB/s)에서 GEMM 하한은 토큰당 약 0.8ms다. TensorRT-LLM의 1.59ms는 대역폭의 약 50%를 쓰는 셈이고, 다른 백엔드는 그보다 비효율적이다.
- 경계 기준: 인코더와 디코더는 "디코더 첫 step에만 나오는 커널이 처음 등장하는 위치"로 나눴다.
  - 이 기준은 경계 앞 GPU 시간이 인코더만 따로 잰 시간과 맞는지로 검증했다(HF 48.6 vs 47.5ms, v1 16.8 vs 17.2ms, CT2 62.5 vs 56.9ms).
  - NNDF non-KV는 디코더 전용 커널이 없어서 따로 잰 인코더 시간(41.6ms)으로 나눴다.
  - TRT-LLM INT8은 인코더도 weight-only GEMM을 써서, 디코더 context의 `paged_kv` attention을 경계로 썼다.
  - v2의 인코더 값에는 cross K/V 복사와 cache 초기화(약 8ms)가 섞여 있다.
- 디코더 GPU 시간은 프로파일러 아래에서 잰 값이라, 프로파일러 없이 잰 E2E보다 조금 클 수 있다(NNDF non-KV). 이 경우 대기는 0으로 표시했다.
- ³ v2는 nsys가 CUDA graph를 노드 단위로 기록하면서 시간을 약 17% 부풀린다(nsys 6.30 vs 실제 5.38ms/step). 그래서 프로파일러 없이 CUDA event로 잰 graph replay 시간(5.38ms/step)과 인코더 시간(17.1ms)으로 바꿔 적었다.

## 1-1. TensorRT-LLM vs 자체 구현 v1/v2: 토큰당 수치 비교

디코더 토큰 1개를 만들 때 무엇이 얼마나 다른지를 나란히 놓았다(한국어, 126토큰 평균, 빈 GPU).

| 항목 (토큰당) | TRT-LLM FP16 | TRT-LLM INT8 | 자체 구현 v2 | 자체 구현 v1 |
|---|---|---|---|---|
| **실제 디코더 시간** ((E2E − 인코더) / 126) | **4.17ms** | 4.40ms | 5.55ms | 8.15ms |
| └ GPU가 일한 시간 | 3.54ms | 3.75ms | **5.38ms**¹ | 6.47ms |
| └ GPU가 CPU를 기다린 시간 | 0.63ms | 0.65ms | **0.17ms** | 1.68ms |
| **행렬곱(GEMM/GEMV) 시간** | **1.59ms** | 1.78ms | 2.54ms¹ (nsys 3.46) | 2.66ms |
| 행렬곱 커널 | `cudaCoreGemm` (M=1 전용, CUDA core) | `weight_only::kernel` (INT8 가중치, FP16 연산) | cuBLAS `gemvx` **FP32** | `xmma_gemm` tensor core 32x32 타일 |
| 행렬곱 호출 수 | 168 | 159 | 234 | 225 |
| 토큰당 읽는 가중치² | 1.60GB (FP16) | 0.87GB (INT8 + FP16 lm_head) | **3.20GB (FP32)** | 1.60GB (FP16) |
| 실효 메모리 대역폭 (가중치 / GEMM 시간) | **1.01TB/s** | 0.49TB/s | 1.26TB/s¹ | 0.60TB/s |
| **attention 시간** | **1.11ms** | 1.22ms | 1.10ms¹ (nsys 1.50) | 2.18ms |
| attention 커널 | `masked_multihead_attention` (디코딩용 fused) | 같음 | `mha_v1_*` | `mha_v2_*` (긴 query용) |
| **LayerNorm, residual, 활성화 (fused)** | **0.68ms / 169개** | 0.60ms / 160개 | 1.23ms¹ / 337개 | 1.61ms / 453개 |
| 기타 elementwise, 변환 | 0.01ms | 0.01ms | 0.49ms¹ (cross K 전치, FP16→FP32 변환, GPU 루프) | 0.01ms |
| **GPU 커널 수** | **418** | 397 | 672 | 746 |
| 인코더 (1회) | 23ms | 26ms | **17ms** | **17ms** |

¹ v2는 CUDA graph라서 nsys 노드 단위 기록이 시간을 약 17% 부풀린다(nsys 6.30 vs 실제 5.38ms/step). 그래서 CUDA event로 graph replay 1회를 프로파일러 없이 잰 **5.38ms**에, nsys에서 얻은 커널 종류별 비율을 곱해 환산했다(GEMM 47%, attention 20%, fused 23%, 기타 9%).

² large-v2 디코더 행렬곱 가중치는 층당 self q/k/v/o + cross q/o + fc1/fc2 = 22.9M개 × 32층 = 734M개에, proj_out 66M개를 더해 약 800M개다. cross K/V projection은 인코더 쪽에서 한 번만 하므로 빠진다. A100-SXM4-80GB 메모리 대역폭은 약 2.0TB/s다.

### 자체 구현 v2가 TRT-LLM FP16보다 토큰당 1.38ms 느린 이유 (5.55 vs 4.17ms)

| 원인 | 토큰당 차이 | 근거 |
|---|---|---|
| 행렬곱 | **+0.95ms** | 가중치를 FP32로 읽는다(3.20GB vs 1.60GB). 커널 대역폭(1.26TB/s)은 TRT-LLM보다 오히려 높지만, 읽는 양이 2배다 |
| LayerNorm, residual 등 fused 조각 | +0.55ms | TRT-LLM은 Myelin 커널 169개, v2는 337개로 2배 잘게 쪼개져 있다 |
| 기타 elementwise | +0.48ms | 매 step cross K 전치(`permutationKernelPLC3`), FP16→FP32 변환(`copyVectorizedKernel`), GPU greedy 루프 연산 |
| attention | 약 0ms | `mha_v1`(1.10ms)과 TRT-LLM masked MHA(1.11ms)가 비슷하다 |
| CPU 대기 | **−0.46ms** | v2가 더 낫다. CUDA graph replay에 EOT 확인만 2 step 늦게 하므로 0.17ms이고, TRT-LLM C++ executor는 0.63ms다 |
| (인코더) | −6ms (1회) | v2 인코더(opset 17 FP16 TRT 엔진) 17ms vs TRT-LLM 23ms |

→ **v2의 GPU 커널이 1.84ms/토큰 더 걸리고, 그중 행렬곱 FP32가 절반이다.** MatMul을 FP16으로 고정하면(가중치 1.60GB, 같은 1.26TB/s를 가정하면 약 1.27ms) 행렬곱은 TRT-LLM(1.59ms)보다 빨라질 수 있다. 여기에 cross K를 미리 전치해 두면 차이 대부분이 없어질 것으로 예상한다. 둘 다 아직 측정하지 않은 예상치다.

### 자체 구현 v1이 TRT-LLM FP16보다 토큰당 3.98ms 느린 이유 (8.15 vs 4.17ms)

| 원인 | 토큰당 차이 | 근거 |
|---|---|---|
| attention | **+1.07ms** | 긴 query용 `mha_v2`를 query 1개에 쓴다(2.18 vs 1.11ms) |
| 행렬곱 | **+1.07ms** | FP16 가중치는 같지만 M=1에 tensor core 32x32 타일 GEMM을 써서 실효 대역폭이 0.60TB/s다(TRT-LLM 1.01TB/s) |
| CPU 대기 | **+1.05ms** | 동적 shape enqueue(step당 4.0ms CPU)와 매 step `.item()` 동기화(1.68 vs 0.63ms) |
| fused 조각 | +0.93ms | Myelin 커널 453개(TRT-LLM 169개) |

### TRT-LLM INT8이 FP16보다 빠르지 않은 이유 (4.40 vs 4.17ms)

- 읽는 가중치는 절반 가까이 줄었다(0.87GB vs 1.60GB).
- 그런데 weight-only INT8 GEMV의 실효 대역폭이 0.49TB/s로 FP16(1.01TB/s)의 절반이다. INT8 → FP16 역양자화를 커널 안에서 하는 비용이 batch 1에서는 메모리 절약분을 상쇄한다. 그래서 행렬곱 시간이 오히려 +0.19ms 늘었다(1.78 vs 1.59ms).

## 2. 백엔드별 근거 (TensorRT-LLM FP16과 비교)

토큰당 수치는 디코더 구간 커널 시간을 126으로 나눈 값이다.

### TensorRT-LLM FP16 (기준): 커널이 M=1에 맞춰져 있음

- **GEMM**: `cudaCoreGemm<__half, __half, 1, 2, 128>` 1.47ms/토큰(165회/토큰). M=1 전용 CUDA-core GEMV로, tensor core 타일을 쓰지 않고 가중치를 한 번만 스트리밍한다.
- **attention**: `masked_multihead_attention_kernel` 1.07ms/토큰(55회/토큰). 토큰 1개의 query로 KV cache를 바로 읽는 fused 디코딩용 커널이다. QK^T, softmax, ·V가 커널 하나다.
- **LayerNorm과 residual**: Myelin이 fuse한 `__myl_AddAddCastMeanSub...Add` 0.46ms/토큰. 커널 하나에 residual, LayerNorm, cast가 묶여 있다.
- **디코더 대기 80ms(0.6ms/토큰)**: Python은 `session.await_responses()`로 기다리기만 하고(cProfile `_initialize_and_fill_output`), C++ executor가 step마다 스케줄링과 샘플링을 한다.

### 자체 구현 v2 (정적 shape + GPU 루프 + CUDA graph): +176ms, 디코더 GPU가 원인

v1의 CPU 쪽 문제(동적 shape enqueue, `.item()` 동기화)는 없어졌다. 디코더 대기가 0이고, CUDA graph replay라 GPU가 쉬지 않는다. 남은 차이는 전부 커널 효율이다.

- **FP16 엔진인데 디코더 GEMM이 FP32로 돈다**: `sm50_xmma_cublas_gemvx_f32f32_f32_f32` nsys 기준 2.64 + 0.67ms/토큰(약 233회/토큰). 실제 시간으로 환산하면 약 2.5ms/토큰이다(1-1절).
  - 정적 shape로 빌드했더니 TRT가 M=1 행렬곱에 cuBLAS **FP32** GEMV tactic을 골랐다.
  - 그래서 가중치를 FP32(3.2GB/토큰)로 읽어 TRT-LLM GEMM(1.47ms)의 2배가 넘는다.
  - FP16 입출력을 FP32로 바꾸는 `copyVectorizedKernel<..., __half, float>`(0.15ms/토큰)도 함께 붙는다.
- **매 step cross K를 전치한다**: `CUTENSOR permutationKernelPLC3` 0.46ms/토큰(2회/토큰). 인코더가 만든 cross K/V(32층×1500프레임)는 고정인데, 디코더 그래프 안의 `K^T` 때문에 매 step 다시 transpose한다.
- **attention**: `mha_v1_*` 1.50ms/토큰. TRT-LLM의 fused masked MHA(1.11ms)보다 느리다.
- **고칠 방향**(같은 TRT 8.6에서 가능):
  - MatMul 레이어 precision을 FP16으로 고정한다(`OBEY_PRECISION_CONSTRAINTS`).
  - 인코더 엔진이 cross K를 미리 전치된 모양으로 출력하게 한다.

### CTranslate2 FP16: +254ms, 커널이 잘게 쪼개져 있음

- **디코더 GPU +119ms**: GEMM 자체는 cuBLAS `gemvx` 1.42ms/토큰으로 나쁘지 않다. 하지만 softmax, LayerNorm, elementwise가 각각 별도 커널이다.
  - `cub::for_each` 등 elementwise 1.06ms/토큰(417회/토큰)
  - LayerNorm 0.41ms, softmax 0.30ms
  - TRT-LLM에서는 이것들이 Myelin과 MHA 커널 안에 fuse돼 있다(0.68ms).
- **디코더 대기 +95ms**: 토큰당 커널 897개를 CPU가 하나씩 띄운다(`cudaLaunchKernel` 약 10만 회). op마다 `cudaMallocAsync/FreeAsync`(각 5.4만 회)도 한다. CUDA graph를 쓰지 않는다.
- **인코더 +40ms**: 인코더도 unfused 커널이다(57ms vs TRT-LLM 23ms).
- cProfile에는 `generate` C++ 호출 하나만 보인다. Python 오버헤드는 없다.

### 자체 구현 v1 (KV, 동적 shape): +495ms, 디코더 GPU와 CPU 대기 모두

- **GEMM**: `sm80_xmma_gemm ... tilesize32x32x64` 2.10ms/토큰(192회/토큰). M=1에 tensor core GEMM 타일을 써서 TRT-LLM GEMV(1.47ms)보다 느리다.
- **attention**: `mha_v2_*` 2.18ms/토큰. 긴 query용 fused MHA를 query 1개에 쓴다.
- **디코더 대기 +133ms**: 매 step KV 길이가 바뀌는 동적 shape라서 `set_input_shape`와 Myelin shape 재계산을 한다(cProfile `TRTModule.__call__` 504ms, step당 4.0ms, `cuLaunchKernel` 6.5만 회). 여기에 매 step `.item()` 동기화가 더해진다(`cudaStreamSynchronize` 254회 309ms). v2가 이 부분을 없앴다.

### NNDF non-KV (opset 12): +544ms, 매 step 다시 계산

- **디코더 GPU +638ms**: KV cache가 없어서 매 step 두 가지를 다시 계산한다.
  - 지금까지 나온 모든 토큰의 self-attention
  - **인코더 1500프레임의 cross K/V projection**: `sm80_xmma_gemm ... 128x128` 1.35ms/토큰(3회/토큰)

  GEMM 5.19ms/토큰, attention(`mha_v2_*`) 2.53ms/토큰이다.
- 커널 수(444/토큰)는 Myelin fuse 덕분에 TRT-LLM과 비슷하다. 하지만 토큰당 연산량 자체가 크다. GPU가 쉬지 않는(대기 0) 유일한 레거시 경로다.

### CTranslate2 INT8: +578ms, batch 1에서 INT8 GEMM이 손해

- **INT8 GEMM**: `gemmSN_kernel_int32` 3.26ms/토큰. FP16 GEMV(1.42ms)의 2배가 넘는다.
- **양자화와 역양자화**: GEMM마다 `quantize_kernel` 0.81ms/토큰과 `dequantize_gemm_output_kernel` 0.57ms/토큰이 붙는다. 그래서 커널이 1206개/토큰까지 늘어난다.
- 반면 TRT-LLM INT8은 weight-only(가중치만 INT8, 연산은 FP16) GEMV 1.50ms/토큰이라 FP16과 거의 같다.

### HF transformers: +2164ms, Python op dispatch가 원인

- **디코더 대기 +1873ms(15ms/토큰)**가 차이의 86%다.
  - 토큰당 Python에서 op 수백 개를 하나씩 호출하고 커널 836개를 하나씩 launch한다.
  - cProfile 상위 함수: `torch._C._nn.linear` 839ms(3.3만 회), `WhisperAttention.forward` 306ms, `WhisperDecoderLayer.forward` 252ms, `torch.bmm` 246ms, `layer_norm` 205ms, `view` 132ms(5만 회), `torch.cat` 107ms
  - nsys `cudaLaunchKernel` 106,185회 770ms(nsys 아래 수치)
- `SuppressTokensLogitsProcessor.__call__`(generation_logits_process.py:732)은 Python 리스트로 `scores[:, list] = -inf`를 해서 매 step H2D 복사와 동기화 지점을 만든다(`cudaMemcpyAsync` 453회). 하지만 빈 GPU에서는 13ms(step당 0.1ms)뿐이다. CPU가 GPU보다 느려서 동기화에서 기다릴 GPU 작업이 거의 없기 때문이다.
  - ⚠️ GPU를 다른 학습과 같이 쓴 10-23 측정에서는 같은 지점이 254ms였다(`cudaStreamSynchronize` 254ms vs 빈 GPU 3.7ms). 밀린 GPU 작업을 이 지점에서 기다린 것이므로, 병목으로 보면 안 된다.
- 디코더 GPU 자체도 +266ms다. cuBLAS `gemvx` 2.43ms/토큰(263회)에, KV cache를 붙이는 `torch.cat`(`CatArrayBatchedCopy`)이 0.39ms/토큰이다.

### `torch.compile(encoder, backend="torch_tensorrt")`: +2363ms

- 디코더는 HF와 같아서 디코더 대기도 같다(+1945ms).
- **인코더 +203ms**: Torch-TensorRT 1.4.0이 `aten.mm/bmm`을 변환하지 못해 행렬곱이 FP32 SGEMM(`ampere_sgemm_*`)으로 남는다. TRT 엔진 448개는 그 사이의 elementwise 조각만 맡는다. 미지원 op 목록은 [../nndf_legacy/README.md](../nndf_legacy/README.md)에 있다.

## 3. 결론

batch 1 Whisper 디코딩은 토큰당 GPU 일이 3.5~9ms 정도다. TensorRT-LLM과의 차이는 세 가지로 갈린다.

1. **디코더 대기(CPU)**: Python 루프(HF, `torch_tensorrt`), 동적 shape enqueue와 동기화(v1), 커널 launch 수(CT2). v2처럼 GPU 루프와 CUDA graph로 바꾸면 0까지 줄어든다.
2. **M=1 GEMM 효율**: TRT-LLM `cudaCoreGemm` 1.47ms/토큰에 비해, FP32 GEMV(v2) 3.3ms, tensor core 32x32 타일(v1) 2.1ms, INT8 GEMM(CT2) 3.3ms다.
3. **attention과 fusion**: TRT-LLM masked MHA 1.1ms/토큰에 비해 `mha_v1/v2`는 1.5~2.5ms다. unfused LayerNorm, softmax, elementwise(CT2, HF)는 토큰당 1~2ms를 더 쓴다.

같은 TensorRT 8.6에서 1번은 이미 해결했고(v2), 2번 중 FP32 GEMV와 cross K 전치는 고칠 수 있다. M=1 전용 GEMV와 masked MHA 커널은 TensorRT-LLM 플러그인 쪽이 앞선다.
