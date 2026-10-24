# Whisper large-v2 추론 백엔드 비교

## 목적

`notebooks/whisper.ipynb`의 TensorRT(NNDF) 경로가 HF 대비 2.3배 정도에서 멈춘 이유를 확인하고, 다음 두 가지를 같은 조건에서 수치로 비교한다.

1. 레거시 TensorRT 경로(NNDF, `torch.compile(backend="torch_tensorrt")`)를 어디까지 끌어올릴 수 있는가
2. 현행 백엔드(CTranslate2, TensorRT-LLM)와 비교하면 어떤가

그리고 각 백엔드가 가장 빠른 백엔드보다 **왜**, **어디서(인코더 / 디코더 GPU / CPU 대기 / 커널 종류)** 느린지 프로파일로 설명한다.

## 실험

### 대상

| 폴더 | conda env | 백엔드 |
|---|---|---|
| [nndf_legacy/](nndf_legacy/README.md) | `trt` (py3.8, TRT 8.6.1, torch 2.0.1) | HF transformers(기준선), `torch.compile(backend="torch_tensorrt")`(Torch-TensorRT 1.4.0), NNDF TensorRT non-KV(opset 12), NNDF 흐름 자체 구현 v1(KV cache, opset 17), v2(+정적 shape, GPU 루프, CUDA graph) |
| [ct2/](ct2/README.md) | `ct2` (py3.10) | CTranslate2 4.8.3 (FP16, INT8) |
| [trtllm/](trtllm/README.md) | `trtllm` (py3.12, TRT 10.14) | TensorRT-LLM 1.2.1 (FP16, INT8 weight-only) |

### 조건

- 모델은 `openai/whisper-large-v2`, batch 1, greedy, 최대 448토큰이다.
- 입력은 `data/inputs.npz`의 HF `WhisperProcessor` log-mel (1, 80, 3000)이고, 모든 백엔드가 같은 텐서를 쓴다. `prepare_inputs.py`로 만든다.
  - EN: librispeech dummy[0], 5.9초, 22토큰
  - KO: `notebooks/korean_news.mp4` 앞 30초, 126토큰
- 프롬프트는 `<|startoftranscript|><|lang|><|transcribe|><|notimestamps|>`이다.
- 정확도는 HF transformers FP32 `generate` 결과(`data/reference.json`)와 텍스트가 완전히 같은지, 그리고 CER로 본다.
- 시간은 warmup 2회 후 10회 반복한 wall-clock p50이다. 매 회 GPU를 sync하고, E2E는 인코더와 디코딩 전체를 포함한다.

### 방법

| 단계 | 내용 | 실행 |
|---|---|---|
| 속도와 정확도 | 백엔드별 `bench.py`. 결과는 `results/*.json` | `GPU=0 bash benchmark/run_all.sh` → `results/RESULTS.md` |
| 프로파일링 | 한국어 E2E 1회를 nsys(TRT-LLM은 torch profiler)와 cProfile로 기록 | `GPU=0 bash benchmark/profiling/run_profile.sh` → `python benchmark/profiling/analyze.py` |
| 개별 실험 | LayerNorm FP16 overflow, Torch-TensorRT 미지원 op와 재컴파일, v1/v2 정확도 검증, v2 step 실측, 토큰당 비교 | [experiments/](experiments/) |

모든 실험의 목적, 조건, 결과, 버린 측정과 그 이유는 [EXPERIMENTS.md](EXPERIMENTS.md)에 시간순으로 기록했다.

## 결과 (빈 GPU, A100-SXM4-80GB)

whisper-large-v2, E2E p50이다. 2026-10-23 20:54~20:59에 gloss_sim GPU 큐로 빈 GPU 2를 받아 연달아 측정했다. 측정 중 다른 프로세스는 없었다(`results/*.json`의 `nvidia_smi`는 벤치 자신의 사용량). Torch-TensorRT 행만 10-24 빈 GPU 0에서 측정했다.

| 구분 | 백엔드 | 정밀도 | 인코더 p50 (ms) | E2E EN p50 (ms) | E2E KO p50 (ms) | HF FP16 대비 (EN / KO) | 텍스트 == HF |
|---|---|---|---|---|---|---|---|
| 기준선 | HF transformers `generate` (KV cache) | FP16 | 49.0 | 562.2 | 2712.5 | 1.0x / 1.0x | ✅ |
| 레거시 | `torch.compile(encoder, backend="torch_tensorrt")` + HF decoder | FP16 | 201.0 | 707.8 | 2911.0 | 0.8x / 0.9x | ✅ |
| 레거시 | NNDF TensorRT (non-KV, opset 12) | FP16 | 41.5 | 244.3 | 1092.0 | 2.3x / 2.5x | ✅ |
| 레거시 | NNDF 흐름 자체 구현 v1 (KV cache, opset 17) | FP16 | 16.9 | 194.8 | 1043.7 | 2.9x / 2.6x | ✅ |
| 레거시 | NNDF 흐름 자체 구현 v2 (+정적 shape, GPU 루프, CUDA graph) | FP16 | 17.3 | 163.1 | 724.2 | 3.4x / 3.7x | ✅ |
| 현행 | CTranslate2 4.8.3 | FP16 | 57.3 | 197.0 | 802.1 | 2.9x / 3.4x | ✅ |
| 현행 | CTranslate2 4.8.3 | INT8 + FP16 | 60.3 | 261.6 | 1126.3 | 2.1x / 2.4x | ✅ |
| 현행 | **TensorRT-LLM 1.2.1** (C++ runtime) | FP16 | - | **113.1** | **548.4** | **5.0x / 4.9x** | ✅ |
| 현행 | TensorRT-LLM 1.2.1 (C++ runtime) | INT8 weight-only | - | 124.6 | 579.9 | 4.5x / 4.7x | ✅ |

- "NNDF 흐름 자체 구현"은 NNDF 노트북의 흐름(HF → ONNX → TRT 엔진 → greedy)과 환경(TensorRT 8.6, torch 2.0.1)을 따라 새로 작성한 [Whisper/trt_kv.py](../Whisper/trt_kv.py)다. 원래 NNDF의 KV cache 경로는 BART/T5 코드가 섞여 동작하지 않았다.
- TensorRT-LLM의 인코더는 C++ runtime이 디코더와 한 번에 실행해서 따로 잴 수 없다. 프로파일에서 잰 인코더 GPU 시간은 23ms다.
- 참고로 GPU를 다른 학습과 같이 쓰던 10-21 측정에서는 모든 백엔드가 느렸다. 특히 커널이 많은 TRT 계열이 더 느렸다(TRT-LLM 996ms, NNDF non-KV 1879ms). 순위는 같았다.

## 원인 분석: TensorRT-LLM FP16과 왜 차이 나는가

[profiling/README.md](profiling/README.md)에 nsys, torch profiler, cProfile 근거가 있다. 한국어 E2E를 **인코더 / 디코더 GPU / 디코더 대기(GPU가 CPU를 기다린 시간)**로 나눠 TensorRT-LLM FP16과 비교하면 다음과 같다.

| 백엔드 | E2E (ms) | ΔE2E | Δ인코더 | Δ디코더 GPU | Δ디코더 대기 | 주원인 (근거 커널, 함수) |
|---|---|---|---|---|---|---|
| **TensorRT-LLM FP16** | **548** | 0 | 0 | 0 | 0 | M=1 GEMV `cudaCoreGemm` 1.47ms/토큰, fused `masked_multihead_attention` 1.07ms/토큰 |
| TensorRT-LLM INT8 | 580 | +32 | +3 | +27 | +2 | weight-only GEMV가 batch 1에서 FP16과 비슷하거나 약간 느림 |
| 자체 구현 v2 | 724 | +176 | −6 | +232 | −59 | 디코더 GEMM이 **FP32** cuBLAS GEMV(가중치 3.2GB/토큰, 2.5ms/토큰), fused 조각이 2배로 쪼개짐, 매 step cross K 전치. CPU 대기는 TRT-LLM보다 적음 |
| CTranslate2 FP16 | 802 | +254 | +40 | +119 | +95 | unfused elementwise, LayerNorm, softmax(1.8ms/토큰), 커널 897개/토큰 launch, op별 `cudaMallocAsync` |
| 자체 구현 v1 | 1044 | +495 | −6 | +369 | +133 | 32x32 tensor core GEMM(2.1ms/토큰), `mha_v2`(2.2ms/토큰), 동적 shape enqueue(step당 4.0ms), `.item()` 동기화 |
| NNDF non-KV | 1092 | +544 | +19 | +638 | −80 | KV cache 없이 매 step self-attention과 cross K/V projection을 다시 계산(GEMM 5.2ms/토큰) |
| CTranslate2 INT8 | 1126 | +578 | +49 | +462 | +66 | INT8 GEMM `gemmSN_kernel_int32` 3.3ms/토큰, quantize와 dequantize 1.4ms/토큰 |
| HF transformers | 2712 | +2164 | +26 | +266 | +1873 | Python op dispatch(`linear` 3.3만 회 839ms, 커널 836개/토큰) |
| `torch_tensorrt` 인코더 | 2911 | +2363 | +203 | +215 | +1945 | HF와 같은 디코더에 더해, 인코더 행렬곱이 FP32 SGEMM(`aten.mm/bmm` 미지원) |

### TensorRT-LLM vs 자체 구현: 토큰당 무엇이 다른가

| 토큰당 | TRT-LLM FP16 | TRT-LLM INT8 | 자체 구현 v2 | 자체 구현 v1 |
|---|---|---|---|---|
| 실제 디코더 시간 | **4.17ms** | 4.40ms | 5.55ms | 8.15ms |
| GPU 일 / CPU 대기 | 3.54 / 0.63ms | 3.75 / 0.65ms | 5.38 / **0.17ms** | 6.47 / 1.68ms |
| 행렬곱 | **1.59ms** (`cudaCoreGemm`) | 1.78ms (INT8 weight-only) | 2.54ms (FP32 `gemvx`) | 2.66ms (tensor core 32x32) |
| 읽는 가중치 → 실효 대역폭 | 1.60GB → 1.01TB/s | 0.87GB → 0.49TB/s | **3.20GB** → 1.26TB/s | 1.60GB → **0.60TB/s** |
| attention | **1.11ms** (fused masked MHA) | 1.22ms | 1.10ms (`mha_v1`) | **2.18ms** (`mha_v2`) |
| fused LayerNorm 등 / 커널 수 | **0.68ms / 418개** | 0.60ms / 397개 | 1.23ms / 672개 | 1.61ms / 746개 |
| 인코더 | 23ms | 26ms | **17ms** | **17ms** |

- **v2 vs TRT-LLM (+1.38ms/토큰)**: 행렬곱 +0.95(FP32라서 가중치를 2배 읽음), fused 조각 +0.55, cross K 전치 등 +0.48, CPU 대기 −0.46(v2가 더 적음)
- **v1 vs TRT-LLM (+3.98ms/토큰)**: attention +1.07, 행렬곱 +1.07(M=1에 비효율적인 타일), CPU 대기 +1.05, fused 조각 +0.93
- **INT8 vs FP16**: 가중치는 절반만 읽지만 weight-only 커널의 실효 대역폭이 절반(0.49TB/s)이라 행렬곱이 오히려 +0.19ms 늘었다.

상세 근거와 환산 방법은 [profiling/README.md](profiling/README.md#1-1-tensorrt-llm-vs-자체-구현-v1v2-토큰당-수치-비교)에 있다.

## 결론

- **가장 빠른 것은 TensorRT-LLM FP16이다.** 빈 GPU에서 HF 대비 약 5배이고 텍스트는 같다. 토큰 1개용 GEMV(1.0TB/s), 디코딩용 fused attention, 적은 커널 수(418개/토큰), C++ runtime 덕분이다.
- **일반 TensorRT도 의미가 있었다.** TensorRT 엔진을 쓴 경로는 모두 HF 대비 2.5~3.7배다. 같은 TRT 8.6에서 실행 방식만 바꾼 v2(정적 shape + CUDA graph)는 CTranslate2보다 빠르고, CPU 대기(0.17ms/토큰)와 인코더(17ms)는 TensorRT-LLM보다도 낫다. 남은 차이는 FP32로 도는 디코더 GEMM과 매 step cross K 전치이고, 둘 다 고칠 수 있다.
- **batch 1에서 INT8은 이득이 없었다**(CT2는 오히려 느리고, TRT-LLM은 FP16과 비슷). **Torch-TensorRT 1.4.0 `torch.compile`은 오히려 느렸다**(`mm/bmm` 미지원, 디코더 재컴파일).
