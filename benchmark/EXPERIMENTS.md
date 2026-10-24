# 실험 로그

whisper-large-v2 TensorRT 추론을 고치고 백엔드를 비교하면서 한 실험을 시간순으로 적었다(2026-10-18 ~ 10-24). 각 항목에 목적, 조건, 결과, 결론을 적고, 수치를 다시 낼 수 있는 스크립트를 붙였다. 최종 결과와 결론은 [README.md](README.md)에 있다.

장비는 A100-SXM4-80GB 8장, 드라이버 580(CUDA 13.0)이다. GPU를 다른 학습과 같이 쓴 측정은 **(GPU 공유)**로 표시했다. 최종 수치는 모두 빈 GPU에서 다시 쟀다.

## 한눈에 보기

| # | 날짜 | 실험 | 핵심 결과 | 스크립트 / 근거 |
|---|---|---|---|---|
| E1 | 10-18 | 원본 노트북 에러 수정 | 버그 3개를 고친 뒤 TRT FP16 243ms vs HF 560ms (2.3배), 텍스트 일치 | `notebooks/whisper2.ipynb` (git `74f9c1c`) |
| E2 | 10-19 | LayerNorm FP16 overflow 측정 | 인코더 마지막 LayerNorm의 (x−mean)² 최대 229,774 > 65504 → opset 17로 해결 | `experiments/layernorm_overflow.py` |
| E3 | 10-19 | KV cache TRT 자체 구현 v1 | HF와 텍스트 일치. 인코더 FP16 42 → 20ms | `Whisper/trt_kv.py`, `experiments/validate_trt_kv.py` |
| E4 | 10-21 | Torch-TensorRT 1.4.0 미지원 op | `aten.mm/bmm`, softmax, LayerNorm, GELU 등 미지원 → 행렬곱이 PyTorch에 남음 | `experiments/torchtrt_unsupported_ops.py` |
| E5 | 10-21 | Torch-TensorRT 디코더 컴파일 | tiny에서도 첫 실행 1165초, 두 번째 79초(재컴파일 171회) → 사용 불가 | `experiments/torchtrt_decoder_recompile.py` |
| E6 | 10-21 | CTranslate2, TensorRT-LLM 환경 구성 | CUDA 13, LD_LIBRARY_PATH, user site 문제를 해결 | `ct2/`, `trtllm/` README |
| E7 | 10-21 | 1차 전체 벤치 **(GPU 공유)** | 순위는 최종과 같지만 절대값이 1.1~1.8배 느림 → 폐기하고 E10으로 대체 | - |
| E8 | 10-23 | 1차 프로파일링 **(GPU 공유)** | 병목 유형을 파악. HF logits processor 원인은 E11에서 정정 | - |
| E9 | 10-23 | 자체 구현 v2 (정적 shape + CUDA graph) | `0×−inf=NaN` 버그 수정 후 텍스트 일치. v1 대비 1.44배 | `Whisper/trt_kv.py`, `experiments/validate_trt_kv.py` |
| E10 | 10-23 | **최종 벤치 (빈 GPU)** | TRT-LLM FP16 548ms(4.9배), v2 724ms(3.7배), CT2 802ms(3.4배) | `run_all.sh`, `results/` |
| E11 | 10-24 | **최종 프로파일링 (빈 GPU)** + 차이 분해 | TRT-LLM 대비 차이를 인코더 / 디코더 GPU / CPU 대기로 분해 | `profiling/` |
| E12 | 10-24 | v2 step 실측 (프로파일러 없음) | 5.38ms/step (nsys 노드 기록은 6.30, 17% 부풀림) | `experiments/v2_step_time.py` |
| E13 | 10-24 | TRT-LLM vs 자체 구현 토큰당 비교 | 가중치 바이트와 실효 대역폭: TRT-LLM 1.01TB/s, v2는 FP32라 3.2GB/토큰 | `experiments/head_to_head.py` |

---

## E1. 원본 노트북 에러 수정 (10-18)

- **목적**: `notebooks/whisper.ipynb` 마지막 부분의 에러와 결과 불일치를 고친다.
- **조건**: 새 conda env `trt`(python 3.8)에 torch 2.0.1, transformers 4.24.0, tensorrt 8.6.1, polygraphy 0.48.1, onnx 1.14.1을 설치했다. NGC 인덱스는 접속이 안 돼서 PyPI만 썼다. PyAV는 소스 빌드가 실패해서 바이너리 휠 12.3.0으로 고정했다. Docker는 필요 없었다.
- **원인 3가지**
  1. `WhisperTRTEncoder.forward`가 CPU 입력을 `.int()`로 캐스팅해서 mel 특징이 잘렸다. 원본의 "Encoder output이 다름!!"이 이것이다. `Whisper/trt.py`를 `.float()`로 고쳤다.
  2. TRT 디코더에 **한국어 뉴스**의 인코더 출력을 넣고 librispeech 정답과 비교했다(그래서 `False`, `" you"`).
  3. `cuda:1`과 `cuda`가 섞여 있었고, NNDF가 `archived_models/`로 이동해 import가 깨져 있었다.
- **결과 (빈 GPU)**: TRT FP16 E2E EN 243ms vs HF FP16 cache 560ms(2.3배). FP32와 FP16 모두 영어·한국어 텍스트가 HF와 같았다. 다만 FP16 인코더 시간이 FP32와 같았다(42ms). 이건 E2에서 이어서 본다.

## E2. LayerNorm FP16 overflow 측정 (10-19)

- **목적**: NNDF 코드가 large 모델 인코더를 일부러 FP32로 빌드하는 이유("FP16 overflow")를 수치로 확인한다.
- **방법**: large-v2 인코더의 모든 LayerNorm 입력에 hook을 걸어 max|x|와 (x−mean)²를 잰다(librispeech 샘플, CPU).
- **결과**

  | LayerNorm | max\|x\| | max (x−mean)² |
  |---|---|---|
  | `layer_norm` (마지막) | 480.8 | **229,774** |
  | `layers.31.*` | 331.9 | **109,374** |
  | `layers.30.*` | 163.3 | 26,377 |

  FP16 최댓값 65504를 넘는 LayerNorm이 3개(합 기준으로는 5개)다.
- **결론**: opset 12는 LayerNorm을 `Pow`/`ReduceMean`으로 풀어서 내보낸다. 그래서 TRT FP16에서 제곱이 inf가 된다. PyTorch는 LayerNorm 내부를 FP32로 계산해서 문제가 없다. **opset 17**로 export하면 `LayerNormalization` → TRT `INormalizationLayer`(내부 FP32)가 되어 해결된다. 실제로 인코더 FP16이 42 → 20ms로 줄었다(GPU 공유 측정).
- **재현**: `CUDA_VISIBLE_DEVICES= python benchmark/experiments/layernorm_overflow.py`

## E3. KV cache TensorRT 자체 구현 v1 (10-19)

- **목적**: non-KV 엔진은 매 step 전체를 다시 계산하므로 KV cache를 붙인다.
- **문제**: NNDF의 KV 경로는 BART/T5 코드가 섞여 동작하지 않았다. export는 `kv/`·`non-kv/` 엔진을 만드는데, runner는 export하지도 않는 `cross_attention_kv_generator`를 찾는다.
- **구현**(`Whisper/trt_kv.py` v1)
  - 인코더 엔진이 디코더 32층의 cross K/V까지 한 번에 출력한다.
  - 디코더 엔진은 self K/V cache를 동적 길이로 입출력한다.
  - 둘 다 opset 17이다.
  - 첫 step의 빈 KV(길이 0)는 `data_ptr()==0`이라 TRT가 거부해서, 더미 주소를 넘기도록 수정했다.
- **검증**: torch 모듈 logits vs HF 오차 1e-5. tiny와 large-v2에서 FP32/FP16 모두 텍스트가 HF와 같았다.
- **당시 관찰 (GPU 공유)**: 디코더 1 step이 KV 8.6ms vs non-KV 7.6ms(최소값)로 KV가 빠르지 않았다. 커널 744개/step이었다. 원인은 E11에서 확정했다(동적 shape enqueue, `.item()` 동기화, M=1에 비효율적인 커널).

## E4. Torch-TensorRT 1.4.0 미지원 op (10-21)

- **목적**: 2023년 당시 `torch.compile(..., backend="tensorrt")`를 썼다면 어땠을지 확인한다.
- **조건**: `trt` env에 torch-tensorrt 1.4.0을 추가했다(torch 2.0.1, TRT 8.6.1 대응 버전, 다른 패키지 변경 없음).
- **결과**
  - **백엔드 이름**: 등록된 이름은 `torch_tensorrt`와 `aot_torch_tensorrt_aten`이다. `"tensorrt"`는 `KeyError`가 난다.
  - **dtype**: `.half()` 모델은 `Dtype mismatch` assert가 난다. FP32 모델에 `enabled_precisions={float, half}`를 줘야 한다.
  - **미지원 aten op**(`partition(verbose=True)`로 확인)
    - `mm`, `bmm`
    - `amax`, `exp`, `sum`, `reciprocal`(softmax)
    - `var_mean`, `sqrt`(LayerNorm)
    - `erf`(GELU)
    - `permute`, `clone`, `_unsafe_view`, `getitem`
    - 디코더 앞부분: `embedding`, `arange`, `where`, `lt`, `slice`, `full`
  - **변환 실패**: 인코더 conv1d에서 `linear convolution_1 has bias of type ITensor` 에러가 나고 PyTorch로 fallback된다.
  - **large-v2**: 그래프 분할 단계(`torch.fx dfs_find_cycle`)에서 `RecursionError`가 나서 인코더 전체가 PyTorch FP32로 돈다. `sys.setrecursionlimit(100000)`으로 우회하면 TRT 엔진 448개(elementwise 조각)가 만들어지지만, 행렬곱은 FP32 SGEMM으로 남는다. 컴파일에 약 6분 걸린다.
- **결론**: 이 버전에서 Whisper는 TRT 가속이 사실상 안 된다. 빈 GPU에서도 HF보다 느렸다(2911 vs 2712ms).
- **재현**: `CUDA_VISIBLE_DEVICES=0 python benchmark/experiments/torchtrt_unsupported_ops.py`

## E5. Torch-TensorRT 디코더 컴파일 (10-21)

- **목적**: 디코더까지 `torch.compile`하면 어떻게 되는지 본다.
- **결과 (whisper-tiny)**

  | 방식 | 첫 실행 | 두 번째 실행 | 그래프 수 |
  |---|---|---|---|
  | eager | 2.2초 | 226ms | - |
  | 인코더만 | 57.7초 | 168ms | 1 |
  | 인코더+디코더 | **1,165초** | **78,794ms** | 171 |

- **결론**: 1.4.0은 정적 shape만 지원한다. 그래서 KV cache 길이가 바뀌는 디코더는 step마다 재컴파일된다. large-v2에서는 측정하지 않았고, 벤치에는 인코더만 컴파일한 경우를 넣었다.
- **재현**: `CUDA_VISIBLE_DEVICES=0 python benchmark/experiments/torchtrt_decoder_recompile.py`

## E6. CTranslate2, TensorRT-LLM 환경 구성 (10-21)

| env | 문제 | 해결 |
|---|---|---|
| `trtllm` | TRT-LLM 1.2.1은 CUDA 13 빌드인데 PyPI torch 2.9.1은 cu128 → `libcublasLt.so.13` 에러 | `torch==2.9.1+cu130` |
| `trtllm` | `libnvrtc.so.13`을 못 찾음(pip `nvidia/cu13/lib`) | `activate.d`에서 `LD_LIBRARY_PATH` 추가 |
| `trtllm` | pyarrow 26이 numpy 2를 요구 | `pyarrow<18` |
| `ct2` | 셸 기본 `LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64` → `__nvJitLink` 심볼 에러 | `activate.d`에서 해제 |
| `ct2`, `trtllm` | pip이 `~/.local`(user site) 패키지를 "설치됨"으로 보고 env에 설치하지 않음 | `PYTHONNOUSERSITE=1` |
| `ct2` | ctranslate2 4.8 converter가 `from_pretrained(dtype=...)`를 씀 | transformers 4.57.3 |

TensorRT-LLM 빌드 설정은 batch 1, beam 1, 디코더 `max_seq_len 448`이다(예제 기본값 114로는 126토큰 출력을 처리하지 못한다).

## E7. 1차 전체 벤치 (10-21, GPU 공유) — 폐기

- 8개 GPU 모두 다른 학습이 48~99%를 쓰는 상태에서 GPU 3으로 측정했다.
- 결과(KO): HF 3048, NNDF non-KV 1879, v1 1895, CT2 1229, TRT-LLM 996ms. 순위는 최종(E10)과 같다. 하지만 절대값이 1.1~1.8배 느리고, 커널이 많은 TRT 계열일수록 더 느려졌다.
- 시분할 때문에 같은 커널이 9µs에서 2.5ms까지 튀었다. 그래서 E10(빈 GPU)으로 대체했다.

## E8. 1차 프로파일링 (10-23, GPU 공유) — 일부 정정

- 병목 유형(Python dispatch, 커널 launch, 재계산, M=1 커널)은 이때 파악했고 E11에서도 유지됐다.
- **정정**: "HF `SuppressTokensLogitsProcessor`가 254ms(E2E의 8%)를 쓴다"고 봤다. 하지만 빈 GPU(E11)에서는 13ms였다. GPU 공유로 밀린 GPU 작업을 이 동기화 지점에서 기다린 시간이었다(`cudaStreamSynchronize` 254 → 3.7ms).
- **도구 문제**: nsys 2023.2는 CUDA 13 앱(TRT-LLM)의 GPU trace를 못 잡았다. 그래서 torch profiler(Kineto/CUPTI cu13)로 대체했다.

## E9. 자체 구현 v2: 정적 shape + GPU 루프 + CUDA graph (10-23)

- **목적**: E3/E8에서 확인한 v1의 CPU 쪽 병목(동적 shape enqueue, 매 step 동기화)을 같은 TRT 8.6에서 없앤다.
- **구현**(`WhisperTRTKVGraph`)
  - self K/V를 고정 길이 448 버퍼로 두고 `pos` 마스크를 쓴다. 엔진은 새 토큰의 K/V만 출력한다.
  - suppress, 강제 프롬프트, argmax, EOT 판정을 모두 GPU 텐서 연산으로 한다.
  - step 하나를 CUDA graph로 캡처해 replay하고, EOT는 2 step 늦게 비동기로 확인한다.
- **버그**: suppress bias를 `-inf`로 두면 첫 토큰이 아닌 step에서 `0 × -inf = NaN`이 된다. torch `argmax`는 NaN을 최댓값으로 골라서 공백(220)만 반복 출력됐다. 유한한 −1e9로 바꿔 해결했다.
- **검증**: tiny에서 FP32/FP16 × graph 유무 4가지, large-v2 FP16 모두 텍스트가 HF와 같았다.
- **재현**: `V=openai/whisper-tiny CUDA_VISIBLE_DEVICES=0 python benchmark/experiments/validate_trt_kv.py`

## E10. 최종 벤치 (10-23, 빈 GPU)

- gloss_sim GPU 큐(`pose/gloss_sim/scripts/gpu_queue.py`, 메모리 2GB 미만 + 사용률 10% 미만일 때만 시작)의 `jobs.json` 맨 앞에 넣어, 빈 GPU 2를 받아 연달아 측정했다(20:54~20:59). Torch-TensorRT만 10-24 빈 GPU 0에서 측정했다.
- 결과는 [README.md](README.md#결과-빈-gpu-a100-sxm4-80gb)에 있다. TRT-LLM FP16 548ms(HF 대비 4.9배), v2 724ms(3.7배), CT2 FP16 802ms(3.4배), v1 1044ms, NNDF non-KV 1092ms, HF 2712ms이고, 모두 텍스트가 HF와 같았다.

## E11. 최종 프로파일링 + 차이 분해 (10-24, 빈 GPU 0)

- 9개 백엔드를 순서대로 nsys(TRT-LLM은 torch profiler)와 cProfile로 측정했다.
- **도구 문제와 해결**
  - `nsys stats`가 같은 이름의 CSV를 덮어쓰지 않아서 이전 실행 값이 남아 있었다 → `--force-overwrite true`
  - CUDA graph(v2)는 기본 설정에서 커널이 보이지 않는다 → `--cuda-graph-trace=node`
  - TRT-LLM INT8 GEMV(`weight_only::kernel`)가 GEMM으로 분류되지 않았다 → 분류 패턴 수정
- **차이 분해**
  - 인코더와 디코더는 "디코더 첫 step에만 나오는 커널"의 첫 등장 위치로 나눴다.
  - 경계 앞 GPU 시간이 인코더만 따로 잰 시간과 맞는지로 검증했다(HF 48.6 vs 47.5ms, v1 16.8 vs 17.2ms).
  - 예외 세 가지: NNDF non-KV는 따로 잰 인코더 시간을 썼다. TRT-LLM INT8은 `paged_kv` attention을 경계로 썼다. v2는 경계 앞에 복사가 섞인 것을 명시했다.
- 결과는 [profiling/README.md](profiling/README.md)에 있다.

## E12. v2 step 실측 (10-24)

- nsys 노드 기록 아래 v2 디코더 GPU는 6.30ms/step이었다. 그런데 실제 디코더 시간이 5.55ms/토큰이라 맞지 않았다(커널 합 7.33 > busy 6.30, graph 내부 커널이 겹침).
- CUDA event로 graph replay 126회를 프로파일러 없이 재니 **5.38ms/step**이었고, 인코더는 17.1ms였다.
- 그래서 v2의 커널 종류별 시간은 nsys 비율 × 5.38ms로 환산했다.
- **재현**: `CUDA_VISIBLE_DEVICES=0 python benchmark/experiments/v2_step_time.py`

## E13. TensorRT-LLM vs 자체 구현 토큰당 비교 (10-24)

- 디코더 행렬곱 가중치는 층당 22.9M개(self q/k/v/o + cross q/o + fc1/fc2) × 32층 = 734M개에, proj_out 66M개를 더해 약 800M개다(cross K/V projection은 인코더 쪽에서 1회).
- 토큰당 읽는 바이트를 GEMM 시간으로 나눠 실효 대역폭을 냈다.

  | | TRT-LLM FP16 | TRT-LLM INT8 | v2 | v1 |
  |---|---|---|---|---|
  | GEMM ms/토큰 | 1.59 | 1.78 | 2.54 (환산) | 2.66 |
  | 가중치 GB/토큰 | 1.60 | 0.87 | 3.20 (FP32) | 1.60 |
  | 실효 대역폭 | 1.01TB/s | 0.49TB/s | 1.26TB/s | 0.60TB/s |

- **결론**
  - v2는 커널 효율은 충분하지만 FP32라서 2배를 읽는다.
  - v1은 M=1에 tensor core 타일을 써서 대역폭이 낮다.
  - TRT-LLM INT8은 역양자화 비용 때문에 읽는 양이 절반이어도 빨라지지 않는다.
- **재현**: `python benchmark/experiments/head_to_head.py`(프로파일 CSV 필요)

## 남은 일

- v2의 MatMul을 FP16으로 고정하고(`OBEY_PRECISION_CONSTRAINTS`), 인코더가 cross K를 미리 전치해서 출력하게 한다. 그러면 TRT-LLM과의 차이(+1.38ms/토큰)의 약 70%가 줄어들 것으로 예상하지만, 아직 측정하지 않았다.
