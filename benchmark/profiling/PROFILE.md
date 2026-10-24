# 프로파일링 결과 (한국어 샘플 E2E 1회, 126토큰)

생성: `benchmark/profiling/analyze.py`. 원본: `benchmark/profiling/<tag>/` (`nsys.nsys-rep`는 Nsight Systems GUI로 열 수 있음, `cprofile.txt`).

- **nsys wall**: nsys 아래에서 잰 E2E 시간. 커널 launch마다 기록 비용이 붙어 실제보다 길다 (launch가 많을수록 더 부풀려짐).
- **GPU busy**: 이 구간에서 GPU 커널/복사가 실제로 돈 시간 (구간 합집합). **유휴%** = GPU가 CPU를 기다린 비율.
- **실제 E2E**: 프로파일러 없이 잰 KO E2E p50 (`results/`, 다른 날/다른 GPU 부하에서 측정). **실제 유휴 % ≈ 1 − GPU busy / 실제 E2E** (GPU 부하 차이로 대략적인 값).
- **커널/토큰**: 토큰 하나 만드는 데 띄운 GPU 커널 수.

| 백엔드 | 실제 E2E (ms) | GPU busy (ms) | 실제 유휴 % | nsys wall (ms) | 커널 수 | 커널/토큰 | 가장 큰 GPU 시간 | cProfile 상위 Python 함수 (자체 시간) |
|---|---|---|---|---|---|---|---|---|
| hf-fp16 | 2712 | 760 | 72% | 4428 | 106,185 | 843 | GEMM/GEMV 486ms (64%) | `{built-in method torch._C._nn.linear}` 839ms<br>`transformers/models/whisper/modeling_whisper.py:143(fo…` 306ms<br>`transformers/models/whisper/modeling_whisper.py:354(fo…` 252ms |
| torchtrt-fp16 | 2911 | 886 | 70% | 4583 | 106,810 | 848 | GEMM/GEMV 605ms (68%) | `{built-in method torch._C._nn.linear}` 880ms<br>`transformers/models/whisper/modeling_whisper.py:143(fo…` 330ms<br>`{built-in method torch.bmm}` 261ms |
| nndf-nonkv-fp16 | 1092 | 1126 | 0% | 1520 | 56,173 | 446 | GEMM/GEMV 685ms (61%) | `Whisper/trt.py:612(forward)` 1286ms<br>`transformers/generation_logits_process.py:732(__call__)` 77ms<br>`Whisper/trt.py:208(forward)` 47ms |
| nndf-port-kv-fp16 | 1044 | 831 | 20% | 1408 | 94,452 | 750 | GEMM/GEMV 346ms (41%) | `Whisper/trt_kv.py:316(generate)` 734ms<br>`Whisper/trt_kv.py:281(__call__)` 504ms<br>`{method 'item' of 'torch._C._TensorBase' objects}` 70ms |
| nndf-port-kv-graph-fp16 | 724 | 819 | 0% | 865 | 85,095 | 675 | GEMM/GEMV 452ms (48%) | `{function Event.synchronize at 0x7fcb4a0bb5e0}` 843ms<br>`{method 'copy_' of 'torch._C._TensorBase' objects}` 16ms<br>`{function Stream.synchronize at 0x7fcb4a0b7ee0}` 16ms |
| ct2-float16 | 802 | 627 | 22% | 1929 | 114,773 | 911 | GEMM/GEMV 367ms (58%) | `benchmark/ct2/bench.py:51(gen)` 989ms<br>`{built-in method torch._C._cuda_synchronize}` 1ms<br>`benchmark/common.py:40(_sync)` 0ms |
| ct2-int8 | 1126 | 980 | 13% | 2631 | 152,729 | 1212 | GEMM/GEMV 637ms (65%) | `benchmark/ct2/bench.py:51(gen)` 1461ms<br>`{built-in method torch._C._cuda_synchronize}` 1ms<br>`benchmark/common.py:40(_sync)` 0ms |
| trtllm-fp16 | 548 | 469 | 15% | 899 (torch profiler) | 52,983 | 420 | GEMM/GEMV 215ms (46%) | `tensorrt_llm/runtime/model_runner_cpp.py:928(_initiali…` 589ms<br>`{built-in method torch.tensor}` 1ms<br>`tensorrt_llm/runtime/model_runner_cpp.py:536(generate)` 0ms |
| trtllm-int8 | 580 | 499 | 14% | 955 (torch profiler) | 50,420 | 400 | GEMM/GEMV 237ms (48%) | `tensorrt_llm/runtime/model_runner_cpp.py:928(_initiali…` 885ms<br>`{built-in method torch.tensor}` 1ms<br>`tensorrt_llm/runtime/model_runner_cpp.py:536(generate)` 0ms |

## TensorRT-LLM FP16 대비 차이 분해 (한국어 30초, 126토큰)

E2E = 인코더 + 디코더(GPU가 일한 시간) + 디코더(GPU가 CPU를 기다린 시간). Δ는 TensorRT-LLM FP16과의 차이(ms). 디코더 GPU 시간은 프로파일러 아래에서 잰 값이라 실제 E2E보다 약간 클 수 있다(이때 대기 = 0).

| 백엔드 | 실제 E2E | ΔE2E | 인코더 (Δ) | 디코더 GPU (Δ) | 디코더 대기 (Δ) | 디코더 GPU ms/토큰 | 커널/토큰 |
|---|---|---|---|---|---|---|---|
| hf-fp16 | 2712 | +2164 | 49 (+26) | 711 (+266) | 1953 (+1873) | 5.65 | 836 |
| torchtrt-fp16 | 2911 | +2363 | 226 (+203) | 660 (+215) | 2025 (+1945) | 5.24 | 836 |
| nndf-nonkv-fp16 | 1092 | +544 | 42 (+19) | 1084 (+638) | 0 (-80) | 8.61 | 444 |
| nndf-port-kv-fp16 | 1044 | +495 | 17 (-6) | 815 (+369) | 212 (+133) | 6.47 | 746 |
| nndf-port-kv-graph-fp16 | 724 | +176 | 25 (+2) | 794 (+348) | 0 (-80) | 6.30 | 672 |
| ct2-float16 | 802 | +254 | 63 (+40) | 565 (+119) | 175 (+95) | 4.48 | 897 |
| ct2-int8 | 1126 | +578 | 72 (+49) | 908 (+462) | 146 (+66) | 7.21 | 1206 |
| trtllm-fp16 | 548 | +0 | 23 (+0) | 446 (+0) | 80 (+0) | 3.54 | 418 |
| trtllm-int8 | 580 | +32 | 26 (+3) | 473 (+27) | 81 (+2) | 3.75 | 397 |

디코더 GPU 시간을 토큰당(ms)으로, 커널 종류별로 나눈 값:

| 백엔드 | GEMM/GEMV | attention | TRT myelin (fused) | norm | softmax | elementwise/copy/etc | decode/sampling | memcpy/memset |
|---|---|---|---|---|---|---|---|---|
| hf-fp16 | 3.62 | 0.00 | 0.00 | 0.67 | 0.39 | 0.94 | 0.03 | 0.01 |
| torchtrt-fp16 | 3.39 | 0.00 | 0.00 | 0.61 | 0.35 | 0.86 | 0.03 | 0.01 |
| nndf-nonkv-fp16 | 5.19 | 2.53 | 0.78 | 0.00 | 0.00 | 0.05 | 0.02 | 0.02 |
| nndf-port-kv-fp16 | 2.66 | 2.18 | 1.61 | 0.00 | 0.00 | 0.01 | 0.02 | 0.06 |
| nndf-port-kv-graph-fp16 | 3.46 | 1.50 | 1.67 | 0.00 | 0.00 | 0.67 | 0.02 | 0.01 |
| ct2-float16 | 2.69 | 0.00 | 0.00 | 0.41 | 0.30 | 1.06 | 0.01 | 0.01 |
| ct2-int8 | 4.81 | 0.00 | 0.00 | 0.47 | 0.30 | 1.60 | 0.01 | 0.01 |
| trtllm-fp16 | 1.59 | 1.11 | 0.68 | 0.00 | 0.04 | 0.01 | 0.07 | 0.05 |
| trtllm-int8 | 1.78 | 1.22 | 0.60 | 0.00 | 0.04 | 0.01 | 0.07 | 0.04 |

## hf-fp16

[nsys] wall 4428.0 ms · GPU busy 759.9 ms · 유휴 83% · 커널 106,185개 (843/토큰) · 커널 시간 합 759.9 ms

| GPU 시간 분류 | ms | % |
|---|---|---|
| GEMM/GEMV | 486.0 | 64% |
| elementwise/copy/etc | 126.8 | 17% |
| norm | 84.7 | 11% |
| softmax | 57.4 | 8% |
| decode/sampling | 3.5 | 0% |
| memcpy/memset | 1.3 | 0% |

| 상위 커널 | ms | 호출 수 |
|---|---|---|
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half,…` | 261.7 | 28,896 |
| `void at::native::<unnamed>::vectorized_layer_norm_kernel<c10::Half, float>(int, T2, const…` | 84.7 | 12,578 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x6_tn` | 61.2 | 4,128 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<c10::Half, unsigned int, (int)4, (int)128…` | 49.5 | 8,192 |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half,…` | 44.4 | 4,128 |
| `void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_64x1_tn_align2>(T…` | 43.4 | 5,376 |
| `void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::Half, float, c10::Half, at::…` | 42.7 | 4,160 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::H…` | 31.7 | 12,513 |

| CUDA API (CPU 쪽) | ms | 호출 수 |
|---|---|---|
| `cudaLaunchKernel` | 770.3 | 106,185 |
| `cudaMemcpyAsync` | 30.1 | 453 |
| `cudaStreamSynchronize` | 3.7 | 453 |
| `cudaMemsetAsync` | 1.9 | 320 |
| `cuCtxSynchronize` | 0.2 | 1 |
| `cudaEventRecord` | 0.0 | 2 |

cProfile wall 2949.9 ms (cProfile 오버헤드 포함)

| Python 함수 (tottime 상위) | 자체 ms | 호출 수 |
|---|---|---|
| `{built-in method torch._C._nn.linear}` | 839.0 | 33409 |
| `transformers/models/whisper/modeling_whisper.py:143(forward)` | 306.0 | 8288 |
| `transformers/models/whisper/modeling_whisper.py:354(forward)` | 252.0 | 4128 |
| `{built-in method torch.bmm}` | 246.0 | 16576 |
| `{built-in method torch.layer_norm}` | 205.0 | 12578 |
| `{method 'view' of 'torch._C._TensorBase' objects}` | 132.0 | 49953 |
| `torch/nn/modules/module.py:1494(_call_impl)` | 114.0 | 63243/130 |
| `{built-in method torch.cat}` | 107.0 | 8321 |
| `{method 'softmax' of 'torch._C._TensorBase' objects}` | 79.0 | 8288 |
| `{method 'transpose' of 'torch._C._TensorBase' objects}` | 72.0 | 33248 |
| `torch/nn/modules/module.py:1601(__getattr__)` | 69.0 | 151324 |
| `torch/nn/functional.py:1235(dropout)` | 57.0 | 25026 |

## torchtrt-fp16

[nsys] wall 4582.6 ms · GPU busy 886.2 ms · 유휴 81% · 커널 106,810개 (848/토큰) · 커널 시간 합 886.2 ms

| GPU 시간 분류 | ms | % |
|---|---|---|
| GEMM/GEMV | 604.6 | 68% |
| elementwise/copy/etc | 154.3 | 17% |
| norm | 77.4 | 9% |
| softmax | 44.6 | 5% |
| decode/sampling | 3.3 | 0% |
| memcpy/memset | 2.0 | 0% |

| 상위 커널 | ms | 호출 수 |
|---|---|---|
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half,…` | 244.6 | 28,896 |
| `ampere_sgemm_128x64_tn` | 84.1 | 160 |
| `void at::native::<unnamed>::vectorized_layer_norm_kernel<c10::Half, float>(int, T2, const…` | 77.4 | 12,513 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x6_tn` | 58.7 | 4,128 |
| `ampere_sgemm_32x128_tn` | 48.3 | 32 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<c10::Half, unsigned int, (int)4, (int)128…` | 45.5 | 8,192 |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half,…` | 41.5 | 4,128 |
| `void cutlass::Kernel<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_64x1_tn_align2>(T…` | 40.1 | 5,376 |

| CUDA API (CPU 쪽) | ms | 호출 수 |
|---|---|---|
| `cudaLaunchKernel` | 852.3 | 106,362 |
| `cudaStreamSynchronize` | 20.9 | 389 |
| `cudaMemsetAsync` | 13.1 | 192 |
| `cudaMemcpyAsync` | 7.0 | 453 |
| `cuLaunchKernel` | 6.7 | 448 |
| `cudaMemcpy2DAsync` | 0.6 | 64 |

cProfile wall 3150.1 ms (cProfile 오버헤드 포함)

| Python 함수 (tottime 상위) | 자체 ms | 호출 수 |
|---|---|---|
| `{built-in method torch._C._nn.linear}` | 880.0 | 33217 |
| `transformers/models/whisper/modeling_whisper.py:143(forward)` | 330.0 | 8256 |
| `{built-in method torch.bmm}` | 261.0 | 16512 |
| `transformers/models/whisper/modeling_whisper.py:354(forward)` | 258.0 | 4128 |
| `{built-in method torch.layer_norm}` | 215.0 | 12513 |
| `{method 'view' of 'torch._C._TensorBase' objects}` | 131.0 | 49729 |
| `torch/nn/modules/module.py:1494(_call_impl)` | 116.0 | 63114/130 |
| `{built-in method torch.cat}` | 107.0 | 8321 |
| `{method 'softmax' of 'torch._C._TensorBase' objects}` | 88.0 | 8256 |
| `{method 'transpose' of 'torch._C._TensorBase' objects}` | 72.0 | 33088 |
| `torch/nn/modules/module.py:1601(__getattr__)` | 69.0 | 150706 |
| `torch/nn/functional.py:1235(dropout)` | 58.0 | 24897 |

## nndf-nonkv-fp16

[nsys] wall 1519.9 ms · GPU busy 1125.8 ms · 유휴 26% · 커널 56,173개 (446/토큰) · 커널 시간 합 1125.8 ms

| GPU 시간 분류 | ms | % |
|---|---|---|
| GEMM/GEMV | 685.3 | 61% |
| attention | 319.2 | 28% |
| TRT myelin (fused) | 109.6 | 10% |
| elementwise/copy/etc | 6.6 | 1% |
| decode/sampling | 2.6 | 0% |
| memcpy/memset | 2.5 | 0% |

| 상위 커널 | ms | 호출 수 |
|---|---|---|
| `trt_ampere_h16816gemm_64x64_sliced1x2_ldg8_relu_stages_64x6_nn_v1` | 200.4 | 12,160 |
| `sm80_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage4_warpsize2x2x1_tensor16x8x…` | 170.2 | 384 |
| `sm80_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x64x64_stage3_warpsize2x2x1_tensor16x8x1…` | 79.9 | 4,096 |
| `trt_ampere_h16816gemm_128x64_ldg8_relu_stages_64x3_tn_v1` | 77.8 | 4,096 |
| `trt_ampere_h16816gemm_64x64_sliced1x2_ldg8_relu_stages_64x5_nn_v1` | 62.0 | 4,224 |
| `trt_ampere_h16816gemm_64x64_ldg8_stages_64x5_nn_v1` | 50.8 | 4,096 |
| `mha_v2_34` | 25.9 | 128 |
| `sm80_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize128x128x16_stage4_warpsize2x2x1_tensor16x8…` | 25.2 | 96 |

| CUDA API (CPU 쪽) | ms | 호출 수 |
|---|---|---|
| `cudaStreamSynchronize` | 845.6 | 515 |
| `cudaLaunchKernel` | 127.5 | 26,505 |
| `cuLaunchKernel` | 114.3 | 24,962 |
| `cuLaunchKernelEx` | 21.5 | 4,706 |
| `cudaMemcpyAsync` | 16.5 | 390 |
| `cudaMemcpy2DAsync` | 1.6 | 128 |

cProfile wall 1478.3 ms (cProfile 오버헤드 포함)

| Python 함수 (tottime 상위) | 자체 ms | 호출 수 |
|---|---|---|
| `Whisper/trt.py:612(forward)` | 1286.0 | 128 |
| `transformers/generation_logits_process.py:732(__call__)` | 77.0 | 128 |
| `Whisper/trt.py:208(forward)` | 47.0 | 1 |
| `transformers/generation_utils.py:2085(greedy_search)` | 31.0 | 1 |
| `/home/klcube/anaconda3/envs/trt/lib/python3.8/inspect.py:2124(_signature_from_function)` | 3.0 | 384 |
| `{built-in method torch.argmax}` | 2.0 | 128 |
| `{built-in method torch.cat}` | 2.0 | 128 |
| `/home/klcube/anaconda3/envs/trt/lib/python3.8/inspect.py:2218(_signature_from_callable)` | 2.0 | 768/384 |
| `/home/klcube/anaconda3/envs/trt/lib/python3.8/inspect.py:2772(__init__)` | 2.0 | 768 |
| `{built-in method torch.rsub}` | 2.0 | 128 |
| `{method 'long' of 'torch._C._TensorBase' objects}` | 2.0 | 128 |
| `{method 'max' of 'torch._C._TensorBase' objects}` | 2.0 | 128 |

## nndf-port-kv-fp16

[nsys] wall 1408.4 ms · GPU busy 831.4 ms · 유휴 41% · 커널 94,452개 (750/토큰) · 커널 시간 합 841.3 ms

| GPU 시간 분류 | ms | % |
|---|---|---|
| GEMM/GEMV | 345.8 | 41% |
| attention | 279.8 | 33% |
| TRT myelin (fused) | 203.9 | 24% |
| memcpy/memset | 8.0 | 1% |
| decode/sampling | 2.5 | 0% |
| elementwise/copy/etc | 1.3 | 0% |

| 상위 커널 | ms | 호출 수 |
|---|---|---|
| `sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16…` | 264.9 | 24,192 |
| `trt_ampere_h16816gemm_64x64_sliced1x2_ldg8_relu_stages_64x6_nn_v1` | 59.6 | 4,032 |
| `mha_v2_46` | 32.1 | 126 |
| `mha_v2_28` | 17.5 | 127 |
| `mha_v2_12` | 11.5 | 127 |
| `trt_ampere_h16816gemm_64x64_sliced1x2_ldg8_stages_64x5_nn_v1` | 10.5 | 126 |
| `mha_v2_14` | 8.9 | 127 |
| `__myl_bb0_316_DivCasErfCasAddMulMul_com` | 8.9 | 4,000 |

| CUDA API (CPU 쪽) | ms | 호출 수 |
|---|---|---|
| `cuLaunchKernel` | 336.3 | 65,304 |
| `cudaStreamSynchronize` | 308.8 | 254 |
| `cuLaunchKernelEx` | 118.0 | 24,355 |
| `cudaMemcpyAsync` | 38.5 | 254 |
| `cudaLaunchKernel` | 28.5 | 4,793 |
| `cudaMemsetAsync` | 20.2 | 4,032 |

cProfile wall 1338.5 ms (cProfile 오버헤드 포함)

| Python 함수 (tottime 상위) | 자체 ms | 호출 수 |
|---|---|---|
| `Whisper/trt_kv.py:316(generate)` | 734.0 | 1 |
| `Whisper/trt_kv.py:281(__call__)` | 504.0 | 127 |
| `{method 'item' of 'torch._C._TensorBase' objects}` | 70.0 | 126 |
| `{built-in method torch.tensor}` | 14.0 | 1 |
| `{built-in method torch.empty}` | 3.0 | 382 |
| `{method 'argmax' of 'torch._C._TensorBase' objects}` | 2.0 | 126 |
| `{method 'to' of 'torch._C._TensorBase' objects}` | 2.0 | 125 |
| `{method 'view' of 'torch._C._TensorBase' objects}` | 1.0 | 125 |
| `torch/_tensor.py:904(__len__)` | 1.0 | 127 |
| `torch/cuda/__init__.py:704(current_stream)` | 1.0 | 127 |
| `Whisper/trt_kv.py:278(_address)` | 0.0 | 1011 |
| `torch/cuda/_utils.py:7(_get_device_index)` | 0.0 | 128 |

## nndf-port-kv-graph-fp16

[nsys] wall 864.8 ms · GPU busy 818.8 ms · 유휴 5% · 커널 85,095개 (675/토큰) · 커널 시간 합 947.9 ms

| GPU 시간 분류 | ms | % |
|---|---|---|
| GEMM/GEMV | 452.2 | 48% |
| TRT myelin (fused) | 211.8 | 22% |
| attention | 195.3 | 21% |
| elementwise/copy/etc | 85.0 | 9% |
| decode/sampling | 2.3 | 0% |
| memcpy/memset | 1.3 | 0% |

| 상위 커널 | ms | 호출 수 |
|---|---|---|
| `sm50_xmma_cublas_gemvx_f32f32_f32_f32_tn_n_int32_unit_n_launch_param4x32x32_strided_unit_…` | 332.5 | 25,152 |
| `sm50_xmma_cublas_gemvx_f32f32_f32_f32_nn_n_int32_unit_n_launch_param64x8x8_strided_unit_s…` | 84.3 | 4,192 |
| `void CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE::VectorWrite2DTensorVie…` | 58.6 | 262 |
| `__myl_bb0_258_Mov` | 29.2 | 8,384 |
| `sm50_xmma_cublas_gemvx_f32f32_f32_f32_nn_n_int32_unit_n_launch_param64x2x1_strided_copyx_…` | 19.4 | 131 |
| `void genericReformat::copyVectorizedKernel<double, __half, float, (bool)1, (bool)0, (int)…` | 18.7 | 262 |
| `__myl_bb0_257_Res` | 16.4 | 4,192 |
| `mha_v1_16` | 14.9 | 131 |

| CUDA API (CPU 쪽) | ms | 호출 수 |
|---|---|---|
| `cudaEventSynchronize` | 697.0 | 129 |
| `cudaGraphLaunch_v10000` | 106.3 | 131 |
| `cudaStreamSynchronize` | 39.1 | 3 |
| `cudaMemcpyAsync` | 5.0 | 136 |
| `cudaLaunchKernel` | 2.3 | 275 |
| `cuLaunchKernel` | 0.9 | 162 |

cProfile wall 886.1 ms (cProfile 오버헤드 포함)

| Python 함수 (tottime 상위) | 자체 ms | 호출 수 |
|---|---|---|
| `{function Event.synchronize at 0x7fcb4a0bb5e0}` | 843.0 | 129 |
| `{method 'copy_' of 'torch._C._TensorBase' objects}` | 16.0 | 135 |
| `{function Stream.synchronize at 0x7fcb4a0b7ee0}` | 16.0 | 1 |
| `torch/cuda/graphs.py:81(replay)` | 3.0 | 131 |
| `Whisper/trt_kv.py:281(__call__)` | 3.0 | 1 |
| `Whisper/trt_kv.py:550(generate)` | 2.0 | 1 |
| `torch/cuda/__init__.py:704(current_stream)` | 0.0 | 133 |
| `torch/cuda/_utils.py:7(_get_device_index)` | 0.0 | 135 |
| `{function Event.record at 0x7fcb4a0bb3a0}` | 0.0 | 131 |
| `torch/cuda/streams.py:173(record)` | 0.0 | 131 |
| `torch/_utils.py:692(_get_device_index)` | 0.0 | 135 |
| `torch/cuda/streams.py:31(__new__)` | 0.0 | 133 |

## ct2-float16

[nsys] wall 1929.1 ms · GPU busy 627.4 ms · 유휴 67% · 커널 114,773개 (911/토큰) · 커널 시간 합 627.4 ms

| GPU 시간 분류 | ms | % |
|---|---|---|
| GEMM/GEMV | 367.0 | 58% |
| elementwise/copy/etc | 143.5 | 23% |
| softmax | 61.0 | 10% |
| norm | 53.1 | 8% |
| memcpy/memset | 1.7 | 0% |
| decode/sampling | 1.1 | 0% |

| 상위 커널 | ms | 호출 수 |
|---|---|---|
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half,…` | 142.3 | 16,000 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x6_tn_align8>(T…` | 91.1 | 8,000 |
| `void at::native::cunn_SoftMaxForward<__half, float, __half, unsigned int, int, at::native…` | 61.0 | 8,096 |
| `void at::native::LayerNormForwardCUDAKernel<__half, unsigned int>(T2, float, const T1 *, …` | 53.1 | 12,286 |
| `void cub::CUB_200700_530_600_610_700_750_800_860_NS::detail::for_each::static_kernel<cub:…` | 40.2 | 16,000 |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half,…` | 37.0 | 4,000 |
| `void cub::CUB_200700_530_600_610_700_750_800_860_NS::detail::for_each::static_kernel<cub:…` | 35.3 | 12,160 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_64x1_tn_align2>(…` | 34.9 | 5,248 |

| CUDA API (CPU 쪽) | ms | 호출 수 |
|---|---|---|
| `cudaLaunchKernel` | 577.5 | 101,300 |
| `cudaFreeAsync_v11020` | 88.0 | 54,160 |
| `cudaMallocAsync_v11020` | 86.6 | 54,160 |
| `cuLaunchKernel` | 77.0 | 13,473 |
| `cudaMemcpyAsync` | 41.1 | 755 |
| `cuKernelGetFunction` | 8.6 | 13,473 |

cProfile wall 990.1 ms (cProfile 오버헤드 포함)

| Python 함수 (tottime 상위) | 자체 ms | 호출 수 |
|---|---|---|
| `benchmark/ct2/bench.py:51(gen)` | 989.0 | 1 |
| `{built-in method torch._C._cuda_synchronize}` | 1.0 | 1 |
| `benchmark/common.py:40(_sync)` | 0.0 | 1 |
| `/home/klcube/anaconda3/envs/ct2/lib/python3.10/os.py:755(encode)` | 0.0 | 2 |
| `torch/cuda/__init__.py:107(_is_compiled)` | 0.0 | 2 |
| `{built-in method torch._C._cuda_getDevice}` | 0.0 | 1 |
| `torch/cuda/__init__.py:944(synchronize)` | 0.0 | 1 |
| `torch/cuda/_utils.py:9(_get_device_index)` | 0.0 | 1 |
| `torch/cuda/__init__.py:116(is_available)` | 0.0 | 2 |
| `/home/klcube/anaconda3/envs/ct2/lib/python3.10/os.py:675(__getitem__)` | 0.0 | 2 |
| `torch/cuda/__init__.py:112(_nvml_based_avail)` | 0.0 | 2 |
| `{built-in method torch._C._cuda_getDeviceCount}` | 0.0 | 2 |

## ct2-int8

[nsys] wall 2630.8 ms · GPU busy 980.3 ms · 유휴 63% · 커널 152,729개 (1212/토큰) · 커널 시간 합 980.3 ms

| GPU 시간 분류 | ms | % |
|---|---|---|
| GEMM/GEMV | 637.5 | 65% |
| elementwise/copy/etc | 209.3 | 21% |
| softmax | 70.9 | 7% |
| norm | 60.0 | 6% |
| memcpy/memset | 1.5 | 0% |
| decode/sampling | 1.1 | 0% |

| 상위 커널 | ms | 호출 수 |
|---|---|---|
| `void gemmSN_kernel_int32<(int)256, (int)32, (int)4, (int)8, (int)8, (int)1, (bool)1, (int…` | 407.9 | 24,318 |
| `void ctranslate2::ops::quantize_kernel<__half>(const T1 *, unsigned int, float *, signed …` | 105.5 | 24,670 |
| `void ctranslate2::ops::dequantize_gemm_output_kernel<thrust::THRUST_200700_530_600_610_70…` | 73.7 | 20,574 |
| `void at::native::cunn_SoftMaxForward<__half, float, __half, unsigned int, int, at::native…` | 70.9 | 8,160 |
| `void at::native::LayerNormForwardCUDAKernel<__half, unsigned int>(T2, float, const T1 *, …` | 60.0 | 12,383 |
| `void cub::CUB_200700_530_600_610_700_750_800_860_NS::detail::for_each::static_kernel<cub:…` | 43.6 | 16,128 |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __half, __half, __half,…` | 37.7 | 4,032 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_64x1_tn_align2>(…` | 35.6 | 5,280 |

| CUDA API (CPU 쪽) | ms | 호출 수 |
|---|---|---|
| `cudaLaunchKernel` | 857.3 | 147,096 |
| `cudaFreeAsync_v11020` | 186.2 | 128,850 |
| `cudaMallocAsync_v11020` | 186.0 | 128,850 |
| `cudaMemcpyAsync` | 118.1 | 762 |
| `cuLaunchKernel` | 35.7 | 5,633 |
| `cudaStreamSynchronize` | 8.2 | 1 |

cProfile wall 1462.4 ms (cProfile 오버헤드 포함)

| Python 함수 (tottime 상위) | 자체 ms | 호출 수 |
|---|---|---|
| `benchmark/ct2/bench.py:51(gen)` | 1461.0 | 1 |
| `{built-in method torch._C._cuda_synchronize}` | 1.0 | 1 |
| `benchmark/common.py:40(_sync)` | 0.0 | 1 |
| `/home/klcube/anaconda3/envs/ct2/lib/python3.10/os.py:675(__getitem__)` | 0.0 | 2 |
| `torch/cuda/__init__.py:944(synchronize)` | 0.0 | 1 |
| `torch/cuda/__init__.py:107(_is_compiled)` | 0.0 | 2 |
| `torch/cuda/_utils.py:9(_get_device_index)` | 0.0 | 1 |
| `{built-in method torch._C._cuda_getDevice}` | 0.0 | 1 |
| `{built-in method torch._C._cuda_getDeviceCount}` | 0.0 | 2 |
| `/home/klcube/anaconda3/envs/ct2/lib/python3.10/os.py:755(encode)` | 0.0 | 2 |
| `torch/cuda/__init__.py:439(__init__)` | 0.0 | 1 |
| `torch/cuda/__init__.py:112(_nvml_based_avail)` | 0.0 | 2 |

## trtllm-fp16

[torchprof] wall 899.2 ms · GPU busy 468.7 ms · 유휴 48% · 커널 52,983개 (420/토큰) · 커널 시간 합 468.7 ms

| GPU 시간 분류 | ms | % |
|---|---|---|
| GEMM/GEMV | 215.5 | 46% |
| attention | 146.0 | 31% |
| TRT myelin (fused) | 86.0 | 18% |
| decode/sampling | 9.2 | 2% |
| memcpy/memset | 5.8 | 1% |
| softmax | 5.2 | 1% |
| elementwise/copy/etc | 1.1 | 0% |

| 상위 커널 | ms | 호출 수 |
|---|---|---|
| `void tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemm<__half, __half, 1, 2, 128>(…` | 185.5 | 20,807 |
| `void tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel<unsigned short, …` | 78.6 | 3,468 |
| `__myl_AddAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x823596fab576cdfadeae72956e9676f0_…` | 58.1 | 10,295 |
| `void tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel<unsigned short, …` | 56.6 | 3,468 |
| `__myl_CastCastAddCast_0x8f7a531de317e4c42dc65f73c8aea1d8_com` | 15.7 | 6,936 |
| `void gemv2T_kernel_val<int, int, __half, __half, __half, float, 128, 16, 4, 4, false, fal…` | 9.5 | 109 |
| `__myl_CastCastAddMulMulMulAddMulTanhAddMulMulCast_0x98855d175352d7b1c475cad368ee3918_com` | 8.3 | 3,468 |
| `void tensorrt_llm::_v1::kernels::batchApplyPenalty<__half>(__half const* const*, __half*,…` | 7.0 | 109 |

| CUDA API (CPU 쪽) | ms | 호출 수 |
|---|---|---|
| `cudaEventSynchronize` | 262.0 | 109 |
| `cudaLaunchKernelExC` | 135.0 | 27,999 |
| `cuLaunchKernel` | 99.1 | 21,319 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | 15.0 | 17,340 |
| `cudaMemcpyAsync` | 11.2 | 2,198 |
| `cudaStreamIsCapturing` | 9.4 | 24,643 |

cProfile wall 591.2 ms (cProfile 오버헤드 포함)

| Python 함수 (tottime 상위) | 자체 ms | 호출 수 |
|---|---|---|
| `tensorrt_llm/runtime/model_runner_cpp.py:928(_initialize_and_fill_output)` | 589.0 | 1 |
| `{built-in method torch.tensor}` | 1.0 | 3 |
| `tensorrt_llm/runtime/model_runner_cpp.py:536(generate)` | 0.0 | 1 |
| `{built-in method torch.floor_divide}` | 0.0 | 1 |
| `tensorrt_llm/runtime/model_runner_cpp.py:1024(_fill_output)` | 0.0 | 1 |
| `benchmark/trtllm/bench.py:47(gen)` | 0.0 | 1 |
| `{method 'disable' of '_lsprof.Profiler' objects}` | 0.0 | 1 |
| `torch/_tensor.py:1161(__iter__)` | 0.0 | 2 |
| `{method 'unbind' of 'torch._C.TensorBase' objects}` | 0.0 | 2 |
| `{built-in method torch._C._cuda_synchronize}` | 0.0 | 2 |
| `torch/utils/_contextlib.py:158(__new__)` | 0.0 | 1 |
| `torch/cuda/__init__.py:1073(synchronize)` | 0.0 | 2 |

## trtllm-int8

[torchprof] wall 955.1 ms · GPU busy 498.6 ms · 유휴 48% · 커널 50,420개 (400/토큰) · 커널 시간 합 498.6 ms

| GPU 시간 분류 | ms | % |
|---|---|---|
| GEMM/GEMV | 237.2 | 48% |
| attention | 162.4 | 33% |
| TRT myelin (fused) | 78.7 | 16% |
| decode/sampling | 8.6 | 2% |
| memcpy/memset | 5.4 | 1% |
| softmax | 4.9 | 1% |
| elementwise/copy/etc | 1.3 | 0% |

| 상위 커널 | ms | 호출 수 |
|---|---|---|
| `void tensorrt_llm::_v1::kernels::weight_only::kernel<tensorrt_llm::_v1::kernels::weight_o…` | 186.7 | 19,750 |
| `void tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel<unsigned short, …` | 111.7 | 3,292 |
| `__myl_AddAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x823596fab576cdfadeae72956e9676f0_…` | 49.7 | 9,772 |
| `void tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel<unsigned short, …` | 38.3 | 3,292 |
| `void gemv2T_kernel_val<int, int, __half, __half, __half, float, 128, 16, 4, 4, false, fal…` | 30.9 | 103 |
| `__myl_CastCastAddCast_0x8f7a531de317e4c42dc65f73c8aea1d8_com` | 14.8 | 6,584 |
| `fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm80_kernel_nl_tiled` | 9.1 | 32 |
| `__myl_CastCastAddMulMulMulAddMulTanhAddMulMulCast_0x98855d175352d7b1c475cad368ee3918_com` | 7.8 | 3,291 |

| CUDA API (CPU 쪽) | ms | 호출 수 |
|---|---|---|
| `cudaEventSynchronize` | 264.1 | 103 |
| `cudaLaunchKernel` | 104.3 | 21,094 |
| `cuLaunchKernel` | 93.3 | 20,313 |
| `cudaLaunchKernelExC` | 33.0 | 6,616 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | 14.5 | 16,460 |
| `cudaMemcpyAsync` | 10.4 | 2,078 |

cProfile wall 888.4 ms (cProfile 오버헤드 포함)

| Python 함수 (tottime 상위) | 자체 ms | 호출 수 |
|---|---|---|
| `tensorrt_llm/runtime/model_runner_cpp.py:928(_initialize_and_fill_output)` | 885.0 | 1 |
| `{built-in method torch.tensor}` | 1.0 | 3 |
| `tensorrt_llm/runtime/model_runner_cpp.py:536(generate)` | 0.0 | 1 |
| `tensorrt_llm/runtime/model_runner_cpp.py:1024(_fill_output)` | 0.0 | 1 |
| `benchmark/trtllm/bench.py:47(gen)` | 0.0 | 1 |
| `{method 'disable' of '_lsprof.Profiler' objects}` | 0.0 | 1 |
| `{method 'unbind' of 'torch._C.TensorBase' objects}` | 0.0 | 2 |
| `torch/_tensor.py:1145(__len__)` | 0.0 | 2 |
| `{built-in method torch._C._cuda_synchronize}` | 0.0 | 2 |
| `torch/_tensor.py:1161(__iter__)` | 0.0 | 2 |
| `{built-in method torch.floor_divide}` | 0.0 | 1 |
| `torch/cuda/__init__.py:1073(synchronize)` | 0.0 | 2 |
