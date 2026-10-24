| 구분 | 백엔드 | 정밀도 | 인코더 p50 (ms) | E2E EN p50 (ms) | E2E KO p50 (ms) | HF FP16 대비 (EN / KO) | 텍스트 == HF | 측정 시각 |
|---|---|---|---|---|---|---|---|---|
| 기준선 | HF transformers generate (cache) | fp16 | 49.0 | 562.2 | 2712.5 | 1.0x / 1.0x | ✅ | 10-23T20:55 |
| 레거시 | torch.compile(encoder, backend="torch_tensorrt") + HF decoder | fp16 | 201.0 | 707.8 | 2911.0 | 0.8x / 0.9x | ✅ | 10-24T16:55 |
| 레거시 | NNDF TensorRT (non-KV, opset 12) | fp16 | 41.5 | 244.3 | 1092.0 | 2.3x / 2.5x | ✅ | 10-23T20:56 |
| 레거시 | NNDF 흐름 자체 구현 TensorRT v1 (KV cache, opset 17) | fp16 | 16.9 | 194.8 | 1043.7 | 2.9x / 2.6x | ✅ | 10-23T20:56 |
| 레거시 | NNDF 흐름 자체 구현 TensorRT v2 (+정적 shape, GPU 루프, CUDA graph) | fp16 | 17.3 | 163.1 | 724.2 | 3.4x / 3.7x | ✅ | 10-23T20:57 |
| 현행 | CTranslate2 | float16 | 57.3 | 197.0 | 802.1 | 2.9x / 3.4x | ✅ | 10-23T20:57 |
| 현행 | CTranslate2 | int8_float16 | 60.3 | 261.6 | 1126.3 | 2.1x / 2.4x | ✅ | 10-23T20:58 |
| 현행 | TensorRT-LLM (C++ runtime) | fp16 | - | 113.1 | 548.4 | 5.0x / 4.9x | ✅ | 10-23T20:58 |
| 현행 | TensorRT-LLM (C++ runtime) | int8 | - | 124.6 | 579.9 | 4.5x / 4.7x | ✅ | 10-23T20:59 |

GPU (측정 시점 nvidia-smi):
  NVIDIA A100-SXM4-80GB, 1315 MiB, 81920 MiB, 44 %
  NVIDIA A100-SXM4-80GB, 18173 MiB, 81920 MiB, 86 %
  NVIDIA A100-SXM4-80GB, 19439 MiB, 81920 MiB, 95 %
  NVIDIA A100-SXM4-80GB, 19711 MiB, 81920 MiB, 0 %
  NVIDIA A100-SXM4-80GB, 2333 MiB, 81920 MiB, 98 %
  NVIDIA A100-SXM4-80GB, 4221 MiB, 81920 MiB, 97 %
  NVIDIA A100-SXM4-80GB, 619 MiB, 81920 MiB, 32 %
  NVIDIA A100-SXM4-80GB, 825 MiB, 81920 MiB, 23 %
  NVIDIA A100-SXM4-80GB, 9579 MiB, 81920 MiB, 4 %
