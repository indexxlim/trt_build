# Whisper TensorRT Inference 🎤

This repository demonstrates TensorRT inference with OpenAI's Whisper model for automatic speech recognition (ASR) using [HuggingFace Transformers](https://huggingface.co/transformers/).

## Supported Models

- [Whisper (automatic speech recognition)](https://huggingface.co/openai/whisper-large-v2)
  - openai/whisper-large-v2
  - ~openai/whisper-large-v3~

## Backend Comparison

whisper-large-v2, batch 1, greedy, same log-mel input for every backend, measured on an idle A100-SXM4-80GB (2026-10-23/24). E2E p50 is encoder plus full decoding. Details and reproduction steps are in [benchmark/README.md](benchmark/README.md).

- EN: librispeech dummy sample, 5.9s audio, 22 tokens
- KO: first 30s of `notebooks/korean_news.mp4`, 126 tokens

| Group | Backend | Precision | Encoder p50 (ms) | E2E EN p50 (ms) | E2E KO p50 (ms) | vs HF FP16 (EN / KO) | Text == HF |
|---|---|---|---|---|---|---|---|
| baseline | HF transformers `generate` (KV cache) | FP16 | 49.0 | 562.2 | 2712.5 | 1.0x / 1.0x | ✅ |
| legacy | `torch.compile(encoder, backend="torch_tensorrt")` + HF decoder | FP16 | 201.0 | 707.8 | 2911.0 | 0.8x / 0.9x | ✅ |
| legacy | NNDF TensorRT (non-KV, ONNX opset 12) | FP16 | 41.5 | 244.3 | 1092.0 | 2.3x / 2.5x | ✅ |
| legacy | Custom TRT v1 following the NNDF flow (KV cache, opset 17, `Whisper/trt_kv.py`) | FP16 | 16.9 | 194.8 | 1043.7 | 2.9x / 2.6x | ✅ |
| legacy | Custom TRT v2 (+ static shapes, on-GPU greedy loop, CUDA graph) | FP16 | 17.3 | 163.1 | 724.2 | 3.4x / 3.7x | ✅ |
| current | CTranslate2 4.8.3 | FP16 | 57.3 | 197.0 | 802.1 | 2.9x / 3.4x | ✅ |
| current | CTranslate2 4.8.3 | INT8 + FP16 | 60.3 | 261.6 | 1126.3 | 2.1x / 2.4x | ✅ |
| current | **TensorRT-LLM 1.2.1** (C++ runtime) | FP16 | - | **113.1** | **548.4** | **5.0x / 4.9x** | ✅ |
| current | TensorRT-LLM 1.2.1 (C++ runtime) | INT8 weight-only | - | 124.6 | 579.9 | 4.5x / 4.7x | ✅ |

"Custom TRT v1/v2" follow the NNDF notebook flow (HF → ONNX → TRT engine → greedy) on the same TensorRT 8.6, but the code is new. The original NNDF KV-cache path mixed BART/T5 code and did not run.

Why each backend is slower than TensorRT-LLM FP16, per profile (KO, 126 tokens):

| Backend | ΔE2E vs TRT-LLM | Main cause |
|---|---|---|
| Custom TRT v2 | +176 ms | Decoder GEMMs run as **FP32** cuBLAS GEMV, so each token reads 3.2 GB of weights instead of 1.6 GB (2.5 vs 1.6 ms/token). Fused LayerNorm/residual pieces are split twice as finely, and cross-K is transposed every step. CPU wait is lower than TRT-LLM's (0.17 vs 0.63 ms/token) |
| CTranslate2 FP16 | +254 ms | Unfused elementwise, LayerNorm and softmax kernels (~900 kernels/token) plus launch overhead |
| Custom TRT v1 | +495 ms | Tensor-core 32x32 GEMM tiles and `mha_v2` for a single query, dynamic-shape enqueue (4 ms/step), `.item()` sync every step |
| NNDF non-KV | +544 ms | No KV cache, so self-attention and the cross K/V projection are recomputed every step |
| HF transformers | +2164 ms | Python op dispatch: the GPU waits on the CPU for 72% of the time |
| `torch_tensorrt` encoder | +2363 ms | Same decoder as HF. In Torch-TensorRT 1.4.0 `aten.mm/bmm` are unsupported, so encoder matmuls stay FP32 |

Full breakdown with kernel and function evidence (nsys, torch profiler, cProfile): [benchmark/profiling/README.md](benchmark/profiling/README.md).

## Setup

Follow the setup steps in the TensorRT OSS repository. It is recommended to experiment inside Docker container.
For a smoother setup experience, it is recommended to use [Poetry](https://python-poetry.org/) to install requirements:

```bash
poetry install # one-time setup
poetry add <path_to_trt_wheel> # see top level repo README.md on how to get TensorRT wheels.
poetry run python run.py <args> # execute program
```

However requirements.txt are also provided:

```bash
pip3 install -r requirements.txt # install requirements
python run.py <args> # execute program
```

Tested with Python 3.8 (conda env `trt`) and NVIDIA driver 580, no Docker needed. The pinned versions matter: `Whisper/*.py` uses old `transformers.generation_utils` APIs and TensorRT 8.6 `PreviewFeature`s.

**Please note that due to end-of-life, Python <= 3.6 is no longer supported.**

## File Structure

```bash
.
├── Whisper              # Whisper directory
│   ├── WhisperModelConfig.py # Model configuration and variant-specific parameters
│   ├── checkpoint.toml       # Example inputs and baseline outputs
│   ├── export.py            # Model conversions between Torch, TRT, ONNX
│   ├── frameworks.py        # PyTorch inference script
│   ├── onnxrt.py           # OnnxRT inference script
│   ├── trt.py              # TensorRT inference script (NNDF, non-KV)
│   ├── trt_kv.py           # Custom TensorRT KV-cache path: v1 (dynamic) and v2 (static shapes + CUDA graph)
│   └── measurements.py     # Performance measurement script
├── notebooks           # Jupyter notebooks for Whisper
│   ├── whisper.ipynb  # Original demo notebook
│   ├── whisper2.ipynb # Fixed notebook: opset 17 + KV-cache TensorRT
│   └── korean_news.mp4 # Sample audio/video file
├── benchmark           # Backend comparison: HF, Torch-TensorRT, NNDF TRT, custom TRT v1/v2, CTranslate2, TensorRT-LLM
│   ├── nndf_legacy/ ct2/ trtllm/   # one folder per backend (each with requirements.txt and README)
│   ├── profiling/      # nsys / torch profiler / cProfile per backend
│   └── experiments/    # standalone experiments behind the numbers in the docs
├── archived_models     # Other models (BART, GPT2, T5, NNDF)
└── run.py             # main entry script
```

## How to run Whisper inference

`run.py` is the main entry point for Whisper demos. `compare` and `run` are two most common actions.

### Compare Performance

The `compare` action compares PyTorch framework and TensorRT inference:

```bash
python3 run.py compare Whisper --variant openai/whisper-large-v2 --working-dir temp
```

### Run Specific Framework

The `run` action executes a specific inference framework:

```bash
# PyTorch inference
python3 run.py run Whisper frameworks --variant openai/whisper-large-v2 --working-dir temp

# TensorRT inference
python3 run.py run Whisper trt --variant openai/whisper-large-v2 --working-dir temp
```

## How to run with different precisions

TensorRT by default uses TF32 precision. To experiment with FP16 precision:

```bash
python3 run.py run Whisper trt --variant openai/whisper-large-v2 --working-dir temp --fp16
```

## Audio Input

Whisper processes audio files. Supported formats include:
- WAV, MP3, FLAC, M4A, and other common audio formats
- Video files (audio will be extracted)

Place your audio/video files in the working directory or specify the path in the checkpoint.toml configuration.

## Performance Measurement

Use timing parameters to control measurement:

```bash
python3 run.py run Whisper trt --variant openai/whisper-large-v2 --working-dir temp --iterations 10 --warmup 3 --percentile 50
```

Parameters:
- `--iterations <int>`: Number of iterations to measure (default 10)
- `--warmup <int>`: Number of warmup iterations (default 3)
- `--percentile <int>`: Percentile for measurement (default 50, i.e. median)

## Jupyter Notebook Demo

For interactive exploration, use the provided Jupyter notebook:

```bash
jupyter notebook notebooks/whisper.ipynb
```

The notebook includes:
- Audio preprocessing examples
- Model inference comparisons
- Performance analysis
- Visualization of results

## Testing

Run tests using pytest:

```bash
pytest tests/
```

## Troubleshooting

### CUDA/cuBLAS Errors

If you encounter CUDA errors, check your LD_LIBRARY_PATH for conflicting CUDA versions:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Audio Loading Issues

Ensure ffmpeg is installed for audio processing:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## License

See LICENSE.txt for details.

## Archived Models

Other TensorRT inference examples (BART, GPT2, T5, NNDF) have been moved to `archived_models/` directory.

## Conclusion

**Best: TensorRT-LLM 1.2.1 FP16, about 5x faster than HF transformers with identical output text** (whisper-large-v2, batch 1, greedy, idle A100).

| Rank | Backend | E2E KO 30s (126 tokens) | vs HF FP16 | Decoder time per token |
|---|---|---|---|---|
| 1 | **TensorRT-LLM FP16** | **548 ms** | **4.9x** | **4.17 ms** |
| 2 | TensorRT-LLM INT8 weight-only | 580 ms | 4.7x | 4.40 ms |
| 3 | Custom TensorRT v2 (TRT 8.6, static shapes + CUDA graph) | 724 ms | 3.7x | 5.55 ms |
| 4 | CTranslate2 FP16 | 802 ms | 3.4x | 5.9 ms |
| 5 | Custom TensorRT v1 (TRT 8.6, KV cache) | 1044 ms | 2.6x | 8.15 ms |
| 6 | NNDF TensorRT (non-KV) | 1092 ms | 2.5x | 8.3 ms |
| 7 | CTranslate2 INT8 | 1126 ms | 2.4x | 8.4 ms |
| - | HF transformers FP16 (baseline) | 2712 ms | 1.0x | 21.1 ms |
| - | `torch.compile(backend="torch_tensorrt")` (Torch-TensorRT 1.4.0) | 2911 ms | 0.9x | 21.3 ms |

**Why TensorRT-LLM wins.** Batch-1 decoding reads all ~1.6 GB of decoder weights for every token, so it is bound by memory bandwidth and per-step overhead. TensorRT-LLM does best on all three fronts:

1. **GEMM:** an M=1-specific CUDA-core GEMV (`cudaCoreGemm`) reaches ~1.0 TB/s, which is 1.59 ms/token.
2. **Attention:** a single fused decoding kernel (`masked_multihead_attention`) takes 1.11 ms/token.
3. **Overhead:** it launches the fewest kernels (418/token), and its C++ runtime leaves the GPU waiting only 0.63 ms/token.

**Plain TensorRT was still worth it.**

- Every TensorRT engine path beat HF by 2.5–3.7x.
- On the same TensorRT 8.6, changing only how the engine is driven (v1 → v2: static shapes, on-GPU greedy loop, CUDA graph) made decoding 1.44x faster.
- v2 overtook CTranslate2 and has the lowest CPU overhead of all backends (0.17 ms/token). It also has the fastest encoder (17 ms vs 23 ms for TensorRT-LLM).
- Its remaining gap to TensorRT-LLM (+1.38 ms/token) comes mostly from decoder GEMMs running in FP32 (twice the weight bytes) and a per-step cross-K transpose. Both are fixable.

**What did not help:**

- INT8 at batch 1 was no faster: CTranslate2 INT8 was slower than FP16, and TensorRT-LLM weight-only INT8 was about the same.
- Torch-TensorRT 1.4.0 `torch.compile` was slower than eager. `aten.mm/bmm` are unsupported, and the decoder recompiles every step.

Details: [benchmark/README.md](benchmark/README.md) · profiling evidence: [benchmark/profiling/README.md](benchmark/profiling/README.md)
