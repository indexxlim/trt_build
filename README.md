# Whisper TensorRT Inference 🎤

This repository demonstrates TensorRT inference with OpenAI's Whisper model for automatic speech recognition (ASR) using [HuggingFace Transformers](https://huggingface.co/transformers/).

## Supported Models

- [Whisper (automatic speech recognition)](https://huggingface.co/openai/whisper-large-v2)
  - openai/whisper-large-v2
  - ~openai/whisper-large-v3~

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
│   ├── trt.py              # TensorRT inference script
│   └── measurements.py     # Performance measurement script
├── notebooks           # Jupyter notebooks for Whisper
│   ├── whisper.ipynb  # Whisper demo notebook
│   └── korean_news.mp4 # Sample audio/video file
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