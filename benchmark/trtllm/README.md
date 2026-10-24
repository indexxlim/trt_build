# trtllm — TensorRT-LLM 1.2.1

conda env `trtllm` (python 3.12, tensorrt_llm 1.2.1, TensorRT 10.14, torch 2.9.1+cu130)

```bash
conda create -n trtllm -c conda-forge python=3.12 openmpi mpi4py && conda activate trtllm
pip install -r benchmark/trtllm/requirements.txt
# 아래 activate 스크립트를 만든 뒤 env를 다시 activate 한다

# assets: OpenAI 원본 체크포인트와 tokenizer
cd benchmark/trtllm/assets
curl -O https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
curl -O https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
curl -o large-v2.pt https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt
cd -

bash benchmark/trtllm/build.sh                   # FP16 -> models/engine_large-v2_fp16
WEIGHT_ONLY=int8 bash benchmark/trtllm/build.sh  # INT8 weight-only -> models/engine_large-v2_int8
CUDA_VISIBLE_DEVICES=0 python benchmark/trtllm/bench.py --engine fp16
```

- `upstream/`은 NVIDIA/TensorRT-LLM `v1.2.1`의 `examples/models/core/whisper` 원본이다. `convert_checkpoint.py`를 그대로 쓴다.
- 빌드 설정은 batch 1, beam 1, 디코더 `max_seq_len 448`이다. 예제 기본값 114로는 한국어 30초(126토큰)를 처리할 수 없다.
- 실행은 예제와 같이 C++ runtime(`ModelRunnerCpp`)을 쓴다. 인코더와 디코더를 한 번에 실행하므로 인코더 시간을 따로 잴 수 없다.
- 예제처럼 plain greedy라서 HF의 `suppress_tokens` 규칙은 적용되지 않는다. 그래도 이번 두 샘플은 결과가 HF와 같았다.
- GPU를 다른 작업과 같이 쓰므로 `kv_cache_free_gpu_memory_fraction`은 예제의 0.9 대신 0.2로 두었다.

## 환경 주의사항

- TensorRT-LLM 1.2.1 wheel은 CUDA 13 빌드다. PyPI 기본 `torch==2.9.1`은 cu128이라 `libcublasLt.so.13` ImportError가 난다. 그래서 `torch==2.9.1+cu130`을 쓴다(requirements.txt 참고).
- CUDA 13 런타임 lib(`site-packages/nvidia/cu13/lib`)이 검색 경로에 없으면 `libnvrtc.so.13` ImportError가 난다. 그래서 `$CONDA_PREFIX/etc/conda/activate.d/cu13_libs.sh`를 만든다.

  ```bash
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONNOUSERSITE=1
  ```

- pyarrow 18 이상은 numpy 2를 요구하는데 tensorrt_llm은 numpy 1.26을 쓴다. 그래서 `pyarrow<18`로 고정한다.
