# ct2 — CTranslate2

conda env `ct2` (python 3.10, ctranslate2 4.8.3, torch 2.5.1+cu124, transformers 4.57.3)

```bash
conda create -n ct2 python=3.10 && conda activate ct2
pip install -r benchmark/ct2/requirements.txt
CUDA_VISIBLE_DEVICES=0 python benchmark/ct2/bench.py --compute-type float16
CUDA_VISIBLE_DEVICES=0 python benchmark/ct2/bench.py --compute-type int8_float16
```

- 처음 실행할 때 `ct2-transformers-converter`로 `openai/whisper-large-v2`를 `models/whisper-large-v2-ct2-float16`에 변환한다. `compute_type`은 로드할 때 바뀐다.
- greedy 디코딩 설정은 `beam_size=1, suppress_blank=True, suppress_tokens=[-1]`이다. HF generate의 suppress 규칙과 같다.
- `encoder` 시간은 `model.encode(..., to_cpu=True)`로 잰다.

## 환경 주의사항

- 셸 기본 `LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64`이 잡혀 있으면 torch import에서 `libcusparse.so.12: undefined symbol __nvJitLinkComplete_12_4`가 난다.
- `~/.local`(user site)에 있는 패키지를 pip이 "이미 설치됨"으로 보고 env 안에 설치하지 않는다.
- 위 두 문제 때문에 `$CONDA_PREFIX/etc/conda/activate.d/isolate.sh`에서 `LD_LIBRARY_PATH`를 해제하고 `PYTHONNOUSERSITE=1`을 설정했다. 그러니 반드시 `conda activate ct2`로 실행해야 한다.
- ctranslate2 4.8 converter는 `from_pretrained(dtype=...)`를 써서 transformers 4.56 이상이 필요하다.
