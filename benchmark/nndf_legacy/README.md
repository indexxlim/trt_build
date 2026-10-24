# nndf_legacy — HF 기준선 + 레거시 TensorRT 경로

conda env `trt` (python 3.8, torch 2.0.1, TensorRT 8.6.1, transformers 4.24, torch-tensorrt 1.4.0)

```bash
conda create -n trt python=3.8 && conda activate trt
pip install --prefer-binary -r benchmark/nndf_legacy/requirements.txt
CUDA_VISIBLE_DEVICES=0 python benchmark/nndf_legacy/bench.py --backend all --precision fp16
```

| backend | 내용 |
|---|---|
| `hf` | HF transformers `generate(use_cache=True)`, FP16. 비교 기준선 |
| `torchtrt` | `torch.compile(encoder, backend="torch_tensorrt")` + HF FP16 디코더 |
| `nndf` | TensorRT OSS `demo/HuggingFace`(NNDF) 기반 non-KV 엔진, ONNX opset 12 (`Whisper/trt.py`) |
| `trtkv` | NNDF 흐름 자체 구현 v1: opset 17 + KV cache, 동적 shape, Python greedy 루프 (`Whisper/trt_kv.py` `WhisperTRTKV`) |
| `trtkv2` | NNDF 흐름 자체 구현 v2: 정적 shape(고정 길이 448 KV 버퍼 + pos 마스크), GPU greedy 루프, step 하나를 CUDA graph로 replay (`WhisperTRTKVGraph`). TRT 8.6 그대로 |

`nndf` 엔진은 v1 노트북(git `74f9c1c`의 `notebooks/whisper2.ipynb`)이 만든 파일을 쓰고, `trtkv`와 `trtkv2` 엔진은 없으면 여기서 export와 빌드를 한다.

"NNDF 흐름 자체 구현"은 NNDF 노트북의 흐름과 환경을 따랐지만 코드는 새로 작성한 것이다. 원래 NNDF의 KV cache 경로(`Whisper/export.py`, `Whisper/trt.py`)는 BART/T5 코드가 섞여 export와 runner가 맞지 않아 동작하지 않았다.

v2 구현 메모: suppress용 bias를 `-inf`로 두면 첫 토큰이 아닌 step에서 `0 × -inf = NaN`이 되고, `argmax`가 NaN을 최댓값으로 골라 공백만 반복 출력된다. 그래서 유한한 큰 음수(-1e9)를 쓴다.

## NNDF가 뭔가

NNDF는 NVIDIA/TensorRT repo의 `demo/HuggingFace`에 있던 데모 프레임워크다(T5/BART/GPT2/Whisper). 2026-10 기준 TensorRT `main` 브랜치의 `demo/`에는 `DeBERTa`와 `EfficientDet`만 남아 있고 HuggingFace 데모는 없다. 반면 `tools/Polygraphy`와 `tools/onnx-graphsurgeon`은 계속 유지되고 있다. LLM과 encoder-decoder 계열은 TensorRT-LLM으로 넘어갔다.

## `torch.compile(backend="torch_tensorrt")` (Torch-TensorRT 1.4.0) 정리

- **백엔드 이름**: torch 2.0.1 + torch-tensorrt 1.4.0에 등록된 dynamo 백엔드는 `torch_tensorrt`와 `aot_torch_tensorrt_aten`뿐이다. `backend="tensorrt"`는 이 조합에서 `KeyError`가 난다. `"tensorrt"` 별칭은 이후 버전에서 생긴 것으로 보인다.
- **dtype**: `.half()` 모델을 넣으면 `AssertionError: Dtype mismatch ... Expect torch.float16, got torch.float32`가 난다. 그래서 FP32 모델에 `enabled_precisions={torch.float, torch.half}`를 주고 TRT 안에서 FP16을 쓰게 했다.
- **미지원 aten op**: 아래 op는 TRT로 바뀌지 않고 PyTorch로 fallback된다. 확인 방법은 `_partition.partition(verbose=True)`로 whisper-tiny 인코더와 디코더를 확인한 것이다.
  - 행렬곱: `aten.mm`, `aten.bmm`. Linear와 attention 계산이 전부 PyTorch에 남는다.
  - softmax 구성 op: `aten.amax`, `aten.exp`, `aten.sum.dim_IntList`, `aten.reciprocal`
  - LayerNorm: `aten.var_mean.correction`, `aten.sqrt`
  - GELU: `aten.erf`
  - shape 관련: `aten.permute`, `aten.clone`, `aten._unsafe_view`, `operator.getitem`
  - 디코더 앞부분: `aten.embedding`, `aten.arange`, `aten.where`, `aten.lt`, `aten.slice`, `aten.full`, `aten.scalar_tensor`, `aten._to_copy`, `aten.unsqueeze`. 이 부분은 TRT 엔진이 0개 만들어진다.
- **변환 실패**: 인코더 conv1d 부분은 `RuntimeError: linear convolution_1 has bias of type ITensor, Expect Optional[Tensor]` 에러가 나고, 해당 subgraph는 PyTorch로 되돌아간다.
- **large-v2 그래프 분할 실패**: 32층 그래프를 미지원 op 기준으로 잘게 나눈 뒤 다시 합치는 단계(`torch.fx ... dfs_find_cycle`)에서 기본 재귀 한도로는 `RecursionError`가 난다. 그러면 인코더 전체가 PyTorch FP32로 돌아간다. `sys.setrecursionlimit(100000)`으로 우회하면 TRT 엔진 448개가 만들어지지만, 행렬곱은 여전히 PyTorch FP32라서 HF FP16보다 느리다. 컴파일에는 약 6분이 걸린다.
- **디코더 컴파일은 사용 불가**: KV cache 길이가 매 step 바뀌는데, 1.4.0은 정적 shape만 지원해서 step마다 재컴파일한다. whisper-tiny에서도 첫 실행 1165초, 두 번째 실행 78.8초, 컴파일된 그래프 171개가 나왔다. 그래서 large-v2에서는 인코더만 컴파일했다.
