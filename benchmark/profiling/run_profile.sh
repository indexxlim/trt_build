#!/usr/bin/env bash
# 백엔드별로 한국어 샘플 E2E 1회(warmup 2회 뒤)를 nsys + cProfile로 프로파일링한다.
#   GPU=4 bash benchmark/profiling/run_profile.sh            # 전부
#   GPU=4 bash benchmark/profiling/run_profile.sh ct2-float16 # 하나만
# 결과: benchmark/profiling/<tag>/{nsys.nsys-rep, nsys_*.csv, cprofile.txt}, 요약은 analyze.py
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PROFILE_TARGET=${PROFILE_TARGET:-e2e:ko}
NSYS=${NSYS:-/opt/nvidia/nsight-systems/2023.2.3/bin/nsys}
source ~/anaconda3/etc/profile.d/conda.sh

# tag | conda env | command
JOBS=(
  "hf-fp16|trt|python benchmark/nndf_legacy/bench.py --backend hf"
  "torchtrt-fp16|trt|python benchmark/nndf_legacy/bench.py --backend torchtrt"
  "nndf-nonkv-fp16|trt|python benchmark/nndf_legacy/bench.py --backend nndf"
  "nndf-port-kv-fp16|trt|python benchmark/nndf_legacy/bench.py --backend trtkv"
  "nndf-port-kv-graph-fp16|trt|python benchmark/nndf_legacy/bench.py --backend trtkv2"
  "ct2-float16|ct2|python benchmark/ct2/bench.py --compute-type float16"
  "ct2-int8|ct2|python benchmark/ct2/bench.py --compute-type int8_float16"
  "trtllm-fp16|trtllm|python benchmark/trtllm/bench.py --engine fp16"
  "trtllm-int8|trtllm|python benchmark/trtllm/bench.py --engine int8"
)
# nsys 2023.2는 CUDA 13 앱(TensorRT-LLM 1.2)의 GPU trace를 못 잡는다 -> torch profiler(Kineto/CUPTI cu13)로 대체
TORCHPROF_TAGS=(trtllm-fp16 trtllm-int8)

for job in "${JOBS[@]}"; do
  IFS="|" read -r tag env cmd <<<"$job"
  [[ $# -gt 0 && " $* " != *" $tag "* ]] && continue
  out=benchmark/profiling/$tag
  mkdir -p "$out"
  conda activate "$env"
  echo "=== $tag ($env)"

  # 1) cProfile: Python 함수별 시간
  PROFILE=cprofile PROFILE_OUT=$out $cmd >"$out/run_cprofile.log" 2>&1 || echo "  cprofile failed (see $out/run_cprofile.log)"

  if [[ " ${TORCHPROF_TAGS[*]} " == *" $tag "* ]]; then
    PROFILE=torchprof PROFILE_OUT=$out $cmd >"$out/run_torchprof.log" 2>&1 || echo "  torchprof failed (see $out/run_torchprof.log)"
    continue
  fi

  # 2) nsys: 지정 구간(cudaProfilerStart~Stop)의 CUDA 커널 / API / NVTX
  # CUDA graph 안의 커널을 노드 단위로 펼쳐 기록 (기본값은 graph 하나로만 기록됨)
  PROFILE=nsys PROFILE_OUT=$out "$NSYS" profile -t cuda,nvtx,osrt --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=stop --force-overwrite true -o "$out/nsys" $cmd >"$out/run_nsys.log" 2>&1 \
    || echo "  nsys failed (see $out/run_nsys.log)"
  if [[ -f $out/nsys.nsys-rep ]]; then
    "$NSYS" stats --force-export true --force-overwrite true --format csv --output "$out/nsys" \
      --report cuda_gpu_trace,cuda_api_sum,nvtx_pushpop_trace "$out/nsys.nsys-rep" >"$out/stats.log" 2>&1 \
      || echo "  nsys stats failed (see $out/stats.log)"
  fi
done
