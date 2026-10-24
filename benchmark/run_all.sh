#!/usr/bin/env bash
# 모든 백엔드를 같은 GPU에서 연달아 측정하고 비교표를 출력한다.
#   GPU=0 bash benchmark/run_all.sh
#   (gloss_sim GPU 큐처럼 CUDA_VISIBLE_DEVICES를 넣어 주면 그 GPU를 쓴다)
#   NNDF_BACKENDS="hf nndf trtkv trtkv2" PROFILE_AFTER="nndf-port-kv-graph-fp16" bash benchmark/run_all.sh
# 사전 준비: 각 폴더 README의 환경 생성 + 엔진/모델 빌드, `python benchmark/prepare_inputs.py` (env trt)
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
NNDF_BACKENDS=${NNDF_BACKENDS:-"hf torchtrt nndf trtkv trtkv2"}
source ~/anaconda3/etc/profile.d/conda.sh
echo "$(date '+%F %T') GPU $CUDA_VISIBLE_DEVICES: $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

conda activate trt
for b in $NNDF_BACKENDS; do
  python benchmark/nndf_legacy/bench.py --backend "$b" --precision fp16
done

conda activate ct2
python benchmark/ct2/bench.py --compute-type float16
python benchmark/ct2/bench.py --compute-type int8_float16

conda activate trtllm
python benchmark/trtllm/bench.py --engine fp16
python benchmark/trtllm/bench.py --engine int8

python benchmark/compare.py | tee benchmark/results/RESULTS.md

# 같은 GPU에서 이어서 프로파일링 (선택)
if [[ -n "${PROFILE_AFTER:-}" ]]; then
  GPU=$CUDA_VISIBLE_DEVICES bash benchmark/profiling/run_profile.sh $PROFILE_AFTER
  python benchmark/profiling/analyze.py >/dev/null
fi
echo "$(date '+%F %T') done"
