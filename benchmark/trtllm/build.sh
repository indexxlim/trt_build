#!/usr/bin/env bash
# TensorRT-LLM 1.2.1 Whisper large-v2 엔진 빌드 (conda env `trtllm`)
#   bash benchmark/trtllm/build.sh            # FP16
#   WEIGHT_ONLY=int8 bash benchmark/trtllm/build.sh   # INT8 weight-only (+FP16 연산)
#
# upstream/ 은 NVIDIA/TensorRT-LLM v1.2.1 examples/models/core/whisper 원본 그대로.
# assets/ 에 large-v2.pt, multilingual.tiktoken, mel_filters.npz 필요 (README 참고).
set -euo pipefail
cd "$(dirname "$0")"

MODEL=large-v2
PRECISION=float16
WEIGHT_ONLY=${WEIGHT_ONLY:-}
TAG=${WEIGHT_ONLY:-fp16}
CKPT=models/ckpt_${MODEL}_${TAG}
ENGINE=models/engine_${MODEL}_${TAG}

WO_ARGS=()
if [[ -n "$WEIGHT_ONLY" ]]; then
  WO_ARGS=(--use_weight_only --weight_only_precision "$WEIGHT_ONLY")
fi

python upstream/convert_checkpoint.py --model_dir assets --model_name "$MODEL" \
  --dtype "$PRECISION" --output_dir "$CKPT" "${WO_ARGS[@]}"

# batch 1, greedy, 디코더 최대 길이 448 (Whisper max_target_positions)
trtllm-build --checkpoint_dir "$CKPT/encoder" --output_dir "$ENGINE/encoder" \
  --moe_plugin disable --max_batch_size 1 \
  --gemm_plugin disable --bert_attention_plugin "$PRECISION" \
  --max_input_len 3000 --max_seq_len 3000

trtllm-build --checkpoint_dir "$CKPT/decoder" --output_dir "$ENGINE/decoder" \
  --moe_plugin disable --max_beam_width 1 --max_batch_size 1 \
  --max_seq_len 448 --max_input_len 14 --max_encoder_input_len 3000 \
  --gemm_plugin "$PRECISION" --bert_attention_plugin "$PRECISION" --gpt_attention_plugin "$PRECISION"

echo "built $ENGINE"
