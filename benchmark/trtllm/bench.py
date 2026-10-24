"""TensorRT-LLM 1.2.1 Whisper 벤치 (conda env `trtllm`, C++ runtime ModelRunnerCpp).

    bash benchmark/trtllm/build.sh
    CUDA_VISIBLE_DEVICES=0 python benchmark/trtllm/bench.py --engine fp16

NVIDIA 예제(upstream/run.py)와 같은 방식으로 실행한다. 예제처럼 plain greedy라서 HF의
suppress_tokens / begin_suppress_tokens 규칙은 적용되지 않는다 (텍스트 차이는 CER로 기록).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import tensorrt_llm  # noqa: E402
from tensorrt_llm.runtime import ModelRunnerCpp  # noqa: E402
from transformers import WhisperTokenizer  # noqa: E402

from common import VARIANT, bench, load_inputs, save_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="fp16", help="models/engine_large-v2_<engine>")
    ap.add_argument("--kv-cache-fraction", type=float, default=0.2,
                    help="남은 GPU 메모리 중 KV cache 비율 (GPU를 다른 작업과 공유하므로 예제의 0.9보다 낮춤)")
    args = ap.parse_args()

    engine_dir = os.path.join(HERE, "models", f"engine_large-v2_{args.engine}")
    runner = ModelRunnerCpp.from_dir(
        engine_dir=engine_dir, is_enc_dec=True, max_batch_size=1, max_input_len=3000,
        max_output_len=448, max_beam_width=1,
        kv_cache_free_gpu_memory_fraction=args.kv_cache_fraction, cross_kv_cache_fraction=0.5)
    tok = WhisperTokenizer.from_pretrained(VARIANT)
    eot = tok.convert_tokens_to_ids("<|endoftext|>")

    per_sample = {}
    for s, x in load_inputs().items():
        # 공통 HF log-mel (1, 80, 3000) -> (1, 3000, 80) fp16 (예제와 같은 입력 형식)
        mel = torch.from_numpy(x["features"]).cuda().half().transpose(1, 2).contiguous()
        prompt = torch.tensor([x["prompt"]], dtype=torch.int32)
        lengths = torch.tensor([mel.shape[1]], dtype=torch.int32, device="cuda")

        def gen():
            with torch.no_grad():
                out = runner.generate(
                    batch_input_ids=prompt, encoder_input_features=mel, encoder_output_lengths=lengths // 2,
                    max_new_tokens=448 - prompt.shape[1], end_id=eot, pad_id=eot, num_beams=1,
                    output_sequence_lengths=True, return_dict=True)
            torch.cuda.synchronize()
            return out

        out = gen()
        seq_len = int(out["sequence_lengths"][0][0])
        ids = out["output_ids"][0][0][:seq_len].tolist()
        gen_ids = ids[prompt.shape[1]:]
        per_sample[s] = {
            "text": tok.decode(gen_ids, skip_special_tokens=True),
            "n_tokens": len(gen_ids) + (0 if gen_ids and gen_ids[-1] == eot else 1),  # EOT 포함
            "encoder": None,  # C++ runtime이 encoder+decoder를 한 번에 실행 (분리 측정 불가)
            "e2e": bench(gen, label=f"e2e:{s}"),
        }

    save_result("tensorrt_llm", args.engine, per_sample,
                extra={"env": f"conda trtllm (py3.12, tensorrt_llm {tensorrt_llm.__version__}, torch {torch.__version__})"})


if __name__ == "__main__":
    main()
