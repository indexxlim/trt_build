"""CTranslate2 Whisper 벤치 (conda env `ct2`).

    CUDA_VISIBLE_DEVICES=0 python benchmark/ct2/bench.py --compute-type float16
    CUDA_VISIBLE_DEVICES=0 python benchmark/ct2/bench.py --compute-type int8_float16

처음 실행 시 HF 체크포인트를 CTranslate2 포맷(float16)으로 변환해 models/에 저장한다.
compute_type은 로드 시점에 바꿀 수 있으므로 변환은 한 번만 한다.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402,F401  CUDA 12 cuBLAS/cuDNN을 먼저 로드해 ctranslate2가 찾게 한다
import ctranslate2  # noqa: E402
import numpy as np  # noqa: E402
from transformers import WhisperTokenizer  # noqa: E402

from common import VARIANT, bench, load_inputs, save_result  # noqa: E402

CT2_DIR = os.path.join(HERE, "models", VARIANT.split("/")[-1] + "-ct2-float16")


def convert():
    if os.path.exists(os.path.join(CT2_DIR, "model.bin")):
        return
    subprocess.check_call([
        "ct2-transformers-converter", "--model", VARIANT, "--output_dir", CT2_DIR,
        "--quantization", "float16", "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-type", default="float16", choices=["float16", "int8_float16", "float32"])
    args = ap.parse_args()

    convert()
    model = ctranslate2.models.Whisper(CT2_DIR, device="cuda", compute_type=args.compute_type)
    tok = WhisperTokenizer.from_pretrained(VARIANT)

    per_sample = {}
    for s, x in load_inputs().items():
        feats = ctranslate2.StorageView.from_array(np.ascontiguousarray(x["features"]))
        prompt = x["prompt"]

        def gen():
            # greedy, HF generate와 같은 suppress 규칙 (suppress_tokens=-1: 모델 config의 suppress_tokens,
            # suppress_blank: 첫 토큰에서 공백/EOT 금지 = HF begin_suppress_tokens)
            return model.generate(feats, [prompt], beam_size=1, max_length=448,
                                  suppress_blank=True, suppress_tokens=[-1])[0].sequences_ids[0]

        ids = gen()  # 프롬프트 제외, EOT 제외
        per_sample[s] = {
            "text": tok.decode(ids, skip_special_tokens=True),
            "n_tokens": len(ids) + 1,  # 다른 백엔드와 같이 EOT 포함
            "encoder": bench(lambda: model.encode(feats, to_cpu=True)),
            "e2e": bench(gen, label=f"e2e:{s}"),
        }

    save_result("ctranslate2", args.compute_type, per_sample,
                extra={"env": f"conda ct2 (py3.10, ctranslate2 {ctranslate2.__version__})"})


if __name__ == "__main__":
    main()
