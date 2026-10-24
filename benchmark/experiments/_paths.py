"""experiments/ 공통 경로 설정."""
import os
import sys

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(EXP_DIR)
ROOT = os.path.dirname(BENCH_DIR)  # trt_build
OUT_DIR = os.path.join(EXP_DIR, "out")
LARGE_V2_DIR = os.path.join(ROOT, "notebooks", "models", "openai", "whisper-large-v2")

for p in (ROOT, BENCH_DIR, os.path.join(ROOT, "archived_models")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.makedirs(OUT_DIR, exist_ok=True)
