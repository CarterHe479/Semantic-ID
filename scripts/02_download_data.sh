#!/usr/bin/env bash
set -e
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023")
print(ds)  # 自动缓存至 ~/.cache；你也可保存到 ./data/amazon23
PY
