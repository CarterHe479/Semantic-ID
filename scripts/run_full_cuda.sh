#!/usr/bin/env bash
set -euo pipefail

# One-command production pipeline for a CUDA workstation such as RTX 3090.
# Override any setting below by exporting it before running this script.

DATA_DIR="${DATA_DIR:-./data/amazon23}"
OUT_ROOT="${OUT_ROOT:-./outputs}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
LOG_DIR="${LOG_DIR:-./logs/full_cuda_$(date +%Y%m%d_%H%M%S)}"

TEXT_MODEL="${TEXT_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
TEXT_COLS="${TEXT_COLS:-title,description}"
EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-256}"
MAX_LEN="${MAX_LEN:-128}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
MAX_TRAIN="${MAX_TRAIN:-200000}"

RQ_L="${RQ_L:-4}"
RQ_K="${RQ_K:-256}"
RQ_EPOCHS="${RQ_EPOCHS:-8}"
RQ_BATCH_SIZE="${RQ_BATCH_SIZE:-4096}"
RQ_ENCODE_BATCH_SIZE="${RQ_ENCODE_BATCH_SIZE:-4096}"
RQ_DIST_CHUNK="${RQ_DIST_CHUNK:-1024}"
RQ_ENTROPY_REG="${RQ_ENTROPY_REG:-1e-4}"

RUN_VQ="${RUN_VQ:-1}"
VQ_K="${VQ_K:-4096}"
VQ_EPOCHS="${VQ_EPOCHS:-10}"
VQ_BATCH_SIZE="${VQ_BATCH_SIZE:-8192}"
VQ_ENCODE_BATCH_SIZE="${VQ_ENCODE_BATCH_SIZE:-4096}"
VQ_DIST_CHUNK="${VQ_DIST_CHUNK:-512}"

EVAL_MAX_TEST="${EVAL_MAX_TEST:-5000}"
EXPAND_RQ_M="${EXPAND_RQ_M:-4}"
EXPAND_VQ_T="${EXPAND_VQ_T:-8}"
USE_FAISS_RERANK="${USE_FAISS_RERANK:-1}"
ALLOW_FULL_SEARCH_FALLBACK="${ALLOW_FULL_SEARCH_FALLBACK:-0}"

EMB_DIR="${OUT_ROOT}/emb"
RQK_DIR="${OUT_ROOT}/rqkmeans_${RQ_L}x${RQ_K}"
RQVAE_DIR="${OUT_ROOT}/rqvae_${RQ_L}x${RQ_K}"
VQVAE_DIR="${OUT_ROOT}/vqvae_${VQ_K}"
FAISS_DIR="${OUT_ROOT}/faiss"
RESULT_CSV="${RESULT_CSV:-${RESULTS_DIR}/offline_eval_cuda.csv}"

mkdir -p "${EMB_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"

run() {
  local name="$1"
  shift
  echo
  echo "[$(date '+%F %T')] >>> ${name}"
  echo "Command: $*"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "${DATA_DIR}/train.parquet"
require_file "${DATA_DIR}/valid.parquet"
require_file "${DATA_DIR}/test.parquet"

DEVICE="${DEVICE}" python3 - <<'PY'
import os
import torch
device = os.environ["DEVICE"]
if device == "cuda":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment. Activate the CUDA-enabled env or set DEVICE=cpu for a dry run.")
    print("CUDA:", torch.cuda.get_device_name(0))
else:
    print("DEVICE:", device)
print("PyTorch:", torch.__version__)
PY

echo "Logs: ${LOG_DIR}"
echo "Outputs: ${OUT_ROOT}"
echo "Results: ${RESULT_CSV}"

run extract_train_embeddings \
  python3 -m src.train.extract_embeddings \
    --data "${DATA_DIR}/train.parquet" \
    --text-cols "${TEXT_COLS}" \
    --model "${TEXT_MODEL}" \
    --batch-size "${EMB_BATCH_SIZE}" \
    --max-len "${MAX_LEN}" \
    --normalize \
    --out "${EMB_DIR}/train_emb.pt"

run extract_item_embeddings \
  python3 -m src.train.extract_embeddings \
    --data "${DATA_DIR}/*.parquet" \
    --text-cols "${TEXT_COLS}" \
    --model "${TEXT_MODEL}" \
    --batch-size "${EMB_BATCH_SIZE}" \
    --max-len "${MAX_LEN}" \
    --normalize \
    --out "${EMB_DIR}/items_emb.pt"

run train_rqkmeans \
  python3 -m src.train.train_tokenizer \
    --method rqkmeans \
    --in "${EMB_DIR}/train_emb.pt" \
    --out "${RQK_DIR}" \
    --L "${RQ_L}" \
    --K "${RQ_K}" \
    --max-train "${MAX_TRAIN}" \
    --seed "${SEED}" \
    --encode-batch-size "${RQ_ENCODE_BATCH_SIZE}"

run encode_rqkmeans_items \
  python3 -m src.train.train_tokenizer \
    --method rqkmeans \
    --no-train \
    --encode "${EMB_DIR}/items_emb.pt" \
    --out "${RQK_DIR}" \
    --load "${RQK_DIR}/codebooks.npz" \
    --encode-batch-size "${RQ_ENCODE_BATCH_SIZE}"

run train_rqvae \
  python3 -m src.train.train_tokenizer \
    --method rqvae \
    --in "${EMB_DIR}/train_emb.pt" \
    --out "${RQVAE_DIR}" \
    --L "${RQ_L}" \
    --K "${RQ_K}" \
    --init "${RQK_DIR}/codebooks.npz" \
    --epochs "${RQ_EPOCHS}" \
    --batch-size "${RQ_BATCH_SIZE}" \
    --encode-batch-size "${RQ_ENCODE_BATCH_SIZE}" \
    --dist-chunk "${RQ_DIST_CHUNK}" \
    --entropy-reg "${RQ_ENTROPY_REG}" \
    --device "${DEVICE}"

run encode_rqvae_items \
  python3 -m src.train.train_tokenizer \
    --method rqvae \
    --no-train \
    --encode "${EMB_DIR}/items_emb.pt" \
    --out "${RQVAE_DIR}" \
    --load "${RQVAE_DIR}/codebooks.npz" \
    --encode-batch-size "${RQ_ENCODE_BATCH_SIZE}" \
    --dist-chunk "${RQ_DIST_CHUNK}" \
    --device "${DEVICE}"

if [[ "${RUN_VQ}" == "1" ]]; then
  run train_vqvae \
    python3 -m src.train.train_tokenizer \
      --method vqvae \
      --in "${EMB_DIR}/train_emb.pt" \
      --out "${VQVAE_DIR}" \
      --K "${VQ_K}" \
      --epochs "${VQ_EPOCHS}" \
      --batch-size "${VQ_BATCH_SIZE}" \
      --encode-batch-size "${VQ_ENCODE_BATCH_SIZE}" \
      --dist-chunk "${VQ_DIST_CHUNK}" \
      --device "${DEVICE}"

  run encode_vqvae_items \
    python3 -m src.train.train_tokenizer \
      --method vqvae \
      --no-train \
      --encode "${EMB_DIR}/items_emb.pt" \
      --out "${VQVAE_DIR}" \
      --load "${VQVAE_DIR}/codebooks.npz" \
      --encode-batch-size "${VQ_ENCODE_BATCH_SIZE}" \
      --dist-chunk "${VQ_DIST_CHUNK}" \
      --device "${DEVICE}"
fi

run export_rqkmeans_codes \
  python3 -m src.train.export_codes \
    --codes "${RQK_DIR}/codes.npy" \
    --ids "${EMB_DIR}/items_emb.ids.npy" \
    --out "${RQK_DIR}/items.parquet" \
    --concat-col

run export_rqvae_codes \
  python3 -m src.train.export_codes \
    --codes "${RQVAE_DIR}/codes.npy" \
    --ids "${EMB_DIR}/items_emb.ids.npy" \
    --out "${RQVAE_DIR}/items.parquet" \
    --concat-col

if [[ "${RUN_VQ}" == "1" ]]; then
  run export_vqvae_codes \
    python3 -m src.train.export_codes \
      --codes "${VQVAE_DIR}/codes.npy" \
      --ids "${EMB_DIR}/items_emb.ids.npy" \
      --out "${VQVAE_DIR}/items.parquet"
fi

run build_rqkmeans_inverted \
  python3 -m src.retrieval.build_inverted \
    --items "${RQK_DIR}/items.parquet" \
    --outdir "${RQK_DIR}"

run build_rqvae_inverted \
  python3 -m src.retrieval.build_inverted \
    --items "${RQVAE_DIR}/items.parquet" \
    --outdir "${RQVAE_DIR}"

if [[ "${RUN_VQ}" == "1" ]]; then
  run build_vqvae_inverted \
    python3 -m src.retrieval.build_inverted \
      --items "${VQVAE_DIR}/items.parquet" \
      --outdir "${VQVAE_DIR}"

VQVAE_DIR="${VQVAE_DIR}" python3 - <<'PY'
import os
import pandas as pd
path = os.path.join(os.environ["VQVAE_DIR"], "items.parquet")
df = pd.read_parquet(path)
vc = df["token"].value_counts()
expected_k = int(os.environ.get("VQ_K", "4096"))
print("VQ token health:")
print("  unique_tokens:", int(vc.size))
print("  min/median/mean/max:", int(vc.min()), float(vc.median()), float(vc.mean()), int(vc.max()))
if vc.size < 0.1 * expected_k:
    print("  note: check the bucket distribution above; very low unique token count may indicate VQ collapse.")
PY
fi

run build_faiss \
  python3 -m src.retrieval.build_faiss \
    --emb "${EMB_DIR}/items_emb.pt" \
    --ids "${EMB_DIR}/items_emb.ids.npy" \
    --outdir "${FAISS_DIR}" \
    --normalize

METHODS=("$(basename "${RQK_DIR}")" "$(basename "${RQVAE_DIR}")")
if [[ "${RUN_VQ}" == "1" ]]; then
  METHODS+=("$(basename "${VQVAE_DIR}")")
fi

EVAL_CMD=(
  python3 -m src.retrieval.eval_offline
  --train "${DATA_DIR}/train.parquet"
  --valid "${DATA_DIR}/valid.parquet"
  --test "${DATA_DIR}/test.parquet"
  --items-emb "${EMB_DIR}/items_emb.pt"
  --items-ids "${EMB_DIR}/items_emb.ids.npy"
  --faiss-index "${FAISS_DIR}/index.faiss"
  --faiss-ids "${FAISS_DIR}/ids.npy"
  --methods "${METHODS[@]}"
  --outputs-root "${OUT_ROOT}"
  --expand-rq-M "${EXPAND_RQ_M}"
  --expand-vq-T "${EXPAND_VQ_T}"
  --max-test "${EVAL_MAX_TEST}"
  --out "${RESULT_CSV}"
)

if [[ "${USE_FAISS_RERANK}" == "1" ]]; then
  EVAL_CMD+=(--use-faiss-rerank)
fi
if [[ "${ALLOW_FULL_SEARCH_FALLBACK}" == "1" ]]; then
  EVAL_CMD+=(--allow-full-search-fallback)
fi

run eval_offline "${EVAL_CMD[@]}"

echo
echo "Done."
echo "Result CSV: ${RESULT_CSV}"
echo "Logs: ${LOG_DIR}"
