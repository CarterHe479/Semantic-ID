# src/train/export_codes.py
# -*- coding: utf-8 -*-
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def _load_ids(ids_path: str):
    ext = os.path.splitext(ids_path)[1].lower()
    if ext == ".npy":
        ids = np.load(ids_path, allow_pickle=True).astype(str).tolist()
        return ids
    if ext == ".pt":
        obj = torch.load(ids_path, map_location="cpu")
        if isinstance(obj, dict) and "ids" in obj:
            return [str(x) for x in obj["ids"]]
        raise ValueError("For .pt, expect a dict with key 'ids'.")
    raise ValueError("Unsupported ids format. Use .npy (recommended) or .pt with {'ids': ...}.")

def main():
    ap = argparse.ArgumentParser(description="Export item_id + codes to parquet")
    ap.add_argument("--codes", type=str, required=True, help="Path to codes.npy (N x L or N x 1)")
    ap.add_argument("--ids", type=str, required=True,
                    help="Path to ids.npy (from extract_embeddings) or the .pt that contains {'ids': ...}")
    ap.add_argument("--out", type=str, required=True, help="Output parquet path")
    ap.add_argument("--id-col", type=str, default="item_id")
    ap.add_argument("--concat-col", action="store_true",
                    help="Also save a code_str column like 'c1-c2-...-cL'")
    args = ap.parse_args()

    codes = np.load(args.codes)
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)
    N, L = codes.shape
    ids = _load_ids(args.ids)
    if len(ids) != N:
        raise ValueError(f"ids length ({len(ids)}) != codes rows ({N}). "
                         "Make sure you used the same source order for embeddings and codes.")

    data = {args.id_col: ids}
    if L == 1:
        data["token"] = codes[:, 0].astype(np.int64)
    else:
        for i in range(L):
            data[f"c{i+1}"] = codes[:, i].astype(np.int64)
        if args.concat_col:
            data["code_str"] = ["-".join(map(str, row)) for row in codes.tolist()]

    df = pd.DataFrame(data)
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved -> {args.out}, shape={df.shape}")

if __name__ == "__main__":
    main()
