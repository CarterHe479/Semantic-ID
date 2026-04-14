# src/retrieval/build_inverted.py
# -*- coding: utf-8 -*-
import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

def infer_code_str(df: pd.DataFrame) -> pd.Series:
    if "code_str" in df.columns:
        return df["code_str"].astype(str)
    if "token" in df.columns:  # VQ
        return df["token"].astype(int).astype(str)
    # RQ: 尝试 c1..cL
    cs = [c for c in df.columns if c.startswith("c") and c[1:].isdigit()]
    if not cs:
        raise ValueError("Cannot find code columns: need 'code_str' or 'token' or 'c1..cL'.")
    cs = sorted(cs, key=lambda x: int(x[1:]))
    return df[cs].astype(int).astype(str).agg("-".join, axis=1)


def infer_rq_code_cols(df: pd.DataFrame):
    cs = [c for c in df.columns if c.startswith("c") and c[1:].isdigit()]
    return sorted(cs, key=lambda x: int(x[1:]))


def build_rq_indexes(df: pd.DataFrame, id_col: str):
    code_cols = infer_rq_code_cols(df)
    if not code_cols:
        raise ValueError("RQ index requires c1..cL columns.")

    exact = defaultdict(list)
    prefixes = {level: defaultdict(list) for level in range(1, len(code_cols) + 1)}
    item_ids = df[id_col].astype(str).tolist()
    codes = df[code_cols].astype(int).astype(str)

    for row, iid in zip(codes.itertuples(index=False, name=None), item_ids):
        parts = list(row)
        exact["-".join(parts)].append(iid)
        for level in range(1, len(parts) + 1):
            prefixes[level]["-".join(parts[:level])].append(iid)

    return {
        "schema_version": 2,
        "type": "rq",
        "exact": dict(exact),
        "prefix": {level: dict(index) for level, index in prefixes.items()},
        "levels": len(code_cols),
    }


def build_vq_index(df: pd.DataFrame, id_col: str):
    code_str = infer_code_str(df)
    item_ids = df[id_col].astype(str).tolist()
    exact = defaultdict(list)
    for cid, iid in zip(code_str.tolist(), item_ids):
        exact[cid].append(iid)
    return {
        "schema_version": 2,
        "type": "vq",
        "exact": dict(exact),
    }


def exact_index_from_payload(payload):
    if isinstance(payload, dict) and payload.get("schema_version") == 2:
        return payload["exact"]
    return payload

def main():
    ap = argparse.ArgumentParser(description="Build inverted index from items.parquet")
    ap.add_argument("--items", required=True, help="Path to items.parquet (RQ/VQ 输出)")
    ap.add_argument("--outdir", required=True, help="Output dir for inverted index")
    ap.add_argument("--id-col", default="item_id")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.items)
    if args.id_col not in df.columns:
        raise KeyError(f"Missing id column {args.id_col}")
    item_ids = df[args.id_col].astype(str).tolist()
    if infer_rq_code_cols(df):
        payload = build_rq_indexes(df, args.id_col)
    else:
        payload = build_vq_index(df, args.id_col)
    inv = exact_index_from_payload(payload)

    inv_path = os.path.join(args.outdir, "inverted.pkl")
    with open(inv_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    # 简要统计
    meta = {
        "items": len(item_ids),
        "unique_codes": len(inv),
        "avg_bucket": sum(len(v) for v in inv.values()) / max(1, len(inv)),
        "items_path": args.items,
        "index_type": payload.get("type", "legacy") if isinstance(payload, dict) else "legacy",
    }
    if isinstance(payload, dict) and payload.get("type") == "rq":
        meta["prefix_unique_codes"] = {
            str(level): len(index) for level, index in payload["prefix"].items()
        }
        meta["prefix_avg_bucket"] = {
            str(level): (sum(len(v) for v in index.values()) / max(1, len(index)))
            for level, index in payload["prefix"].items()
        }
    with open(os.path.join(args.outdir, "inverted.meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved inverted -> {inv_path}")
    print("meta:", meta)

if __name__ == "__main__":
    main()
