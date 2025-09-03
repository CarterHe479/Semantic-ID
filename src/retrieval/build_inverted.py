# src/retrieval/build_inverted.py
# -*- coding: utf-8 -*-
import argparse, os, json, pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

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
    code_str = infer_code_str(df)
    item_ids = df[args.id_col].astype(str).tolist()

    inv = defaultdict(list)
    for cid, iid in zip(code_str.tolist(), item_ids):
        inv[cid].append(iid)

    inv_path = os.path.join(args.outdir, "inverted.pkl")
    with open(inv_path, "wb") as f:
        pickle.dump(dict(inv), f, protocol=4)

    # 简要统计
    meta = {
        "items": len(item_ids),
        "unique_codes": len(inv),
        "avg_bucket": sum(len(v) for v in inv.values()) / max(1, len(inv)),
        "items_path": args.items,
    }
    with open(os.path.join(args.outdir, "inverted.meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved inverted -> {inv_path}")
    print("meta:", meta)

if __name__ == "__main__":
    main()
