# src/train/extract_embeddings.py
# -*- coding: utf-8 -*-
import argparse, os
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def _to_text(v) -> str:
    """把任意类型（None/NaN/list/ndarray/dict/bytes/str）安全地转为干净字符串。"""
    import numpy as np
    import pandas as pd

    if v is None:
        return ""
    # NaN
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass

    # bytes
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode("utf-8", errors="ignore")
        except Exception:
            v = str(v)

    # ndarray / list / tuple
    if isinstance(v, (list, tuple, np.ndarray)):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        parts = []
        for x in v:
            sx = _to_text(x)
            if sx:
                parts.append(sx)
        return " ".join(parts).strip()

    # dict：尝试抓常见字段
    if isinstance(v, dict):
        for k in ("text", "title", "description", "summary"):
            if k in v:
                s = _to_text(v[k])
                if s:
                    return s
        return ""

    # 其他直接转字符串
    s = str(v).strip()
    return s


def _mean_pool(last_hidden_state, attention_mask):
    # masked mean pooling
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counted = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counted

def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _read_many(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if os.path.isdir(p):
            for fp in Path(p).glob("*.parquet"):
                dfs.append(pd.read_parquet(fp))
        elif "*" in p:
            for fp in Path().glob(p):
                dfs.append(pd.read_parquet(fp))
        else:
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise FileNotFoundError("No parquet files found for --data")
    return pd.concat(dfs, ignore_index=True)

def main():
    ap = argparse.ArgumentParser(description="Extract TEXT-only item embeddings to .pt")
    ap.add_argument("--data", type=str, required=True,
                    help="Parquet path(s). Comma-separated or glob or directory. "
                         "e.g., ./data/amazon23/train.parquet,./data/amazon23/valid.parquet")
    ap.add_argument("--id-col", type=str, default="item_id")
    ap.add_argument("--ts-col", type=str, default="timestamp",
                    help="If present, for duplicated items keep the latest by this column.")
    ap.add_argument("--text-cols", type=str, default="title,description",
                    help="Comma-separated text columns to concatenate.")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--normalize", action="store_true", help="L2-normalize output vectors.")
    ap.add_argument("--out", type=str, required=True, help="Output .pt path (directory will be created).")
    args = ap.parse_args()

    paths = [s for s in (args.data.split(",") if "," in args.data else [args.data])]
    df = _read_many(paths)

    # 仅保留有 id 的行
    if args.id_col not in df.columns:
        raise KeyError(f"Missing id column: {args.id_col}")
    df = df.dropna(subset=[args.id_col])

    # 去重策略：若有时间戳，取每个 item 最新的一条；否则取第一条
    if args.ts_col in df.columns:
        df[args.ts_col] = pd.to_numeric(df[args.ts_col], errors="coerce")
        df = df.sort_values(args.ts_col).groupby(args.id_col, as_index=False).tail(1)
    else:
        df = df.drop_duplicates(subset=[args.id_col], keep="first")

    # 组装文本
    # 组装文本（鲁棒处理 list/ndarray/dict/bytes）
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""

    def _join_text(row):
        chunks = []
        for c in text_cols:
            val = row[c]
            s = _to_text(val)
            if s:
                chunks.append(s)
        return " [SEP] ".join(chunks) if chunks else "[EMPTY]"

    df["__text__"] = df.apply(_join_text, axis=1)


    ids = df[args.id_col].astype(str).tolist()
    texts = df["__text__"].tolist()
    device = _auto_device()
    print(f"Loaded {len(texts)} unique items. Using device: {device}")

    # 模型
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModel.from_pretrained(args.model).to(device).eval()

    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i+args.batch_size]
            enc = tok(
                batch, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output
            else:
                vec = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            if args.normalize:
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            all_vecs.append(vec.cpu())
    emb = torch.cat(all_vecs, dim=0).contiguous()
    print(f"Embeddings shape: {tuple(emb.shape)}")

    # 保存
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    torch.save({"emb": emb, "ids": ids}, args.out)
    ids_npy = os.path.splitext(args.out)[0] + ".ids.npy"
    np.save(ids_npy, np.array(ids, dtype=object))
    print(f"Saved -> {args.out} and {ids_npy}")

if __name__ == "__main__":
    main()
