# src/retrieval/build_faiss.py
# -*- coding: utf-8 -*-
import argparse, os, numpy as np, torch
from pathlib import Path

def load_emb(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "emb" in obj:
        return obj["emb"].float().numpy()
    if isinstance(obj, torch.Tensor):
        return obj.float().numpy()
    raise ValueError("items_emb.pt should be a dict with key 'emb' or a Tensor.")

def main():
    ap = argparse.ArgumentParser(description="Build FAISS index for item embeddings")
    ap.add_argument("--emb", required=True, help="Path to items_emb.pt")
    ap.add_argument("--ids", required=True, help="Path to items_emb.ids.npy")
    ap.add_argument("--outdir", required=True, help="Output dir")
    ap.add_argument("--normalize", action="store_true", help="L2 normalize (cosine)")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    X = load_emb(args.emb)  # [N,d]
    ids = np.load(args.ids, allow_pickle=True).astype(str)
    assert X.shape[0] == len(ids), "emb / ids length mismatch"

    if args.normalize:
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        X = X / n

    # ---- 尝试使用 FAISS；失败则回退为 NumPy 数组落盘 ----
    try:
        import faiss
        d = X.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(X.astype(np.float32))
        faiss.write_index(index, os.path.join(args.outdir, "index.faiss"))
        np.save(os.path.join(args.outdir, "ids.npy"), ids)
        print(f"Saved FAISS -> {os.path.join(args.outdir, 'index.faiss')}  (N={len(ids)}, d={d})")
    except Exception as e:
        # 回退：保存原始数组 + 哨兵文件
        np.save(os.path.join(args.outdir, "X.npy"), X.astype(np.float32))
        np.save(os.path.join(args.outdir, "ids.npy"), ids)
        with open(os.path.join(args.outdir, "NO_FAISS"), "w") as f:
            f.write(str(e))
        print("[WARN] FAISS unavailable; saved raw arrays for fallback.")
        print("Reason:", e)

if __name__ == "__main__":
    main()
