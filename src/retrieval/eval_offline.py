# src/retrieval/eval_offline.py
# -*- coding: utf-8 -*-
import argparse, os, pickle, math, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def l2norm(x, axis=1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def load_items_emb(emb_path, ids_path, normalize=True):
    obj = torch.load(emb_path, map_location="cpu")
    X = (obj["emb"] if isinstance(obj, dict) else obj).float().numpy()
    ids = np.load(ids_path, allow_pickle=True).astype(str)
    assert X.shape[0] == len(ids)
    if normalize:
        X = l2norm(X, axis=1)
    id2idx = {i: k for k, i in enumerate(ids)}
    return X, ids, id2idx

def vq_quantize(z, codebook):  # z: [d], codebook: [K,d]
    # 返回 int
    z = z[None, :]
    c_norm = (codebook**2).sum(1)[None, :]
    z_norm = (z**2).sum(1, keepdims=True)
    dists = z_norm - 2.0 * (z @ codebook.T) + c_norm
    return int(dists.argmin(axis=1)[0])

def rq_quantize(z, codebooks):  # codebooks: [L,K,d]
    residual = z.copy()
    codes = []
    for l in range(codebooks.shape[0]):
        C = codebooks[l]
        c_norm = (C**2).sum(1)[None, :]
        r = residual[None, :]
        r_norm = (r**2).sum(1, keepdims=True)
        dists = r_norm - 2.0 * (r @ C.T) + c_norm
        idx = int(dists.argmin(axis=1)[0])
        codes.append(idx)
        residual = residual - C[idx]
    return codes  # list[int] len=L

def code_key_from_codes(codes):
    if isinstance(codes, (list, tuple)):
        return "-".join(str(int(x)) for x in codes)
    return str(int(codes))

def build_user_profiles(train_df, valid_df, id2idx, X, id_col="item_id", user_col="user_id"):
    df_hist = pd.concat([train_df[[user_col, id_col]], valid_df[[user_col, id_col]]], ignore_index=True)
    df_hist[id_col] = df_hist[id_col].astype(str)
    users = df_hist[user_col].unique().tolist()
    profiles = {}
    for u, g in df_hist.groupby(user_col):
        idxs = [id2idx[i] for i in g[id_col].tolist() if i in id2idx]
        if not idxs:
            continue
        v = X[idxs].mean(axis=0)
        profiles[u] = v / (np.linalg.norm(v) + 1e-9)
    return profiles

def ndcg_at_k(rank, k):
    if rank is None or rank >= k:
        return 0.0
    return 1.0 / math.log2(rank + 2.0)  # rank从0开始

def evaluate_method(
    name, codebooks, inverted, test_df, profiles, X, id2idx, faiss_index=None, faiss_ids=None,
    topk_list=(10,50,100), use_faiss_rerank=False
):
    is_vq = (codebooks.ndim == 2)  # [K,d]
    L = None if is_vq else codebooks.shape[0]

    results = {f"Recall@{k}": 0.0 for k in topk_list}
    results.update({f"NDCG@{k}": 0.0 for k in topk_list})
    n_eval = 0
    n_cold = 0
    cand_sizes = []

    import faiss

    for _, row in test_df.iterrows():
        u = row["user_id"]
        gt = str(row["item_id"])
        if u not in profiles or gt not in id2idx:
            n_cold += 1
            continue
        z_u = profiles[u]  # [d]

        # 1) 候选：倒排
        if is_vq:
            tok = vq_quantize(z_u, codebooks)
            key = code_key_from_codes(tok)
        else:
            codes = rq_quantize(z_u, codebooks)
            key = code_key_from_codes(codes)
        cands = inverted.get(key, [])

        # 2) 重排
        topk = max(topk_list)
        if len(cands) == 0:
            # 回退：FAISS 全库近邻（Approximate）
            if faiss_index is None:
                # 退而求其次：全量点积（慢，但最简）
                scores = X @ z_u
                idxs = np.argsort(-scores)[:topk]
                ranked_ids = [faiss_ids[i] if faiss_ids is not None else None for i in idxs]
                # 如果没有 faiss_ids，直接从 id2idx 的 keys 无序拿不到，还是建议提前传 faiss 索引
            else:
                q = z_u.astype(np.float32)[None, :]
                D, I = faiss_index.search(q, topk)
                ranked_ids = [str(faiss_ids[i]) for i in I[0]]
        else:
            cand_sizes.append(len(cands))
            cand_idx = [id2idx[i] for i in cands if i in id2idx]
            if not cand_idx:
                # 极端情况：倒排有id但无向量（不太可能）
                if faiss_index is None:
                    scores = X @ z_u
                    idxs = np.argsort(-scores)[:topk]
                    ranked_ids = [faiss_ids[i] if faiss_ids is not None else None for i in idxs]
                else:
                    q = z_u.astype(np.float32)[None, :]
                    D, I = faiss_index.search(q, topk)
                    ranked_ids = [str(faiss_ids[i]) for i in I[0]]
            else:
                V = X[cand_idx]  # [C,d]
                if use_faiss_rerank and len(cand_idx) > 1:
                    # 临时建一个小索引做“FAISS重排”
                    d = V.shape[1]
                    sub = faiss.IndexFlatIP(d)
                    sub.add(V.astype(np.float32))
                    q = z_u.astype(np.float32)[None, :]
                    D, I = sub.search(q, min(topk, len(cand_idx)))
                    take = [cand_idx[i] for i in I[0]]
                else:
                    # 直接点积重排
                    scores = V @ z_u
                    order = np.argsort(-scores)[:topk]
                    take = [cand_idx[i] for i in order]
                ranked_ids = [list(id2idx.keys())[list(id2idx.values()).index(i)] for i in take]  # 慢但简洁
                # 更快：直接由 ids 数组取：我们没有 ids 数组？可以不转换，后面比较用 id2idx 位置也行

        # 3) 计算指标
        try:
            rank = ranked_ids.index(gt)
        except ValueError:
            rank = None
        for k in topk_list:
            if rank is not None and rank < k:
                results[f"Recall@{k}"] += 1.0
                results[f"NDCG@{k}"] += ndcg_at_k(rank, k)
        n_eval += 1

    if n_eval == 0:
        raise RuntimeError("No eligible test samples (all cold-start).")

    for k in topk_list:
        results[f"Recall@{k}"] /= n_eval
        results[f"NDCG@{k}"] /= n_eval

    results.update({
        "evaluated": n_eval,
        "cold_start_skipped": n_cold,
        "avg_cand_size": (sum(cand_sizes)/len(cand_sizes) if cand_sizes else 0.0),
    })
    return results

def main():
    ap = argparse.ArgumentParser(description="Offline eval: inverted candidates + FAISS rerank")
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--items-emb", required=True)
    ap.add_argument("--items-ids", required=True)
    ap.add_argument("--faiss-index", default="./outputs/faiss/index.faiss")
    ap.add_argument("--faiss-ids",    default="./outputs/faiss/ids.npy")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="Method names under outputs/, e.g. rqkmeans_4x256 rqvae_4x256 vqvae_4096")
    ap.add_argument("--use-faiss-rerank", action="store_true")
    ap.add_argument("--max-test", type=int, default=None, help="Subsample test rows for quick run")
    ap.add_argument("--out", default="./results/offline_eval.csv")
    args = ap.parse_args()

    # 数据
    train = pd.read_parquet(args.train)
    valid = pd.read_parquet(args.valid)
    test = pd.read_parquet(args.test)
    if args.max_test and len(test) > args.max_test:
        test = test.sample(args.max_test, random_state=42).reset_index(drop=True)

    # 向量与映射
    X, ids, id2idx = load_items_emb(args.items_emb, args.items_ids, normalize=True)

    # 用户画像
    profiles = build_user_profiles(train, valid, id2idx, X)

    # FAISS（全库回退）
    faiss_index = None
    faiss_ids = None
    if os.path.exists(args.faiss_index):
        import faiss
        faiss_index = faiss.read_index(args.faiss_index)
        faiss_ids = np.load(args.faiss_ids, allow_pickle=True)

    # 评测每个方法
    rows = []
    for name in args.methods:
        base = os.path.join("outputs", name)
        inv_path = os.path.join(base, "inverted.pkl")
        cb_path  = os.path.join(base, "codebooks.npz")

        if not (os.path.exists(inv_path) and os.path.exists(cb_path)):
            print(f"[WARN] skip {name}: missing inverted or codebooks.")
            continue

        with open(inv_path, "rb") as f:
            inverted = pickle.load(f)
        codebooks = np.load(cb_path)["codebooks"]  # [L,K,d] or [K,d]

        res = evaluate_method(
            name=name,
            codebooks=codebooks,
            inverted=inverted,
            test_df=test,
            profiles=profiles,
            X=X,
            id2idx=id2idx,
            faiss_index=faiss_index,
            faiss_ids=faiss_ids,
            use_faiss_rerank=args.use_faiss_rerank,
        )
        print(f"{name}: {json.dumps(res, indent=2)}")
        row = {"method": name}
        row.update(res)
        rows.append(row)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()
