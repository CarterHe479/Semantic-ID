# src/train/train_tokenizer.py
# -*- coding: utf-8 -*-
"""
Usage examples:

# RQ-KMeans（也可作 RQ-VAE 初始化）
python -m src.train.train_tokenizer \
  --method rqkmeans --in ./outputs/emb/train_emb.pt \
  --out ./outputs/rqkmeans_4x256 --L 4 --K 256 --max-train 200000

# VQ-VAE（EMA 版；单码本）
python -m src.train.train_tokenizer \
  --method vqvae --in ./outputs/emb/train_emb.pt \
  --out ./outputs/vqvae_4096 --K 4096 --epochs 3 --batch-size 8192

# RQ-VAE（建议用 RQ-KMeans 初始化）
python -m src.train.train_tokenizer \
  --method rqvae --in ./outputs/emb/train_emb.pt \
  --out ./outputs/rqvae_4x256 --L 4 --K 256 --epochs 5 --init ./outputs/rqkmeans_4x256/codebooks.npz

# 只编码另一份全量向量（例如 items_emb.pt），不再训练：
python -m src.train.train_tokenizer \
  --method rqvae --encode ./outputs/emb/items_emb.pt \
  --out ./outputs/rqvae_4x256 --L 4 --K 256 --load ./outputs/rqvae_4x256/codebooks.npz --no-train
"""
import argparse
import json
import os
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 依赖你此前放在 src/models 下的实现
from src.models.RQ_KMeans import ResidualKMeans
from src.models.VQ_VAE import VectorQuantizerEMA
from src.models.RQ_VAE import ResidualQuantizer


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_embeddings(path: str, max_rows: Optional[int] = None) -> torch.Tensor:
    """
    支持 .pt / .npy / .npz
    返回 float32 的 [N, d] torch.Tensor (CPU)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            Z = obj
        elif isinstance(obj, dict):
            # 常见键名
            for k in ["emb", "Z", "embeddings", "features"]:
                if k in obj:
                    Z = obj[k]
                    break
            else:
                # 尝试取第一个 tensor
                Z = next((v for v in obj.values() if isinstance(v, torch.Tensor)), None)
                if Z is None:
                    raise ValueError(f"Cannot find tensor in dict from {path}")
        else:
            raise ValueError(f"Unsupported .pt content type: {type(obj)}")
        Z = Z.detach().cpu().float()
    elif ext == ".npy":
        Z = torch.from_numpy(np.load(path)).float()
    elif ext == ".npz":
        npz = np.load(path)
        if "emb" in npz:
            arr = npz["emb"]
        elif "Z" in npz:
            arr = npz["Z"]
        elif "arr_0" in npz:
            arr = npz["arr_0"]
        else:
            raise ValueError(f"Unsupported .npz keys: {list(npz.keys())}")
        Z = torch.from_numpy(arr).float()
    else:
        raise ValueError(f"Unsupported embedding file extension: {ext}")
    if Z.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {tuple(Z.shape)} from {path}")
    if max_rows is not None and Z.shape[0] > max_rows:
        Z = Z[:max_rows]
    return Z.contiguous()


def save_codebooks_npz(out_dir: str, arr: np.ndarray):
    """
    arr:
      - RQ 系: [L, K, d]
      - VQ 系: [K, d]
    """
    np.savez(os.path.join(out_dir, "codebooks.npz"), codebooks=arr)


def save_codes(out_dir: str, codes: np.ndarray):
    np.save(os.path.join(out_dir, "codes.npy"), codes)


def write_metrics(out_dir: str, metrics: Dict):
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def batch_iterator(Z: torch.Tensor, batch_size: int, shuffle: bool = True):
    ds = TensorDataset(Z)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# -------------------------
# Trainers
# -------------------------
def train_vqvae(
    Z: torch.Tensor,
    K: int = 4096,
    beta: float = 0.25,
    decay: float = 0.99,
    epochs: int = 3,
    batch_size: int = 8192,
    device: str = "cpu",
    k_chunk: int = 1024,
) -> Tuple[VectorQuantizerEMA, Dict]:
    """
    轻量 VQ-VAE 训练：仅量化器 + 承诺损失 + (z_q 与 z_e 的重构 MSE)
    采用 EMA 更新码本，无需 optimizer。
    """
    # quant = VectorQuantizerEMA(K=K, d=Z.shape[1], beta=beta, decay=decay).to(device)
    quant = VectorQuantizerEMA(K=K, d=Z.shape[1], beta=beta, decay=decay, k_chunk=k_chunk).to(device)
    quant.train()
    metrics = {"epoch_losses": []}

    for ep in range(epochs):
        running = 0.0
        n = 0
        for (batch,) in batch_iterator(Z, batch_size, shuffle=True):
            batch = batch.to(device, non_blocking=True)
            z_q, idx, commit = quant(batch)
            recon = F.mse_loss(z_q, batch)  # 直接以 z_e 为重构目标
            loss = recon + commit
            # EMA 版本不需要 optimizer；前向里已更新 codebook
            running += loss.item() * batch.size(0)
            n += batch.size(0)
        metrics["epoch_losses"].append(running / max(n, 1))
        print(f"[VQ-VAE] epoch {ep+1}/{epochs} loss={metrics['epoch_losses'][-1]:.6f}")

    # 导出 codebook
    codebook = quant.codebook.detach().cpu().numpy()
    return quant, {"loss": metrics["epoch_losses"][-1], "codebook": codebook}


@torch.no_grad()
def encode_vqvae(quant: VectorQuantizerEMA, Z: torch.Tensor, device: str = "cpu") -> np.ndarray:
    quant.eval()
    quant.to(device)  # <- 新增
    all_idx = []
    for (batch,) in batch_iterator(Z, batch_size=8192, shuffle=False):
        batch = batch.to(device)
        _, idx, _ = quant(batch)
        all_idx.append(idx.detach().cpu().numpy()[:, None])  # N×1
    return np.concatenate(all_idx, axis=0)


def train_rqvae(
    Z: torch.Tensor,
    L: int = 4,
    K: int = 256,
    beta: float = 0.25,
    entropy_reg: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 1e-3,               # 保留签名但不用
    device: str = "cpu",
    init_codebooks: Optional[np.ndarray] = None,  # [L, K, d]
    k_chunk: int = 1024,
) -> Tuple[ResidualQuantizer, Dict]:
    rq = ResidualQuantizer(L=L, K=K, d=Z.shape[1], beta=beta,
                           entropy_reg=entropy_reg, k_chunk=k_chunk).to(device)
    if init_codebooks is not None:
        rq.init_from_rqkmeans(init_codebooks)

    rq.train()
    metrics = {"epoch_losses": []}

    # EMA 版：不需要反传/优化器
    for ep in range(epochs):
        running = 0.0
        n = 0
        with torch.no_grad():
            for (batch,) in batch_iterator(Z, batch_size, shuffle=True):
                batch = batch.to(device)
                z_q, ids, commit, reg = rq(batch)
                recon = F.mse_loss(z_q, batch)
                loss = recon + commit + reg
                running += float(loss.item()) * batch.size(0)
                n += batch.size(0)

        avg = running / max(n, 1)
        metrics["epoch_losses"].append(avg)
        print(f"[RQ-VAE/EMA] epoch {ep+1}/{epochs} loss={avg:.6f}")

    # 导出 codebooks
    with torch.no_grad():
        stacks = []
        for cb in rq.codebooks:
            stacks.append(cb.detach().cpu().numpy()[None, ...])  # 1×K×d
        codebooks = np.concatenate(stacks, axis=0)  # L×K×d
    # return rq, {"loss": metrics["epoch_losses"][-1], "codebooks": codebooks}
    last = metrics["epoch_losses"][-1] if metrics["epoch_losses"] else None
    return rq, {"loss": last, "codebooks": codebooks}




@torch.no_grad()
def encode_rq(rq: ResidualQuantizer, Z: torch.Tensor, device: str = "cpu") -> np.ndarray:
    rq.eval()
    L = rq.L
    codes = []
    for (batch,) in batch_iterator(Z, batch_size=8192, shuffle=False):
        batch = batch.to(device)
        residual = batch
        codes_b = []
        for C in rq.codebooks:
            # 最近邻索引
            dist = (residual.pow(2).sum(1, keepdim=True) - 2 * residual @ C.t() + C.pow(2).sum(1))
            idx = torch.argmin(dist, dim=1)
            q = C[idx]
            residual = residual - q
            codes_b.append(idx.detach().cpu().numpy())
        codes.append(np.stack(codes_b, axis=1))  # (L, N_b) → N_b×L
    return np.concatenate(codes, axis=0)


def fit_rqkmeans(Z: torch.Tensor, L: int = 4, K: int = 256, random_state: int = 0) -> Tuple[ResidualKMeans, np.ndarray]:
    rk = ResidualKMeans(L=L, K=K, random_state=random_state)
    rk.fit(Z.numpy())
    # 导出 codebooks
    C_list = []
    for C in rk.codebooks:
        C_list.append(C[None, ...])  # 1×K×d
    codebooks = np.concatenate(C_list, axis=0)  # L×K×d
    return rk, codebooks


def encode_rqkmeans(rk: ResidualKMeans, Z: torch.Tensor, batch_size: int = 2048) -> np.ndarray:
    return rk.encode(Z.numpy(), batch_size=batch_size)  # N×L


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train/fit tokenizer and export semantic IDs.")
    parser.add_argument("--method", type=str, required=True, choices=["rqkmeans", "vqvae", "rqvae"])
    parser.add_argument("--in", dest="train_in", type=str, default=None, help="Embeddings for training.")
    parser.add_argument("--encode", type=str, default=None, help="Embeddings file to encode/export codes (default: use --in).")
    parser.add_argument("--out", type=str, required=True, help="Output directory.")
    parser.add_argument("--load", type=str, default=None, help="Load codebooks from npz (for encoding or continuing).")
    parser.add_argument("--init", type=str, default=None, help="Initializer codebooks npz for RQ-VAE (from RQ-KMeans).")
    parser.add_argument("--max-train", type=int, default=None, help="Subsample rows for training.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--entropy-reg", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=0.99, help="EMA decay for VQ-VAE.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-train", action="store_true", help="Skip training and only encode using --load.")
    parser.add_argument("--dist-chunk", type=int, default=1024,
    help="Chunk size along K when computing distances to reduce memory.")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out)

    # 设备
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # ---------------------
    # 载入数据
    # ---------------------
    Z_train = None
    if not args.no_train:
        if not args.train_in:
            raise ValueError("--in is required unless --no-train is set.")
        Z_train = load_embeddings(args.train_in, max_rows=args.max_train)
        print(f"Loaded train embeddings from {args.train_in}, shape={tuple(Z_train.shape)}")

    encode_path = args.encode or args.train_in
    if encode_path is None:
        raise ValueError("No file to encode. Provide --encode or --in.")
    Z_encode = load_embeddings(encode_path)
    print(f"Loaded encode embeddings from {encode_path}, shape={tuple(Z_encode.shape)}")

    metrics = {}

    # ---------------------
    # 方法分支
    # ---------------------    
    if args.method == "rqkmeans":
        if args.no_train and args.load:
            # 仅编码：加载 RQ-KMeans 的 L×K×d 码本
            codebooks = np.load(args.load)["codebooks"]  # [L, K, d]
            # 构造 ResidualKMeans 容器并注入码本（CPU/NumPy 实现，无需 device）
            rk = ResidualKMeans(L=int(codebooks.shape[0]), K=int(codebooks.shape[1]))
            rk.codebooks = [codebooks[i].astype(np.float32, copy=False) for i in range(codebooks.shape[0])]

            # 形状自检（可留可删）
            d_cb = int(codebooks.shape[2])
            d_z  = int(Z_encode.shape[1])
            if d_cb != d_z:
                raise ValueError(f"Codebook dim (d={d_cb}) != embedding dim (d={d_z}). "
                                f"请确认训练的嵌入维度与当前编码用的向量一致。")
        else:
            if Z_train is None:
                raise ValueError("rqkmeans requires training data unless --no-train with --load is used.")
            rk, codebooks = fit_rqkmeans(Z_train, L=args.L, K=args.K, random_state=args.seed)
            metrics["train_size"] = int(Z_train.shape[0])
            save_codebooks_npz(args.out, codebooks)
            print(f"Saved codebooks -> {os.path.join(args.out, 'codebooks.npz')}")

        # 编码全量 items（小批量，低内存）
        codes = encode_rqkmeans(rk, Z_encode, batch_size=2048)
        save_codes(args.out, codes)
        metrics.update({
            "L": int(codebooks.shape[0]),
            "K": int(codebooks.shape[1]),
            "embedding_dim": int(codebooks.shape[2]),
            "encoded": int(codes.shape[0]),
        })
        write_metrics(args.out, metrics)
        print(f"Saved codes -> {os.path.join(args.out, 'codes.npy')} shape={codes.shape}")


    elif args.method == "vqvae":
        if args.no_train and args.load:
            # 仅编码：加载 codebook
            npz = np.load(args.load)
            codebook = npz["codebooks"]  # [K, d]
            quant = VectorQuantizerEMA(K=codebook.shape[0], d=codebook.shape[1])
            with torch.no_grad():
                quant.codebook.data.copy_(torch.tensor(codebook))
        else:
            if Z_train is None:
                raise ValueError("vqvae requires training data unless --no-train with --load is used.")
            quant, train_out = train_vqvae(
                Z_train, K=args.K, beta=args.beta, decay=args.decay,
                epochs=args.epochs, batch_size=args.batch_size, device=device, k_chunk=args.dist_chunk
            )
            codebook = train_out["codebook"]
            metrics["loss"] = float(train_out["loss"])
            save_codebooks_npz(args.out, codebook)
            print(f"Saved codebook -> {os.path.join(args.out, 'codebooks.npz')}")

        # 构造量化器用于编码        
        if "quant" not in locals():
            quant = VectorQuantizerEMA(K=codebook.shape[0], d=codebook.shape[1]).to(device)   # <- 加 .to(device)
            with torch.no_grad():
                quant.codebook.data.copy_(torch.tensor(codebook, device=device))              # <- 指定 device


        codes = encode_vqvae(quant, Z_encode, device=device)
        save_codes(args.out, codes)
        metrics.update({"K": int(quant.codebook.shape[0]), "embedding_dim": int(quant.codebook.shape[1]), "encoded": int(codes.shape[0])})
        write_metrics(args.out, metrics)
        print(f"Saved codes -> {os.path.join(args.out, 'codes.npy')} shape={codes.shape}")

    elif args.method == "rqvae":
        # 准备初始化或加载
        init_cbs = None
        if args.init:
            init_npz = np.load(args.init)
            init_cbs = init_npz["codebooks"]  # [L, K, d]
        if args.no_train and args.load:
            loaded_cbs = np.load(args.load)["codebooks"]  # [L, K, d]
            rq = ResidualQuantizer(
                L=loaded_cbs.shape[0],
                K=loaded_cbs.shape[1],
                d=loaded_cbs.shape[2],
                k_chunk=args.dist_chunk
            ).to(device)
            with torch.no_grad():
                rq.init_from_rqkmeans(loaded_cbs)
        else:
            if Z_train is None:
                raise ValueError("rqvae requires training data unless --no-train with --load is used.")
            rq, train_out = train_rqvae(
                Z_train, L=args.L, K=args.K, beta=args.beta, entropy_reg=args.entropy_reg,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                device=device, init_codebooks=init_cbs, k_chunk=args.dist_chunk
            )
            codebooks = train_out["codebooks"]
            metrics["loss"] = float(train_out["loss"])
            save_codebooks_npz(args.out, codebooks)
            print(f"Saved codebooks -> {os.path.join(args.out, 'codebooks.npz')}")

        # 编码
        if "rq" not in locals():
            # 用加载的 codebooks 构建 rq（无需训练）
            loaded = np.load(args.load)["codebooks"]
            rq = ResidualQuantizer(L=loaded.shape[0], K=loaded.shape[1], d=loaded.shape[2], k_chunk=args.dist_chunk)
            with torch.no_grad():
                for i, cb in enumerate(rq.codebooks):
                    cb.copy_(torch.tensor(loaded[i]))
            rq = rq.to(device)

        codes = encode_rq(rq, Z_encode, device=device)
        save_codes(args.out, codes)
        # 导出 L/K/d
        with torch.no_grad():
            L = rq.L
            K = rq.codebooks[0].shape[0]
            d = rq.codebooks[0].shape[1]
        metrics.update({"L": int(L), "K": int(K), "embedding_dim": int(d), "encoded": int(codes.shape[0])})
        write_metrics(args.out, metrics)
        print(f"Saved codes -> {os.path.join(args.out, 'codes.npy')} shape={codes.shape}")

    else:
        raise ValueError(f"Unknown method: {args.method}")

    print("Done.")


if __name__ == "__main__":
    main()
