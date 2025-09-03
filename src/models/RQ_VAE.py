# src/models/RQ_VAE.py
import torch, torch.nn as nn, torch.nn.functional as F

class ResidualQuantizer(nn.Module):
    """
    RQ-VAE 残差量化（低内存 + EMA 版）：
      - K 维分块计算最近邻，避免 B×K×d 广播
      - 码本用 EMA 更新（无需反传 / optimizer）
      - 使用批内直方图熵做均匀使用正则
    """
    def __init__(self, L=4, K=256, d=256, beta=0.25, entropy_reg=1e-4, k_chunk=1024, ema_decay=0.99, eps=1e-5):
        super().__init__()
        self.L, self.K, self.d = L, K, d
        self.beta = beta
        self.entropy_reg = entropy_reg
        self.k_chunk = max(1, int(k_chunk))
        self.ema_decay = ema_decay
        self.eps = eps

        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(K, d)) for _ in range(L)])
        # EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(L, K))
        self.register_buffer("ema_weight", torch.randn(L, K, d))

    @torch.no_grad()
    def init_from_rqkmeans(self, codebooks_np):
        # codebooks_np: [L,K,d]
        for p, C in zip(self.codebooks, codebooks_np):
            p.copy_(torch.tensor(C, dtype=p.dtype, device=p.device))
        self.ema_weight.copy_(torch.stack([p.data for p in self.codebooks], dim=0))

    def _argmin_chunked(self, R: torch.Tensor, C: torch.Tensor):
        """
        单层最近邻（在 K 上分块）:
        R: [B,d], C: [K,d] -> idx[B], q[B,d]
        """
        B, d = R.shape
        r_norm = (R * R).sum(1, keepdim=True)          # [B,1]
        idx_min, dist_min = None, None
        K = C.shape[0]
        for s in range(0, K, self.k_chunk):
            e = min(s + self.k_chunk, K)
            Cj = C[s:e]                                # [Ck,d]
            c_norm = (Cj * Cj).sum(1)                  # [Ck]
            dists = r_norm - 2.0 * (R @ Cj.t()) + c_norm[None, :]
            d_chunk, i_chunk = torch.min(dists, dim=1) # [B]
            i_chunk = i_chunk + s
            if dist_min is None:
                dist_min, idx_min = d_chunk, i_chunk
            else:
                m = d_chunk < dist_min
                dist_min = torch.where(m, d_chunk, dist_min)
                idx_min = torch.where(m, i_chunk, idx_min)
        q = C.index_select(0, idx_min)
        return idx_min, q

    def forward(self, z_e: torch.Tensor):
        """
        z_e: [B,d]
        返回: z_q_total(STE), ids(list[L] of idx[B]), commitment_loss, entropy_usage_reg
        训练期内部用 EMA 更新码本；推理期只做最近邻量化。
        """
        z_e = z_e.float()
        residual = z_e
        zs, ids = [], []
        ent_sum = 0.0

        for l in range(self.L):
            C = self.codebooks[l]
            idx_l, q_l = self._argmin_chunked(residual, C)  # [B], [B,d]
            zs.append(q_l)
            ids.append(idx_l)

            # ===== EMA 更新 =====
            with torch.no_grad():
                decay, eps = self.ema_decay, self.eps
                counts = torch.bincount(idx_l, minlength=self.K).to(z_e.dtype)     # [K]
                self.ema_cluster_size[l].mul_(decay).add_(counts, alpha=1-decay)   # [K]

                dw = torch.zeros_like(self.ema_weight[l])                          # [K,d]
                dw.index_add_(0, idx_l, residual)                                   # 累加 residual（该层拟合对象）
                self.ema_weight[l].mul_(decay).add_(dw, alpha=1-decay)

                n = self.ema_cluster_size[l].sum()
                cluster_size = (self.ema_cluster_size[l] + eps) / (n + self.K*eps) * n
                new_cb = self.ema_weight[l] / cluster_size.unsqueeze(1)
                C.data.copy_(new_cb)  # 更新码本权重

            # ===== 残差推进 =====
            residual = residual - q_l

            # ===== 批内熵正则（越大越好 → loss 里取负）=====
            if self.entropy_reg > 0:
                with torch.no_grad():
                    p = counts / counts.sum().clamp_min(1e-9)
                    ent = -(p * p.clamp_min(1e-9).log()).sum()
                ent_sum = ent_sum + ent

        # 最终量化向量 & 损失
        z_q_total = sum(zs)
        z_q_total = z_e + (z_q_total - z_e).detach()               # STE
        commit = self.beta * F.mse_loss(z_e, z_q_total.detach())
        loss_reg = - self.entropy_reg * (ent_sum / max(1, self.L)) # 最大化熵 → 负号
        return z_q_total, ids, commit, loss_reg
