# src/models/vq_vae.py
import torch, torch.nn as nn, torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    """
    低内存 VQ-VAE 量化器：
    - 距离计算按 codebook 的 K 维分块（k_chunk）进行，避免 B×K×d 广播
    - EMA 更新使用 bincount + index_add_，不再构造 one-hot [B,K]
    """
    def __init__(self, K=4096, d=256, beta=0.25, decay=0.99, eps=1e-5, k_chunk=1024):
        super().__init__()
        self.K, self.d, self.beta, self.decay, self.eps = K, d, beta, decay, eps
        self.k_chunk = max(1, int(k_chunk))
        self.codebook = nn.Parameter(torch.randn(K, d))
        self.register_buffer("ema_cluster_size", torch.zeros(K))
        self.register_buffer("ema_weight", torch.randn(K, d))

    def _argmin_chunked(self, z_e: torch.Tensor):
        """
        在 codebook 维度上分块找最小距离的索引。
        返回: (idx[B], dist_min[B])
        """
        B, d = z_e.shape
        e_norm = (z_e * z_e).sum(1, keepdim=True)  # [B,1]
        idx_min = None
        dist_min = None
        K = self.codebook.shape[0]

        for start in range(0, K, self.k_chunk):
            end = min(start + self.k_chunk, K)
            C = self.codebook[start:end]                          # [Ck,d]
            c_norm = (C * C).sum(1)                               # [Ck]
            dists = e_norm - 2.0 * (z_e @ C.t()) + c_norm[None,:] # [B,Ck]
            d_chunk, i_chunk = torch.min(dists, dim=1)            # [B]
            i_chunk = i_chunk + start
            if dist_min is None:
                dist_min = d_chunk
                idx_min = i_chunk
            else:
                mask = d_chunk < dist_min
                dist_min = torch.where(mask, d_chunk, dist_min)
                idx_min = torch.where(mask, i_chunk, idx_min)
        return idx_min, dist_min

    def forward(self, z_e: torch.Tensor):
        """
        z_e: [B,d]
        返回: z_q (STE), idx, commitment_loss
        """
        z_e = z_e.float()
        # 1) 找最近的 code（分块）
        idx, _ = self._argmin_chunked(z_e)
        z_q = self.codebook.index_select(0, idx)

        # 2) EMA 更新，不用 one-hot
        with torch.no_grad():
            decay, eps = self.decay, self.eps
            counts = torch.bincount(idx, minlength=self.K).to(z_e.dtype)   # [K]
            self.ema_cluster_size.mul_(decay).add_(counts, alpha=1-decay)

            dw = torch.zeros_like(self.ema_weight)                         # [K,d]
            dw.index_add_(0, idx, z_e)                                     # sum z_e per code
            self.ema_weight.mul_(decay).add_(dw, alpha=1-decay)

            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + eps) / (n + self.K*eps) * n  # smooth
            new_codebook = self.ema_weight / cluster_size.unsqueeze(1)
            self.codebook.data.copy_(new_codebook)

        # 3) STE & commitment
        z_q_ste = z_e + (z_q - z_e).detach()
        commit = self.beta * F.mse_loss(z_e, z_q.detach())
        return z_q_ste, idx, commit
