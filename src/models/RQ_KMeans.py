# src/models/rq_kmeans.py
import numpy as np
from sklearn.cluster import KMeans

class ResidualKMeans:
    def __init__(self, L=4, K=256, random_state=0):
        self.L, self.K = L, K
        self.codebooks = []

    def fit(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        residual = Z.copy()
        self.codebooks = []
        for _ in range(self.L):
            km = KMeans(n_clusters=self.K, n_init="auto", random_state=0).fit(residual)
            C = km.cluster_centers_.astype(np.float32, copy=False)
            idx = km.predict(residual)
            Q = C[idx]
            self.codebooks.append(C)
            residual = residual - Q

    def encode(self, Z, batch_size: int = 2048):
        """
        内存友好的编码：分批计算 (||x||^2 - 2 x C^T + ||C||^2)，避免 [N,K,d] 广播。
        返回形状 [N, L] 的 int32。
        """
        Z = np.asarray(Z, dtype=np.float32)
        N, d = Z.shape
        L = len(self.codebooks)
        codes = np.empty((N, L), dtype=np.int32)
        residual = Z.copy()

        for l, C in enumerate(self.codebooks):
            C = C.astype(np.float32, copy=False)           # [K, d]
            c_norm = (C * C).sum(axis=1)                   # [K]

            for i in range(0, N, batch_size):
                R = residual[i:i+batch_size]               # [B, d]
                r_norm = (R * R).sum(axis=1, keepdims=True)  # [B,1]
                # 距离矩阵 [B,K]
                dists = r_norm - 2.0 * (R @ C.T) + c_norm[None, :]
                idx = dists.argmin(axis=1).astype(np.int32)  # [B]
                codes[i:i+batch_size, l] = idx
                residual[i:i+batch_size] = R - C[idx]
        return codes
