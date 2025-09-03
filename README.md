# Semantic ID Tokenizers for Generative Recommenders

**RQ-KMeans / RQ-VAE / VQ-VAE** 的一体化实现：将 item 连续表征压缩为**离散语义 ID**，并提供倒排检索 + FAISS 重排 + 离线评测管线。

> 已在 Amazon Reviews 2023（样例子集）上打通 **Step 4–8**：数据预处理 → 嵌入抽取 → 分词器训练/编码 → 语义 ID 导出 → 倒排索引 → FAISS 重排 → 评测。

---

## ✨ 特性
- **三种分词器**：`RQ-KMeans`（残差 kmeans）、`RQ-VAE`（EMA 版、K 维分块近邻）、`VQ-VAE`（EMA 版、K 维分块近邻）
- **低内存实现**：所有距离计算支持 **K 维分块** 和 **按批**，避免构造 `B×K×d` 巨矩阵
- **一键评测**：倒排候选 +（可选）FAISS 重排，输出 Recall@K / NDCG@K 等
- **跨设备**：CPU / macOS MPS / CUDA（VAE 训练与编码均可）

---

## 📦 目录结构（关键文件）
src/
data/
preprocess_amazon23.py # Step 4：数据预处理（统一 schema）
models/
RQ_KMeans.py # 残差 KMeans（fit/encode 低内存）
RQ_VAE.py # RQ-VAE（EMA + 分块近邻 + 批内熵）
VQ_VAE.py # VQ-VAE（EMA + 分块近邻）
train/
extract_embeddings.py # Step 5：抽取 item 连续表征（BERT/SBERT）
train_tokenizer.py # Step 7：训练/编码（rqkmeans | rqvae | vqvae）
export_codes.py # Step 8：导出语义 ID 表（parquet）
retrieval/
build_inverted.py # 倒排索引（code → item_ids）
build_faiss.py # 全库 FAISS 索引（可选）
eval_offline.py # 候选 + 重排 + 线下评测


---

## 🛠️ 环境安装

> **建议 conda**（解决 FAISS 与 NumPy 兼容性问题）。macOS / Linux 皆可。

```bash
conda create -n semantic-id python=3.10 -y
conda activate semantic-id

# CPU 版 PyTorch（macOS MPS 用 pip 官方轮子也可）
pip install torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu

# 基础依赖
pip install "numpy<2" pandas pyarrow scikit-learn tqdm
pip install transformers==4.41.2 datasets==2.21.0

# FAISS（可选，但推荐；会自动带来兼容的 numpy 1.26.x）
conda install -c conda-forge "faiss-cpu>=1.7.4,<2.0.0" -y


如果你想用 MPS（Apple 芯片）：按 PyTorch 官方文档安装支持 MPS 的 torch，运行脚本时加 --device mps。

⚡ 快速开始
0) 数据准备（Amazon Reviews 2023）

按 src/data/preprocess_amazon23.py 将数据规整到：

data/amazon23/{train,valid,test}.parquet
# 至少应包含：user_id, item_id, timestamp, title, description（其他列可空）

1) 抽取 item 连续表征（Step 5）
# 训练用（去重后每 item 一条）
python -m src.train.extract_embeddings \
  --data ./data/amazon23/train.parquet \
  --text-cols title,description \
  --out ./outputs/emb/train_emb.pt

# 全量 items（train/valid/test 合并）
python -m src.train.extract_embeddings \
  --data "./data/amazon23/*.parquet" \
  --text-cols title,description \
  --out ./outputs/emb/items_emb.pt

2) 分词器训练与编码（Step 7）

RQ-KMeans（可作为 RQ-VAE 初始化）

# 训练
python -m src.train.train_tokenizer --method rqkmeans \
  --in ./outputs/emb/train_emb.pt --out ./outputs/rqkmeans_4x256 \
  --L 4 --K 256 --max-train 200000

# 仅编码全量 items
python -m src.train.train_tokenizer --method rqkmeans --no-train \
  --encode ./outputs/emb/items_emb.pt \
  --out    ./outputs/rqkmeans_4x256 \
  --load   ./outputs/rqkmeans_4x256/codebooks.npz


RQ-VAE（EMA 版；建议用 RQ-KMeans 初始化）

# 训练
python -m src.train.train_tokenizer --method rqvae \
  --in ./outputs/emb/train_emb.pt --out ./outputs/rqvae_4x256 \
  --L 4 --K 256 --init ./outputs/rqkmeans_4x256/codebooks.npz \
  --epochs 5 --batch-size 4096 --dist-chunk 1024 --device mps

# 仅编码全量 items
python -m src.train.train_tokenizer --method rqvae --no-train \
  --encode ./outputs/emb/items_emb.pt \
  --out   ./outputs/rqvae_4x256 \
  --load  ./outputs/rqvae_4x256/codebooks.npz \
  --encode-batch-size 4096 --device mps


VQ-VAE（EMA 版）

# 训练
python -m src.train.train_tokenizer --method vqvae \
  --in ./outputs/emb/train_emb.pt --out ./outputs/vqvae_4096 \
  --K 4096 --epochs 6 --batch-size 8192 --dist-chunk 512 --device mps

# 仅编码全量 items
python -m src.train.train_tokenizer --method vqvae --no-train \
  --encode ./outputs/emb/items_emb.pt \
  --out   ./outputs/vqvae_4096 \
  --load  ./outputs/vqvae_4096/codebooks.npz \
  --encode-batch-size 4096 --device mps

3) 导出语义 ID 表（Step 8）
# RQ 系（会输出 c1..cL + code_str）
python -m src.train.export_codes \
  --codes ./outputs/rqvae_4x256/codes.npy \
  --ids   ./outputs/emb/items_emb.ids.npy \
  --out   ./outputs/rqvae_4x256/items.parquet --concat-col

# VQ 系（输出 token）
python -m src.train.export_codes \
  --codes ./outputs/vqvae_4096/codes.npy \
  --ids   ./outputs/emb/items_emb.ids.npy \
  --out   ./outputs/vqvae_4096/items.parquet

4) 构建倒排 & FAISS 索引（可选） & 评测
# 倒排（每种方法一次）
python -m src.retrieval.build_inverted --items ./outputs/rqkmeans_4x256/items.parquet --outdir ./outputs/rqkmeans_4x256
python -m src.retrieval.build_inverted --items ./outputs/rqvae_4x256/items.parquet   --outdir ./outputs/rqvae_4x256
python -m src.retrieval.build_inverted --items ./outputs/vqvae_4096/items.parquet    --outdir ./outputs/vqvae_4096

# 全库 FAISS（若 faiss 安装失败可跳过；评测会自动回退到全库点积）
python -m src.retrieval.build_faiss \
  --emb ./outputs/emb/items_emb.pt \
  --ids ./outputs/emb/items_emb.ids.npy \
  --outdir ./outputs/faiss --normalize

# 评测（会输出 ./results/offline_eval.csv）
python -m src.retrieval.eval_offline \
  --train ./data/amazon23/train.parquet \
  --valid ./data/amazon23/valid.parquet \
  --test  ./data/amazon23/test.parquet \
  --items-emb ./outputs/emb/items_emb.pt \
  --items-ids ./outputs/emb/items_emb.ids.npy \
  --faiss-index ./outputs/faiss/index.faiss \
  --faiss-ids   ./outputs/faiss/ids.npy \
  --methods rqkmeans_4x256 rqvae_4x256 vqvae_4096 \
  --use-faiss-rerank \
  --max-test 5000 \
  --out ./results/offline_eval.csv

🔍 常见问题（FAQ）

Q1. faiss 导入报 NumPy 2.x 兼容问题？
A：使用 conda-forge 安装 faiss-cpu，并确保 numpy<2：

conda install -c conda-forge "faiss-cpu>=1.7.4,<2.0.0" "numpy<2" -y


安装失败时可先跳过 FAISS 步骤，评测会自动回退到全库点积（更慢但可用）。

Q2. VQ 的 token 都集中到一个桶？

确认训练/编码使用的是同一 codebook（--load 正确）且量化器在相同 device（我们已修复 .to(device)）。

训练 --epochs 适当提高（5–10），--dist-chunk 调小（256/512）。

Q3. RQ 候选太小导致 Recall 低？

目前倒排用“精确 code 串”。后续可以开启 Hamming-1 扩展 或 分层并集扩大候选（TODO）。

📊 结果复现与对比

以 相近码长对比：

RQ 4×256 ≈ 32bit

VQ K=4096 ≈ 12bit（可再跑 8192/32768 做更公平比较）

指标：Recall@K、NDCG@K、avg_cand_size、cold_start_skipped。

🔖 License

本仓库代码采用 MIT License。
使用的公开数据（如 Amazon Reviews 2023）请遵守其原始许可与使用条款。

🙏 Acknowledgements

McAuley Lab — Amazon Reviews 2023

VQ-VAE / RQ-VAE 相关工作与社区实现

FAISS


---

需要我再帮你补一个 `scripts/run_all.sh`（一键从预处理到评测）或 GitHub Actions（lint/基础检查）的模板吗？我可以直接按你当前目录结构产出。
::contentReference[oaicite:0]{index=0}