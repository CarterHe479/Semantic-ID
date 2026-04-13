# Semantic ID Tokenizers for Generative Recommenders

**RQ-KMeans / RQ-VAE / VQ-VAE** 一体化实现：将物品（Item）的连续表征压缩为**离散语义 ID**，并提供完整的「倒排检索 + FAISS 重排 + 离线评测」端到端管线。  

> 已在 **Amazon Reviews 2023（样例子集）** 上打通核心流程：
> 准备 parquet 数据 → 嵌入抽取 → 分词器训练/编码 → 语义 ID 导出 → 倒排索引构建 → FAISS 重排 → 离线评测


## ✨ 核心特性
- **多分词器支持**：集成 `RQ-KMeans`（残差 KMeans，轻量高效）、`RQ-VAE`（EMA 优化 + K 维分块近邻）、`VQ-VAE`（EMA 优化 + K 维分块近邻）三种离散化方案，适配不同精度/效率需求。
- **低内存优化**：所有距离计算支持「K 维分块」与「按批处理」，避免构造 `B×K×d` 级别的巨型矩阵，适配中小显存设备（如 macOS M1/M2）。
- **一键式评测**：内置「倒排候选召回 + （可选）FAISS 重排」逻辑，自动输出 Recall@K、NDCG@K 等核心推荐指标，无需额外开发。
- **跨设备兼容**：支持 CPU / macOS MPS / CUDA 三种运行环境，VAE 训练与编码流程在各设备上均已验证。


## 📂 目录结构（核心文件说明）
```
semantic-id/
├── src/
│   ├── models/
│   │   ├── RQ_KMeans.py           # RQ-KMeans 模型（fit/encode 低内存实现）
│   │   ├── RQ_VAE.py              # RQ-VAE 模型（EMA + 分块近邻 + 批内熵）
│   │   └── VQ_VAE.py              # VQ-VAE 模型（EMA + 分块近邻）
│   ├── train/
│   │   ├── extract_embeddings.py  # 抽取 Item 连续表征（基于 BERT/SBERT）
│   │   ├── train_tokenizer.py     # 分词器训练/编码（支持 rqkmeans/rqvae/vqvae）
│   │   └── export_codes.py        # 导出语义 ID 表（输出 parquet 格式）
│   └── retrieval/
│       ├── build_inverted.py      # 构建倒排索引（code → item_ids 映射）
│       ├── build_faiss.py         # 构建全库 FAISS 索引（可选，用于重排）
│       └── eval_offline.py        # 离线评测（候选召回 + 重排 + 指标计算）
├── data/                          # 数据存储目录（需自行放置/生成）
├── outputs/                       # 输出目录（自动生成：嵌入、模型、索引等）
├── results/                       # 评测结果目录（自动生成：离线指标 CSV）
└── env.yml                        # 环境配置文件
```


## 🛠️ 环境安装
**推荐使用 Conda**（解决 FAISS 与 NumPy 的版本兼容问题），支持 macOS / Linux 系统。

### 1. 基础环境搭建（通用）
```bash
# 1. 创建并激活 Conda 环境
conda create -n semantic-id python=3.10 -y
conda activate semantic-id

# 2. 安装 PyTorch（根据设备选择对应命令）
## 选项 A：CPU 版（通用，macOS/Linux 均适用）
pip install torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu

## 选项 B：macOS MPS 版（Apple 芯片专属，支持 GPU 加速）
pip install torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cpu

## 选项 C：CUDA 版（NVIDIA 显卡，需替换 <cu121> 为对应 CUDA 版本）
pip install torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cu121

# 3. 安装基础依赖
pip install "numpy<2" pandas pyarrow scikit-learn tqdm
pip install transformers==4.41.2 datasets==2.21.0

# 4. 安装 FAISS（可选，用于重排；Conda 安装可自动适配 NumPy 版本）
conda install -c conda-forge "faiss-cpu>=1.7.4,<2.0.0" -y
```

### 2. 设备专属配置
- **macOS MPS 用户**：无需额外操作，安装上述「选项 B」的 PyTorch 后，运行脚本时添加 `--device mps` 即可启用 GPU 加速。
- **CUDA 用户**：确保已安装对应版本的 CUDA Toolkit（如 CUDA 12.1），运行脚本时添加 `--device cuda` 启用 GPU 加速。


## ⚡ 快速开始（全流程）
以下步骤以 **Amazon Reviews 2023 数据集**为例，从数据准备到离线评测完整演示。

### 0. 数据准备（前置步骤）
准备三份 parquet 文件：

```text
data/amazon23/train.parquet
data/amazon23/valid.parquet
data/amazon23/test.parquet
```

必需字段：`user_id`, `item_id`, `timestamp`, `title`, `description`。`title` 和 `description` 可为空，脚本会安全转成文本。

### 3090/CUDA 一键运行
如果你已经准备好真实 parquet 数据，并在 NVIDIA 3090 的 CUDA 环境中运行，可以直接执行：

```bash
cd ~/Semantic-ID
./scripts/run_full_cuda.sh
```

脚本默认使用正式参数：

```text
RQ-KMeans: L=4, K=256, max_train=200000
RQ-VAE:    L=4, K=256, epochs=8, batch_size=4096, dist_chunk=1024
VQ-VAE:    K=4096, epochs=10, batch_size=8192, dist_chunk=512
Eval:      max_test=5000, expand_rq_M=4, expand_vq_T=8
```

输出位置：

```text
outputs/rqkmeans_4x256/
outputs/rqvae_4x256/
outputs/vqvae_4096/
outputs/faiss/
results/offline_eval_cuda.csv
logs/full_cuda_*/
```

运行前脚本会检查当前 Python 环境是否能看到 CUDA；如果检查失败，请先切换到安装了 CUDA 版 PyTorch 的 Conda/venv。运行中每个阶段都会写入独立日志，若中途失败，优先查看 `logs/full_cuda_*/<step>.log`。

常用覆盖项：

```bash
# 跳过 VQ-VAE，只跑 RQ-KMeans/RQ-VAE
RUN_VQ=0 ./scripts/run_full_cuda.sh

# 全量评测，不限制 test 样本数
EVAL_MAX_TEST=0 ./scripts/run_full_cuda.sh

# 如果显存紧张，调小 batch
RQ_BATCH_SIZE=2048 VQ_BATCH_SIZE=4096 ./scripts/run_full_cuda.sh

# 使用不同文本模型
TEXT_MODEL=BAAI/bge-small-en-v1.5 ./scripts/run_full_cuda.sh
```

VQ-VAE 训练后脚本会打印 token 分布健康度。如果 `unique_tokens` 很低或最大桶接近全库大小，说明 VQ 发生 collapse；这种情况下建议先用 `RUN_VQ=0` 跑 RQ 系列，或者增大 `VQ_K`、增加 `VQ_EPOCHS` 后重新训练。


### 1. 抽取 Item 连续表征（Step 5）
使用 BERT/SBERT 模型将 Item 的文本信息（标题+描述）转换为连续向量：
```bash
# ① 抽取训练用表征（去重后每 Item 一条，用于分词器训练）
python -m src.train.extract_embeddings \
  --data ./data/amazon23/train.parquet \
  --text-cols title,description \
  --out ./outputs/emb/train_emb.pt

# ② 抽取全量 Item 表征（合并 train/valid/test，用于后续编码）
python -m src.train.extract_embeddings \
  --data "./data/amazon23/*.parquet" \
  --text-cols title,description \
  --out ./outputs/emb/items_emb.pt
```

- 输出说明：`train_emb.pt`（训练用嵌入）、`items_emb.pt`（全量 Item 嵌入），同时会生成 `*.ids.npy` 文件存储对应的 Item ID。


### 2. 分词器训练与编码（Step 7）
支持三种分词器（RQ-KMeans / RQ-VAE / VQ-VAE），选择对应方法执行训练与编码。

#### 方法 1：RQ-KMeans（轻量，可作为 RQ-VAE 初始化）
```bash
# ① 训练 RQ-KMeans 分词器
python -m src.train.train_tokenizer --method rqkmeans \
  --in ./outputs/emb/train_emb.pt \
  --out ./outputs/rqkmeans_4x256 \
  --L 4 --K 256 --max-train 200000  # L=4层，每层K=256，最大训练样本数20万

# ② 编码全量 Item（仅编码，不重新训练）
python -m src.train.train_tokenizer --method rqkmeans --no-train \
  --encode ./outputs/emb/items_emb.pt \
  --out    ./outputs/rqkmeans_4x256 \
  --load   ./outputs/rqkmeans_4x256/codebooks.npz  # 加载已训练的码本
```

#### 方法 2：RQ-VAE（EMA 优化，建议用 RQ-KMeans 初始化）
```bash
# ① 训练 RQ-VAE 分词器（用 RQ-KMeans 码本初始化）
python -m src.train.train_tokenizer --method rqvae \
  --in ./outputs/emb/train_emb.pt \
  --out ./outputs/rqvae_4x256 \
  --L 4 --K 256 \
  --init ./outputs/rqkmeans_4x256/codebooks.npz \
  --epochs 5 --batch-size 4096 --dist-chunk 1024 \
  --device mps

# ② 编码全量 Item
python -m src.train.train_tokenizer --method rqvae --no-train \
  --encode ./outputs/emb/items_emb.pt \
  --out    ./outputs/rqvae_4x256 \
  --load   ./outputs/rqvae_4x256/codebooks.npz \
  --encode-batch-size 4096 --device mps
```

#### 方法 3：VQ-VAE（EMA 优化，适用于单码本场景）
```bash
# ① 训练 VQ-VAE 分词器
python -m src.train.train_tokenizer --method vqvae \
  --in ./outputs/emb/train_emb.pt \
  --out ./outputs/vqvae_4096 \
  --K 4096 \
  --epochs 6 --batch-size 8192 --dist-chunk 512 \
  --device mps

# ② 编码全量 Item
python -m src.train.train_tokenizer --method vqvae --no-train \
  --encode ./outputs/emb/items_emb.pt \
  --out    ./outputs/vqvae_4096 \
  --load   ./outputs/vqvae_4096/codebooks.npz \
  --encode-batch-size 4096 --device mps
```

- 输出说明：`codebooks.npz`（训练好的码本）、`codes.npy`（全量 Item 的编码结果）。


### 3. 导出语义 ID 表（Step 8）
将编码结果（codes）与 Item ID 关联，导出为 parquet 格式（方便后续检索）：
```bash
# ① RQ 系列（RQ-KMeans/RQ-VAE）：输出 c1~cL 分层 ID + 合并的 code_str
python -m src.train.export_codes \
  --codes ./outputs/rqvae_4x256/codes.npy \
  --ids   ./outputs/emb/items_emb.ids.npy \
  --out   ./outputs/rqvae_4x256/items.parquet \
  --concat-col  # 生成合并后的 code_str 字段

# ② VQ 系列（VQ-VAE）：输出单 token ID
python -m src.train.export_codes \
  --codes ./outputs/vqvae_4096/codes.npy \
  --ids   ./outputs/emb/items_emb.ids.npy \
  --out   ./outputs/vqvae_4096/items.parquet
```


### 4. 构建索引与离线评测
#### ① 构建倒排索引（用于候选召回）
```bash
# 为三种分词器结果分别构建倒排索引
python -m src.retrieval.build_inverted \
  --items ./outputs/rqkmeans_4x256/items.parquet \
  --outdir ./outputs/rqkmeans_4x256

python -m src.retrieval.build_inverted \
  --items ./outputs/rqvae_4x256/items.parquet \
  --outdir ./outputs/rqvae_4x256

python -m src.retrieval.build_inverted \
  --items ./outputs/vqvae_4096/items.parquet \
  --outdir ./outputs/vqvae_4096
```

#### ② （可选）构建 FAISS 索引（用于重排）
```bash
python -m src.retrieval.build_faiss \
  --emb ./outputs/emb/items_emb.pt \
  --ids ./outputs/emb/items_emb.ids.npy \
  --outdir ./outputs/faiss \
  --normalize  # 对嵌入向量归一化（提升检索精度）
```

#### ③ 离线评测（输出核心指标）
```bash
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
```

- 结果说明：`offline_eval.csv` 包含每种方法的 Recall@10/50/100、NDCG@10/50/100、平均候选集大小、空候选率、全库回退率等指标。
- 默认行为：semantic-ID 候选为空时计为 miss，不再自动回退全库 dense search。若要评测「semantic-ID 候选 + dense fallback」混合策略，可显式添加 `--allow-full-search-fallback`。


## 🔍 常见问题（FAQ）
### Q1：FAISS 导入报错，提示 NumPy 2.x 兼容问题？
A：通过 Conda 安装 FAISS 并锁定 NumPy 版本：
```bash
conda install -c conda-forge "faiss-cpu>=1.7.4,<2.0.0" "numpy<2" -y
```
若仍失败，可跳过 FAISS 相关步骤，候选集内部重排会自动使用 NumPy 点积。只有显式添加 `--allow-full-search-fallback` 时，空候选才会回退到全库点积。

### Q2：VQ 分词器训练后，token 集中到少数几个桶（码本利用率低）？
A：1. 确认训练/编码使用**同一码本**（`--load` 路径正确）且量化器在**相同设备**上；
   2. 适当增加训练轮次（`--epochs 8~10`）；  
   3. 调小 `--dist-chunk`（如 256/512），降低单次距离矩阵内存峰值。

## ✅ 验证
```bash
python3 -m compileall -q src tests
python3 -m unittest discover -s tests
```

当前测试覆盖：
- `eval()` / 编码阶段不会更新 VQ/RQ 码本；
- 加载 VQ 码本时会同步 EMA buffer；
- semantic-ID 空候选默认计 miss，只有显式开启 `--allow-full-search-fallback` 才走全库回退。

### Q3：RQ 系列方法的召回候选集太小，导致 Recall 偏低？
A：当前倒排索引基于 code 串召回，评测时已支持每层近邻并集扩展。可以增大 `--expand-rq-M` 或 `--expand-vq-T`，也可以适当增大码本大小（如 `--K 512`）。


## 📊 结果对比参考
以「码长相近」为前提，三种方法的指标对比（示例）：
| 方法          | 码长（近似） | Recall@10 | Recall@50 | NDCG@10 | 平均候选大小 |
|---------------|--------------|-----------|-----------|---------|--------------|
| RQ-KMeans 4×256 | 32 bit       | 0.68      | 0.82      | 0.52    | 120          |
| RQ-VAE 4×256   | 32 bit       | 0.72      | 0.85      | 0.56    | 115          |
| VQ-VAE 4096    | 12 bit       | 0.61      | 0.75      | 0.45    | 98           |

- 注：VQ-VAE 码长更短，若需公平对比，可训练 `K=8192`（13 bit）或 `K=32768`（15 bit）的 VQ-VAE 模型。


## 🔖 License
本仓库代码采用 **MIT License**，可自由修改与商用。  
使用的公开数据集（如 Amazon Reviews 2023）需遵守其原始许可条款（参考 [McAuley Lab 数据集协议](https://cseweb.ucsd.edu/~jmcauley/datasets.html)）。


## 🙏 Acknowled
