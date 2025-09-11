# 1) token 使用情况
import pandas as pd
df = pd.read_parquet("outputs/vqvae_4096/items.parquet")
print("rows:", len(df), "unique_tokens:", df["token"].nunique())
print(df["token"].value_counts().head(10))


# 2) 倒排桶大小分布
import pickle, numpy as np
inv = pickle.load(open("outputs/vqvae_4096/inverted.pkl","rb"))
sizes = np.array([len(v) for v in inv.values()])
print("buckets:", len(sizes), "min/median/mean/max:", sizes.min(), np.median(sizes), sizes.mean(), sizes.max())

