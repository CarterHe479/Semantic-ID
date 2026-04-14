# src/data/split_interactions.py
# -*- coding: utf-8 -*-
import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def read_inputs(patterns):
    dfs = []
    for pattern in patterns:
        paths = sorted(glob.glob(pattern))
        if not paths and os.path.exists(pattern):
            paths = [pattern]
        for path in paths:
            dfs.append(pd.read_parquet(path))
    if not dfs:
        raise FileNotFoundError(f"No parquet files matched: {patterns}")
    return pd.concat(dfs, ignore_index=True)


def kcore_filter(df, user_col, item_col, min_user_interactions, min_item_interactions, max_iter=20):
    out = df.copy()
    for _ in range(max_iter):
        before = len(out)
        if min_user_interactions > 1:
            uc = out[user_col].value_counts()
            out = out[out[user_col].isin(uc[uc >= min_user_interactions].index)]
        if min_item_interactions > 1:
            ic = out[item_col].value_counts()
            out = out[out[item_col].isin(ic[ic >= min_item_interactions].index)]
        if len(out) == before:
            break
    return out.reset_index(drop=True)


def leave_one_out_split(df, user_col, item_col, ts_col, valid_per_user=1, test_per_user=1):
    if ts_col in df.columns:
        df = df.sort_values([user_col, ts_col]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        df["__order__"] = df.groupby(user_col).cumcount()
        ts_col = "__order__"

    train_parts, valid_parts, test_parts = [], [], []
    min_needed = valid_per_user + test_per_user + 1
    for _, g in df.groupby(user_col, sort=False):
        if len(g) < min_needed:
            continue
        train_end = len(g) - valid_per_user - test_per_user
        valid_end = len(g) - test_per_user
        train_parts.append(g.iloc[:train_end])
        valid_parts.append(g.iloc[train_end:valid_end])
        test_parts.append(g.iloc[valid_end:])

    if not train_parts:
        raise RuntimeError("No users left after leave-one-out split. Lower k-core or split requirements.")

    train = pd.concat(train_parts, ignore_index=True)
    valid = pd.concat(valid_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    for split in (train, valid, test):
        if "__order__" in split.columns:
            split.drop(columns=["__order__"], inplace=True)
    return train, valid, test


def describe(name, df, user_col, item_col):
    print(
        f"{name}: rows={len(df)} users={df[user_col].nunique()} "
        f"items={df[item_col].nunique()}"
    )


def main():
    ap = argparse.ArgumentParser(description="Build user-level leave-one-out parquet splits.")
    ap.add_argument("--input", nargs="+", required=True,
                    help="Input parquet path(s) or glob(s), e.g. data/raw/*.parquet")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--item-col", default="item_id")
    ap.add_argument("--ts-col", default="timestamp")
    ap.add_argument("--min-user-interactions", type=int, default=5)
    ap.add_argument("--min-item-interactions", type=int, default=5)
    ap.add_argument("--valid-per-user", type=int, default=1)
    ap.add_argument("--test-per-user", type=int, default=1)
    args = ap.parse_args()

    df = read_inputs(args.input)
    for col in [args.user_col, args.item_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    df = df.dropna(subset=[args.user_col, args.item_col]).copy()
    df[args.user_col] = df[args.user_col].astype(str)
    df[args.item_col] = df[args.item_col].astype(str)

    print("Before filtering:")
    describe("all", df, args.user_col, args.item_col)

    df = kcore_filter(
        df,
        user_col=args.user_col,
        item_col=args.item_col,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )

    print("After k-core filtering:")
    describe("all", df, args.user_col, args.item_col)

    train, valid, test = leave_one_out_split(
        df,
        user_col=args.user_col,
        item_col=args.item_col,
        ts_col=args.ts_col,
        valid_per_user=args.valid_per_user,
        test_per_user=args.test_per_user,
    )

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    train.to_parquet(os.path.join(args.outdir, "train.parquet"), index=False)
    valid.to_parquet(os.path.join(args.outdir, "valid.parquet"), index=False)
    test.to_parquet(os.path.join(args.outdir, "test.parquet"), index=False)

    print("Saved splits:")
    describe("train", train, args.user_col, args.item_col)
    describe("valid", valid, args.user_col, args.item_col)
    describe("test", test, args.user_col, args.item_col)


if __name__ == "__main__":
    main()
