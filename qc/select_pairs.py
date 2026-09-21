#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def safe_link_or_copy(src, dst, mode):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return

    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)


def materialize(df, out_dir, name, mode, ihc_dir_name="@105"):
    if mode == "none" or df.empty:
        return

    he_dir = Path(out_dir) / name / "HE"
    ihc_dir = Path(out_dir) / name / ihc_dir_name

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"materialize {name}"):
        safe_link_or_copy(row.he_path, he_dir / Path(row.he_path).name, mode)
        safe_link_or_copy(row.ihc_path, ihc_dir / Path(row.ihc_path).name, mode)


def balanced_group(df, group, neg_ratio, seed):
    x = df[(df["reg_group"] == group) & df["erg_status"].isin(["positive", "negative"])].copy()
    pos = x[x["erg_status"] == "positive"]
    neg = x[x["erg_status"] == "negative"]

    if pos.empty:
        return x.iloc[0:0].copy()

    n_neg = min(len(neg), round(len(pos) * neg_ratio))
    neg = neg.sample(n=n_neg, random_state=seed) if n_neg > 0 else neg.iloc[0:0]
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def select_top_k(df, top_k, min_tissue):
    x = df[
        (df["error"].fillna("") == "")
        & (df["he_tissue_ratio"] >= min_tissue)
        & (df["ihc_tissue_ratio"] >= min_tissue)
    ].sort_values("reg_score", ascending=False)

    pos = x[x["erg_status"] == "positive"]
    neg = x[x["erg_status"] == "negative"]
    amb = x[x["erg_status"] == "ambiguous"]

    n_pos = min(len(pos), top_k // 2)
    n_neg = min(len(neg), top_k - n_pos)
    selected = pd.concat([pos.head(n_pos), neg.head(n_neg)])

    if len(selected) < top_k:
        remaining = x[~x["stem"].isin(selected["stem"])]
        selected = pd.concat([selected, remaining.head(top_k - len(selected))])

    return selected.sort_values("reg_score", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Select/materialize pairs from pair_scores.tsv.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--materialize", choices=["none", "symlink", "hardlink", "copy"], default="hardlink")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-k-min-tissue", type=float, default=0.30)
    parser.add_argument("--neg-ratio", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    # The IHC leaf folder used to be hardcoded as "@105", which is the old
    # pre-migration name. Kept as the default so the existing qc/105/selected
    # tree and prepare_cut_dataset.py's --stain-subdir default keep matching.
    parser.add_argument("--ihc-dir-name", default="@105",
                        help="leaf folder name for the IHC side inside each group")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.scores, sep="\t", keep_default_na=False)
    valid = df[df["error"].fillna("") == ""].copy()

    if args.top_k > 0:
        selected = select_top_k(valid, args.top_k, args.top_k_min_tissue)
        selected["training_mode"] = "top_k_selected"
        selected.to_csv(out_dir / "top_k_pairs.tsv", sep="\t", index=False, float_format="%.6f")
        materialize(selected, out_dir, "top_k_pairs", args.materialize, args.ihc_dir_name)
        print(f"Top-K selected: {len(selected)}")
        return

    strong = balanced_group(valid, "A", args.neg_ratio, args.seed)
    weak = balanced_group(valid, "B", args.neg_ratio, args.seed)

    strong.to_csv(out_dir / "strong_A.tsv", sep="\t", index=False, float_format="%.6f")
    weak.to_csv(out_dir / "weak_B.tsv", sep="\t", index=False, float_format="%.6f")

    materialize(strong, out_dir, "strong_A", args.materialize, args.ihc_dir_name)
    materialize(weak, out_dir, "weak_B", args.materialize, args.ihc_dir_name)

    print(f"Strong A: {len(strong)}")
    print(f"Weak B  : {len(weak)}")


if __name__ == "__main__":
    main()
