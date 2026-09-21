#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math

import numpy as np
import pandas as pd


def erg_status(row, min_components, min_area_ratio):
    count = int(row["dab_component_count"])
    ratio = float(row["dab_area_ratio"])

    if count >= min_components or ratio >= min_area_ratio:
        return "positive"
    if count == 0 and ratio < min_area_ratio * 0.5:
        return "negative"
    return "ambiguous"


def reg_score(row, args):
    shift_term = math.exp(-float(row["residual_shift_qc_px"]) / args.shift_scale)
    score = (
        args.w_iou * float(row["tissue_iou"])
        + args.w_h_ncc * max(0.0, float(row["h_ncc"]))
        + args.w_h_nmi * float(row["h_nmi"])
        + args.w_edge_ncc * max(0.0, float(row["edge_ncc"]))
        + args.w_nuclei * float(row["nuclei_match"])
        + args.w_shift * shift_term
    )
    return float(np.clip(score, 0, 1))


def reg_group(row, args):
    if row["error"]:
        return "C"

    he_ratio = float(row["he_tissue_ratio"])
    ihc_ratio = float(row["ihc_tissue_ratio"])
    iou = float(row["tissue_iou"])
    score = float(row["reg_score"])
    shift = float(row["residual_shift_qc_px"])

    if he_ratio < args.min_tissue_ratio or ihc_ratio < args.min_tissue_ratio:
        return "C"
    if iou >= args.high_min_iou and score >= args.high_min_score and shift <= args.high_max_shift:
        return "A"
    if iou >= args.mid_min_iou and score >= args.mid_min_score and shift <= args.mid_max_shift:
        return "B"
    return "C"


def main():
    parser = argparse.ArgumentParser(description="Convert saved pair metrics into scores/groups.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", default="pair_scores.tsv")

    parser.add_argument("--min-tissue-ratio", type=float, default=0.15)
    parser.add_argument("--erg-min-components", type=int, default=2)
    parser.add_argument("--erg-min-area-ratio", type=float, default=0.00015)

    parser.add_argument("--high-min-iou", type=float, default=0.55)
    parser.add_argument("--high-min-score", type=float, default=0.50)
    parser.add_argument("--high-max-shift", type=float, default=18.0)
    parser.add_argument("--mid-min-iou", type=float, default=0.35)
    parser.add_argument("--mid-min-score", type=float, default=0.30)
    parser.add_argument("--mid-max-shift", type=float, default=40.0)
    parser.add_argument("--shift-scale", type=float, default=10.0)

    # 与原脚本完全相同的默认权重；如需改权重，不需要重跑图像指标。
    parser.add_argument("--w-iou", type=float, default=0.20)
    parser.add_argument("--w-h-ncc", type=float, default=0.25)
    parser.add_argument("--w-h-nmi", type=float, default=0.20)
    parser.add_argument("--w-edge-ncc", type=float, default=0.15)
    parser.add_argument("--w-nuclei", type=float, default=0.10)
    parser.add_argument("--w-shift", type=float, default=0.10)
    args = parser.parse_args()

    df = pd.read_csv(args.metrics, sep="\t", keep_default_na=False)

    weight_sum = (
        args.w_iou
        + args.w_h_ncc
        + args.w_h_nmi
        + args.w_edge_ncc
        + args.w_nuclei
        + args.w_shift
    )
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"score 权重之和必须为 1.0，当前为 {weight_sum:.6f}")

    df["erg_status"] = df.apply(
        erg_status,
        axis=1,
        args=(args.erg_min_components, args.erg_min_area_ratio),
    )
    df["reg_score"] = df.apply(reg_score, axis=1, args=(args,))
    df["reg_group"] = df.apply(reg_group, axis=1, args=(args,))

    group_order = pd.Categorical(df["reg_group"], categories=["A", "B", "C"], ordered=True)
    df = df.assign(_group_order=group_order).sort_values(
        ["_group_order", "reg_score"], ascending=[True, False]
    )
    df = df.drop(columns="_group_order")

    df.to_csv(args.out, sep="\t", index=False, float_format="%.6f")

    print(df["reg_group"].value_counts().reindex(["A", "B", "C"], fill_value=0).to_string())
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
