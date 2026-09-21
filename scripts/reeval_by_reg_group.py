#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-aggregates the already-computed per-image metrics (evaluate_erg_sweep.py
and evaluate_seg_head.py), restricted to reg_group=='A' stems -- the subset
whose HE/ERG registration actually passed score_pairs.py's strict thresholds
(tissue_iou>=0.55, reg_score>=0.50, residual_shift<=18px). All the paired
pixel metrics (SSIM/PSNR/DAB MAE/r, and the segmentation head's Dice/IoU/
precision/recall) assume real_B is pixel-aligned with fake_B/real_A; on a
reg_group='B' (or worse) pair that assumption is false, so those metrics are
measuring registration error as much as model quality. This does NOT
retrain or re-run inference -- it re-reads results already on disk. See
qc/erg_v2_sweep_per_image/, qc/erg_v2_seg_per_image/, and
qc/105/pair_scores.tsv.

Caveat printed in the output: only ~15-19 of the 603-pair test split are
reg_group A (the "strong" bucket has 107 pairs project-wide and most went to
training in the random 85/15 split) -- this is a small-N sanity check, not a
replacement for a properly stratified held-out set.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

SWEEP_EXPERIMENTS = [
    "ERG_v2_baseline", "ERG_v2_dab_l1", "ERG_v2_dab_l2", "ERG_v2_dab_l5",
    "ERG_v2_dab_l2_ambig01", "ERG_v2_dab_l2_ambig1", "ERG_v2_fastcut_dab_l2",
    "ERG_v2_dab_l2_netG6", "ERG_v2_dab_l2_noidt",
]
SEG_EXPERIMENTS = ["ERG_v2_dab_l1_seg", "ERG_v2_dab_l2_ambig01_seg", "ERG_v2_baseline_seg"]


def load_reg_group(pair_scores_tsv):
    group = {}
    with open(pair_scores_tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            group[row["stem"]] = row.get("reg_group", "")
    return group


def load_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def agg(rows, keys, reg_group, target_group):
    all_rows = rows
    clean_rows = [r for r in rows if reg_group.get(r["stem"], "") == target_group]
    out = {"n_all": len(all_rows), "n_clean": len(clean_rows)}
    for k, agg_fn, label in keys:
        vals_all = np.array([float(r[k]) for r in all_rows if r[k] not in ("", "nan")], dtype=float)
        vals_clean = np.array([float(r[k]) for r in clean_rows if r[k] not in ("", "nan")], dtype=float)
        vals_all = vals_all[~np.isnan(vals_all)]
        vals_clean = vals_clean[~np.isnan(vals_clean)]
        out[f"{label}_all"] = float(agg_fn(vals_all)) if vals_all.size else float("nan")
        out[f"{label}_clean"] = float(agg_fn(vals_clean)) if vals_clean.size else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-scores", type=Path,
                         default=Path("/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/qc/105/pair_scores.tsv"))
    parser.add_argument("--sweep-per-image", type=Path, default=Path("qc/erg_v2_sweep_per_image"))
    parser.add_argument("--seg-per-image", type=Path, default=Path("qc/erg_v2_seg_per_image"))
    parser.add_argument("--out", type=Path, default=Path("qc/erg_v2_reg_group_a_comparison.tsv"))
    parser.add_argument("--target-group", default="A")
    args = parser.parse_args()

    reg_group = load_reg_group(args.pair_scores)
    rows_out = []

    print("=== CUT sweep (SSIM higher=better, PSNR higher=better, DAB MAE / area err lower=better, r closer to 1=better) ===")
    for name in SWEEP_EXPERIMENTS:
        p = args.sweep_per_image / f"{name}.tsv"
        if not p.is_file():
            continue
        rows = load_tsv(p)
        a = agg(rows, [
            ("ssim", np.mean, "ssim"), ("psnr", np.mean, "psnr"),
            ("dab_mae", np.mean, "dab_mae"), ("dab_area_abs_diff", np.mean, "area_err"),
            ("r", np.nanmedian, "r"),
        ], reg_group, args.target_group)
        a["name"] = name
        rows_out.append(("sweep", a))
        print(f"{name:<26} n={a['n_clean']:>3}/{a['n_all']:<4} "
              f"ssim {a['ssim_all']:.3f}->{a['ssim_clean']:.3f}  "
              f"psnr {a['psnr_all']:.2f}->{a['psnr_clean']:.2f}  "
              f"dab_mae {a['dab_mae_all']:.4f}->{a['dab_mae_clean']:.4f}  "
              f"r {a['r_all']:.3f}->{a['r_clean']:.3f}")

    print("\n=== Segmentation head (all higher=better) ===")
    for name in SEG_EXPERIMENTS:
        p = args.seg_per_image / f"{name}.tsv"
        if not p.is_file():
            continue
        rows = load_tsv(p)
        a = agg(rows, [
            ("dice", np.mean, "dice"), ("iou", np.mean, "iou"),
            ("precision", np.mean, "precision"), ("recall", np.mean, "recall"),
        ], reg_group, args.target_group)
        a["name"] = name
        rows_out.append(("seg", a))
        print(f"{name:<26} n={a['n_clean']:>3}/{a['n_all']:<4} "
              f"dice {a['dice_all']:.3f}->{a['dice_clean']:.3f}  "
              f"iou {a['iou_all']:.3f}->{a['iou_clean']:.3f}  "
              f"prec {a['precision_all']:.3f}->{a['precision_clean']:.3f}  "
              f"rec {a['recall_all']:.3f}->{a['recall_clean']:.3f}")

    fieldnames = sorted({k for _, a in rows_out for k in a})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind"] + fieldnames, delimiter="\t")
        w.writeheader()
        for kind, a in rows_out:
            w.writerow({"kind": kind, **a})
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
