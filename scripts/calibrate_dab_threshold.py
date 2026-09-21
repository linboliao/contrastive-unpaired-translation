#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-hoc threshold calibration for the ERG_v2 sweep's r metric
(r = fake DAB-positive pixel count / ground-truth DAB-positive pixel count,
see scripts/evaluate_erg_sweep.py). No retraining: r is defined by a fixed
DAB optical-density threshold (0.15) applied to BOTH fake and real. If a
model's fake_B is systematically fainter/brighter in the DAB channel than
real_B, a *different* threshold on fake_B alone can bring r_median back to
1.0 without touching the model. This script searches for that per-experiment
threshold and reports how much of the miscalibration it explains, plus how
r looks stratified by registration confidence (qc/erg_v2_sweep_scores.tsv's
reg_group, from pair_scores.tsv): if the under/over-calling is concentrated
in low-confidence (weak) pairs, that points at noisy supervision rather than
a model problem a global threshold shift can fix.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hed

REAL_THRESHOLD = 0.15
CANDIDATE_THRESHOLDS = np.round(np.arange(0.05, 0.301, 0.01), 3)

EXPERIMENTS = [
    "ERG_v2_baseline", "ERG_v2_dab_l1", "ERG_v2_dab_l2", "ERG_v2_dab_l5",
    "ERG_v2_dab_l2_ambig01", "ERG_v2_dab_l2_ambig1", "ERG_v2_fastcut_dab_l2",
    "ERG_v2_dab_l2_netG6", "ERG_v2_dab_l2_noidt",
]


def dab_channel(img_uint8):
    rgb01 = img_uint8.astype(np.float64) / 255.0
    return np.clip(rgb2hed(rgb01)[..., 2], 0, None)


def load_reg_group(pair_scores_tsv):
    group = {}
    with open(pair_scores_tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            group[row["stem"]] = row.get("reg_group", "")
    return group


def main():
    parser = argparse.ArgumentParser(description="Calibrate the DAB-positive threshold on fake_B to fix r's bias.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--per-image-dir", type=Path, default=Path("qc/erg_v2_sweep_per_image"))
    parser.add_argument("--pair-scores", type=Path,
                         default=Path("/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/qc/105/pair_scores.tsv"))
    parser.add_argument("--out", type=Path, default=Path("qc/erg_v2_sweep_calibration.tsv"))
    args = parser.parse_args()

    reg_group = load_reg_group(args.pair_scores)
    rows_out = []

    for name in EXPERIMENTS:
        fake_dir = args.results_root / name / "test_latest" / "images" / "fake_B"
        per_image_tsv = args.per_image_dir / f"{name}.tsv"
        if not fake_dir.is_dir() or not per_image_tsv.is_file():
            print(f"[skip] {name}")
            continue

        real_pos = {}
        with open(per_image_tsv, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                real_pos[row["stem"]] = int(row["dab_pos_real_px"])

        stems = sorted(real_pos)
        fake_counts = np.zeros((len(stems), len(CANDIDATE_THRESHOLDS)), dtype=np.int64)
        for i, stem in enumerate(stems):
            dab = dab_channel(np.array(Image.open(fake_dir / f"{stem}.png").convert("RGB")))
            fake_counts[i] = [(dab > t).sum() for t in CANDIDATE_THRESHOLDS]

        real_counts = np.array([real_pos[s] for s in stems], dtype=np.float64)
        valid = real_counts > 0

        def r_median_at(threshold_idx):
            r = fake_counts[valid, threshold_idx] / real_counts[valid]
            return float(np.median(r))

        r_at_default = r_median_at(int(np.argmin(np.abs(CANDIDATE_THRESHOLDS - REAL_THRESHOLD))))
        errs = [abs(np.log2(r_median_at(i))) for i in range(len(CANDIDATE_THRESHOLDS))]
        best_idx = int(np.argmin(errs))
        best_threshold = float(CANDIDATE_THRESHOLDS[best_idx])
        r_at_best = r_median_at(best_idx)

        # stratify the *default*-threshold r by registration confidence group
        strat = {}
        for grp in sorted(set(reg_group.get(s, "") for s in stems)):
            idx = [i for i, s in enumerate(stems) if reg_group.get(s, "") == grp and valid[i]]
            if idx:
                r_grp = fake_counts[idx, int(np.argmin(np.abs(CANDIDATE_THRESHOLDS - REAL_THRESHOLD)))] / real_counts[idx]
                strat[grp] = float(np.median(r_grp))

        print(f"{name:<26} r@0.15={r_at_default:6.3f}  best_thr={best_threshold:.2f} -> r={r_at_best:6.3f}  "
              f"by_group={ {k: round(v, 3) for k, v in strat.items()} }")
        rows_out.append({
            "name": name, "r_median_default": r_at_default, "best_threshold": best_threshold,
            "r_median_best_threshold": r_at_best,
            **{f"r_median_group_{k}": v for k, v in strat.items()},
        })

    fieldnames = sorted({k for r in rows_out for k in r})
    fieldnames = ["name", "r_median_default", "best_threshold", "r_median_best_threshold"] + \
                 [f for f in fieldnames if f.startswith("r_median_group_")]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
