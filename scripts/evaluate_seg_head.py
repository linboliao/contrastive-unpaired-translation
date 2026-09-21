#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scores CUTSegModel's actual deliverable -- the nucleus-probability head,
not fake_B -- against its ground-truth mask, using the seg_prob_vis /
seg_target_vis images test.py already saved (see
scripts/run_erg_v2_seg_test.sh). Both were remapped [0,1]->[-1,1] before
saving (CUTSegModel.forward) specifically so tensor2im's [-1,1]->[0,255]
mapping round-trips exactly back to the original probability/mask: reloaded
pixel value / 255 = original value. No color-deconvolution recompute needed
here, unlike evaluate_erg_sweep.py (which scores fake_B instead).

Reports, per experiment, at a 0.5 probability threshold:
  dice / iou / precision / recall  -- mean over 200 test pairs
  r_seg_median                     -- predicted / ground-truth positive
                                       pixel count ratio (median), the
                                       segmentation-head analogue of
                                       evaluate_erg_sweep.py's r
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

EXPERIMENTS = ["ERG_v2_dab_l1_seg", "ERG_v2_dab_l2_ambig01_seg", "ERG_v2_baseline_seg"]
PROB_THRESHOLD = 0.5
CANDIDATE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def load_gray01(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def metrics_at(pred, target, target_sum):
    inter = float((pred * target).sum())
    pred_sum = float(pred.sum())
    union = pred_sum + target_sum - inter
    dice = 1.0 if (pred_sum + target_sum) == 0 else 2 * inter / (pred_sum + target_sum)
    iou = 1.0 if union == 0 else inter / union
    precision = 1.0 if pred_sum == 0 else inter / pred_sum
    recall = 1.0 if target_sum == 0 else inter / target_sum
    r = pred_sum / target_sum if target_sum > 0 else float("nan")
    return dice, iou, precision, recall, r


def score_pair(prob_path, target_path):
    prob = load_gray01(prob_path)
    target = (load_gray01(target_path) > 0.5).astype(np.float64)
    target_sum = float(target.sum())
    pred = (prob > PROB_THRESHOLD).astype(np.float64)
    dice, iou, precision, recall, r = metrics_at(pred, target, target_sum)
    at_other_thresh = {t: metrics_at((prob > t).astype(np.float64), target, target_sum) for t in CANDIDATE_THRESHOLDS}
    return dice, iou, precision, recall, r, target_sum, at_other_thresh


def main():
    parser = argparse.ArgumentParser(description="Score CUTSegModel's seg-head output against its ground-truth mask.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("qc/erg_v2_seg_scores.tsv"))
    parser.add_argument("--per-image-dir", type=Path, default=Path("qc/erg_v2_seg_per_image"))
    args = parser.parse_args()
    args.per_image_dir.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for name in EXPERIMENTS:
        img_dir = args.results_root / name / "test_latest" / "images"
        prob_dir, target_dir = img_dir / "seg_prob_vis", img_dir / "seg_target_vis"
        if not prob_dir.is_dir():
            print(f"[skip] {name}: no results under {prob_dir}")
            continue

        stems = sorted(p.stem for p in prob_dir.glob("*.png") if (target_dir / p.name).exists())
        dices, ious, precs, recs, rs = [], [], [], [], []
        by_thresh = {t: [] for t in CANDIDATE_THRESHOLDS}
        n_empty_gt = 0
        per_image = []
        for stem in stems:
            dice, iou, prec, rec, r, target_sum, at_other = score_pair(prob_dir / f"{stem}.png", target_dir / f"{stem}.png")
            dices.append(dice); ious.append(iou); precs.append(prec); recs.append(rec)
            for t, vals in at_other.items():
                by_thresh[t].append(vals[0])  # dice at that threshold
            if target_sum > 0:
                rs.append(r)
            else:
                n_empty_gt += 1
            per_image.append({"stem": stem, "dice": dice, "iou": iou, "precision": prec, "recall": rec, "r": r})

        with open(args.per_image_dir / f"{name}.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "dice", "iou", "precision", "recall", "r"], delimiter="\t")
            w.writeheader()
            w.writerows(per_image)

        best_t = max(CANDIDATE_THRESHOLDS, key=lambda t: np.mean(by_thresh[t]))
        row = {
            "name": name, "n": len(stems),
            "dice_mean": float(np.mean(dices)), "iou_mean": float(np.mean(ious)),
            "precision_mean": float(np.mean(precs)), "recall_mean": float(np.mean(recs)),
            "r_seg_median": float(np.median(rs)) if rs else float("nan"),
            "n_empty_gt": n_empty_gt,
            "best_threshold": best_t, "dice_at_best_threshold": float(np.mean(by_thresh[best_t])),
        }
        rows_out.append(row)
        print(f"{name:<28} dice@0.5={row['dice_mean']:.3f} iou={row['iou_mean']:.3f} "
              f"prec={row['precision_mean']:.3f} rec={row['recall_mean']:.3f} r_seg={row['r_seg_median']:.3f} "
              f"| best_thr={best_t} dice@best={row['dice_at_best_threshold']:.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
