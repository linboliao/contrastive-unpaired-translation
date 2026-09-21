#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quantitative comparison of the ERG_v2 sweep checkpoints on the held-out
CUT test split. Reads the fake_B/real_B images test.py already saved under
results/<name>/test_latest/images/ (see scripts/run_erg_v2_test_sweep.sh) and
scores each pair on:

  ssim / psnr   -- standard paired image-fidelity metrics (RGB)
  dab_mae       -- mean abs error of the DAB optical-density channel
                    (skimage.color.rgb2hed, same stain matrix the DAB loss
                    trains against), i.e. how faithfully the ERG stain
                    pattern itself is reproduced, not just overall look
  dab_area_bias / dab_area_abs_diff
                -- signed / unsigned difference in ERG-positive area
                    fraction (DAB OD > threshold) between fake and real,
                    i.e. does the model over- or under-call positivity
  r             -- dab_pos_fake_px / dab_pos_real_px, the ratio of
                    DAB-positive pixel counts (fake over ground truth).
                    r=1 is perfect, r>1 over-calls positivity, r<1
                    under-calls it. Undefined (skipped) when the real
                    patch has zero DAB-positive pixels.

Requires all experiments to have been tested with the same --num_test so the
image sets are directly comparable (see run_erg_v2_test_sweep.sh).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hed
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DAB_OD_THRESHOLD = 0.15

EXPERIMENTS = [
    "ERG_v2_baseline",
    "ERG_v2_dab_l1",
    "ERG_v2_dab_l2",
    "ERG_v2_dab_l5",
    "ERG_v2_dab_l2_ambig01",
    "ERG_v2_dab_l2_ambig1",
    "ERG_v2_fastcut_dab_l2",
    "ERG_v2_dab_l2_netG6",
    "ERG_v2_dab_l2_noidt",
]


def dab_channel(img_uint8):
    rgb01 = img_uint8.astype(np.float64) / 255.0
    hed = rgb2hed(rgb01)
    return np.clip(hed[..., 2], 0, None)


def score_pair(fake_path, real_path):
    fake = np.array(Image.open(fake_path).convert("RGB"))
    real = np.array(Image.open(real_path).convert("RGB"))

    ssim = structural_similarity(real, fake, channel_axis=-1, data_range=255)
    psnr = peak_signal_noise_ratio(real, fake, data_range=255)

    dab_fake = dab_channel(fake)
    dab_real = dab_channel(real)
    dab_mae = float(np.mean(np.abs(dab_fake - dab_real)))

    pos_fake_px = int(np.sum(dab_fake > DAB_OD_THRESHOLD))
    pos_real_px = int(np.sum(dab_real > DAB_OD_THRESHOLD))
    area_fake = pos_fake_px / dab_fake.size
    area_real = pos_real_px / dab_real.size
    r = pos_fake_px / pos_real_px if pos_real_px > 0 else float("nan")

    return {
        "ssim": ssim,
        "psnr": psnr,
        "dab_mae": dab_mae,
        "dab_area_bias": area_fake - area_real,
        "dab_area_abs_diff": abs(area_fake - area_real),
        "dab_pos_fake_px": pos_fake_px,
        "dab_pos_real_px": pos_real_px,
        "r": r,
    }


def main():
    parser = argparse.ArgumentParser(description="Score ERG_v2 sweep checkpoints on the CUT test split.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("qc/erg_v2_sweep_scores.tsv"))
    parser.add_argument("--per-image-dir", type=Path, default=Path("qc/erg_v2_sweep_per_image"))
    parser.add_argument("--names", default=None, help="comma-separated experiment names (default: the v2 sweep)")
    args = parser.parse_args()
    experiments = args.names.split(",") if args.names else EXPERIMENTS

    args.per_image_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = ["ssim", "psnr", "dab_mae", "dab_area_bias", "dab_area_abs_diff"]
    extra_keys = ["dab_pos_fake_px", "dab_pos_real_px", "r"]

    summary_rows = []
    for name in experiments:
        img_dir = args.results_root / name / "test_latest" / "images"
        fake_dir, real_dir = img_dir / "fake_B", img_dir / "real_B"
        if not fake_dir.is_dir() or not real_dir.is_dir():
            print(f"[skip] {name}: no results under {img_dir}")
            continue

        stems = sorted(p.stem for p in fake_dir.glob("*.png") if (real_dir / p.name).exists())
        print(f"[{name}] scoring {len(stems)} pairs...")

        per_image = []
        for stem in stems:
            row = {"stem": stem}
            row.update(score_pair(fake_dir / f"{stem}.png", real_dir / f"{stem}.png"))
            per_image.append(row)

        with open(args.per_image_dir / f"{name}.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stem"] + metric_keys + extra_keys, delimiter="\t")
            w.writeheader()
            w.writerows(per_image)

        agg = {"name": name, "n": len(per_image)}
        for k in metric_keys:
            vals = np.array([row[k] for row in per_image])
            agg[f"{k}_mean"] = float(vals.mean())
            agg[f"{k}_std"] = float(vals.std())

        r_vals = np.array([row["r"] for row in per_image], dtype=float)
        valid = r_vals[~np.isnan(r_vals)]
        agg["r_n_valid"] = int(valid.size)
        agg["r_n_skipped"] = int(r_vals.size - valid.size)  # real patch had 0 DAB-positive px
        agg["r_mean"] = float(valid.mean()) if valid.size else float("nan")
        agg["r_median"] = float(np.median(valid)) if valid.size else float("nan")
        agg["r_std"] = float(valid.std()) if valid.size else float("nan")
        summary_rows.append(agg)

    if not summary_rows:
        raise RuntimeError("No experiments scored -- did run_erg_v2_test_sweep.sh finish?")

    fieldnames = (["name", "n"] + [f"{k}_{s}" for k in metric_keys for s in ("mean", "std")]
                  + ["r_mean", "r_median", "r_std", "r_n_valid", "r_n_skipped"])
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nWrote {args.out}")
    print(f"{'name':<26}{'ssim':>8}{'psnr':>8}{'dab_mae':>10}{'area_bias':>12}{'area_abs':>10}{'r_mean':>10}{'r_median':>10}{'r_n_skip':>10}")
    for r in sorted(summary_rows, key=lambda r: -r["ssim_mean"]):
        print(f"{r['name']:<26}{r['ssim_mean']:>8.4f}{r['psnr_mean']:>8.2f}"
              f"{r['dab_mae_mean']:>10.4f}{r['dab_area_bias_mean']:>+12.4f}{r['dab_area_abs_diff_mean']:>10.4f}"
              f"{r['r_mean']:>10.3f}{r['r_median']:>10.3f}{r['r_n_skipped']:>10}")


if __name__ == "__main__":
    main()
