#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Object-level evaluation of virtual ERG staining: does the generated image
give you the same *vessels* as the real IHC?

Why this replaces SSIM/PSNR as the headline metric
--------------------------------------------------
On the v3 sweep the eight configurations scored SSIM 0.494-0.521 -- a spread of
0.027 against a within-configuration std of 0.062. SSIM on an H&E->IHC
translation is dominated by the haematoxylin background, which every model
reproduces, so it cannot separate models. The DAB area ratio r was worse: real
DAB covers a median 0.85% of a tile, so the denominator is tiny and r's std
came out at 9.3-51.

The downstream task is counting and measuring vessels, so the metric should be
too. This script runs the same detector (scripts/vessel_features.py, optical
density so one threshold works on both real and generated images) over real_B
and fake_B, matches instances by centroid proximity, and reports:

  detection      precision / recall / F1 of vessels found in fake vs real
  counting       Spearman rho and MAE between per-tile vessel counts
  area           Bland-Altman bias and 95% limits of agreement on vessel
                 area fraction -- the form a method-comparison study needs
  morphology     median equivalent diameter and circularity on each side, so
                 a model that finds the right *number* of wrong-shaped
                 vessels is visible

Unlike SSIM these tolerate a few pixels of registration error (matching runs at
--match-dist, default 40px) but not gross misalignment, so --reg-group A
restricts to well-registered tiles by default when a manifest is supplied.

Caveat that applies to any run against datasets/cut2: 89.6% of its test tiles
share pixels with training tiles, so numbers from it are optimistic regardless
of which metric is used. Rerun on the slide-disjoint datasets/cut3 test split
once GPU time is available; the machinery is the point here.
"""
import argparse
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vessel_features import detect_vessels, vessel_instances, tissue_mask  # noqa: E402

MATCH_DIST = 40  # px between centroids for a real/fake vessel to count as the same one


class DetectArgs:
    """detect_vessels/vessel_instances read their knobs off an argparse
    namespace; this mirrors the subset they touch."""

    def __init__(self, a):
        for k in ("dab_threshold", "ring_link_radius", "ring_min_fragments",
                  "ring_max_fill_ratio", "smooth_ksize", "min_area", "max_area", "min_extent"):
            setattr(self, k, getattr(a, k))


def match_instances(real, fake, max_dist=MATCH_DIST):
    """Greedy nearest-centroid one-to-one matching. Returns (tp, fp, fn)."""
    if not real or not fake:
        return 0, len(fake), len(real)
    r = np.array([[v["cx"], v["cy"]] for v in real], dtype=np.float32)
    f = np.array([[v["cx"], v["cy"]] for v in fake], dtype=np.float32)
    d = np.linalg.norm(r[:, None, :] - f[None, :, :], axis=-1)

    tp = 0
    used_r, used_f = set(), set()
    for ri, fi in sorted(np.ndindex(d.shape), key=lambda ix: d[ix]):
        if d[ri, fi] > max_dist:
            break
        if ri in used_r or fi in used_f:
            continue
        used_r.add(ri)
        used_f.add(fi)
        tp += 1
    return tp, len(fake) - tp, len(real) - tp


def spearman(a, b):
    """Rank correlation, implemented here so the script has no scipy dependency."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")

    def rank(x):
        order = x.argsort()
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        # average ranks within ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def median(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return float(np.median(xs)) if xs else float("nan")


def eval_one_tile(rp_str, fp_str, stem, dargs, match_dist):
    real_rgb = cv2.cvtColor(cv2.imread(rp_str), cv2.COLOR_BGR2RGB)
    fake_rgb = cv2.cvtColor(cv2.imread(fp_str), cv2.COLOR_BGR2RGB)

    rring, rdab = detect_vessels(real_rgb, dargs)
    fring, fdab = detect_vessels(fake_rgb, dargs)
    rinst = vessel_instances(rring, rdab, dargs)
    finst = vessel_instances(fring, fdab, dargs)

    tp, fp, fn = match_instances(rinst, finst, match_dist)
    tis = max(1, int((tissue_mask(real_rgb) > 0).sum()))
    return {
        "stem": stem,
        "n_real": len(rinst), "n_fake": len(finst),
        "tp": tp, "fp": fp, "fn": fn,
        "real_area_frac": sum(v["area_px"] for v in rinst) / tis,
        "fake_area_frac": sum(v["area_px"] for v in finst) / tis,
        "real_diam_median": median([v["equiv_diam_px"] for v in rinst]),
        "fake_diam_median": median([v["equiv_diam_px"] for v in finst]),
        "real_circ_median": median([v["circularity"] for v in rinst]),
        "fake_circ_median": median([v["circularity"] for v in finst]),
    }


def evaluate(name, args, stems, pool):
    img_dir = args.results_root / name / "test_latest" / "images"
    real_dir, fake_dir = img_dir / "real_B", img_dir / "fake_B"
    dargs = DetectArgs(args)

    futs = []
    for stem in stems:
        rp, fp_ = real_dir / f"{stem}.png", fake_dir / f"{stem}.png"
        if not (rp.exists() and fp_.exists()):
            continue
        futs.append(pool.submit(eval_one_tile, str(rp), str(fp_), stem, dargs, args.match_dist))
    rows = [f.result() for f in futs]
    rows.sort(key=lambda r: r["stem"])
    tp_t = sum(r["tp"] for r in rows)
    fp_t = sum(r["fp"] for r in rows)
    fn_t = sum(r["fn"] for r in rows)

    prec = tp_t / (tp_t + fp_t) if tp_t + fp_t else float("nan")
    rec = tp_t / (tp_t + fn_t) if tp_t + fn_t else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not math.isnan(prec + rec) else float("nan")

    nr = [r["n_real"] for r in rows]
    nf = [r["n_fake"] for r in rows]
    diff = np.array([r["fake_area_frac"] - r["real_area_frac"] for r in rows], dtype=float)
    return rows, {
        "name": name,
        "n_tiles": len(rows),
        "n_vessels_real": sum(nr),
        "n_vessels_fake": sum(nf),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "count_spearman": round(spearman(nr, nf), 4),
        "count_mae": round(float(np.mean(np.abs(np.array(nf) - np.array(nr)))), 3) if rows else "",
        "area_bias": round(float(diff.mean()), 6) if rows else "",
        "area_loa_lo": round(float(diff.mean() - 1.96 * diff.std()), 6) if rows else "",
        "area_loa_hi": round(float(diff.mean() + 1.96 * diff.std()), 6) if rows else "",
        "diam_median_real": round(median([r["real_diam_median"] for r in rows]), 2),
        "diam_median_fake": round(median([r["fake_diam_median"] for r in rows]), 2),
        "circ_median_real": round(median([r["real_circ_median"] for r in rows]), 4),
        "circ_median_fake": round(median([r["fake_circ_median"] for r in rows]), 4),
    }


def main():
    p = argparse.ArgumentParser(description="Object-level real-vs-virtual vessel agreement.")
    p.add_argument("--names", nargs="+", required=True, help="experiment names under --results-root")
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--out-dir", type=Path, default=Path("qc/vessel_agreement"))
    p.add_argument("--manifest", type=Path, default=None,
                   help="split_manifest.tsv, to restrict to a reg_group")
    p.add_argument("--reg-group", default="A", help="reg_group to keep when --manifest is given; 'all' to skip")
    p.add_argument("--match-dist", type=int, default=MATCH_DIST)
    p.add_argument("--dab-threshold", type=float, default=0.10)
    p.add_argument("--ring-link-radius", type=int, default=20)
    p.add_argument("--ring-min-fragments", type=int, default=2)
    p.add_argument("--ring-max-fill-ratio", type=float, default=0.5)
    p.add_argument("--smooth-ksize", type=int, default=9)
    p.add_argument("--min-area", type=int, default=30)
    p.add_argument("--max-area", type=int, default=12000)
    p.add_argument("--min-extent", type=int, default=6)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    keep = None
    if args.manifest:
        with open(args.manifest, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        rows = [r for r in rows if r.get("split", "test") == "test"]
        if args.reg_group != "all":
            rows = [r for r in rows if r.get("reg_group") == args.reg_group]
        keep = {r["stem"] for r in rows}
        print(f"manifest: keeping {len(keep)} test stems"
              + ("" if args.reg_group == "all" else f" in reg_group {args.reg_group}"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    pool = ProcessPoolExecutor(max_workers=args.workers)
    for name in args.names:
        img_dir = args.results_root / name / "test_latest" / "images" / "fake_B"
        stems = sorted(p_.stem for p_ in img_dir.glob("*.png"))
        if keep is not None:
            stems = [s for s in stems if s in keep]
        print(f"{name}: {len(stems)} tiles", flush=True)
        rows, summary = evaluate(name, args, stems, pool)
        summaries.append(summary)

        with open(args.out_dir / f"{name}.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"  P={summary['precision']} R={summary['recall']} F1={summary['f1']} "
              f"rho={summary['count_spearman']} real/fake vessels="
              f"{summary['n_vessels_real']}/{summary['n_vessels_fake']}", flush=True)

    pool.shutdown()
    out = args.out_dir / "vessel_agreement_scores.tsv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(summaries)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
