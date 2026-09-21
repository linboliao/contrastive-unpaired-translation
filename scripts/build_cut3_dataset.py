#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Slide-disjoint CUT dataset split, replacing scripts/build_cut2_dataset.py's
registration-only stratification.

Why cut2 had to be rebuilt
--------------------------
cut2 split by reg_group alone (test = all reg_group A, train = all B), which
fixed the *registration* confound but left a much larger one untouched: it
never looked at which slide a tile came from. Measured on the shipped
cut2/split_manifest.tsv:

  * 104 of the 106 test tiles come from 20-03444HE -- a slide that also
    contributes 1540 training tiles.
  * Tiles are 2048px cut on a 1024px stride, i.e. 50% overlap in both axes,
    so neighbouring tiles literally share pixels. 95 of the 106 test tiles
    (89.6%) overlap at least one *training* tile in raw pixel coordinates.

So cut2's SSIM/PSNR measure reconstruction of pixels the generator was
trained on. This script splits at the slide level instead, which is the only
level at which "unseen" means anything here: different scanning session,
different staining batch, different patient.

The cost, stated plainly
------------------------
reg_group A is concentrated almost entirely on one slide (104 of the 120 A
tiles across all 15 slides are 20-03444HE). Under a slide-disjoint split you
cannot have both a large training set and a large well-registered test set.
Two consequences, both intended:

  * 20-03444HE is placed in test by default, so *some* paired-pixel eval
    stays possible (104 A tiles) as a secondary reference.
  * Everything else must be evaluated with registration-free, object-level
    metrics (vessel counts/areas/morphology -- see
    scripts/evaluate_vessel_agreement.py). Paired SSIM/PSNR on the other
    test slides would be measuring registration error, not model quality.

Splits written: train / val / test (slide-disjoint), plus a manifest carrying
slide, tile coordinates, reg_group and reg_score so downstream evaluation can
filter to A tiles without re-reading the QC tables.
"""
import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

BASE = Path("/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准")

# Slide-disjoint assignment. Held fixed (not seeded-random) so every rerun and
# every experiment compares on exactly the same slides.
#   test: 20-03444HE is the only slide with a usable reg_group A set (104
#         tiles), so it anchors test; the two supp slides add unseen-batch
#         diversity without costing much training data.
#   val:  one slide from each batch, used for checkpoint selection only.
DEFAULT_TEST_SLIDES = ["20-03444HE", "20-07345HE", "20-23577HE"]
DEFAULT_VAL_SLIDES = ["20-05843HE", "20-10294HE"]

TILE_SIZE = 2048  # tiles are cut 2048px on a 1024px stride -> 50% overlap


def parse_stem(stem):
    """'20-03444HE_21504_15360' -> ('20-03444HE', 21504, 15360)."""
    parts = stem.split("_")
    return parts[0], int(parts[1]), int(parts[2])


def transfer_file(src: Path, dst: Path, mode="hardlink"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def thin_overlapping(rows, stride_mult=2):
    """Keep only tiles on a coarser grid so the kept tiles no longer share
    pixels with each other. Tiles sit on a 1024px lattice at 2048px each, so
    taking every other position in both axes (stride_mult=2) gives a
    non-overlapping cover. Applied to val/test only: overlapping test tiles
    are not independent samples, and treating them as such makes every
    confidence interval too narrow."""
    step = 1024 * stride_mult
    kept = []
    for r in rows:
        _, x, y = parse_stem(r["stem"])
        if x % step == 0 and y % step == 0:
            kept.append(r)
    return kept


def main():
    parser = argparse.ArgumentParser(description="Slide-disjoint CUT dataset split.")
    parser.add_argument("--pair-scores", type=Path, nargs="+",
                        default=[BASE / "qc/105/pair_scores.tsv", BASE / "qc/105supp/pair_scores.tsv"],
                        help="one or more pair_scores.tsv; both batches are pooled")
    parser.add_argument("--output", type=Path, default=BASE / "datasets/cut3")
    parser.add_argument("--test-slides", nargs="+", default=DEFAULT_TEST_SLIDES)
    parser.add_argument("--val-slides", nargs="+", default=DEFAULT_VAL_SLIDES)
    parser.add_argument("--train-groups", nargs="+", default=["A", "B"],
                        help="reg_groups usable for training (CUT tolerates weak pairing)")
    parser.add_argument("--eval-groups", nargs="+", default=["A", "B"],
                        help="reg_groups kept in val/test; filter to A at eval time for paired metrics")
    parser.add_argument("--thin-eval", type=int, default=2,
                        help="keep val/test tiles on a 1024*N grid so they don't overlap each other; 1 disables")
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="hardlink")
    parser.add_argument("--dry-run", action="store_true", help="print the split table and exit")
    args = parser.parse_args()

    rows = []
    for path in args.pair_scores:
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("error"):
                    continue
                r["slide"], r["x"], r["y"] = parse_stem(r["stem"])
                rows.append(r)

    test_slides, val_slides = set(args.test_slides), set(args.val_slides)
    overlap = test_slides & val_slides
    if overlap:
        raise SystemExit(f"test and val share slides: {sorted(overlap)}")
    all_slides = {r["slide"] for r in rows}
    missing = (test_slides | val_slides) - all_slides
    if missing:
        raise SystemExit(f"unknown slides requested: {sorted(missing)}")

    buckets = defaultdict(list)
    for r in rows:
        if r["slide"] in test_slides:
            split, groups = "test", args.eval_groups
        elif r["slide"] in val_slides:
            split, groups = "val", args.eval_groups
        else:
            split, groups = "train", args.train_groups
        if r["reg_group"] in groups:
            buckets[split].append(r)

    if args.thin_eval > 1:
        for split in ("val", "test"):
            before = len(buckets[split])
            buckets[split] = thin_overlapping(buckets[split], args.thin_eval)
            print(f"{split}: thinned {before} -> {len(buckets[split])} tiles (non-overlapping grid)")

    print(f"\n{'split':<6} {'slide':<12} {'A':>5} {'B':>5} {'total':>6}")
    for split in ("train", "val", "test"):
        per = defaultdict(Counter)
        for r in buckets[split]:
            per[r["slide"]][r["reg_group"]] += 1
        for slide in sorted(per):
            c = per[slide]
            print(f"{split:<6} {slide:<12} {c['A']:>5} {c['B']:>5} {sum(c.values()):>6}")
    print()
    for split in ("train", "val", "test"):
        c = Counter(r["reg_group"] for r in buckets[split])
        print(f"{split:<6} total={len(buckets[split]):>5}  A={c['A']:<5} B={c['B']:<5} "
              f"slides={len({r['slide'] for r in buckets[split]})}")

    if args.dry_run:
        return

    dir_names = {(s, side): f"{s}{side}" for s in ("train", "val", "test") for side in ("A", "B")}
    for name in dir_names.values():
        (args.output / name).mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        dir_a = args.output / dir_names[(split, "A")]
        dir_b = args.output / dir_names[(split, "B")]
        for row in tqdm(buckets[split], desc=f"{split:<5}", unit="pair", dynamic_ncols=True):
            he_path, ihc_path = Path(row["he_path"]), Path(row["ihc_path"])
            transfer_file(he_path, dir_a / he_path.name, args.mode)
            transfer_file(ihc_path, dir_b / ihc_path.name, args.mode)

    manifest = args.output / "split_manifest.tsv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["split", "stem", "slide", "x", "y", "reg_group", "reg_score"])
        for split in ("train", "val", "test"):
            for r in buckets[split]:
                w.writerow([split, r["stem"], r["slide"], r["x"], r["y"], r["reg_group"], r["reg_score"]])

    print(f"\nOutput:   {args.output}\nManifest: {manifest}\nA: HE -> B: dhr (real ERG IHC)")


if __name__ == "__main__":
    main()
