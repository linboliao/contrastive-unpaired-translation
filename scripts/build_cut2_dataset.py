#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuilds the CUT dataset with a *stratified* (not random) train/test split
by registration confidence, fixing the eval-methodology bug flagged in
review: the old CUT_105 split randomly mixed reg_group A (well-registered,
strict thresholds in qc/score_pairs.py: tissue_iou>=0.55, reg_score>=0.50,
residual_shift<=18px) and B (weak) pairs into both train AND test, so
pixel-paired eval metrics (SSIM/PSNR/DAB MAE/r, seg Dice/IoU) were partly
measuring registration error, not model quality -- only ~15/200 sampled test
pairs turned out to be reg_group A.

New split:
  testA/testB  = ALL reg_group A pairs (106) -- registration is trustworthy,
                 so paired pixel metrics on this set actually mean something.
  trainA/trainB = ALL reg_group B pairs (3919) -- CUT tolerates imperfect
                 pairing by design (unpaired GAN + NCE), and the DAB loss
                 already down-weights by reg_score, so weak pairs are fine
                 for training but not for evaluation.
  reg_group C (1078, failed tissue/QC) is excluded from both, as before.

No random split, no overlap between train and test by construction.
"""
import argparse
import csv
import os
import shutil
from pathlib import Path

from tqdm import tqdm

IGNORE_DIRS = {"@eaDir", ".DS_Store", "__MACOSX", ".ipynb_checkpoints"}


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


def main():
    parser = argparse.ArgumentParser(description="Stratified (reg_group) CUT dataset split.")
    parser.add_argument("--pair-scores", type=Path,
                         default=Path("/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/qc/105/pair_scores.tsv"))
    parser.add_argument("--output", type=Path,
                         default=Path("/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/datasets/cut2"))
    parser.add_argument("--test-group", default="A", help="reg_group sent to test (trustworthy for paired eval)")
    parser.add_argument("--train-group", default="B", help="reg_group sent to train (CUT tolerates the noise)")
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="hardlink")
    args = parser.parse_args()

    with open(args.pair_scores, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if not r.get("error")]

    train_rows = [r for r in rows if r["reg_group"] == args.train_group]
    test_rows = [r for r in rows if r["reg_group"] == args.test_group]
    other = len(rows) - len(train_rows) - len(test_rows)
    print(f"pairs: train(group {args.train_group})={len(train_rows)}  "
          f"test(group {args.test_group})={len(test_rows)}  excluded(other groups)={other}")

    dirs = {d: args.output / d for d in ["trainA", "trainB", "testA", "testB"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows, dir_a, dir_b, desc in [
        ("train", train_rows, dirs["trainA"], dirs["trainB"], "Train"),
        ("test", test_rows, dirs["testA"], dirs["testB"], "Test "),
    ]:
        for row in tqdm(split_rows, desc=desc, unit="pair", dynamic_ncols=True):
            he_path, ihc_path = Path(row["he_path"]), Path(row["ihc_path"])
            transfer_file(he_path, dir_a / he_path.name, args.mode)
            transfer_file(ihc_path, dir_b / ihc_path.name, args.mode)

    manifest = args.output / "split_manifest.tsv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["split", "stem", "reg_group", "reg_score"])
        for r in train_rows:
            w.writerow(["train", r["stem"], r["reg_group"], r["reg_score"]])
        for r in test_rows:
            w.writerow(["test", r["stem"], r["reg_group"], r["reg_score"]])

    print(f"\nFinished!\nOutput: {args.output}\nTrain: {len(train_rows)} | Test: {len(test_rows)}\n"
          f"Manifest: {manifest}\nA: HE -> B: @105_dhr (ERG)")


if __name__ == "__main__":
    main()
