#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from qc_utils import imread_rgb, read_path_list, resize_for_qc, tissue_mask


def measure_one(job):
    stain, stem, path, qc_size = job
    image = imread_rgb(path)
    if image is None:
        return {
            "stain": stain,
            "stem": stem,
            "path": path,
            "tissue_ratio": 0.0,
            "error": "read_failed",
        }

    image = resize_for_qc(image, qc_size)
    ratio = float(tissue_mask(image).mean())
    return {
        "stain": stain,
        "stem": stem,
        "path": path,
        "tissue_ratio": ratio,
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Measure tissue ratio for every H&E/IHC image.")
    parser.add_argument("--he-list", required=True)
    parser.add_argument("--ihc-list", required=True)
    parser.add_argument("--out", default="tissue_qc.tsv")
    parser.add_argument("--qc-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    args = parser.parse_args()

    he = read_path_list(args.he_list)
    ihc = read_path_list(args.ihc_list)

    jobs = [("HE", stem, path, args.qc_size) for stem, path in he.items()]
    jobs += [("IHC", stem, path, args.qc_size) for stem, path in ihc.items()]

    if args.workers == 1:
        rows = [measure_one(job) for job in tqdm(jobs, desc="Tissue QC")]
    else:
        with Pool(args.workers) as pool:
            rows = list(
                tqdm(
                    pool.imap_unordered(measure_one, jobs, chunksize=64),
                    total=len(jobs),
                    desc="Tissue QC",
                )
            )

    df = pd.DataFrame(rows).sort_values(["stain", "stem"])
    df.to_csv(args.out, sep="\t", index=False, float_format="%.6f")

    print(f"HE images : {len(he)}")
    print(f"IHC images: {len(ihc)}")
    print(f"Saved     : {args.out}")


if __name__ == "__main__":
    main()
