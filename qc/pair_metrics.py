#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from qc_utils import (
    dab_stats,
    detect_nuclei,
    edge_map,
    imread_rgb,
    nmi,
    pad_same_shape,
    pearson_ncc,
    point_match_ratio,
    read_path_list,
    residual_shift,
    resize_for_qc,
    robust_norm,
    stain_channels,
    tissue_mask,
)


def empty_metrics(stem, he_path, ihc_path, error):
    return {
        "stem": stem,
        "he_path": he_path,
        "ihc_path": ihc_path,
        "he_tissue_ratio": 0.0,
        "ihc_tissue_ratio": 0.0,
        "tissue_iou": 0.0,
        "dab_area_ratio": 0.0,
        "dab_component_count": 0,
        "h_ncc": 0.0,
        "h_nmi": 0.0,
        "edge_ncc": 0.0,
        "nuclei_match": 0.0,
        "residual_shift_qc_px": 999.0,
        "phase_response": 0.0,
        "error": error,
    }


def measure_pair(job):
    (
        stem,
        he_path,
        ihc_path,
        qc_size,
        dab_od_threshold,
        dab_min_area,
        dab_max_area,
        nuclei_min_area,
        nuclei_max_area,
        nuclei_match_dist,
    ) = job

    he = imread_rgb(he_path)
    ihc = imread_rgb(ihc_path)
    if he is None or ihc is None:
        return empty_metrics(stem, he_path, ihc_path, "read_failed")

    he = resize_for_qc(he, qc_size)
    ihc = resize_for_qc(ihc, qc_size)
    he, ihc = pad_same_shape(he, ihc)

    he_tissue = tissue_mask(he)
    ihc_tissue = tissue_mask(ihc)
    he_ratio = float(he_tissue.mean())
    ihc_ratio = float(ihc_tissue.mean())

    intersection = he_tissue & ihc_tissue
    union = he_tissue | ihc_tissue
    tissue_iou = float(intersection.sum() / max(int(union.sum()), 1))

    he_h, _ = stain_channels(he)
    ihc_h, ihc_dab = stain_channels(ihc)

    dab_ratio, dab_count = dab_stats(
        ihc_dab,
        ihc_tissue,
        dab_od_threshold,
        dab_min_area,
        dab_max_area,
    )

    h_ncc = 0.0
    h_nmi = 0.0
    edge_ncc = 0.0
    nuclei_match = 0.0
    shift = 999.0
    phase_response = 0.0

    # 只在没有足够重叠像素时停止；不再按可调 QC 阈值提前退出。
    if np.count_nonzero(intersection) >= 128:
        he_hn = robust_norm(he_h, he_tissue)
        ihc_hn = robust_norm(ihc_h, ihc_tissue)

        h_ncc = pearson_ncc(he_hn, ihc_hn, intersection)
        h_nmi = nmi(he_hn, ihc_hn, intersection)
        edge_ncc = pearson_ncc(edge_map(he_hn), edge_map(ihc_hn), intersection)

        he_nuclei = detect_nuclei(he_h, he_tissue, nuclei_min_area, nuclei_max_area)
        ihc_nuclei = detect_nuclei(ihc_h, ihc_tissue, nuclei_min_area, nuclei_max_area)
        nuclei_match = point_match_ratio(he_nuclei, ihc_nuclei, nuclei_match_dist)
        shift, phase_response = residual_shift(he_h, ihc_h, intersection)

    return {
        "stem": stem,
        "he_path": he_path,
        "ihc_path": ihc_path,
        "he_tissue_ratio": he_ratio,
        "ihc_tissue_ratio": ihc_ratio,
        "tissue_iou": tissue_iou,
        "dab_area_ratio": dab_ratio,
        "dab_component_count": dab_count,
        "h_ncc": h_ncc,
        "h_nmi": h_nmi,
        "edge_ncc": edge_ncc,
        "nuclei_match": nuclei_match,
        "residual_shift_qc_px": shift,
        "phase_response": phase_response,
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Compute raw metrics for every paired H&E-IHC image.")
    parser.add_argument("--he-list", required=True)
    parser.add_argument("--ihc-list", required=True)
    parser.add_argument("--out", default="pair_metrics.tsv")
    parser.add_argument("--qc-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))

    parser.add_argument("--dab-od-threshold", type=float, default=0.06)
    parser.add_argument("--dab-min-area", type=int, default=3)
    parser.add_argument("--dab-max-area", type=int, default=300)
    parser.add_argument("--nuclei-min-area", type=int, default=3)
    parser.add_argument("--nuclei-max-area", type=int, default=250)
    parser.add_argument("--nuclei-match-dist", type=float, default=6.0)
    args = parser.parse_args()

    he = read_path_list(args.he_list)
    ihc = read_path_list(args.ihc_list)
    common = sorted(set(he) & set(ihc))

    print(f"HE: {len(he)}, IHC: {len(ihc)}, paired: {len(common)}")

    jobs = [
        (
            stem,
            he[stem],
            ihc[stem],
            args.qc_size,
            args.dab_od_threshold,
            args.dab_min_area,
            args.dab_max_area,
            args.nuclei_min_area,
            args.nuclei_max_area,
            args.nuclei_match_dist,
        )
        for stem in common
    ]

    if args.workers == 1:
        rows = [measure_pair(job) for job in tqdm(jobs, desc="Pair metrics")]
    else:
        with Pool(args.workers) as pool:
            rows = list(
                tqdm(
                    pool.imap_unordered(measure_pair, jobs, chunksize=32),
                    total=len(jobs),
                    desc="Pair metrics",
                )
            )

    df = pd.DataFrame(rows).sort_values("stem")
    df.to_csv(args.out, sep="\t", index=False, float_format="%.6f")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
