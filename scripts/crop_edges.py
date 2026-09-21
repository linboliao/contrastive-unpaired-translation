#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""以 HE 图片为准，同时裁切 HE 和 @105_dhr 目录中同名图片的四周边缘，
解决 DHR 配准后图像边缘拉伸的问题。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# lower compress_level (0-9): trades smaller files for CPU time; irrelevant to
# quality since PNG compression is lossless, only matters when NAS I/O is the
# bottleneck and we want to spend less time encoding.
PNG_COMPRESS_LEVEL = 1


def crop_pair(he_path: Path, dhr_path: Path, out_he: Path, out_dhr: Path, margin: int):
    if out_he.exists() and out_dhr.exists():
        return  # resume support: skip work already done in a previous run

    he_img = Image.open(he_path)
    dhr_img = Image.open(dhr_path)

    w, h = he_img.size
    if dhr_img.size != (w, h):
        raise ValueError(f"Size mismatch for {he_path.name}: HE={he_img.size} dhr={dhr_img.size}")

    box = (margin, margin, w - margin, h - margin)
    out_he.parent.mkdir(parents=True, exist_ok=True)
    out_dhr.parent.mkdir(parents=True, exist_ok=True)
    he_img.crop(box).save(out_he, compress_level=PNG_COMPRESS_LEVEL)
    dhr_img.crop(box).save(out_dhr, compress_level=PNG_COMPRESS_LEVEL)


def main():
    parser = argparse.ArgumentParser(description="Crop HE/@105_dhr image pairs by a fixed edge margin.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/cases/105"),
    )
    parser.add_argument("--he-subdir", default="raw/HE", help="input dir, relative to --root")
    parser.add_argument("--dhr-subdir", default="raw/dhr", help="input dir, relative to --root")
    parser.add_argument("--out-subdir", default="crop1748")
    parser.add_argument("--out-he-name", default="HE", help="output leaf dir name, under --out-subdir")
    parser.add_argument("--out-dhr-name", default="dhr", help="output leaf dir name, under --out-subdir")
    parser.add_argument("--margin", type=int, default=150, help="pixels to crop off each of the 4 edges")
    parser.add_argument("--workers", type=int, default=8, help="parallel threads (I/O-bound, helps when NAS is slow)")
    args = parser.parse_args()

    he_dir = args.root / args.he_subdir
    dhr_dir = args.root / args.dhr_subdir
    out_dir = args.root / args.out_subdir
    out_he_dir = out_dir / args.out_he_name
    out_dhr_dir = out_dir / args.out_dhr_name

    he_files = sorted(p for p in he_dir.iterdir() if p.suffix.lower() in EXTS)
    print(f"Found {len(he_files)} HE images under {he_dir}")

    missing = []
    jobs = []
    for he_path in he_files:
        dhr_path = dhr_dir / he_path.name
        if not dhr_path.exists():
            missing.append(he_path.name)
            continue
        jobs.append((he_path, dhr_path))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                crop_pair,
                he_path,
                dhr_path,
                out_he_dir / he_path.name,
                out_dhr_dir / he_path.name,
                args.margin,
            )
            for he_path, dhr_path in jobs
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="cropping", unit="img", dynamic_ncols=True):
            f.result()  # re-raise any worker exception

    if missing:
        print(f"Skipped {len(missing)} images with no matching @105_dhr file, e.g. {missing[:5]}")


if __name__ == "__main__":
    main()
