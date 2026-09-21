#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
IGNORE_DIRS = {"@eaDir", ".DS_Store", "__MACOSX", ".ipynb_checkpoints"}
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def collect_images(root, images=None):
    root = Path(root)
    if images is None:
        images = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in EXTS:
            continue
        if IGNORE_DIRS & set(path.parts):
            continue
        if path.name.startswith(("SYNOFILE_THUMB", "._")):
            continue
        if path.stem in images:
            raise RuntimeError(f"Duplicate stem: {path.stem} ({path} vs {images[path.stem]})")
        images[path.stem] = path
    return images          # ← 返回 dict，key 是 stem


def transfer_file(src: Path, dst: Path, mode="copy"):
    """复制/链接文件到目标路径"""
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


def process_split(keys, he_imgs, stain_imgs, dir_a, dir_b, mode, desc):
    """处理并转移一个 split 的数据"""
    for key in tqdm(keys, desc=desc, unit="pair", dynamic_ncols=True):
        transfer_file(he_imgs[key], dir_a / he_imgs[key].name, mode)
        transfer_file(stain_imgs[key], dir_b / stain_imgs[key].name, mode)


def main():
    parser = argparse.ArgumentParser(description="Convert paired HE/@105 images into CUT dataset format.")
    parser.add_argument("--root", type=Path, default=Path("/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/qc/105/selected"))
    parser.add_argument("--output", type=Path, default=Path("/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/datasets/_deprecated/cut105"))
    parser.add_argument("--groups", default="strong_A,weak_B",
                         help="comma-separated subdirs under --root, each holding HE/ and @105/ (empty to treat --root itself as the single group)")
    parser.add_argument("--stain-subdir", default="@105", help="name of the non-HE subdir inside each group")
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="hardlink")
    args = parser.parse_args()

    if not 0 <= args.test_ratio < 1:
        raise ValueError("--test-ratio must be in [0, 1)")

    group_names = [g for g in args.groups.split(",") if g] if args.groups else [None]
    group_roots = [args.root / g for g in group_names] if group_names[0] is not None else [args.root]

    he_imgs, stain_imgs = {}, {}
    for group_root in group_roots:
        he_dir, stain_dir = group_root / "HE", group_root / args.stain_subdir
        if not he_dir.is_dir() or not stain_dir.is_dir():
            raise FileNotFoundError(f"HE or {args.stain_subdir} folder not found under {group_root}.")
        print(f"Scanning {he_dir} and {stain_dir}...")
        collect_images(he_dir, he_imgs)
        collect_images(stain_dir, stain_imgs)

    he_keys, stain_keys = set(he_imgs), set(stain_imgs)
    common_keys = sorted(he_keys & stain_keys)

    print(f"Matched: {len(common_keys)} | HE only: {len(he_keys - stain_keys)} | @105 only: {len(stain_keys - he_keys)}")
    if not common_keys:
        raise RuntimeError("No matched images found by stem.")

    random.seed(args.seed)
    random.shuffle(common_keys)
    n_total = len(common_keys)

    # 计算 test 数量，保证边界条件
    n_test = int(n_total * args.test_ratio)
    if args.test_ratio > 0 and n_total > 1:
        n_test = max(1, n_test)
    n_test = min(n_test, n_total - 1) if n_total > 0 else 0

    test_keys, train_keys = common_keys[:n_test], common_keys[n_test:]

    # 创建目录
    dirs = [args.output / d for d in ["trainA", "trainB", "testA", "testB"]]
    for d in dirs: d.mkdir(parents=True, exist_ok=True)

    # 处理 train 和 test
    splits = [
        (train_keys, dirs[0], dirs[1], "Train"),
        (test_keys, dirs[2], dirs[3], "Test ")
    ]
    for keys, a, b, desc in splits:
        if keys:
            process_split(keys, he_imgs, stain_imgs, a, b, args.mode, desc)

    print(f"\nFinished!\nOutput: {args.output}\nTrain: {len(train_keys)} | Test: {len(test_keys)}\nA: HE -> B: @105")


if __name__ == "__main__":
    main()