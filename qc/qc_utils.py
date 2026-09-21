#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2hed

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
cv2.setNumThreads(0)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def read_path_list(txt_path):
    """Read one-image-path-per-line txt and return {stem: path}."""
    items = {}
    duplicates = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            if not path:
                continue

            p = Path(path)
            if p.suffix.lower() not in IMG_EXTS:
                continue

            if p.stem in items:
                duplicates.append(p.stem)
            items[p.stem] = path

    if duplicates:
        raise RuntimeError(f"{txt_path} 存在重复 stem: {duplicates[:10]}")

    return items


def imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_for_qc(rgb, qc_size):
    h, w = rgb.shape[:2]
    scale = qc_size / max(h, w)
    if scale >= 1:
        return rgb

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def pad_same_shape(a, b):
    height = max(a.shape[0], b.shape[0])
    width = max(a.shape[1], b.shape[1])

    def pad(x):
        out = np.full((height, width, 3), 255, dtype=np.uint8)
        out[: x.shape[0], : x.shape[1]] = x
        return out

    return pad(a), pad(b)


def tissue_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)[..., 1] / 255.0

    # 新增:局部纹理判据,过滤"发灰但平滑"的空白背景
    mean = cv2.blur(gray, (9, 9))
    sq_mean = cv2.blur(gray * gray, (9, 9))
    local_std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))

    mask = (((gray < 0.93) | (sat > 0.08)) & (local_std > 0.02)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 64
    return keep[labels]


def stain_channels(rgb):
    hed = rgb2hed(rgb.astype(np.float32) / 255.0)
    h = np.maximum(hed[..., 0], 0).astype(np.float32)
    dab = np.maximum(hed[..., 2], 0).astype(np.float32)
    return h, dab


def robust_norm(x, mask=None):
    if mask is not None and np.count_nonzero(mask) >= 64:
        values = x[mask]
    else:
        values = x.reshape(-1)

    lo, hi = np.percentile(values, [1.0, 99.5])
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def pearson_ncc(a, b, mask):
    if np.count_nonzero(mask) < 128:
        return 0.0

    x = a[mask].astype(np.float32)
    y = b[mask].astype(np.float32)
    x -= x.mean()
    y -= y.mean()

    den = np.sqrt((x * x).sum() * (y * y).sum())
    if den <= 1e-12:
        return 0.0
    return float(np.clip((x * y).sum() / den, -1, 1))


def nmi(a, b, mask, bins=32):
    if np.count_nonzero(mask) < 128:
        return 0.0

    x = np.clip(a[mask], 0, 1).astype(np.float32)
    y = np.clip(b[mask], 0, 1).astype(np.float32)

    hist = cv2.calcHist([x, y], [0, 1], None, [bins, bins], [0, 1, 0, 1])
    pxy = hist / max(float(hist.sum()), 1.0)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    mi = np.sum(pxy[nz] * np.log((pxy[nz] + 1e-12) / (denom[nz] + 1e-12)))
    hx = -np.sum(px[px > 0] * np.log(px[px > 0] + 1e-12))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0] + 1e-12))

    if hx + hy <= 1e-12:
        return 0.0
    return float(np.clip(2.0 * mi / (hx + hy), 0, 1))


def edge_map(x):
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    max_value = float(mag.max())
    return mag / max_value if max_value > 1e-8 else mag


def component_centroids(binary, min_area, max_area):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    keep = [
        i
        for i in range(1, n)
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area
    ]

    if keep:
        points = np.asarray([centroids[i] for i in keep], dtype=np.float32)
    else:
        points = np.empty((0, 2), dtype=np.float32)

    total_area = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in keep)
    return points, len(keep), total_area


def detect_nuclei(h_channel, tissue, min_area, max_area):
    h_norm = robust_norm(h_channel, tissue)
    values = h_norm[tissue]
    if values.size < 128:
        return np.empty((0, 2), dtype=np.float32)

    threshold = max(0.35, float(np.percentile(values, 70)))
    binary = ((h_norm >= threshold) & tissue).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    points, _, _ = component_centroids(binary.astype(bool), min_area, max_area)
    return points


def point_match_ratio(p1, p2, max_dist):
    if len(p1) == 0 or len(p2) == 0:
        return 0.0

    if len(p1) > 512 or len(p2) > 512:
        from scipy.spatial import cKDTree

        tree1 = cKDTree(p1)
        tree2 = cKDTree(p2)
        d1, _ = tree2.query(p1, k=1, workers=-1)
        d2, _ = tree1.query(p2, k=1, workers=-1)
    else:
        distances = np.sqrt(((p1[:, None] - p2[None, :]) ** 2).sum(axis=2))
        d1 = distances.min(axis=1)
        d2 = distances.min(axis=0)

    return 0.5 * (float(np.mean(d1 <= max_dist)) + float(np.mean(d2 <= max_dist)))


def dab_stats(dab, tissue, od_threshold, min_area, max_area):
    binary = ((dab >= od_threshold) & tissue).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _, count, area = component_centroids(binary.astype(bool), min_area, max_area)
    tissue_area = max(int(np.count_nonzero(tissue)), 1)
    return area / tissue_area, count


def residual_shift(h1, h2, mask):
    if np.count_nonzero(mask) < 128:
        return 999.0, 0.0

    a = robust_norm(h1, mask) * mask.astype(np.float32)
    b = robust_norm(h2, mask) * mask.astype(np.float32)
    height, width = a.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    shift = math.sqrt(dx * dx + dy * dy)
    return float(shift), float(response)
