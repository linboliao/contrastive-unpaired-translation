#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vessel instance detection + morphometry on ERG IHC tiles -- Stage 1-2 of the
downstream quantification pipeline, run directly on the *real* IHC so it does
not depend on virtual staining quality at all.

Detection reuses the ring-linking machinery tuned in
scripts/vessel_rbc_ring_detect.py (fragment grouping -> convex hull -> hollow
check), but swaps the DAB detector. That script's RGB box was hand-tuned on
*virtually* stained output, which is pale and low-contrast; real IHC DAB is
far more saturated, and an RGB box calibrated on one does not transfer to the
other. So the default backend here is optical-density colour deconvolution
(skimage rgb2hed, the Ruifrok & Johnston matrix already used project-wide via
models/stain_utils.py), thresholded at DAB_OD_THRESHOLD -- the same 0.15 cutoff
scripts/evaluate_erg_sweep.py uses for "ERG-positive". OD is a physical
quantity, so one threshold works across staining intensities and across real
vs virtual images, which is exactly what scripts/evaluate_vessel_agreement.py
needs when it compares the two.

Three tables come out:

  vessel_instances.tsv  one row per detected vessel: shape morphometry
                        (area/perimeter/circularity/solidity/eccentricity),
                        lumen vs wall split, and -- when --he-dir is given --
                        the red-blood-cell fraction inside the lumen, i.e. a
                        perfusion proxy separating open from collapsed vessels.
  vessel_tiles.tsv      one row per tile: vessel count, vessel area fraction,
                        tissue area, nearest-neighbour spacing.
  vessel_slides.tsv     one row per slide: MVD, area fraction, the median of
                        each shape feature, and the across-tile coefficient of
                        variation of each (intratumoural heterogeneity).

Tiles are cut 2048px on a 1024px stride, so adjacent tiles share half their
pixels and would double-count vessels. Slide-level aggregation therefore uses
only tiles on the non-overlapping 2048px lattice by default (--all-tiles to
override); per-tile and per-instance tables still cover everything.

MPP is NOT recorded anywhere in this dataset, so densities in mm^-2 depend on
--mpp being correct. Pixel-unit columns are always written alongside; check
--mpp against the scanner before quoting any MVD number.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2hed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vessel_rbc_ring_detect import (  # noqa: E402
    RBC_HSV_RANGES,
    denoise_mask,
    link_ring_fragments,
    rbc_mask,
    smooth_mask,
)

# 0.15 is what scripts/evaluate_erg_sweep.py uses for "ERG-positive pixel",
# but checked against overlays it misses the faintly stained vessel walls that
# a pathologist still counts, so detection runs a little lower. PROVISIONAL --
# this cutoff, and RING_LINK_RADIUS below, together define what counts as "one
# microvessel", which is a clinical definition and needs a pathologist to
# arbitrate against annotated tiles before any MVD number is quoted.
DAB_OD_THRESHOLD = 0.10
TISSUE_GRAY_MAX = 220      # background glass is near-white
TISSUE_SAT_MIN = 15

# Stained endothelial nuclei along one vessel wall sit apart from each other;
# at radius 7 a single vessel was being split into several "vessels". 20px
# merges them without joining genuinely separate vessels.
RING_LINK_RADIUS = 20
RING_MIN_FRAGMENTS = 2
RING_MAX_FILL_RATIO = 0.5
SMOOTH_KSIZE = 9
MIN_AREA = 30      # a single stained endothelial cell already counts as one microvessel
MAX_AREA = 12000
MIN_EXTENT = 6

DEFAULT_MPP = 0.5          # unverified -- see module docstring

INSTANCE_FIELDS = [
    "slide", "stem", "tile_x", "tile_y", "cx", "cy",
    "area_px", "perimeter_px", "equiv_diam_px", "circularity", "solidity",
    "eccentricity", "aspect_ratio", "dab_area_px", "lumen_area_px",
    "wall_frac", "rbc_frac", "n_fragments", "fill_ratio", "is_ring",
]
SHAPE_FEATURES = ["area_px", "equiv_diam_px", "circularity", "solidity",
                  "eccentricity", "aspect_ratio", "wall_frac", "rbc_frac",
                  "fill_ratio"]
# is_ring is 0/1, so its median over instances is just 0 whenever fewer than
# half of them are rings -- which is always (about a quarter are). It is
# aggregated as a mean instead, i.e. the fraction of vessels showing a lumen.
FRACTION_FEATURES = ["is_ring"]


def dab_mask_od(rgb, threshold=DAB_OD_THRESHOLD):
    """DAB mask by colour deconvolution: rgb2hed channel 2 is the DAB
    optical density. Robust to staining intensity in a way an RGB box is not."""
    dab = rgb2hed(rgb.astype(np.float32) / 255.0)[..., 2]
    return ((dab > threshold).astype(np.uint8)) * 255


def tissue_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = ((gray < TISSUE_GRAY_MAX) & (hsv[..., 1] > TISSUE_SAT_MIN)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def detect_vessels(rgb, args):
    """DAB mask -> denoise -> cluster nearby fragments -> one hull per cluster.
    Returns (vessel_mask, dab_mask).

    Note the deliberate difference from scripts/vessel_rbc_ring_detect.py, which
    keeps *only* clusters whose hull is hollow (a ring) and throws the rest
    away. That rule exists because on virtually stained images brown is
    unreliable, so a ring shape is the evidence that something is really a
    vessel. On real IHC the premise is inverted: DAB brown *is* the ERG signal,
    and standard microvessel counting (Weidner) scores any stained endothelial
    cell or cluster as one microvessel whether or not a lumen is visible.
    Applying the ring filter here was checked against overlays and dropped most
    of the plainly-stained vessels in a tile.

    So ring-ness is recorded as a feature (fill_ratio, is_ring, n_fragments)
    rather than used as a filter. --ring-only restores the old behaviour for
    the virtual-image case."""
    dab = denoise_mask(dab_mask_od(rgb, args.dab_threshold))

    if getattr(args, "ring_only", False):
        ring, _ = link_ring_fragments(dab, args.ring_link_radius,
                                      args.ring_min_fragments, args.ring_max_fill_ratio)
        return smooth_mask(ring, args.smooth_ksize), dab

    if args.ring_link_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * args.ring_link_radius + 1,) * 2)
        linked = cv2.dilate(dab, k)
    else:
        linked = dab
    n, labels, stats, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)

    # Work inside each cluster's bounding box. Doing `(dab > 0) & (labels == cid)`
    # over the whole 1748x1748 frame once per cluster is ~1e9 element ops per
    # tile with a few hundred clusters, which dominated runtime.
    out = np.zeros_like(dab)
    for cid in range(1, n):
        x, y, w, h, _ = stats[cid]
        sub_lab = labels[y:y + h, x:x + w]
        sub_dab = dab[y:y + h, x:x + w]
        px = ((sub_lab == cid) & (sub_dab > 0)).astype(np.uint8)
        if not px.any():
            continue
        pts = cv2.findNonZero(px)
        hull = cv2.convexHull(pts) + np.array([[[x, y]]], dtype=pts.dtype)
        cv2.drawContours(out, [hull], -1, 255, thickness=-1)
    return smooth_mask(out, args.smooth_ksize), dab


def vessel_instances(vessel_mask, dab, args, he_rbc=None):
    """One record per vessel contour that passes the size/extent filters."""
    contours, _ = cv2.findContours(vessel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, frag_labels = cv2.connectedComponents(dab, connectivity=8)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (args.min_area <= area <= args.max_area):
            continue
        _, _, bw, bh = cv2.boundingRect(c)
        if bw < args.min_extent or bh < args.min_extent:
            continue

        perim = cv2.arcLength(c, True)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        m = cv2.moments(c)
        cx = m["m10"] / m["m00"] if m["m00"] else float(c[:, 0, 0].mean())
        cy = m["m01"] / m["m00"] if m["m00"] else float(c[:, 0, 1].mean())

        ecc, aspect = float("nan"), float("nan")
        if len(c) >= 5:
            (_, _), (ma, MA), _ = cv2.fitEllipse(c)
            major, minor = max(ma, MA), min(ma, MA)
            if major > 0:
                ecc = math.sqrt(max(0.0, 1 - (minor / major) ** 2))
                aspect = major / max(minor, 1e-6)

        # Same bounding-box restriction as detect_vessels: rasterising each
        # contour into a full-frame buffer is what made this O(contours x pixels).
        bx, by, bw2, bh2 = cv2.boundingRect(c)
        filled = np.zeros((bh2, bw2), dtype=np.uint8)
        cv2.drawContours(filled, [c - np.array([[bx, by]])], -1, 255, thickness=-1)
        inside = filled > 0
        sub_dab = dab[by:by + bh2, bx:bx + bw2]
        inside_px = int(inside.sum())
        dab_px = int(((sub_dab > 0) & inside).sum())
        lumen_px = inside_px - dab_px
        rbc_frac = float("nan")
        if he_rbc is not None and lumen_px > 0:
            lumen = inside & (sub_dab == 0)
            rbc_frac = float((he_rbc[by:by + bh2, bx:bx + bw2][lumen] > 0).sum()) / lumen_px

        # How many separate stained fragments make up this vessel, and how much
        # of its outline is hollow -- a solid blob is a single stained cell, a
        # hollow multi-fragment outline is a lumen ringed by endothelium.
        labs = np.unique(frag_labels[by:by + bh2, bx:bx + bw2][inside])
        n_frag = int((labs != 0).sum())
        fill_ratio = dab_px / max(1, inside_px)

        out.append({
            "cx": round(cx, 1), "cy": round(cy, 1),
            "area_px": round(area, 1),
            "perimeter_px": round(perim, 1),
            "equiv_diam_px": round(2 * math.sqrt(area / math.pi), 2),
            "circularity": round(4 * math.pi * area / perim ** 2, 4) if perim > 0 else float("nan"),
            "solidity": round(area / hull_area, 4) if hull_area > 0 else float("nan"),
            "eccentricity": round(ecc, 4),
            "aspect_ratio": round(aspect, 3),
            "dab_area_px": dab_px,
            "lumen_area_px": lumen_px,
            "wall_frac": round(dab_px / area, 4) if area > 0 else float("nan"),
            "rbc_frac": round(rbc_frac, 4),
            "n_fragments": n_frag,
            "fill_ratio": round(fill_ratio, 4),
            "is_ring": int(n_frag >= args.ring_min_fragments and fill_ratio <= args.ring_max_fill_ratio),
        })
    return out


def nn_distances(instances, chunk=512):
    """Nearest-neighbour distance for every vessel centroid.

    Chunked because this is also called on a whole slide's instances: the
    largest slide has ~7200 vessels, and materialising the full pairwise
    difference tensor there is ~400MB, which pushed the aggregation into
    swap. Row blocks keep it to a few MB at the same result."""
    pts = np.array([[v["cx"], v["cy"]] for v in instances], dtype=np.float32)
    n = len(pts)
    if n < 2:
        return []
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        block = pts[i:i + chunk]
        d = np.linalg.norm(block[:, None, :] - pts[None, :, :], axis=-1)
        d[np.arange(len(block)), np.arange(i, i + len(block))] = np.inf
        out[i:i + chunk] = d.min(axis=1)
    return out.tolist()


def save_overlay(rgb, ring_mask, out_path):
    """Red contours on the IHC tile, so the detector can be eyeballed before
    any number it produces is quoted."""
    contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    cv2.drawContours(vis, contours, -1, (30, 30, 230), 3)
    cv2.imwrite(str(out_path), vis)


def process_tile(path_str, he_path_str, args, vis_dir=None):
    path = Path(path_str)
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    he_rbc = None
    if he_path_str:
        he = cv2.cvtColor(cv2.imread(he_path_str), cv2.COLOR_BGR2RGB)
        he_rbc = rbc_mask(he, RBC_HSV_RANGES)

    ring, dab = detect_vessels(rgb, args)
    inst = vessel_instances(ring, dab, args, he_rbc)
    if vis_dir is not None:
        save_overlay(rgb, ring, Path(vis_dir) / f"{path.stem}.png")
    tis = int((tissue_mask(rgb) > 0).sum())

    slide, x, y = path.stem.split("_")[0], int(path.stem.split("_")[1]), int(path.stem.split("_")[2])
    for v in inst:
        v.update(slide=slide, stem=path.stem, tile_x=x, tile_y=y)

    nn = nn_distances(inst)
    tile = {
        "slide": slide, "stem": path.stem, "tile_x": x, "tile_y": y,
        "n_vessels": len(inst),
        "tissue_px": tis,
        "vessel_area_px": int(sum(v["area_px"] for v in inst)),
        "dab_area_px": int(((dab > 0)).sum()),
        "vessel_area_frac": round(sum(v["area_px"] for v in inst) / tis, 6) if tis else 0.0,
        "dab_area_frac": round(float((dab > 0).sum()) / tis, 6) if tis else 0.0,
        "nn_dist_median_px": round(float(np.median(nn)), 1) if nn else "",
    }
    return inst, tile


def summarize(values):
    v = [x for x in values if isinstance(x, (int, float)) and not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    med = float(np.median(v))
    mean = float(np.mean(v))
    cv = float(np.std(v) / mean) if mean else float("nan")
    return med, cv


def main():
    p = argparse.ArgumentParser(description="Vessel detection + morphometry on ERG IHC tiles.")
    p.add_argument("--ihc-dir", type=Path, nargs="+", default=None,
                   help="directories of real ERG IHC tiles (cases/*/crop1748/dhr)")
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip detection; rebuild vessel_slides.tsv from the instance/tile "
                        "tables already in --out-dir (use after correcting --mpp)")
    p.add_argument("--he-dir", type=Path, nargs="+", default=None,
                   help="matching HE dirs, enables the RBC/perfusion feature")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mpp", type=float, default=DEFAULT_MPP,
                   help="microns per pixel; densities in mm^-2 are only as right as this")
    p.add_argument("--all-tiles", action="store_true",
                   help="aggregate over overlapping tiles too (double-counts vessels)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="debug: only process N tiles per dir")
    p.add_argument("--vis-dir", type=Path, default=None,
                   help="also write red-contour overlays here, to sanity-check the detector by eye")
    p.add_argument("--vis-every", type=int, default=1,
                   help="with --vis-dir, only render every Nth tile")
    p.add_argument("--dab-threshold", type=float, default=DAB_OD_THRESHOLD)
    p.add_argument("--ring-link-radius", type=int, default=RING_LINK_RADIUS)
    p.add_argument("--ring-min-fragments", type=int, default=RING_MIN_FRAGMENTS)
    p.add_argument("--ring-max-fill-ratio", type=float, default=RING_MAX_FILL_RATIO)
    p.add_argument("--smooth-ksize", type=int, default=SMOOTH_KSIZE)
    p.add_argument("--ring-only", action="store_true",
                   help="keep only hollow ring-shaped clusters (the virtual-image rule; "
                        "drops most genuinely stained vessels on real IHC -- see detect_vessels)")
    p.add_argument("--min-area", type=int, default=MIN_AREA)
    p.add_argument("--max-area", type=int, default=MAX_AREA)
    p.add_argument("--min-extent", type=int, default=MIN_EXTENT)
    args = p.parse_args()

    if args.aggregate_only:
        with open(args.out_dir / "vessel_instances.tsv", newline="") as f:
            inst = list(csv.DictReader(f, delimiter="\t"))
        with open(args.out_dir / "vessel_tiles.tsv", newline="") as f:
            tiles = [dict(r, tile_x=int(r["tile_x"]), tile_y=int(r["tile_y"]),
                          tissue_px=int(r["tissue_px"]), vessel_area_px=int(r["vessel_area_px"]),
                          dab_area_px=int(r["dab_area_px"]))
                     for r in csv.DictReader(f, delimiter="\t")]
        for v in inst:
            v["cx"], v["cy"] = float(v["cx"]), float(v["cy"])
        print(f"aggregate-only: {len(inst)} instances, {len(tiles)} tiles, mpp={args.mpp}")
        aggregate_slides(inst, tiles, args)
        return

    if not args.ihc_dir:
        raise SystemExit("--ihc-dir is required unless --aggregate-only is given")

    he_lookup = {}
    for d in (args.he_dir or []):
        for f in d.glob("*.png"):
            he_lookup[f.name] = str(f)

    jobs = []
    for d in args.ihc_dir:
        files = sorted(d.glob("*.png"))
        if args.limit:
            files = files[:args.limit]
        for f in files:
            jobs.append((str(f), he_lookup.get(f.name)))
    print(f"{len(jobs)} tiles, {len(he_lookup)} HE tiles available for the perfusion feature")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.vis_dir:
        args.vis_dir.mkdir(parents=True, exist_ok=True)
    all_inst, all_tiles = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_tile, ihc, he, args,
                            str(args.vis_dir) if args.vis_dir and i % args.vis_every == 0 else None)
                for i, (ihc, he) in enumerate(jobs)]
        for i, fut in enumerate(as_completed(futs), 1):
            inst, tile = fut.result()
            all_inst.extend(inst)
            all_tiles.append(tile)
            if i % 200 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}", flush=True)

    inst_path = args.out_dir / "vessel_instances.tsv"
    with open(inst_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INSTANCE_FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(all_inst, key=lambda r: (r["stem"], r["cy"], r["cx"])))

    tile_fields = ["slide", "stem", "tile_x", "tile_y", "n_vessels", "tissue_px",
                   "vessel_area_px", "dab_area_px", "vessel_area_frac", "dab_area_frac",
                   "nn_dist_median_px"]
    all_tiles.sort(key=lambda r: r["stem"])
    tile_path = args.out_dir / "vessel_tiles.tsv"
    with open(tile_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tile_fields, delimiter="\t")
        w.writeheader()
        w.writerows(all_tiles)

    aggregate_slides(all_inst, all_tiles, args)

    print(f"\ninstances: {len(all_inst)}  ->  {inst_path}")
    print(f"tiles    : {len(all_tiles)}  ->  {tile_path}")


def aggregate_slides(all_inst, all_tiles, args):
    """Slide-level table. Split out from main() so --aggregate-only can redo it
    from the saved per-instance/per-tile tables -- re-running detection over
    6877 tiles to change --mpp (which every mm^-2 column scales by) would be
    an hour of NAS I/O for arithmetic.

    Overlapping tiles share half their pixels, so this restricts to the
    non-overlapping 2048px lattice unless told otherwise."""
    def independent(t):
        return args.all_tiles or (t["tile_x"] % 2048 == 0 and t["tile_y"] % 2048 == 0)

    px_mm2 = (args.mpp / 1000.0) ** 2
    by_slide_tiles, by_slide_inst = defaultdict(list), defaultdict(list)
    keep_stems = {t["stem"] for t in all_tiles if independent(t)}
    for t in all_tiles:
        if independent(t):
            by_slide_tiles[t["slide"]].append(t)
    for v in all_inst:
        if v["stem"] in keep_stems:
            by_slide_inst[v["slide"]].append(v)

    slide_fields = (["slide", "n_tiles", "n_vessels", "tissue_mm2", "mvd_per_mm2",
                     "vessel_area_frac", "dab_area_frac", "nn_dist_median_px"]
                    + [f"{k}_frac" for k in FRACTION_FEATURES]
                    + [f"{k}_median" for k in SHAPE_FEATURES]
                    + [f"{k}_cv" for k in SHAPE_FEATURES])
    slide_path = args.out_dir / "vessel_slides.tsv"
    with open(slide_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=slide_fields, delimiter="\t")
        w.writeheader()
        for slide in sorted(by_slide_tiles):
            tiles, inst = by_slide_tiles[slide], by_slide_inst[slide]
            tissue_mm2 = sum(t["tissue_px"] for t in tiles) * px_mm2
            row = {
                "slide": slide,
                "n_tiles": len(tiles),
                "n_vessels": len(inst),
                "tissue_mm2": round(tissue_mm2, 4),
                "mvd_per_mm2": round(len(inst) / tissue_mm2, 2) if tissue_mm2 else "",
                "vessel_area_frac": round(sum(t["vessel_area_px"] for t in tiles)
                                          / max(1, sum(t["tissue_px"] for t in tiles)), 6),
                "dab_area_frac": round(sum(t["dab_area_px"] for t in tiles)
                                       / max(1, sum(t["tissue_px"] for t in tiles)), 6),
                "nn_dist_median_px": round(float(np.median([v for v in nn_distances(inst)])), 1) if len(inst) > 1 else "",
            }
            for k in FRACTION_FEATURES:
                vals = [float(v[k]) for v in inst if v[k] not in ("", None)]
                row[f"{k}_frac"] = round(float(np.mean(vals)), 4) if vals else ""
            for k in SHAPE_FEATURES:
                med, cv = summarize([float(v[k]) for v in inst if v[k] not in ("", "nan", None)])
                row[f"{k}_median"] = "" if math.isnan(med) else round(med, 4)
                row[f"{k}_cv"] = "" if math.isnan(cv) else round(cv, 4)
            w.writerow(row)

    print(f"slides   : {len(by_slide_tiles)}  ->  {slide_path}")
    print(f"(slide aggregation used {len(keep_stems)} non-overlapping tiles of {len(all_tiles)})")


if __name__ == "__main__":
    main()
