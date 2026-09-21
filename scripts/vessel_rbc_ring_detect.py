#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug/tuning pipeline for framing DAB(ERG)-positive vessel regions,
built on top of scripts/lumen_detect_debug.py's RGB-threshold + denoise +
ring-linking stages (separate file -- that script is left untouched), adding
red-blood-cell confirmation and dropping anything that never forms a ring:

    1. RGB range filter -- pixels within [--rgb-low, --rgb-high] on all
       three channels count as DAB, on the virtually-stained (fake_B) image.
       Defaults are the manually sampled DAB pixel values:
       low=(32,11,18), high=(183,171,151). See dab_mask().
    1b. exclude hematoxylin blue -- the DAB RGB box above is wide enough to
       also catch blue/purple nucleus pixels, so they're cut back out via an
       HSV hue range instead of RGB (hue stays stable across light/dark
       staining intensity where RGB/saturation don't). See exclude_blue_mask().
    2. denoise -- morphological open (drop small noise) then close (fill
       small gaps) on the binary mask. See denoise_mask().
    2a. link ring fragments -- ERG only marks endothelial *nuclei*, so a
       vessel wall often shows up as several separate DAB fragments a short
       distance apart, arranged in a ring or open ring (an arc) rather than
       one solid blob. Fragments within --ring-link-radius of each other are
       grouped, and a group becomes one shape (the convex hull of all its
       fragments) only if that hull is mostly *empty* in the middle
       (--ring-max-fill-ratio) -- that hollow center is what tells a ring
       apart from a solid clump of nuclei. See link_ring_fragments(), which
       returns the ring hulls and the leftover unlinked fragments separately.
    2b. confirm *unlinked* fragments by nearby red blood cells -- a shape
       that already traces a ring is evidence enough on its own and is kept
       unconditionally; RBC confirmation now only gates the leftover
       fragments that never formed a ring (relevant only if
       --keep-unlinked 1 -- with the default 0 they're dropped either way).
       This ordering matters: confirming RBC *before* linking (an earlier
       version of this script) required every individual fragment to have
       its own nearby RBC, which killed real rings whose fragments only had
       RBC nearby *as a whole* (or none at all -- plenty of real vessel
       cross-sections have no visible RBC in them). A real vessel usually
       has RBCs (strongly eosinophilic red) in or near its lumen on HE; RBC
       red is matched with two HSV ranges because it sits right on OpenCV's
       0/179 hue seam (a naive single range calibrated from "pure red" RGB
       samples misses it -- checked empirically against this data, see
       rbc_mask()). --require-rbc 0 disables the check entirely.
    2c. smooth -- Gaussian-blur the mask then re-threshold it (the classic
       cheap way to round off a binary shape's corners) so ring hulls read
       as smooth curves instead of jagged polygons. --smooth-ksize 0
       disables it.
    3. frame -- cv2.findContours on the final mask, keeping contours whose
       area falls in [--min-area, --max-area] AND whose bounding box is at
       least --min-extent px in both width and height. See dab_contours().

Renders the same 3-panel deliverable as scripts/build_vessel_seg_visualization.py
(HE / virtual IHC, both with the contours drawn / real IHC, plain).

Standalone debug tool. Debug dataset defaults to ERG_v3_dab_l2_ambig01's test
results (already on disk from scripts/run_erg_v3_test_sweep.sh) -- no
retraining, no GPU.

Example tuning loop:
    python scripts/vessel_rbc_ring_detect.py --out-dir /tmp/vessel_debug --num-samples 12
    python scripts/vessel_rbc_ring_detect.py --out-dir /tmp/vessel_debug \\
        --rbc-margin 30 --ring-link-radius 10 --keep-unlinked 1
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RGB_LOW = (32, 11, 18)
RGB_HIGH = (183, 171, 151)

# The RGB box above only constrains each channel *independently* -- it never
# checks the *relationship* between them, so a neutral gray or blue-gray
# pixel (R roughly equal to G roughly equal to B, or B the largest) can
# satisfy all three per-channel ranges and get picked up as "DAB" even
# though it plainly isn't brown. Brown means R meaningfully bigger than B
# (checked against sampled swatches: brown R-B is +70 to +85, neutral gray
# is ~0, blue-gray is -15 to -20) -- so dab_mask() now also requires
# R-B >= DAB_MIN_R_MINUS_B, which grays and blues fail regardless of how
# wide the box is.
DAB_MIN_R_MINUS_B = 23

# Hematoxylin blue, expressed as an HSV hue range instead of RGB -- see the
# module docstring for how these bounds were derived from two sampled blue
# pixels (a small margin is added around both: H -10/+6, S -5/+24, V -23/+18).
BLUE_HSV_LOW = (28, 24, 90)
BLUE_HSV_HIGH = (111, 126, 180)

# Red blood cells (strongly eosinophilic red) on HE. OpenCV hue wraps at
# 0/179, and real scanned RBC red sits right on that seam (checked
# empirically: 1st-99th pct H=168-178 on a sampled RBC-like region) -- NOT
# near H=0 the way a "pure red" RGB triple naively converts. So this is two
# ranges (one on each side of the wrap), OR'd together in rbc_mask().
RBC_HSV_RANGES = [
    ((0, 80, 135), (10, 190, 215)),
    ((165, 80, 135), (179, 190, 215)),
]
RBC_MARGIN = 20            # a DAB fragment is confirmed if an RBC pixel is within this many px

OPEN_KSIZE = 3
OPEN_ITER = 1
CLOSE_KSIZE = 5
CLOSE_ITER = 1

RING_LINK_RADIUS = 7       # fragments within this many px of each other are grouped
RING_MIN_FRAGMENTS = 2     # need at least this many separate fragments to consider linking
RING_MAX_FILL_RATIO = 0.5  # group's (fragment area / hull area) must be BELOW this to
                           # count as a ring (hollow middle); above it, it's a solid clump

SMOOTH_KSIZE = 9           # Gaussian blur kernel (odd) before re-thresholding the mask, to
                           # round off jagged corners on hulls; 0 disables it

CONTOUR_MIN_AREA = 40      # ignore contours smaller than this (noise)
CONTOUR_MAX_AREA = 12000   # ignore contours bigger than this (background, not DAB)
CONTOUR_MIN_EXTENT = 15    # ignore contours whose bounding box is narrower than this in
                           # either dimension (small dot/sliver fragments; area alone can miss
                           # these since a thin elongated sliver can still have moderate area)

CONTOUR_COLOR = (230, 30, 30)
CONTOUR_WIDTH = 2
LABEL_H = 28
PANEL_GAP = 4
CJK_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def dab_mask(rgb_img, low=RGB_LOW, high=RGB_HIGH, min_r_minus_b=DAB_MIN_R_MINUS_B):
    """Stage 1: uint8 0/255 mask, 255 where all three RGB channels fall
    within [low, high] (the DAB color range) AND R exceeds B by at least
    min_r_minus_b -- the per-channel box alone doesn't stop a gray/blue-gray
    pixel (R~=G~=B, or B the largest) from qualifying, since it never checks
    the relationship between channels the way this second condition does."""
    box = cv2.inRange(rgb_img, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    if min_r_minus_b > 0:
        r_minus_b = rgb_img[..., 0].astype(np.int16) - rgb_img[..., 2].astype(np.int16)
        brownish = (r_minus_b >= min_r_minus_b).astype(np.uint8) * 255
        box = cv2.bitwise_and(box, brownish)
    return box


def exclude_blue_mask(rgb_img, low=BLUE_HSV_LOW, high=BLUE_HSV_HIGH):
    """Stage 1b: uint8 0/255 mask, 255 where the pixel's HSV hue/sat/val
    fall in the hematoxylin-blue range (to be subtracted out of dab_mask)."""
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    lower = np.array(low, dtype=np.uint8)
    upper = np.array(high, dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def rbc_mask(rgb_img, ranges=RBC_HSV_RANGES):
    """uint8 0/255 mask, 255 where the pixel's HSV hue/sat/val fall in any
    of the red-blood-cell ranges (on the HE image) -- two ranges by default
    since RBC red sits right on OpenCV's 0/179 hue seam."""
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    out = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        out = cv2.bitwise_or(out, cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)))
    return out


def confirm_by_rbc(mask, rbc, margin=RBC_MARGIN):
    """Stage 2a: keep a DAB fragment only if an RBC pixel is within margin
    px of it -- confirms the fragment is actually vessel-associated."""
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    if margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        rbc_zone = cv2.dilate(rbc, k)
    else:
        rbc_zone = rbc

    out = np.zeros_like(mask)
    for i in range(1, n):
        frag = labels == i
        if (rbc_zone[frag] > 0).any():
            out[frag] = 255
    return out


def denoise_mask(mask, open_ksize=OPEN_KSIZE, open_iter=OPEN_ITER, close_ksize=CLOSE_KSIZE, close_iter=CLOSE_ITER):
    """Stage 2: morphological open (remove small noise) then close (fill
    small gaps) on the binary mask."""
    out = mask
    if open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return out


def link_ring_fragments(mask, link_radius=RING_LINK_RADIUS, min_fragments=RING_MIN_FRAGMENTS,
                         max_fill_ratio=RING_MAX_FILL_RATIO):
    """Stage 2b: group DAB fragments within link_radius of each other; a
    group of >= min_fragments separate pieces becomes the convex hull of all
    its pixels *only if* that hull is mostly empty (fill_ratio <=
    max_fill_ratio) -- i.e. it traces a ring or open ring (arc), not a solid
    clump. Returns (ring_mask, unlinked_mask) separately: a shape that
    successfully formed a ring is evidence enough on its own (see the
    caller, confirm_by_rbc only ever gates unlinked_mask now -- requiring
    RBC confirmation *before* linking was killing real rings whose
    individual fragments had no RBC pixel within margin of *each one
    separately*, even though the finished ring plainly traced a vessel)."""
    _, frag_labels = cv2.connectedComponents(mask, connectivity=8)

    if link_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * link_radius + 1, 2 * link_radius + 1))
        linked = cv2.dilate(mask, k)
    else:
        linked = mask
    n_clusters, cluster_labels = cv2.connectedComponents(linked, connectivity=8)

    ring_mask = np.zeros_like(mask)
    unlinked_mask = np.zeros_like(mask)
    for cid in range(1, n_clusters):
        in_cluster = cluster_labels == cid
        original_px = ((mask > 0) & in_cluster)
        if not original_px.any():
            continue
        original_u8 = (original_px * 255).astype(np.uint8)

        n_fragments = len(set(frag_labels[original_px].tolist()) - {0})
        if n_fragments >= min_fragments:
            pts = cv2.findNonZero(original_u8)
            hull = cv2.convexHull(pts)
            hull_mask = np.zeros_like(mask)
            cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
            fill_ratio = float(original_px.sum()) / max(1, int((hull_mask > 0).sum()))
            if fill_ratio <= max_fill_ratio:
                ring_mask = cv2.bitwise_or(ring_mask, hull_mask)
                continue
        unlinked_mask = cv2.bitwise_or(unlinked_mask, original_u8)
    return ring_mask, unlinked_mask


def smooth_mask(mask, ksize=SMOOTH_KSIZE):
    """Stage 2d: Gaussian-blur then re-threshold at the midpoint -- the
    cheap classic trick for rounding a binary mask's corners, so ring hulls
    trace smooth curves instead of straight hull edges."""
    if ksize <= 0:
        return mask
    k = ksize | 1  # cv2.GaussianBlur requires an odd kernel size
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    _, out = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return out


def dab_contours(mask, min_area=CONTOUR_MIN_AREA, max_area=CONTOUR_MAX_AREA, min_extent=CONTOUR_MIN_EXTENT):
    """Stage 3: cv2.findContours on the mask, kept if their area falls in
    [min_area, max_area] AND their bounding box is at least min_extent px
    wide and tall. Returns a list of (N, 2) point arrays."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        _, _, w, h = cv2.boundingRect(c)
        if w < min_extent or h < min_extent:
            continue
        kept.append(c.reshape(-1, 2))
    return kept


def draw_contours(img, contours, color=CONTOUR_COLOR, width=CONTOUR_WIDTH):
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for c in contours:
        pts = [(float(x), float(y)) for x, y in c]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=color, width=width, joint="curve")
    return out


def label_panel(img, text, font):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + LABEL_H), (255, 255, 255))
    canvas.paste(img, (0, LABEL_H))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, 6), text, fill=(20, 20, 20), font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="RGB DAB filter -> denoise -> RBC confirmation -> ring-linking -> smooth -> contours, 3-panel visualization.")
    parser.add_argument("--name", default="ERG_v3_dab_l2_ambig01")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stems", default=None, help="comma-separated stems (default: first --num-samples test images)")
    parser.add_argument("--num-samples", type=int, default=12)

    parser.add_argument("--rgb-low", type=int, nargs=3, default=list(RGB_LOW), metavar=("R", "G", "B"))
    parser.add_argument("--rgb-high", type=int, nargs=3, default=list(RGB_HIGH), metavar=("R", "G", "B"))
    parser.add_argument("--dab-min-r-minus-b", type=int, default=DAB_MIN_R_MINUS_B,
                         help="also require R-B >= this (rejects gray/blue-gray the RGB box alone lets through), 0 to disable")

    parser.add_argument("--exclude-blue", type=int, default=1, choices=[0, 1],
                         help="subtract the hematoxylin-blue HSV range from the DAB mask (1=on, default)")
    parser.add_argument("--blue-hsv-low", type=int, nargs=3, default=list(BLUE_HSV_LOW), metavar=("H", "S", "V"))
    parser.add_argument("--blue-hsv-high", type=int, nargs=3, default=list(BLUE_HSV_HIGH), metavar=("H", "S", "V"))

    parser.add_argument("--open-ksize", type=int, default=OPEN_KSIZE, help="0 to disable")
    parser.add_argument("--open-iter", type=int, default=OPEN_ITER)
    parser.add_argument("--close-ksize", type=int, default=CLOSE_KSIZE, help="0 to disable")
    parser.add_argument("--close-iter", type=int, default=CLOSE_ITER)

    parser.add_argument("--require-rbc", type=int, default=1, choices=[0, 1],
                         help="keep a DAB fragment only if a red blood cell is nearby on HE (1=on, default)")
    parser.add_argument("--rbc-hsv-low", type=int, nargs=3, default=list(RBC_HSV_RANGES[0][0]), metavar=("H", "S", "V"),
                         help="RBC red range, low-hue side of OpenCV's 0/179 seam")
    parser.add_argument("--rbc-hsv-high", type=int, nargs=3, default=list(RBC_HSV_RANGES[0][1]), metavar=("H", "S", "V"))
    parser.add_argument("--rbc-hsv-low2", type=int, nargs=3, default=list(RBC_HSV_RANGES[1][0]), metavar=("H", "S", "V"),
                         help="RBC red range, high-hue side of the seam (real scanned RBC red sits right on it)")
    parser.add_argument("--rbc-hsv-high2", type=int, nargs=3, default=list(RBC_HSV_RANGES[1][1]), metavar=("H", "S", "V"))
    parser.add_argument("--rbc-margin", type=int, default=RBC_MARGIN,
                         help="a DAB fragment is confirmed if an RBC pixel is within this many px")

    parser.add_argument("--link-rings", type=int, default=1, choices=[0, 1],
                         help="link nearby DAB fragments that trace a ring/open ring (1=on, default)")
    parser.add_argument("--ring-link-radius", type=int, default=RING_LINK_RADIUS,
                         help="fragments within this many px of each other are grouped")
    parser.add_argument("--ring-min-fragments", type=int, default=RING_MIN_FRAGMENTS,
                         help="minimum separate fragments in a group before it's considered for linking")
    parser.add_argument("--ring-max-fill-ratio", type=float, default=RING_MAX_FILL_RATIO,
                         help="group's (fragment area / hull area) must be below this to count as a ring")
    parser.add_argument("--keep-unlinked", type=int, default=0, choices=[0, 1],
                         help="keep fragments that never formed a qualifying ring (0=drop them, default)")

    parser.add_argument("--smooth-ksize", type=int, default=SMOOTH_KSIZE,
                         help="Gaussian blur kernel to round mask corners before contouring, 0 to disable")

    parser.add_argument("--min-area", type=int, default=CONTOUR_MIN_AREA)
    parser.add_argument("--max-area", type=int, default=CONTOUR_MAX_AREA)
    parser.add_argument("--min-extent", type=int, default=CONTOUR_MIN_EXTENT,
                         help="drop contours whose bounding box is narrower than this in either dimension")

    args = parser.parse_args()

    img_dir = args.results_root / args.name / "test_latest" / "images"
    he_dir, fake_dir, real_dir = img_dir / "real_A", img_dir / "fake_B", img_dir / "real_B"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype(CJK_FONT_PATH, 20)
    except OSError:
        font = ImageFont.load_default()

    rgb_low, rgb_high = tuple(args.rgb_low), tuple(args.rgb_high)
    blue_low, blue_high = tuple(args.blue_hsv_low), tuple(args.blue_hsv_high)
    rbc_ranges = [(tuple(args.rbc_hsv_low), tuple(args.rbc_hsv_high)),
                  (tuple(args.rbc_hsv_low2), tuple(args.rbc_hsv_high2))]
    stems = args.stems.split(",") if args.stems else sorted(p.stem for p in he_dir.glob("*.png"))[:args.num_samples]
    print(f"Framing {len(stems)} images from {args.name} -> {args.out_dir} "
          f"(rgb range {rgb_low}-{rgb_high}, require_rbc={'off' if not args.require_rbc else 'on'}, "
          f"keep_unlinked={'yes' if args.keep_unlinked else 'no'})")

    for stem in stems:
        he = Image.open(he_dir / f"{stem}.png").convert("RGB")
        fake = Image.open(fake_dir / f"{stem}.png").convert("RGB")
        real = Image.open(real_dir / f"{stem}.png").convert("RGB")

        he_rgb = np.array(he)
        fake_rgb = np.array(fake)

        mask = dab_mask(fake_rgb, rgb_low, rgb_high, args.dab_min_r_minus_b)
        if args.exclude_blue:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_blue_mask(fake_rgb, blue_low, blue_high)))
        mask = denoise_mask(mask, args.open_ksize, args.open_iter, args.close_ksize, args.close_iter)
        if args.link_rings:
            ring_mask, unlinked_mask = link_ring_fragments(mask, args.ring_link_radius,
                                                             args.ring_min_fragments, args.ring_max_fill_ratio)
        else:
            ring_mask, unlinked_mask = np.zeros_like(mask), mask
        if args.keep_unlinked:
            if args.require_rbc:
                unlinked_mask = confirm_by_rbc(unlinked_mask, rbc_mask(he_rgb, rbc_ranges), args.rbc_margin)
            mask = cv2.bitwise_or(ring_mask, unlinked_mask)
        else:
            mask = ring_mask
        mask = smooth_mask(mask, args.smooth_ksize)
        contours = dab_contours(mask, args.min_area, args.max_area, args.min_extent)

        he_panel = label_panel(draw_contours(he, contours), "HE", font)
        fake_panel = label_panel(draw_contours(fake, contours), "虚拟IHC (ERG)", font)
        real_panel = label_panel(real, "真实IHC (ERG)", font)

        w, h = he_panel.size
        combined = Image.new("RGB", (w * 3 + PANEL_GAP * 2, h), (255, 255, 255))
        combined.paste(he_panel, (0, 0))
        combined.paste(fake_panel, (w + PANEL_GAP, 0))
        combined.paste(real_panel, (2 * (w + PANEL_GAP), 0))
        combined.save(args.out_dir / f"{stem}.png")

        print(f"  {stem}: contours={len(contours)}")

    print("done")


if __name__ == "__main__":
    main()
