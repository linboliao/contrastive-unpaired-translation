#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple, non-learned vessel segmentation combining the HE and virtually-
stained (fake_B) images: no training, no model.

v1 of this script just clustered DAB(ERG)-positive specks in fake_B and took
their convex hull -- but ERG only marks endothelial cell *nuclei*, so that
traces a belt through the nuclei, not the actual vessel (lumen + wall) a
pathologist would circle. v2 looked for enclosed white/RBC-filled blobs in
HE, but that misses vessels cut by the crop edge or without a clean closed
lumen shape. v3 added Frangi vesselness to catch those -- but Frangi is a
*ridge* detector: it responds to elongated tubes and under-responds to
round/short vessel cross-sections (a vessel cut perpendicular to its axis
looks like a small "O", not a ridge, so its two principal curvatures are
similar and Frangi's response stays weak). So this version keeps *both*
candidate generators and takes their union:

  1. Frangi vesselness on HE (tube/ridge-like structures, any orientation,
     including ones running off the crop edge) -- see frangi_candidates()
  2. enclosed white/RBC-filled blobs on HE (round or short vessel
     cross-sections Frangi misses) -- see blob_candidates()
  3. keeps only candidates (from either generator) whose boundary is
     actually lined by DAB+ (ERG-positive) endothelial nuclei from fake_B --
     see confirmed_vessel_mask() -- which is what tells a random tissue gap
     or gland lumen apart from a real vessel, since ERG is
     endothelium-specific. Caveat: if the generated image has no DAB signal
     at all near a real vessel, no post-processing can recover it -- that's
     a generation-fidelity gap, not a segmentation one.
  4. outlines the confirmed region (dilated by a couple px to include the
     wall) instead of a hull of the sparse DAB dots, so the result traces
     the whole anatomical vessel the way a pathologist would.

Still fully classical: two filters + thresholds + connected components + one
overlap test, no training, no learned features.

Renders the same 3-panel deliverable as scripts/build_seg_visualization.py
(HE / virtual IHC, both with the mask's contour in red / real IHC, plain).
No retraining, no GPU -- only re-reads results/<name>/test_latest images
already on disk (see scripts/run_erg_v3_test_sweep.sh).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import measure
from skimage.color import rgb2hed, rgb2hsv
from skimage.filters import frangi
from skimage.morphology import (
    remove_small_objects, remove_small_holes, binary_dilation, convex_hull_image, disk,
)

DAB_OD_THRESHOLD = 0.15
MIN_BLOB_PX = 6

# Frangi vesselness: enhances tube/ridge-like structures at each scale in
# FRANGI_SIGMAS regardless of whether the vessel is a closed round lumen or
# an elongated one running off the edge of the crop -- unlike a plain
# "enclosed white blob" detector, it doesn't need the lumen to be a closed
# shape. Vessels are brighter than the surrounding tissue, hence
# black_ridges=False. FRANGI_PERCENTILE keeps only the strongest response
# per image (robust to frangi's data-dependent output scale).
FRANGI_SIGMAS = range(3, 15, 3)
FRANGI_PERCENTILE = 98.5

# Blob candidates: round/short vessel cross-sections -- enclosed empty
# (white) or RBC-filled (strongly red/eosinophilic) regions in HE. Frangi
# alone under-responds to these because they're round, not ridge-like.
WHITE_V_MIN = 0.85         # HSV value above which a pixel counts as "empty/white" lumen
WHITE_S_MAX = 0.15         # HSV saturation below which a pixel counts as "empty/white" lumen
RBC_R_MIN = 0.55           # red blood cells: strongly red/eosinophilic, not just pink tissue
RBC_R_MINUS_G = 0.15
RBC_S_MIN = 0.35

LUMEN_MIN_PX = 40          # ignore lumen candidates smaller than this (noise, not a real space)
LUMEN_MAX_PX = 12000       # ignore candidates this big (background / torn tissue, not a vessel)
LUMEN_FILL_AREA = 400      # fill enclosed dark specks/holes up to this size (nuclei sitting inside
                           # a lumen shouldn't fragment it into a ring)
LUMEN_LINK_RADIUS = 15     # merge lumen fragments this close together (a vessel's lumen often
                           # shows up as several adjacent gaps, not one clean blob -- an
                           # RBC-packed vessel especially, since each red cell is its own
                           # small blob with only a few px of gap to its neighbors)

WALL_MARGIN = 14           # how far (px) around a lumen to look for confirming ERG+ nuclei
MIN_CONFIRM_PX = 1         # how many DAB+ pixels in that margin count as "confirmed vessel"
                           # (ERG staining on the generated image is often faint/sparse, so this
                           # stays permissive -- see the module docstring's caveat about recall)
OUTLINE_GROW = 2           # grow the confirmed lumen by this much so the outline includes the wall

CONTOUR_COLOR = (230, 30, 30)
CONTOUR_WIDTH = 2
LABEL_H = 28
PANEL_GAP = 4
CJK_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def dab_channel(img_uint8):
    rgb01 = img_uint8.astype(np.float64) / 255.0
    return np.clip(rgb2hed(rgb01)[..., 2], 0, None)


def dab_speck_mask(fake_img):
    return remove_small_objects(dab_channel(np.array(fake_img.convert("RGB"))) > DAB_OD_THRESHOLD,
                                 min_size=MIN_BLOB_PX)


def _clean_and_hull(candidate):
    """Shared cleanup for either candidate generator: fill small enclosed
    holes, drop noise, merge nearby fragments of the same vessel, then take
    the convex hull of each merged cluster's *original* pixels (fills gaps
    without inheriting the link dilation's extra fatness), dropping clusters
    that end up implausibly large (background, not a vessel)."""
    candidate = remove_small_holes(candidate, area_threshold=LUMEN_FILL_AREA)
    candidate = remove_small_objects(candidate, min_size=LUMEN_MIN_PX)

    linked = binary_dilation(candidate, footprint=disk(LUMEN_LINK_RADIUS)) if LUMEN_LINK_RADIUS > 0 else candidate
    linked = remove_small_holes(linked, area_threshold=LUMEN_FILL_AREA)

    labels, n = ndi.label(linked)
    out = np.zeros_like(candidate)
    for i in range(1, n + 1):
        cluster = labels == i
        original_px = candidate & cluster
        if not original_px.any() or cluster.sum() > LUMEN_MAX_PX:
            continue
        out |= convex_hull_image(original_px)
    return out


def frangi_candidates(he_img):
    """Tube/ridge-like bright structures (Frangi vesselness on the HSV value
    channel) -- catches elongated vessels regardless of orientation, or
    running off the crop edge. Under-responds to round/short cross-sections;
    see blob_candidates() for those."""
    rgb = np.array(he_img.convert("RGB")).astype(np.float64) / 255.0
    v = rgb2hsv(rgb)[..., 2]

    vesselness = frangi(v, sigmas=FRANGI_SIGMAS, black_ridges=False)
    cutoff = np.percentile(vesselness, FRANGI_PERCENTILE)
    return _clean_and_hull(vesselness > max(cutoff, 1e-12))


def blob_candidates(he_img):
    """Enclosed empty (white) or RBC-filled (strongly red) regions -- round
    or short vessel cross-sections, which frangi_candidates() misses."""
    rgb = np.array(he_img.convert("RGB")).astype(np.float64) / 255.0
    hsv = rgb2hsv(rgb)
    v, s = hsv[..., 2], hsv[..., 1]
    r, g = rgb[..., 0], rgb[..., 1]

    white = (v > WHITE_V_MIN) & (s < WHITE_S_MAX)
    rbc = (r > RBC_R_MIN) & ((r - g) > RBC_R_MINUS_G) & (s > RBC_S_MIN)
    return _clean_and_hull(white | rbc)


def lumen_candidates(he_img):
    """Union of both candidate generators -- every vessel-shaped space a
    vessel *could* be, before checking for ERG confirmation."""
    return frangi_candidates(he_img) | blob_candidates(he_img)


def confirmed_vessel_mask(he_img, fake_img, wall_margin=WALL_MARGIN, min_confirm_px=MIN_CONFIRM_PX):
    """Lumen candidates whose boundary is actually lined by ERG+ (DAB)
    nuclei -- the real discriminator between a vessel and any other gap."""
    lumens = lumen_candidates(he_img)
    specks = dab_speck_mask(fake_img)

    labels, n = ndi.label(lumens)
    out = np.zeros_like(lumens)
    for i in range(1, n + 1):
        comp = labels == i
        wall_zone = binary_dilation(comp, footprint=disk(wall_margin)) & ~comp
        if (wall_zone & specks).sum() >= min_confirm_px:
            out |= binary_dilation(comp, footprint=disk(OUTLINE_GROW))
    return out


def vessel_mask(he_img, fake_img, wall_margin=WALL_MARGIN, min_confirm_px=MIN_CONFIRM_PX):
    return confirmed_vessel_mask(he_img, fake_img, wall_margin, min_confirm_px)


def draw_contours(img, mask, color=CONTOUR_COLOR, width=CONTOUR_WIDTH):
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for contour in measure.find_contours(mask.astype(np.float64), 0.5):
        pts = [(float(x), float(y)) for y, x in contour]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width, joint="curve")
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
    parser = argparse.ArgumentParser(description="Classical DAB-threshold vessel segmentation on fake_B, 3-panel visualization.")
    parser.add_argument("--name", default="ERG_v3_dab_l2")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wall-margin", type=int, default=WALL_MARGIN,
                         help="how far (px) around a candidate lumen to look for confirming ERG+ nuclei")
    parser.add_argument("--min-confirm-px", type=int, default=MIN_CONFIRM_PX,
                         help="DAB+ pixels required in the wall margin to confirm a lumen as a vessel")
    parser.add_argument("--stems", default=None, help="comma-separated stems to render (default: all)")
    args = parser.parse_args()

    img_dir = args.results_root / args.name / "test_latest" / "images"
    he_dir, fake_dir, real_dir = img_dir / "real_A", img_dir / "fake_B", img_dir / "real_B"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype(CJK_FONT_PATH, 20)
    except OSError:
        font = ImageFont.load_default()

    stems = args.stems.split(",") if args.stems else sorted(p.stem for p in fake_dir.glob("*.png"))
    print(f"Building {len(stems)} vessel-segmentation visualizations from {args.name} -> {args.out_dir} "
          f"(wall_margin={args.wall_margin}, min_confirm_px={args.min_confirm_px})")

    for i, stem in enumerate(stems):
        he = Image.open(he_dir / f"{stem}.png").convert("RGB")
        fake = Image.open(fake_dir / f"{stem}.png").convert("RGB")
        real = Image.open(real_dir / f"{stem}.png").convert("RGB")
        mask = vessel_mask(he, fake, wall_margin=args.wall_margin, min_confirm_px=args.min_confirm_px)

        he_panel = label_panel(draw_contours(he, mask), "HE", font)
        fake_panel = label_panel(draw_contours(fake, mask), "虚拟IHC (ERG)", font)
        real_panel = label_panel(real, "真实IHC (ERG)", font)

        w, h = he_panel.size
        combined = Image.new("RGB", (w * 3 + PANEL_GAP * 2, h), (255, 255, 255))
        combined.paste(he_panel, (0, 0))
        combined.paste(fake_panel, (w + PANEL_GAP, 0))
        combined.paste(real_panel, (2 * (w + PANEL_GAP), 0))
        combined.save(args.out_dir / f"{stem}.png")

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(stems)}")

    print("done")


if __name__ == "__main__":
    main()
