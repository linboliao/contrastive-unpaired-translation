#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Three-panel qualitative deliverable for CUTSegModel: HE input and
generated ERG (fake_B), both with the segmentation head's predicted nucleus
contour overlaid in red, next to the plain ground-truth ERG (real_B). No
retraining -- this only re-renders results/<name>/test_latest images already
produced by scripts/run_erg_v2_seg_test.sh + evaluate_seg_head.py.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

CONTOUR_COLOR = (230, 30, 30)
CONTOUR_WIDTH = 2
LABEL_H = 28
PANEL_GAP = 4
CJK_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def load_prob(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float64) / 255.0


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
    parser = argparse.ArgumentParser(description="Build 3-panel HE / generated-ERG(+seg contour) / real-ERG visualizations.")
    parser.add_argument("--name", default="ERG_v2_dab_l2_ambig01_seg")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    img_dir = args.results_root / args.name / "test_latest" / "images"
    he_dir = img_dir / "real_A"
    fake_dir = img_dir / "fake_B"
    real_dir = img_dir / "real_B"
    prob_dir = img_dir / "seg_prob_vis"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype(CJK_FONT_PATH, 20)
    except OSError:
        font = ImageFont.load_default()
    stems = sorted(p.stem for p in fake_dir.glob("*.png"))
    print(f"Building {len(stems)} visualizations from {args.name} -> {args.out_dir}")

    for i, stem in enumerate(stems):
        he = Image.open(he_dir / f"{stem}.png").convert("RGB")
        fake = Image.open(fake_dir / f"{stem}.png").convert("RGB")
        real = Image.open(real_dir / f"{stem}.png").convert("RGB")
        mask = load_prob(prob_dir / f"{stem}.png") > args.threshold

        he_panel = label_panel(draw_contours(he, mask), "HE", font)
        fake_panel = label_panel(draw_contours(fake, mask), "虚拟IHC (ERG)", font)
        real_panel = label_panel(real, "真实IHC (ERG)", font)

        w, h = he_panel.size
        combined = Image.new("RGB", (w * 3 + PANEL_GAP * 2, h), (255, 255, 255))
        combined.paste(he_panel, (0, 0))
        combined.paste(fake_panel, (w + PANEL_GAP, 0))
        combined.paste(real_panel, (2 * (w + PANEL_GAP), 0))
        combined.save(args.out_dir / f"{stem}.png")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(stems)}")

    print("done")


if __name__ == "__main__":
    main()
