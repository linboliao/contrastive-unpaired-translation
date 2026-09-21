"""Derives the endothelial-nucleus supervision mask for CUTSegModel from the
registered real_B (ERG) image: real_B is DHR-registered and pixel-aligned
with real_A/fake_B, so its thresholded DAB optical-density channel doubles as
a per-pixel nucleus mask with no extra annotation. Same threshold convention
as scripts/evaluate_erg_sweep.py's DAB_OD_THRESHOLD, so segmentation
supervision and the sweep's r metric agree on what counts as ERG-positive.
"""
import torch

from .stain_utils import rgb_tensor_to_dab

DAB_OD_THRESHOLD = 0.15


def nucleus_mask_from_rgb(real_B):
    with torch.no_grad():
        dab = rgb_tensor_to_dab(real_B)
        mask = (dab > DAB_OD_THRESHOLD).float()
    return mask
