"""Differentiable RGB->DAB color deconvolution plus registration-quality-based
per-sample loss weights.

Used by CUTModel to add a paired, DAB-channel (ERG/vessel) supervision term on
top of the vanilla unpaired GAN + PatchNCE losses, so training capacity is
pulled toward getting the vessel signal right instead of only the overall
staining style. Only valid when trainA/trainB are filename-aligned registered
pairs fed with matching indices (see data/unaligned_dataset.py + serial_batches).
"""
import csv
import math
import os

import torch

# Ruifrok & Johnston stain matrix, same convention as skimage.color.rgb2hed
# (channel order: Hematoxylin, Eosin, DAB) -- matches filter_erg_pairs.py's
# stain_channels(), which reads hed[..., 0] (H) and hed[..., 2] (DAB).
_RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29],
                               [0.07, 0.99, 0.11],
                               [0.27, 0.57, 0.78]], dtype=torch.float32)
_HED_FROM_RGB = torch.inverse(_RGB_FROM_HED)
_LOG_ADJUST = math.log(1e-6)


def rgb_tensor_to_dab(x):
    """x: (B, 3, H, W) in [-1, 1] (dataset uses Normalize((.5,.5,.5), (.5,.5,.5))).

    Returns the DAB optical-density channel, (B, 1, H, W), clamped to >= 0.
    """
    rgb01 = ((x + 1.0) / 2.0).clamp(min=1e-6)
    od = torch.log(rgb01) / _LOG_ADJUST  # (B, 3, H, W)
    od = od.permute(0, 2, 3, 1)  # (B, H, W, 3)
    hed = od @ _HED_FROM_RGB.to(device=x.device, dtype=x.dtype)
    dab = hed[..., 2].clamp(min=0)
    return dab.unsqueeze(1)  # (B, 1, H, W)


class RegistrationWeightTable:
    """Looks up per-pair registration confidence / ERG status from a QC tsv
    (see qc/pair_scores.tsv, columns include stem/reg_score/erg_status) and
    turns it into a scalar weight for the paired DAB loss, so poorly
    registered or ambiguous pairs contribute less.
    """

    def __init__(self, tsv_path, ambiguous_weight=0.3, min_weight=0.05, min_reg_score=0.0):
        """min_reg_score: hard-exclude pairs whose raw reg_score falls below this
        (weight forced to 0.0, bypassing min_weight and the erg_status multiplier)
        instead of only down-weighting them. 0.0 preserves the original
        soft-weighting-only behaviour. See e-series registration-hard-filter study.
        """
        self.ambiguous_weight = ambiguous_weight
        self.min_weight = min_weight
        self.min_reg_score = min_reg_score
        self._weights = {}
        if tsv_path and os.path.isfile(tsv_path):
            with open(tsv_path, newline='') as f:
                for row in csv.DictReader(f, delimiter='\t'):
                    stem = row.get('stem')
                    if not stem:
                        continue
                    try:
                        raw_score = float(row.get('reg_score', 1.0))
                    except (TypeError, ValueError):
                        raw_score = 1.0
                    if raw_score < self.min_reg_score:
                        self._weights[stem] = 0.0
                        continue
                    weight = raw_score
                    if row.get('erg_status') == 'ambiguous':
                        weight *= self.ambiguous_weight
                    self._weights[stem] = max(self.min_weight, min(1.0, weight))

    def get(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return self._weights.get(stem, 1.0)

    def get_batch(self, paths, device, dtype=torch.float32):
        return torch.tensor([self.get(p) for p in paths], device=device, dtype=dtype)
