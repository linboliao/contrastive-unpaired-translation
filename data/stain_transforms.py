"""Stain normalization and HED colour augmentation for the H&E -> ERG task.

Both address the same problem from opposite directions: the 15 slides in this
dataset were stained and scanned in batches, so the same tissue structure can
arrive in noticeably different colours. Nothing in the pipeline told the model
to ignore that, which is a plausible contributor to it failing to generalise
across slides (see qc/vessel_agreement -- object-level vessel F1 0.13-0.19).

  Stain normalization    maps every image onto one reference colour target, so
                         the model sees a consistent input distribution. This
                         is *preprocessing*: it must be applied identically at
                         train and test time. Two methods; on this dataset
                         Reinhard cuts cross-slide colour variance by 64% and
                         Macenko does nothing at all -- see ReinhardNormalizer
                         for the measurements, and prefer reinhard.
  HED augmentation       randomly perturbs stain concentrations during
                         training, so the model cannot key on absolute colour.
                         This is *augmentation*: train only.

Kept in its own module and wired in through UnalignedDataset's
modify_commandline_options, so models/ and options/ are untouched and the
defaults (--stain_norm none, --hed_aug 0) reproduce the previous behaviour
exactly.

Both operate on PIL RGB images and are applied before get_transform()'s
ToTensor, on the already-resized image (see UnalignedDataset.__getitem__) --
estimating a stain matrix on 1748x1748 pixels every sample would cost more
than the training step itself.
"""
import os

import numpy as np
from PIL import Image
from skimage.color import hed2rgb, lab2rgb, rgb2hed, rgb2lab

# Macenko defaults, following the reference implementation.
OD_BETA = 0.15      # optical-density floor: below this a pixel is glass, not tissue
ANGLE_ALPHA = 1.0   # percentile trimmed off each end when picking the two stain vectors
MAX_C_PCTL = 99.0   # concentration percentile used as the per-image scale
IO = 255.0          # illumination / white level
FIT_PIXELS = 40000  # pixels subsampled for the SVD; the estimate is stable well below this


def _to_od(rgb_u8):
    """RGB uint8 -> optical density, one row per pixel."""
    rgb = rgb_u8.reshape(-1, 3).astype(np.float64)
    np.maximum(rgb, 1.0, out=rgb)          # log(0) guard
    return -np.log10(rgb / IO)


def _estimate_stain_matrix(od, beta=OD_BETA, alpha=ANGLE_ALPHA, rng=None):
    """Macenko's two-stain basis: project tissue pixels onto the plane spanned
    by the top two OD eigenvectors, then take the extreme angles in that plane
    as the stain directions. Returns a 3x2 matrix, haematoxylin first."""
    tissue = od[~np.any(od < beta, axis=1)]
    if len(tissue) < 100:                  # near-empty tile: nothing to fit
        return None
    if len(tissue) > FIT_PIXELS:
        rng = rng or np.random.default_rng(0)
        tissue = tissue[rng.choice(len(tissue), FIT_PIXELS, replace=False)]

    # eigh returns ascending eigenvalues, so the last two columns are the plane
    _, eigvecs = np.linalg.eigh(np.cov(tissue.T))
    plane = eigvecs[:, 1:3]

    proj = tissue.dot(plane)
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    v_min = plane.dot([np.cos(np.percentile(phi, alpha)), np.sin(np.percentile(phi, alpha))])
    v_max = plane.dot([np.cos(np.percentile(phi, 100 - alpha)), np.sin(np.percentile(phi, 100 - alpha))])

    # Haematoxylin absorbs more in red than the second stain does; that is what
    # fixes which extreme is which, and the sign convention keeps OD positive.
    he = np.array([v_min, v_max] if v_min[0] > v_max[0] else [v_max, v_min]).T
    return he * np.sign(he.sum(axis=0))


def _concentrations(od, stain_matrix):
    """Per-pixel stain concentrations, 2 x N."""
    return np.linalg.lstsq(stain_matrix, od.T, rcond=None)[0]


def fit_reference(paths, max_side=1024):
    """Pool pixels from one or more images and fit a single reference basis.

    Pooling several tiles rather than trusting one is deliberate: a lone
    reference tile bakes its own staining idiosyncrasies into every image the
    normalizer ever touches.
    """
    ods = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.BICUBIC)
        od = _to_od(np.asarray(img))
        ods.append(od[~np.any(od < OD_BETA, axis=1)])
    if not ods:
        raise ValueError("no reference images given")
    od = np.concatenate(ods)

    stain_matrix = _estimate_stain_matrix(od)
    if stain_matrix is None:
        raise ValueError("could not fit a stain matrix from the reference images")
    max_c = np.percentile(_concentrations(od, stain_matrix), MAX_C_PCTL, axis=1)
    return stain_matrix, max_c


class MacenkoNormalizer:
    """Re-renders an image in the reference stain basis, rescaling each stain's
    concentration so its 99th percentile matches the reference's.

    Tiles the estimator cannot fit (almost pure background, no tissue pixels
    above the OD floor) are returned unchanged rather than forced through a
    garbage basis.
    """

    def __init__(self, stain_matrix_ref, max_c_ref):
        self.stain_matrix_ref = np.asarray(stain_matrix_ref, dtype=np.float64)
        self.max_c_ref = np.asarray(max_c_ref, dtype=np.float64)

    def __call__(self, img):
        arr = np.asarray(img.convert("RGB"))
        h, w = arr.shape[:2]
        od = _to_od(arr)

        stain_matrix = _estimate_stain_matrix(od)
        if stain_matrix is None:
            return img

        conc = _concentrations(od, stain_matrix)
        max_c = np.percentile(conc, MAX_C_PCTL, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(max_c > 1e-6, self.max_c_ref / max_c, 1.0)
        conc *= scale[:, None]

        out = IO * np.power(10.0, -self.stain_matrix_ref.dot(conc))
        out = np.clip(out.T.reshape(h, w, 3), 0, 255).astype(np.uint8)
        return Image.fromarray(out)


class HedJitter:
    """Tellez-style HED augmentation: decompose into haematoxylin / eosin / DAB
    optical densities, scale and shift each independently, recompose.

    Perturbing the stains separately is the point -- a plain RGB brightness or
    hue jitter moves all three together and so cannot mimic the way one stain
    runs darker than another between batches.

    `bias` is a FRACTION of each channel's own standard deviation, not an
    absolute optical density. Tellez's formulation shifts stain concentrations
    that are O(1), but skimage's rgb2hed returns much smaller numbers -- on
    this data the channel stds are H 0.055 / E 0.044 / D 0.025, so the
    conventional bias of 0.05 taken as an absolute value is 0.9-2.0x an entire
    channel's spread. Applied that way it turned H&E tiles yellow and green.
    Scaling by the per-image std keeps the same knob dataset-independent.
    """

    def __init__(self, sigma=0.05, bias=0.05, rng=None):
        self.sigma = float(sigma)
        self.bias = float(bias)
        self.rng = rng or np.random.default_rng()

    def __call__(self, img):
        if self.sigma <= 0 and self.bias <= 0:
            return img
        hed = rgb2hed(np.asarray(img.convert("RGB")).astype(np.float64) / 255.0)
        alpha = 1.0 + self.rng.uniform(-self.sigma, self.sigma, 3)
        beta = self.rng.uniform(-self.bias, self.bias, 3) * hed.reshape(-1, 3).std(axis=0)
        out = hed2rgb(hed * alpha + beta)
        return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8))


class ReinhardNormalizer:
    """Matches the per-channel mean and std of the image's CIELAB distribution
    to a reference's.

    This is the one that actually works on this dataset, and by a wide margin.
    Measured over 10 slides x 4 tiles, cross-slide std of the tissue mean RGB:

        raw             10.14
        macenko         10.59   (+4%)
        fixed Ruifrok   10.51   (+4%)
        fixed pooled    11.26   (+11%)
        reinhard         3.66   (-64%)

    The deconvolution methods all rescale each stain's *concentration
    percentile*, which is set by the darkest nuclei -- and those are already
    similar between slides here. What actually differs is a global tone/tint
    shift across the whole tile, which is exactly what matching LAB moments
    corrects and what concentration rescaling leaves alone. Macenko is kept as
    an option because it is the literature default, not because it helped.

    tissue_only takes the moments over tissue pixels instead of the whole tile.
    It sounds like the safer choice -- a tile that is mostly glass should not
    drag its tissue around -- but measured over 60 tiles spanning 0.22-0.94
    tissue fraction, it is clearly worse:

                     corr(output brightness, tissue fraction)   output std
        tissue-only              -0.642                            8.57
        all pixels               -0.004                            4.88

    Matching only the tissue moments pins the tissue mean near the OD
    threshold that defines the mask, so which pixels count as tissue shifts
    with how far the transform stretched the tile -- and that shift tracks
    tissue fraction. Including the background gives each tile a large, stable
    anchor instead. Hence the default is off; it is kept only to make the
    comparison reproducible.
    """

    def __init__(self, mean_ref, std_ref, tissue_only=False):
        self.mean_ref = np.asarray(mean_ref, dtype=np.float64)
        self.std_ref = np.asarray(std_ref, dtype=np.float64)
        self.tissue_only = tissue_only

    def __call__(self, img):
        arr = np.asarray(img.convert("RGB"))
        lab = rgb2lab(arr.astype(np.float64) / 255.0)
        flat = lab.reshape(-1, 3)

        sel = slice(None)
        if self.tissue_only:
            mask = ~np.any(_to_od(arr) < OD_BETA, axis=1)
            if mask.sum() >= 100:
                sel = mask
        mean = flat[sel].mean(axis=0)
        std = np.maximum(flat[sel].std(axis=0), 1e-6)

        out = (flat - mean) / std * self.std_ref + self.mean_ref
        rgb = lab2rgb(out.reshape(lab.shape))     # may go out of gamut; clipped below
        return Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))


def fit_reinhard_reference(paths, tissue_only=False, max_side=1024):
    """Pooled LAB moments over the reference images."""
    chunks = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.BICUBIC)
        arr = np.asarray(img)
        flat = rgb2lab(arr.astype(np.float64) / 255.0).reshape(-1, 3)
        if tissue_only:
            mask = ~np.any(_to_od(arr) < OD_BETA, axis=1)
            if mask.sum() >= 100:
                flat = flat[mask]
        chunks.append(flat)
    if not chunks:
        raise ValueError("no reference images given")
    pooled = np.concatenate(chunks)
    return pooled.mean(axis=0), pooled.std(axis=0)


def _reference_paths(ref, n, seed=0):
    if os.path.isdir(ref):
        names = sorted(f for f in os.listdir(ref)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")))
        if not names:
            raise ValueError(f"no images under --stain_ref {ref}")
        idx = np.random.default_rng(seed).choice(len(names), min(n, len(names)), replace=False)
        return [os.path.join(ref, names[i]) for i in sorted(idx)]
    return [ref]


def load_normalizer(ref, method="reinhard", n=24, seed=0, tissue_only=False, cache=True):
    """Fit (or load a cached) reference. Cached beside the reference so every
    dataloader worker and every later run share one target -- refitting per
    worker would silently give each of them a different one."""
    tag = f"stain_{method}_n{n}_s{seed}{'_tissue' if tissue_only else ''}.npz"
    cache_path = os.path.join(ref, tag) if os.path.isdir(ref) else f"{ref}.{tag}"
    if cache and os.path.isfile(cache_path):
        d = np.load(cache_path)
        if method == "macenko":
            return MacenkoNormalizer(d["a"], d["b"])
        return ReinhardNormalizer(d["a"], d["b"], tissue_only)

    paths = _reference_paths(ref, n, seed)
    if method == "macenko":
        a, b = fit_reference(paths)
        norm = MacenkoNormalizer(a, b)
    else:
        a, b = fit_reinhard_reference(paths, tissue_only)
        norm = ReinhardNormalizer(a, b, tissue_only)
    if cache:
        try:
            np.savez(cache_path, a=a, b=b)
        except OSError:
            pass
    return norm


def build_stain_pipeline(opt, domain):
    """Return a PIL->PIL callable for domain 'A' or 'B', or None if this domain
    has neither normalization nor augmentation enabled."""
    steps = []

    method = getattr(opt, "stain_norm", "none")
    if method != "none" and domain in getattr(opt, "stain_norm_domains", "A"):
        ref = getattr(opt, f"stain_ref_{domain}")
        if not ref:
            raise ValueError(f"--stain_norm {method} needs --stain_ref_{domain}")
        steps.append(load_normalizer(ref, method=method, n=opt.stain_ref_n,
                                     tissue_only=bool(getattr(opt, "stain_tissue_only", 0))))

    # Augmentation is training-only; at test time the same weights must see
    # deterministically preprocessed input.
    if opt.isTrain and getattr(opt, "hed_aug", 0.0) > 0 and domain in getattr(opt, "hed_aug_domains", "A"):
        steps.append(HedJitter(opt.hed_aug, opt.hed_aug_bias))

    if not steps:
        return None

    def pipeline(img):
        for s in steps:
            img = s(img)
        return img
    return pipeline
