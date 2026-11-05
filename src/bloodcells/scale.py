"""Scale estimation by granulometry: the typical RBC radius ``r`` (project.md, Phase 3).

Every size parameter of the pipeline is a multiple of ``r``, so the pipeline
works at any magnification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import filters

from .preprocessing import denoise, green_channel


def opening_by_disk(mask: np.ndarray, radius: float, edt: np.ndarray | None = None) -> np.ndarray:
    """Binary opening by a Euclidean disk, computed with two distance transforms.

    erosion  = pixels whose nearest background pixel is farther than radius
               (the closed disk {d <= radius} then lies inside the mask)
    dilation = pixels within distance radius of the eroded set
    Same result as ``skimage.morphology.opening`` with ``disk(radius)``, but fast
    for any radius (no large structuring element). ``edt`` (distance transform of
    ``mask``) can be passed to reuse it across radii.
    """
    if edt is None:
        edt = ndi.distance_transform_edt(mask)
    eroded = edt > radius
    if not eroded.any():
        return np.zeros_like(mask, dtype=bool)
    return (ndi.distance_transform_edt(~eroded) <= radius) & mask


def pattern_spectrum(mask: np.ndarray, r_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Granulometry of a binary mask with disks of radius 0..r_max.

    Returns ``radii`` (1..r_max) and ``spectrum`` where spectrum[k] is the area
    removed when the opening radius grows from radii[k] - 1 to radii[k]: the area
    of objects whose largest inscribed disk has a radius of about radii[k] - 1.
    """
    mask = mask.astype(bool)
    edt = ndi.distance_transform_edt(mask)
    areas = [mask.sum()] + [opening_by_disk(mask, k, edt).sum() for k in range(1, r_max + 1)]
    spectrum = -np.diff(np.asarray(areas, dtype=float))
    return np.arange(1, r_max + 1), spectrum


def smooth_spectrum(
    spectrum: np.ndarray,
    weight: float = 1.0,
    sigma_log: float = 0.08,
    r_min: float = 2.0,
    n_grid: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted pattern spectrum, smoothed in log-radius.

    spectrum[k] is the area of objects whose inscribed radius lies in (k, k + 1],
    so it is placed at rho = k + 0.5 and divided by ``rho ** weight``:
    weight 0 keeps the raw (area-weighted) spectrum, where one big WBC or clump
    outweighs many RBCs; weight 2 counts objects (area / rho^2), which lets
    small debris dominate; weight 1 is the compromise found on the train set.
    The Gaussian smoothing has a constant width in log(rho), i.e. a constant
    *relative* width, so the result does not depend on the magnification.
    Radii below ``r_min`` pixels (noise, thin ring walls) are ignored.

    Returns ``grid`` (radii, log-spaced) and ``density`` (same length).
    """
    rho = np.arange(len(spectrum)) + 0.5
    keep = (rho >= r_min) & (spectrum > 0)
    grid = np.exp(np.linspace(np.log(r_min), np.log(max(rho[-1], r_min * 2)), n_grid))
    if not keep.any():
        return grid, np.zeros(n_grid)
    w = spectrum[keep] / rho[keep] ** weight
    z = (np.log(grid)[:, None] - np.log(rho[keep])[None, :]) / sigma_log
    return grid, (w[None, :] * np.exp(-0.5 * z**2)).sum(axis=1)


@dataclass
class ScaleEstimate:
    r: float  # estimated RBC radius in pixels (calibrated to the box radius)
    ok: bool  # False when the default radius was used
    r_inscribed: float  # raw spectrum peak: inscribed-disk radius, before calibration
    radii: np.ndarray
    spectrum: np.ndarray
    grid: np.ndarray  # radii of the smoothed spectrum
    density: np.ndarray  # smoothed, weighted spectrum
    mask: np.ndarray  # coarse cell mask the estimate was computed on


def coarse_cell_mask(img: np.ndarray, median_r: int = 2) -> np.ndarray:
    """Bootstrapping mask (no r needed): Otsu on the green channel, holes filled.

    Plain hole filling is acceptable here: it only has to give the size of
    typical cells, not a clean segmentation.
    """
    g = denoise(green_channel(img), median_r)
    return ndi.binary_fill_holes(g < filters.threshold_otsu(g))


def estimate_rbc_radius(
    mask: np.ndarray,
    r_max: int,
    r_default: float,
    weight: float = 1.0,
    sigma_log: float = 0.08,
    r_min: float = 2.0,
    calibration: float = 1.0,
    min_peak_share: float = 0.1,
) -> ScaleEstimate:
    """RBC radius = position of the peak of the smoothed pattern spectrum.

    RBCs are the most numerous cells, so they make the main peak once the
    spectrum is weighted and smoothed (``smooth_spectrum``). The peak gives the
    radius of the largest disk inscribed in a typical RBC; ``calibration``
    (box radius / inscribed radius, fitted on the train set by
    ``scripts/check_scale.py``) converts it to the box-based radius in which
    ``config.yaml`` parameters are expressed. Falls back to ``r_default`` when the
    mask is empty or the peak region (+-2 sigma) holds less than
    ``min_peak_share`` of the weighted spectrum (no dominant size).
    """
    radii, spectrum = pattern_spectrum(mask, r_max)
    grid, density = smooth_spectrum(spectrum, weight, sigma_log, r_min)
    if density.sum() <= 0:
        return ScaleEstimate(float(r_default), False, np.nan, radii, spectrum, grid, density, mask)
    peak = float(grid[np.argmax(density)])
    near = np.abs(np.log(grid / peak)) <= 2 * sigma_log
    if density[near].sum() < min_peak_share * density.sum():
        return ScaleEstimate(float(r_default), False, peak, radii, spectrum, grid, density, mask)
    return ScaleEstimate(calibration * peak, True, peak, radii, spectrum, grid, density, mask)


def estimate_scale(img: np.ndarray, cfg: dict) -> ScaleEstimate:
    """End-to-end scale estimate from an RGB image, using ``config.yaml['scale']``."""
    sc = cfg["scale"]
    mask = coarse_cell_mask(img, cfg["preprocessing"]["median_r"])
    r_max = max(4, int(sc["r_max_fraction"] * min(mask.shape)))
    return estimate_rbc_radius(
        mask,
        r_max,
        sc["r_default"],
        weight=sc["weight"],
        sigma_log=sc["sigma_log"],
        r_min=sc["r_min"],
        calibration=sc["calibration"],
    )
