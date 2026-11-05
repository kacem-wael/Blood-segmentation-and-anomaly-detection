"""Preprocessing: channels, denoising, illumination correction (project.md, Phase 3)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, morphology, transform


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 -> float in [0, 1]; float input is returned as float64."""
    return img.astype(float) / 255.0 if img.dtype == np.uint8 else img.astype(float)


def green_channel(img: np.ndarray) -> np.ndarray:
    """Green channel in [0, 1]. Hemoglobin absorbs green: cells are dark on a bright background."""
    return to_float(img)[..., 1]


def purple_channel(img: np.ndarray, kind: str = "lab_b") -> np.ndarray:
    """Stain channel where WBC nuclei and platelets are bright.

    ``lab_b``: -b* of CIE Lab (purple/blue = negative b*), best WBC/RBC separation
    on BCCD (notebook 01, results/tables/channel_separability.csv). ``hsv_s``: HSV saturation.
    """
    rgb = to_float(img)
    if kind == "lab_b":
        return -color.rgb2lab(rgb)[..., 2]
    if kind == "hsv_s":
        return color.rgb2hsv(rgb)[..., 1]
    raise ValueError(f"unknown purple channel '{kind}'")


def denoise(gray: np.ndarray, radius: int = 2) -> np.ndarray:
    """Median filter with a disk: removes salt-and-pepper noise, keeps edges."""
    return ndi.median_filter(gray, footprint=morphology.disk(radius))


def grey_morphology(gray: np.ndarray, op: int, radius: int) -> np.ndarray:
    """Grayscale morphology by ``skimage.morphology.disk(radius)`` computed with OpenCV.

    Same result as ``skimage.morphology`` (to float32 precision) but about 15x
    faster for large disks. ``op`` is a ``cv2.MORPH_*`` constant (CLOSE,
    GRADIENT, ...). Borders are replicated.
    """
    se = morphology.disk(radius).astype(np.uint8)
    out = cv2.morphologyEx(np.asarray(gray, np.float32), op, se, borderType=cv2.BORDER_REPLICATE)
    return out.astype(np.float64)


def background_closing(gray: np.ndarray, radius: float, downsample: int = 4) -> np.ndarray:
    """Background estimate: grayscale closing by a disk larger than any cell.

    Cells are dark, so a closing by a disk that does not fit inside them removes
    them and leaves the (bright) background. The closing is computed on a
    downsampled image (the background is smooth) and resized back, which keeps
    large structuring elements fast.
    """
    small = transform.downscale_local_mean(gray, (downsample, downsample))
    rad = max(1, int(round(radius / downsample)))
    # Edge padding: without padding (or with mirror padding, which turns a lighting
    # gradient into a V) the closing overestimates the background near the border.
    padded = np.pad(small, rad, mode="edge")
    closed = grey_morphology(padded, cv2.MORPH_CLOSE, rad)[rad:-rad, rad:-rad]
    closed = filters.gaussian(closed, sigma=1)
    return transform.resize(closed, gray.shape, order=1, mode="edge", anti_aliasing=False)


def tophat_from_background(gray: np.ndarray, background: np.ndarray, relative: bool = False) -> np.ndarray:
    """(background - image), clipped at 0, divided by the background if ``relative``."""
    tophat = np.clip(background - gray, 0, None)
    return tophat / np.maximum(background, 1e-3) if relative else tophat


def black_tophat(gray: np.ndarray, radius: float, downsample: int = 4, relative: bool = False) -> np.ndarray:
    """Black top-hat = closing - image: cells become bright on a flat dark background.

    Removes uneven illumination (vignetting, gradients). Clipped at 0.
    ``relative=True`` divides by the background, (closing - image) / closing:
    in transmitted light the image is illumination x transmittance, so a darker
    region also has a lower cell contrast; dividing removes this multiplicative
    effect too (flat-field correction), which the plain top-hat cannot.
    """
    return tophat_from_background(gray, background_closing(gray, radius, downsample), relative)


def normalize(x: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    """Percentile contrast stretch to [0, 1] (robust to a few extreme pixels)."""
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def preprocess(img: np.ndarray, cfg: dict, r: float) -> dict[str, np.ndarray]:
    """Full Phase 3 preprocessing for a known RBC radius ``r`` (pixels).

    Returns (main outputs first, then intermediates kept for figures and ``outputs/``):
        green:       denoised green channel in [0, 1] (cells dark)
        cells:       normalized black top-hat of ``green`` (cells bright, flat background),
                     relative to the background if ``tophat_relative``
        purple:      normalized, denoised stain channel (nuclei and platelets bright)
        green_raw:   green channel before the median filter
        background:  closing of ``green`` (background estimate)
        tophat:      top-hat before normalization
        purple_raw:  stain channel before the median filter
    """
    p = cfg["preprocessing"]
    green_raw = green_channel(img)
    green = denoise(green_raw, p["median_r"])
    background = background_closing(green, p["tophat_r"] * r, p["tophat_downsample"])
    tophat = tophat_from_background(green, background, p["tophat_relative"])
    purple_raw = purple_channel(img, p["purple_channel"])
    purple = denoise(purple_raw, p["median_r"])
    return {
        "green": green,
        "cells": normalize(tophat),
        "purple": normalize(purple),
        "green_raw": green_raw,
        "background": background,
        "tophat": tophat,
        "purple_raw": purple_raw,
    }
