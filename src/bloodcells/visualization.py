"""Images and figures for inspection: step images, overlays, spectrum plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy import ndimage as ndi
from skimage import io, morphology

CELL_COLOR = (0, 220, 0)
PURPLE_COLOR = (255, 0, 255)
# Class colours of the final overlay (project.md, Phase 9).
CLASS_COLORS = {
    "RBC": (230, 30, 30),
    "WBC": (30, 90, 255),
    "Platelet": (0, 200, 60),
    "Artefact": (255, 200, 0),
}


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Displayable uint8 image of any step output.

    RGB uint8 is returned as is; masks become 0/255; float channels in [0, 1]
    are scaled by 255, other float ranges (e.g. -b*, top-hat) are min-max
    stretched for display only.
    """
    if x.dtype == np.uint8:
        return x
    if x.dtype == bool:
        return x.astype(np.uint8) * 255
    x = x.astype(float)
    lo, hi = float(x.min()), float(x.max())
    if lo >= 0 and hi <= 1:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        return np.zeros(x.shape, np.uint8)
    return np.round((x - lo) / (hi - lo) * 255).astype(np.uint8)


def mask_contours(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Inner boundary of a mask (internal gradient: mask minus its erosion by a disk of ``width``).

    Erosion via the distance transform; the image border is not a boundary.
    """
    mask = mask.astype(bool)
    return mask & (ndi.distance_transform_edt(mask) <= width)


def overlay(img: np.ndarray, cell_mask: np.ndarray, purple_mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Original image with the cell mask contour in green and the purple mask contour in magenta."""
    out = img.copy()
    out[mask_contours(cell_mask, width)] = CELL_COLOR
    out[mask_contours(purple_mask, width)] = PURPLE_COLOR
    return out


def label_contours(labels: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological gradient of the label map: dilation(labels) != erosion(labels).

    Unlike the contour of the binary mask, this also draws the watershed lines
    between touching cells (two labels meet there). Background-only pixels are
    never contours.
    """
    se = morphology.disk(radius)
    edge = morphology.dilation(labels, se) != morphology.erosion(labels, se)
    return edge & (morphology.dilation(labels, se) > 0)


def class_overlay(
    img: np.ndarray,
    labels: np.ndarray,
    classes: dict[int, str],
    radius: int = 1,
    show_ids: bool = False,
) -> np.ndarray:
    """Original image with the contour of every labelled object in its class colour.

    Contours come from the label-map gradient, so split cells are visibly
    separated. With ``show_ids`` the label number is written at each centroid.
    """
    out = img.copy()
    edge = label_contours(labels, radius)
    # Each contour pixel takes the colour of the object it belongs to (or of
    # the neighbour that dilated into the background).
    owner = np.where(labels > 0, labels, morphology.dilation(labels, morphology.disk(radius)))
    lut = np.zeros((int(labels.max()) + 1, 3), np.uint8)
    for k, cls in classes.items():
        lut[k] = CLASS_COLORS.get(cls, (255, 255, 255))
    out[edge] = lut[owner[edge]]
    if show_ids:
        out = _draw_ids(out, labels)
    return out


def _draw_ids(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    import cv2

    out = np.ascontiguousarray(img)
    centres = ndi.center_of_mass(labels > 0, labels, range(1, int(labels.max()) + 1))
    for k, (cy, cx) in enumerate(centres, start=1):
        if np.isfinite(cy):
            cv2.putText(out, str(k), (int(cx) - 6, int(cy) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    return out


def spectrum_figure(scale, r_ref: float | None = None, title: str = "") -> Figure:
    """Pattern spectrum (raw and smoothed) with the estimated radius.

    Built without pyplot, so it works with any backend and is not shown twice in notebooks.
    """
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    rho = scale.radii - 0.5
    if scale.spectrum.max() > 0:
        ax.bar(rho, scale.spectrum / scale.spectrum.max(), width=1, color="lightgray", label="raw spectrum")
    if scale.density.max() > 0:
        ax.plot(scale.grid, scale.density / scale.density.max(), color="tab:blue", label="weighted, smoothed")
    ax.axvline(scale.r_inscribed, color="tab:blue", ls="--", label=f"inscribed r = {scale.r_inscribed:.1f}")
    ax.axvline(scale.r, color="tab:red", label=f"r = {scale.r:.1f} px" + ("" if scale.ok else " (default)"))
    if r_ref is not None and np.isfinite(r_ref):
        ax.axvline(r_ref, color="k", lw=2, alpha=0.4, label=f"box r = {r_ref:.1f}")
    ax.set_xlabel("radius (px)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def save_image(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(path, to_uint8(x), check_contrast=False)
