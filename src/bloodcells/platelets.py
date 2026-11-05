"""Platelet branch: small purple components outside the WBCs (project.md, Phase 6).

Platelets are taken from the purple mask, not from the cell mask, so a platelet
stuck to an RBC is found even when the cell mask merges the two; it is then
removed from the RBC mask before the watershed (``pipeline.make_rbc_mask``).

Two morphological filters turn purple specks into platelets:

* **grouping**: the granules of one platelet are separate components of the
  purple mask; a closing by ``group_r`` x r decides which belong together
  (as the lobes of a WBC nucleus).
* **enclosure**: a purple speck *inside* an RBC (stain deposit, bluish pale
  centre) is surrounded by non-purple cell pixels, whereas a platelet lies on
  the background or on the edge of an RBC. The ring between ``ring_in`` and
  ``ring_out`` x r around each candidate is measured; candidates whose ring is
  more than ``max_enclosed`` cell pixels are rejected (on BCCD train: keeps
  179 / 187 labelled platelets and removes 80 % of the other candidates).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from .segmentation import closing_by_disk, disk_radius


def group_components(mask: np.ndarray, radius: int) -> np.ndarray:
    """Label map in which components closer than ~2 ``radius`` share a label (pixels of ``mask`` only)."""
    if not mask.any():
        return np.zeros(mask.shape, np.int32)
    groups = measure.label(closing_by_disk(mask, radius) | mask, connectivity=2)
    return np.where(mask, groups, 0).astype(np.int32)


def enclosure(labels: np.ndarray, cell_mask: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    """Per label, the fraction of the ring (r_in, r_out] px around it that lies in ``cell_mask``.

    One feature transform for all labels: each ring pixel belongs to its
    nearest object, so the rings of close objects do not overlap.
    Index 0 of the result is unused.
    """
    n = int(labels.max())
    if n == 0:
        return np.zeros(1)
    dist, (iy, ix) = ndi.distance_transform_edt(labels == 0, return_indices=True)
    ring = (dist > r_in) & (dist <= r_out)
    owner = labels[iy[ring], ix[ring]]
    total = np.bincount(owner, minlength=n + 1).astype(float)
    inside = np.bincount(owner, weights=cell_mask[ring].astype(float), minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, inside / total, 0.0)


def find_platelets(
    purple_mask: np.ndarray,
    wbc_cells: np.ndarray,
    cell_mask: np.ndarray,
    r: float,
    cfg: dict,
) -> np.ndarray:
    """Label map of platelets (stained cores), consecutive labels 1..n.

    Candidates: purple pixels outside the WBC cells, grouped by a closing of
    ``platelets.group_r`` x r. A candidate is kept when its area lies in
    [area_min, area_max] x pi r^2, it does not touch a WBC cell (precedence:
    WBC, then platelet, then RBC) and it is not enclosed in a cell (see
    module docstring).
    """
    p = cfg["platelets"]
    unit = np.pi * r**2
    wbc = wbc_cells > 0
    candidates = purple_mask.astype(bool) & ~wbc
    lab = group_components(candidates, disk_radius(p["group_r"], r))
    n = int(lab.max())
    if n == 0:
        return lab
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    keep = (areas >= p["area_min"] * unit) & (areas <= p["area_max"] * unit)
    if wbc.any():
        near_wbc = ndi.binary_dilation(wbc, structure=np.ones((3, 3), bool))
        keep[np.unique(lab[near_wbc])] = False
    # Ring in non-purple cell pixels: a speck inside an RBC is surrounded by them.
    enclosed = enclosure(
        lab, cell_mask.astype(bool) & ~purple_mask.astype(bool), p["ring_in"] * r, p["ring_out"] * r
    )
    keep &= enclosed <= p["max_enclosed"]
    keep[0] = False
    lut = np.zeros(n + 1, np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1)
    return lut[lab]
