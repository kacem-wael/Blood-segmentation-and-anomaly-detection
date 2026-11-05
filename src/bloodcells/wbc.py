"""WBC branch: nuclei from the purple mask, whole cell by gradient watershed (project.md, Phase 6).

A WBC = a large purple nucleus + a pale cytoplasm around it. The nucleus is
easy (strongly stained); the cytoplasm is found by a watershed of the
morphological gradient, flooded from the nucleus and competing with a
background marker and with the neighbouring RBCs, so that it stops on the
first strong edge around the nucleus.

Every size is a multiple of the RBC radius ``r`` (pixels), areas of pi r^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from .preprocessing import grey_morphology
from .segmentation import closing_by_disk, disk_radius


def morphological_gradient(gray: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilation - erosion by a disk: high on edges, ~0 inside flat regions."""
    return grey_morphology(gray, cv2.MORPH_GRADIENT, radius)


def find_nuclei(purple_mask: np.ndarray, r: float, cfg: dict) -> np.ndarray:
    """Label map of WBC nuclei: large components of the purple mask.

    The lobes of a segmented nucleus (neutrophil) are separate components of
    the purple mask; a closing by ``wbc.lobe_close_r`` x r groups them, so one
    WBC gives one nucleus. The nucleus pixels are the purple pixels of each
    group plus its small holes (a closing only decides which lobes belong
    together, it does not enlarge the nucleus). Groups smaller than
    ``wbc.nucleus_min_area`` x pi r^2 are platelet candidates, not nuclei.
    """
    w = cfg["wbc"]
    purple = purple_mask.astype(bool)
    grouped = closing_by_disk(purple, disk_radius(w["lobe_close_r"], r)) if purple.any() else purple
    groups = measure.label(grouped, connectivity=2)
    min_area = w["nucleus_min_area"] * np.pi * r**2
    nuclei = np.zeros(purple.shape, np.int32)
    n = 0
    for region in measure.regionprops(groups):
        sl = region.slice
        member = (groups[sl] == region.label) & purple[sl]
        if member.sum() < min_area:
            continue
        n += 1
        # Small holes inside the nucleus (pale chromatin) belong to it.
        filled = ndi.binary_fill_holes(member)
        holes = filled & ~member
        small = morphology.remove_small_objects(holes, max_size=int(0.05 * np.pi * r**2))
        nuclei[sl][member | (holes & ~small)] = n
    return nuclei


@dataclass
class WbcResult:
    nuclei: np.ndarray  # label map, one label per WBC
    cells: np.ndarray  # label map of whole WBCs (nucleus + cytoplasm), same labels
    gradient: np.ndarray  # gradient image the watershed flooded
    markers: np.ndarray  # watershed markers (nuclei, rbc competitors, background)


def wbc_cells(
    green: np.ndarray,
    purple: np.ndarray,
    nuclei: np.ndarray,
    cell_mask: np.ndarray,
    r: float,
    cfg: dict,
    cells: np.ndarray | None = None,
) -> WbcResult:
    """Whole WBC (nucleus + cytoplasm) by a marker-controlled watershed of the gradient.

    Gradient: morphological gradient of the green channel (cell / background
    edges) and of the purple channel (cytoplasm is lilac, RBCs are pink: the
    edge between a WBC and a touching RBC is in the colour, not in the
    brightness). Both are scaled to [0, 1] and the maximum is flooded.

    Markers:
      * each nucleus (label 1..n),
      * background: farther than ``wbc.search`` x r from every nucleus, or
        *surely* background: outside the cell mask and, when the top-hat
        channel ``cells`` is given, below ``wbc.background_level`` x its Otsu
        threshold. The pale cytoplasm is often fainter than the Otsu threshold
        (missing from the cell mask); left unassigned, the gradient decides
        where the cell ends instead of the threshold,
      * RBC competitors: cell-mask pixels at least ``wbc.rbc_seed_dist`` x r away
        from the nucleus and deeper than ``wbc.rbc_seed_depth`` x r inside the
        cell mask: centres of the neighbouring RBCs. Without them a touching RBC
        has no seed of its own and is absorbed into the WBC.
    """
    w = cfg["wbc"]
    n = int(nuclei.max())
    gradient = np.maximum(_unit(morphological_gradient(green)), _unit(morphological_gradient(purple)))
    if n == 0:
        empty = np.zeros_like(nuclei)
        return WbcResult(nuclei, empty, gradient, empty)
    cell_mask = cell_mask.astype(bool) | (nuclei > 0)
    # Distance to the nearest nucleus and which nucleus it is (one feature transform).
    dist_nucleus, (iy, ix) = ndi.distance_transform_edt(nuclei == 0, return_indices=True)
    depth = ndi.distance_transform_edt(cell_mask)
    background, rbc = n + 1, n + 2
    sure_bg = ~cell_mask
    if cells is not None and w["background_level"] < 1:
        sure_bg &= cells < w["background_level"] * filters.threshold_otsu(cells)
    # Each nucleus marker is grown by a few pixels past its own edge: the
    # nucleus / cytoplasm boundary is the strongest ridge of the gradient, and a
    # marker stopped inside it would lose the cytoplasm to the background marker.
    grow = w["marker_grow_px"]
    if grow > 0:
        markers = np.where((dist_nucleus <= grow) & ~sure_bg, nuclei[iy, ix], 0).astype(np.int32)
    else:
        markers = nuclei.astype(np.int32).copy()
    markers[(sure_bg & (markers == 0)) | (dist_nucleus > w["search"] * r)] = background
    seeds = cell_mask & (dist_nucleus >= w["rbc_seed_dist"] * r) & (depth >= w["rbc_seed_depth"] * r)
    markers[seeds & (markers == 0)] = rbc
    flooded = segmentation.watershed(gradient, markers)
    # One connected piece per WBC (the one holding the nucleus), holes filled.
    out = np.zeros_like(nuclei)
    for k in range(1, n + 1):
        piece = flooded == k
        lab = measure.label(piece)
        keep = np.isin(lab, np.unique(lab[(nuclei == k) & piece]))
        out[ndi.binary_fill_holes(keep) & (out == 0)] = k
    return WbcResult(nuclei, out, gradient, markers)


def nc_ratio(nucleus_area: float, cell_area: float, min_cytoplasm: float = 0.1) -> tuple[float, bool]:
    """N/C ratio = nucleus area / cytoplasm area (cytoplasm = cell - nucleus).

    Returns (ratio, cytoplasm_uncertain). When the cytoplasm is less than
    ``min_cytoplasm`` of the cell (pale cytoplasm lost by the segmentation) the
    ratio would be huge and meaningless: it is NaN and the flag is set.
    """
    cytoplasm = cell_area - nucleus_area
    if cell_area <= 0 or cytoplasm < min_cytoplasm * cell_area:
        return float("nan"), True
    return float(nucleus_area / cytoplasm), False


def _unit(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] by the 99.5th percentile (a few extreme edges do not set the range)."""
    hi = np.percentile(x, 99.5)
    return np.clip(x / hi, 0, 1) if hi > 0 else np.zeros_like(x, dtype=float)
