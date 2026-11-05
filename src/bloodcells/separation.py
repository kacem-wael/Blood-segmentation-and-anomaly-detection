"""Splitting touching RBCs: distance transform + markers + watershed (project.md, Phase 5).

The distance to the background is high at the centre of each cell and drops
at the neck between two touching cells. Each cell centre becomes a marker
(h-maxima of the distance, or local peaks, or ultimate erosion) and the
watershed of the inverted distance, flooded from the markers inside the mask,
cuts the clumps along the necks.

Every size is a multiple of the RBC radius ``r`` (pixels), areas of pi r^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, measure, morphology, segmentation

MARKERS = ("hmaxima", "peaks", "ultimate")


@dataclass
class Separation:
    labels: np.ndarray  # one integer per RBC, 0 = background
    distance: np.ndarray  # smoothed distance transform of the RBC mask
    markers: np.ndarray  # labelled markers the watershed was flooded from
    n_orphans: int  # components that received no marker (kept as their own label)
    n_merged: int  # small fragments merged into a neighbour


def distance_map(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Euclidean distance to the background, Gaussian-smoothed (removes tiny ridges)."""
    dist = ndi.distance_transform_edt(mask)
    if sigma > 0:
        dist = ndi.gaussian_filter(dist, sigma=sigma)
        dist[~mask] = 0
    return dist


def find_markers(
    dist: np.ndarray, mask: np.ndarray, r: float, cfg: dict, method: str | None = None
) -> np.ndarray:
    """Labelled cell centres.

    ``hmaxima``: regional maxima of the distance with a dynamic >= h x r: a
    peak counts only if it rises at least h above the neck that joins it to a
    higher peak (h too small: over-segmentation; too large: merged cells).
    ``peaks``: local maxima at least ``min_distance`` x r apart.
    ``ultimate``: ultimate erosion = regional maxima of the distance (h -> 0):
    the last pieces left by successive erosions; it over-splits elongated cells
    and is kept for comparison.
    """
    s = cfg["separation"]
    method = method or s["markers"]
    if method == "hmaxima":
        peaks = morphology.h_maxima(dist, s["h"] * r) > 0
    elif method == "ultimate":
        peaks = morphology.local_maxima(dist, connectivity=2) & mask
    elif method == "peaks":
        comps = measure.label(mask)
        coords = feature.peak_local_max(
            dist, min_distance=max(1, int(round(s["min_distance"] * r))), labels=comps, exclude_border=False
        )
        peaks = np.zeros(mask.shape, bool)
        peaks[tuple(coords.T)] = True
    else:
        raise ValueError(f"unknown marker method '{method}', expected one of {MARKERS}")
    return measure.label(peaks & mask, connectivity=2)


def merge_small_fragments(labels: np.ndarray, min_area: float) -> tuple[np.ndarray, int]:
    """Merge every label smaller than ``min_area`` px into the neighbour sharing its longest border.

    Fragments with no labelled neighbour are kept (an isolated small cell is a
    cell, not a fragment). Smallest fragments first; areas are updated after
    each merge.
    """
    labels = labels.copy()
    areas = np.bincount(labels.ravel())
    merged = 0
    for lab in np.argsort(areas):
        if lab == 0 or areas[lab] == 0 or areas[lab] >= min_area:
            continue
        region = labels == lab
        ring = ndi.binary_dilation(region, structure=np.ones((3, 3), bool)) & ~region
        neigh = labels[ring]
        neigh = neigh[neigh > 0]
        if neigh.size == 0:
            continue
        target = np.bincount(neigh).argmax()
        labels[region] = target
        areas[target] += areas[lab]
        areas[lab] = 0
        merged += 1
    return relabel(labels), merged


def relabel(labels: np.ndarray) -> np.ndarray:
    """Consecutive labels 1..n (0 stays background)."""
    return segmentation.relabel_sequential(labels)[0].astype(np.int32)


def separate(
    rbc_mask: np.ndarray, r: float, cfg: dict, method: str | None = None, watershed: bool = True
) -> Separation:
    """Label map of individual RBCs from the RBC mask.

    Watershed of the inverted distance from the markers, restricted to the
    mask. Connected components that hold no marker (too small or too flat for
    the h-maxima) are not dropped: each becomes its own label. Fragments
    smaller than ``min_fragment_area`` x pi r^2 are merged into a neighbour.
    ``watershed=False`` (ablation) returns the connected components.
    """
    s = cfg["separation"]
    mask = rbc_mask.astype(bool)
    dist = distance_map(mask, s["dist_sigma"] * r)
    if not watershed:
        return Separation(
            relabel(measure.label(mask, connectivity=1)), dist, np.zeros(mask.shape, np.int32), 0, 0
        )
    markers = find_markers(dist, mask, r, cfg, method)
    labels = segmentation.watershed(-dist, markers, mask=mask).astype(np.int32)
    orphans = measure.label(mask & (labels == 0), connectivity=1)
    n_orphans = int(orphans.max())
    if n_orphans:
        labels[orphans > 0] = orphans[orphans > 0] + labels.max()
    labels, n_merged = merge_small_fragments(labels, s["min_fragment_area"] * np.pi * r**2)
    return Separation(labels, dist, markers, n_orphans, n_merged)


# ---- Bonus: SKIZ and ultimate erosion (project.md, Phase 6) ----------------


def skiz(markers: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Skeleton by influence zones of labelled markers.

    The influence zone of a marker is the set of pixels closer to it than to
    any other marker (watershed of the distance to the markers); the SKIZ is
    the set of boundaries between zones. Restricted to ``mask`` if given, it
    is the geodesic SKIZ, the morphological basis of the watershed.
    Returns (zones label map, SKIZ boolean lines).
    """
    dist = ndi.distance_transform_edt(markers == 0)
    zones = segmentation.watershed(dist, markers, mask=mask).astype(np.int32)
    se = morphology.disk(1)
    lines = (morphology.dilation(zones, se) != morphology.erosion(zones, se)) & (zones > 0)
    return zones, lines


def ultimate_erosion_count(mask: np.ndarray, r: float, h: float = 0.05) -> int:
    """Number of cells estimated in a clump: components of the ultimate erosion.

    Ultimate erosion = regional maxima of the distance transform (the last
    piece of each object before it vanishes under successive erosions); a small
    dynamic ``h`` x r ignores the plateaus created by the pixel grid.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return 0
    dist = ndi.distance_transform_edt(mask)
    peaks = morphology.h_maxima(dist, max(h * r, 0.5)) > 0
    return int(measure.label(peaks & mask, connectivity=2).max())
