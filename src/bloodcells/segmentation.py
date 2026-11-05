"""Segmentation and morphological cleaning (project.md, Phase 4).

Two binary masks per image: the cell mask (RBC + WBC + platelets vs background)
and the purple mask (WBC nuclei + platelets). Every size is a multiple of the
RBC radius ``r`` (pixels) from ``scale.estimate_scale``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from .preprocessing import normalize, preprocess
from .scale import ScaleEstimate, estimate_scale, opening_by_disk


def threshold_cells(cells: np.ndarray, method: str = "otsu", block: int = 0) -> np.ndarray:
    """Cell mask from the (relative) black top-hat: cells are bright, background ~0.

    ``otsu``: one global threshold. ``local``: Gaussian-weighted local mean over
    ``block`` pixels (odd), the adaptive alternative of project.md Phase 4.
    """
    if method == "otsu":
        return cells > filters.threshold_otsu(cells)
    if method == "local":
        return cells > filters.threshold_local(cells, block_size=block | 1)
    raise ValueError(f"unknown cell threshold '{method}'")


def threshold_purple(purple: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Purple stain (nuclei + platelets) on the -b* channel.

    ``otsu``: two classes. ``multiotsu``: three classes (background, RBC, stain),
    the top class is kept; useful when RBCs are bluish enough to pull a two-class
    threshold down.
    """
    if method == "otsu":
        return purple > filters.threshold_otsu(purple)
    if method == "multiotsu":
        return purple > filters.threshold_multiotsu(purple, classes=3)[-1]
    raise ValueError(f"unknown purple threshold '{method}'")


def purple_coverage(purple: np.ndarray, cells: np.ndarray) -> float:
    """Share of the cell area (cells | purple) that the stain mask covers.

    BCCD: median 0.11 (nuclei and platelets only); Raabin-WBC: median 0.95
    (the RBCs are stained too).
    """
    union = cells | purple
    return float(purple.sum() / max(union.sum(), 1))


def circularity(region) -> float:
    """4 pi area / perimeter^2 with the Crofton perimeter: ~1 for a disc.

    ``regionprops.perimeter`` underestimates the perimeter of small shapes, which
    makes them look round (the gap between three overlapping discs scores 0.68
    with it, 0.55 with Crofton). Ideal tangent discs enclose a gap of 0.2.
    """
    return 4 * np.pi * region.area / max(measure.perimeter_crofton(region.image), 1) ** 2


def fill_round_holes(
    mask: np.ndarray,
    r: float,
    max_area: float = 0.5,
    min_circ: float = 0.6,
    min_solidity: float = 0.9,
    always_fill: float = 0.005,
) -> np.ndarray:
    """Fill RBC pale centres (small, round, convex holes), not the gaps between touching cells.

    Areas are in units of pi r^2. A gap enclosed by touching cells has concave
    sides: low circularity and low solidity (area / convex hull area: <= 0.8
    for 3 or 4 touching discs, >= 0.97 for pale centres). Such gaps stay
    background; a plain ``binary_fill_holes`` would glue the cells together.
    Holes below ``always_fill`` (a few pixels) are noise and are always filled.
    """
    holes = measure.label(ndi.binary_fill_holes(mask) & ~mask, connectivity=1)
    out = mask.copy()
    unit = np.pi * r**2
    for region in measure.regionprops(holes):
        tiny = region.area <= always_fill * unit
        pale_centre = (
            region.area <= max_area * unit
            and circularity(region) >= min_circ
            and region.solidity >= min_solidity
        )
        if tiny or pale_centre:
            out[tuple(region.coords.T)] = True
    return out


def disk_radius(size_in_r: float, r: float) -> int:
    """Structuring-element radius in pixels for a size given in units of r (at least 1)."""
    return max(1, int(round(size_in_r * r)))


def closing_by_disk(mask: np.ndarray, radius: float) -> np.ndarray:
    """Binary closing by a Euclidean disk with two distance transforms (dual of ``opening_by_disk``).

    dilation = pixels within ``radius`` of the mask; erosion of it = pixels whose
    nearest non-dilated pixel is farther than ``radius``. The image border is
    not treated as background, so cells cut by the border are not eroded.
    """
    dilated = ndi.distance_transform_edt(~mask) <= radius
    if dilated.all():
        return dilated
    return ndi.distance_transform_edt(dilated) > radius


def clean_mask(
    mask: np.ndarray,
    r: float,
    cfg: dict,
    fill_holes: bool = True,
    steps: dict | None = None,
    prefix: str = "",
    open_r: float | None = None,
    close_r: float | None = None,
    min_area: float | None = None,
) -> np.ndarray:
    """Shape-filtered hole fill, opening, closing, small-object removal, in this order.

    The hole fill comes first so that the opening cannot break the thin ring
    of a pale-centred RBC (it would then never be filled). Each intermediate
    mask is stored in ``steps`` (if given) as ``prefix + step name``.
    ``open_r``, ``close_r`` (x r) and ``min_area`` (x pi r^2) default to the
    ``segmentation`` values of the config (the cell mask ones).
    """
    s = cfg["segmentation"]
    open_r = s["open_r"] if open_r is None else open_r
    close_r = s["close_r"] if close_r is None else close_r
    min_area = s["min_area"] if min_area is None else min_area
    record = steps.__setitem__ if steps is not None else (lambda *_: None)
    out = mask.astype(bool)
    if fill_holes:
        # Seal the rim of RBCs whose pale centre leaks to the background through a
        # thin gap ("C" shapes): the centre then becomes a hole the filter can fill.
        # Gaps this closing creates between neighbouring cells are concave and stay unfilled.
        before = out
        if s["gap_close_r"] > 0:
            out = closing_by_disk(out, disk_radius(s["gap_close_r"], r))
        record(prefix + "gap_closing", out)
        closed = out
        if s["hole_fill"] == "shape":
            out = fill_round_holes(
                out,
                r,
                s["hole_max_area"],
                s["hole_min_circularity"],
                s["hole_min_solidity"],
                s["hole_always_fill"],
            )
        elif s["hole_fill"] == "plain":  # every hole, for comparison only
            out = ndi.binary_fill_holes(out)
        elif s["hole_fill"] != "none":
            raise ValueError(f"unknown hole_fill '{s['hole_fill']}'")
        if s["gap_close_r"] > 0 and not s.get("gap_close_keep", False):
            # The closing only served to find the holes: keep the filled holes but
            # not the bridges it built between neighbouring cells (they merge cells).
            out = before | (out & ~closed)
        record(prefix + "hole_fill", out)
    if not stage(cfg, "morphology"):  # ablation: no opening / closing / small-object removal
        for name in ("opening", "closing", "remove_small_objects"):
            record(prefix + name, out)
        return out
    out = opening_by_disk(out, disk_radius(open_r, r))
    record(prefix + "opening", out)
    out = closing_by_disk(out, disk_radius(close_r, r))
    record(prefix + "closing", out)
    min_size = max(1, int(min_area * np.pi * r**2))
    out = morphology.remove_small_objects(out, max_size=min_size)
    record(prefix + "remove_small_objects", out)
    return out


def stage(cfg: dict, name: str) -> bool:
    """Whether a pipeline stage is enabled (``config.yaml['pipeline']``, all on by default).

    The switches exist for the ablation study (project.md 10.2): preprocessing,
    morphology (opening / closing / small objects), wbc_first, watershed.
    """
    return bool(cfg.get("pipeline", {}).get(name, True))


# Every step of the pipeline up to Phase 4, in execution order: (name, description).
# ``segment`` stores each output under its name; scripts/export_steps.py writes
# them to outputs/<NN>-<name>/<image_id>.png.
STEPS: list[tuple[str, str]] = [
    ("original", "input RGB image"),
    ("green_channel", "G channel of RGB (hemoglobin absorbs green: cells dark)"),
    ("median_filter", "median filter, disk radius preprocessing.median_r px"),
    ("scale_coarse_mask", "scale bootstrap: Otsu on the median-filtered green channel + fill_holes"),
    ("scale_granulometry", "pattern spectrum of the coarse mask (openings by disks) -> RBC radius r"),
    ("background_closing", "grayscale closing by a disk of tophat_r x r: background estimate"),
    ("relative_tophat", "(closing - image) / closing: cells bright, flat background"),
    ("cells_normalized", "percentile contrast stretch of the top-hat to [0, 1]"),
    ("cells_otsu", "Otsu threshold of the cells channel: raw cell mask"),
    ("purple_lab_b", "-b* of CIE Lab: purple stain (nuclei, platelets) bright"),
    ("purple_median_normalized", "median filter + contrast stretch of -b*"),
    ("purple_otsu", "Otsu threshold of the purple channel: raw purple mask"),
    ("purple_opening", "purple mask: opening by a disk of purple_open_r x r"),
    ("purple_closing", "purple mask: closing by a disk of purple_close_r x r"),
    ("purple_remove_small_objects", "purple mask: remove objects < purple_min_area x pi r^2 -> PURPLE MASK"),
    (
        "cells_union_purple",
        "raw cell mask OR purple mask (a nucleus is a cell even with weak green contrast)",
    ),
    ("gap_closing", "closing by a disk of gap_close_r x r: seals open RBC rims (C shapes)"),
    ("hole_fill", "fill small, round, convex holes (RBC pale centres), not gaps between cells"),
    ("opening", "opening by a disk of open_r x r: noise, thin bridges"),
    ("closing", "closing by a disk of close_r x r: smooth contours"),
    ("remove_small_objects", "remove objects < min_area x pi r^2 -> CELL MASK"),
    ("segmentation_overlay", "final masks on the original: cell mask (green), purple mask (magenta)"),
]


@dataclass
class Segmentation:
    r: float
    scale: ScaleEstimate | None  # None when r was given
    channels: dict[str, np.ndarray]  # preprocess output (green, cells, purple + intermediates)
    cell_mask: np.ndarray  # RBC + WBC + platelets
    purple_mask: np.ndarray  # WBC nuclei + platelets
    raw_cell_mask: np.ndarray  # threshold before cleaning
    raw_purple_mask: np.ndarray
    steps: dict[str, np.ndarray]  # every intermediate image, keyed by STEPS names


def segment(img: np.ndarray, cfg: dict, r: float | None = None) -> Segmentation:
    """Full pipeline up to Phase 4: scale (if ``r`` is not given), preprocessing, both masks.

    With ``r`` given, the scale step is skipped (``scale`` is None and the
    ``scale_coarse_mask`` step is absent).

    The purple mask is added to the cell mask before cleaning: a nucleus or a
    platelet is a cell even where its green contrast is weak. The two figure
    steps (``scale_granulometry``, ``segmentation_overlay``) are drawn by the
    exporter from ``scale`` and the masks, so they are not in ``steps``.
    """
    scale = estimate_scale(img, cfg) if r is None else None
    r = scale.r if scale is not None else float(r)
    ch = preprocess(img, cfg, r)
    if not stage(cfg, "preprocessing"):
        # Ablation baseline: raw inverted green channel and raw -b*, no median, no top-hat.
        ch = {**ch, "cells": normalize(1 - ch["green_raw"]), "purple": normalize(ch["purple_raw"])}
    s = cfg["segmentation"]
    steps = {"original": img, "green_channel": ch["green_raw"], "median_filter": ch["green"]}
    if scale is not None:
        steps["scale_coarse_mask"] = scale.mask
    steps.update(
        {
            "background_closing": ch["background"],
            "relative_tophat": ch["tophat"],
            "cells_normalized": ch["cells"],
        }
    )
    raw_cells = threshold_cells(ch["cells"], s["cell_threshold"], disk_radius(s["local_block"], r))
    # Safety check (project.md 8.3): a blank or uniform image has no contrast,
    # and the percentile stretch would turn its rounding noise into "cells".
    if np.percentile(ch["tophat"], 99) < s["min_contrast"]:
        raw_cells = np.zeros_like(raw_cells)
    steps["cells_otsu"] = raw_cells
    steps["purple_lab_b"] = ch["purple_raw"]
    steps["purple_median_normalized"] = ch["purple"]
    method = s["purple_threshold"]
    raw_purple = threshold_purple(ch["purple"], "otsu" if method == "auto" else method)
    if method == "auto" and purple_coverage(raw_purple, raw_cells) > s["purple_auto_cover"]:
        # The whole cell population is stained (e.g. Raabin-WBC, bluish RBCs): two
        # classes split cells from background, not nuclei from cells. Three classes
        # (background, cells, nuclei) keep only the top one.
        raw_purple = threshold_purple(ch["purple"], "multiotsu")
    lo, hi = np.percentile(ch["purple_raw"], [1, 99])
    if hi - lo < s["purple_min_range"]:
        raw_purple = np.zeros_like(raw_purple)
    steps["purple_otsu"] = raw_purple
    purple = clean_mask(
        raw_purple,
        r,
        cfg,
        fill_holes=False,
        steps=steps,
        prefix="purple_",
        open_r=s["purple_open_r"],
        close_r=s["purple_close_r"],
        min_area=s["purple_min_area"],
    )
    steps["cells_union_purple"] = raw_cells | purple
    cells = clean_mask(raw_cells | purple, r, cfg, steps=steps)
    return Segmentation(r, scale, ch, cells, purple, raw_cells, raw_purple, steps)
