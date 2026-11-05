"""End-to-end pipeline (project.md, section 6.1).

scale -> preprocessing -> cell / purple masks -> WBC (nucleus + gradient
watershed) -> platelets -> RBC mask -> distance + h-maxima + watershed ->
one label map in which every object knows the branch that produced it
(``analyze``) -> features, classes, anomaly flags, counts, overlay (``run``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, transform

from .anomaly import flag_anomalies, rbc_statistics
from .classifier import classify
from .features import object_features
from .io_utils import ensure_rgb
from .platelets import find_platelets
from .scale import estimate_scale, opening_by_disk
from .segmentation import Segmentation, disk_radius, segment, stage
from .separation import Separation, separate
from .visualization import class_overlay, label_contours
from .wbc import WbcResult, find_nuclei, wbc_cells

BRANCHES = ("RBC", "WBC", "Platelet")


@dataclass
class Analysis:
    r: float
    seg: Segmentation
    wbc: WbcResult
    platelets: np.ndarray  # label map
    rbc_mask: np.ndarray  # cell mask minus WBCs minus (dilated) platelets
    sep: Separation
    labels: np.ndarray  # all objects: WBCs, then platelets, then RBCs
    branch: dict[int, str] = field(default_factory=dict)  # label -> RBC / WBC / Platelet

    @property
    def counts(self) -> dict[str, int]:
        values = list(self.branch.values())
        return {b: values.count(b) for b in BRANCHES}


def make_rbc_mask(
    seg: Segmentation, wbc: np.ndarray, platelets: np.ndarray, r: float, cfg: dict
) -> np.ndarray:
    """Cell mask without the WBCs and the platelets, before the RBC watershed.

    Platelets are removed dilated by ``separation.platelet_dilate`` px, so a
    platelet stuck to an RBC does not leave a notch that the distance
    transform would turn into a false marker. Slivers left around a WBC are
    removed by an opening (``segmentation.open_r``) and by the minimum fragment area.
    """
    s = cfg["separation"]
    plt_mask = platelets > 0
    if s["platelet_dilate"] > 0 and plt_mask.any():
        plt_mask = ndi.distance_transform_edt(~plt_mask) <= s["platelet_dilate"]
    mask = seg.cell_mask & (wbc == 0) & ~plt_mask
    if (not wbc.any() and not plt_mask.any()) or not stage(cfg, "morphology"):
        return mask
    mask = opening_by_disk(mask, disk_radius(cfg["segmentation"]["open_r"], r))
    lab = measure.label(mask, connectivity=1)
    areas = np.bincount(lab.ravel())
    small = areas < s["min_fragment_area"] * np.pi * r**2
    small[0] = False
    return mask & ~small[lab]


def combine(wbc: np.ndarray, platelets: np.ndarray, rbc: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """One label map: WBCs 1..a, platelets a+1..b, RBCs b+1..n; precedence WBC > platelet > RBC."""
    labels = np.zeros(wbc.shape, np.int32)
    branch: dict[int, str] = {}
    offset = 0
    for name, lab in (("WBC", wbc), ("Platelet", platelets), ("RBC", rbc)):
        ids = np.unique(lab[lab > 0])
        free = (labels == 0) & (lab > 0)
        lut = np.zeros(int(lab.max()) + 1, np.int32)
        lut[ids] = np.arange(1, len(ids) + 1) + offset
        labels[free] = lut[lab[free]]
        # A label fully hidden by a higher-precedence object disappears.
        for k in np.unique(labels[free]):
            branch[int(k)] = name
        offset += len(ids)
    return relabel_with_branch(labels, branch)


def relabel_with_branch(labels: np.ndarray, branch: dict[int, str]) -> tuple[np.ndarray, dict[int, str]]:
    present = np.unique(labels[labels > 0])
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    lut[present] = np.arange(1, len(present) + 1)
    return lut[labels], {int(lut[k]): branch[int(k)] for k in present}


def wbc_from_watershed(sep_labels: np.ndarray, nuclei: np.ndarray) -> np.ndarray:
    """Ablation without "WBC first": the WBCs are the watershed objects that hold most of a nucleus."""
    out = np.zeros_like(sep_labels)
    for k, sl in enumerate(ndi.find_objects(nuclei), start=1):
        if sl is None:
            continue
        inside = sep_labels[sl][nuclei[sl] == k]
        inside = inside[inside > 0]
        if inside.size:
            target = np.bincount(inside).argmax()
            out[sep_labels == target] = k
    return out


def analyze(img: np.ndarray, cfg: dict, r: float | None = None) -> Analysis:
    """Run every morphological stage on an RGB image (``r`` overrides the granulometry)."""
    img = ensure_rgb(img)
    seg = segment(img, cfg, r=r)
    r = seg.r
    nuclei = find_nuclei(seg.purple_mask, r, cfg)
    watershed = stage(cfg, "watershed")
    if stage(cfg, "wbc_first"):
        wbc = wbc_cells(
            seg.channels["green"],
            seg.channels["purple"],
            nuclei,
            seg.cell_mask,
            r,
            cfg,
            cells=seg.channels["cells"],
        )
        platelets = find_platelets(seg.purple_mask, wbc.cells, seg.cell_mask, r, cfg)
        rbc_mask = make_rbc_mask(seg, wbc.cells, platelets, r, cfg)
        sep = separate(rbc_mask, r, cfg, watershed=watershed)
        labels, branch = combine(wbc.cells, platelets, sep.labels)
    else:
        # Ablation: one distance watershed over every cell, WBCs recognised afterwards by their nucleus.
        platelets = find_platelets(seg.purple_mask, nuclei, seg.cell_mask, r, cfg)
        rbc_mask = make_rbc_mask(seg, np.zeros_like(nuclei), platelets, r, cfg) | (nuclei > 0)
        sep = separate(rbc_mask, r, cfg, watershed=watershed)
        cells = wbc_from_watershed(sep.labels, nuclei)
        empty = np.zeros_like(nuclei)
        wbc = WbcResult(nuclei, cells, empty.astype(float), empty)
        labels, branch = combine(cells, platelets, np.where(cells > 0, 0, sep.labels))
    return Analysis(r, seg, wbc, platelets, rbc_mask, sep, labels, branch)


# ---- Full result: features, classes, anomalies (project.md, Phase 9) -------


@dataclass
class Result:
    """Everything about one image, at the resolution of the input image.

    ``analysis`` holds the morphological intermediates at the working
    resolution (``factor`` x the input size, see ``working_factor``); ``labels``
    and ``features`` are mapped back to the input image.
    """

    image: np.ndarray
    analysis: Analysis
    features: pd.DataFrame  # one row per object: features, branch, cls, anomaly flags
    labels: np.ndarray  # label map at the input resolution
    r: float  # RBC radius in input pixels
    factor: float = 1.0  # working resolution / input resolution

    @property
    def classes(self) -> dict[int, str]:
        return dict(zip(self.features["label"].astype(int), self.features["cls"]))

    @property
    def counts(self) -> dict[str, int]:
        """Objects per class (artefacts excluded) and the RBC count corrected for clumps."""
        cls = self.features["cls"]
        out = {c: int((cls == c).sum()) for c in BRANCHES}
        out["RBC_with_clumps"] = int(self.features.loc[cls == "RBC", "clump_cells"].sum())
        return out

    def summary(self) -> dict:
        """Image-level numbers for tables and the app."""
        f = self.features
        wbc = f[f["cls"] == "WBC"]
        return {
            "r": self.r,
            "r_estimated": self.analysis.seg.scale is not None and self.analysis.seg.scale.ok,
            "resampling": self.factor,
            **{f"n_{k}": v for k, v in self.counts.items()},
            **rbc_statistics(f),
            "wbc_nc_ratio_mean": float(wbc["nc_ratio"].mean()) if len(wbc) else np.nan,
        }

    def overlay(self, show_ids: bool = False, radius: int = 1) -> np.ndarray:
        return class_overlay(self.image, self.labels, self.classes, radius=radius, show_ids=show_ids)


def working_factor(r: float, cfg: dict) -> float:
    """Resampling factor that brings the RBC radius to ``scale.working_r`` (1 within the tolerance).

    Every parameter is a multiple of r, but a few operations are defined on the
    pixel grid (median 2 px, gradient 1 px, platelet dilation 2 px) and the
    distance map is quantized: at r = 25 px they are twice as coarse as at
    r = 50 px. Resampling to a fixed working radius removes this dependence.
    """
    sc = cfg["scale"]
    if not sc.get("normalize", False) or r <= 0:
        return 1.0
    f = sc["working_r"] / r
    return 1.0 if abs(1 / f - 1) <= sc["tolerance"] else float(f)


def _resize(x: np.ndarray, shape: tuple[int, int], order: int) -> np.ndarray:
    out = transform.resize(x, shape, order=order, preserve_range=True, anti_aliasing=False, mode="edge")
    return out.astype(x.dtype) if order == 0 else out


def describe(
    img: np.ndarray, a: Analysis, cfg: dict, model: dict | None = None, factor: float = 1.0
) -> Result:
    """Features, final classes and anomaly flags for an ``Analysis`` of ``img`` resampled by ``factor``."""
    img = ensure_rgb(img)
    labels, purple_mask, purple, nuclei = a.labels, a.seg.purple_mask, a.seg.channels["purple"], a.wbc.nuclei
    if factor != 1.0:
        shape = img.shape[:2]
        labels, purple_mask, nuclei = (_resize(x, shape, 0) for x in (labels, purple_mask, nuclei))
        purple = _resize(purple, shape, 1)
        present = set(np.unique(labels[labels > 0]).tolist())
        branch = {k: v for k, v in a.branch.items() if k in present}  # tiny objects may vanish
    else:
        branch = a.branch
    r = a.r / factor
    feats = object_features(
        img, labels, branch, r, purple_mask, purple, nuclei=nuclei, min_cytoplasm=cfg["wbc"]["min_cytoplasm"]
    )
    feats["cls"] = classify(feats, cfg, model) if len(feats) else pd.Series(dtype=object)
    feats = flag_anomalies(feats, cfg)
    return Result(img, a, feats, labels, r, factor)


def run(img: np.ndarray, cfg: dict, r: float | None = None, model: dict | None = None) -> Result:
    """Whole pipeline on an RGB image: scale, (resampling), morphology (``analyze``), ``describe``.

    ``r`` (input pixels) overrides the granulometry. When it is far from
    ``scale.working_r`` the morphology runs on a resampled copy of the image
    and the label map is mapped back (nearest neighbour).
    """
    img = ensure_rgb(img)
    scale = estimate_scale(img, cfg) if r is None else None
    r_in = scale.r if scale is not None else float(r)
    factor = working_factor(r_in, cfg)
    work = img
    if factor != 1.0:
        work = transform.rescale(img, factor, channel_axis=-1, anti_aliasing=factor < 1, preserve_range=True)
        work = np.clip(np.round(work), 0, 255).astype(np.uint8)
    a = analyze(work, cfg, r=r_in * factor)
    a.seg.scale = scale  # keep the granulometry of the input image (for display and summary)
    return describe(img, a, cfg, model, factor)


# ---- Step images after the segmentation (scripts/export_steps.py) ---------

# Continues ``segmentation.STEPS``: (name, description), in execution order.
PIPELINE_STEPS: list[tuple[str, str]] = [
    ("wbc_lobe_grouping", "purple mask closed by wbc.lobe_close_r x r: lobes of one nucleus grouped"),
    ("wbc_nuclei", "nuclei = groups >= wbc.nucleus_min_area x pi r^2 (one colour per WBC)"),
    ("wbc_gradient", "morphological gradient (dilation - erosion) of the green and purple channels"),
    ("wbc_markers", "watershed markers: nucleus (grown), background (dark), RBC competitors (light)"),
    ("wbc_cells", "gradient watershed: whole WBC (cyan) and nucleus (magenta) -> N/C ratio"),
    ("platelet_candidates", "purple components outside the WBCs, grouped by platelets.group_r x r"),
    ("platelet_enclosure", "candidates kept (green), enclosed in a cell (red), wrong size (grey)"),
    ("platelets", "final platelets (stained cores)"),
    ("rbc_mask", "cell mask - WBCs - dilated platelets, opened: input of the RBC watershed"),
    ("distance_transform", "Euclidean distance to the background, Gaussian-smoothed"),
    ("ultimate_erosion", "regional maxima of the distance (ultimate erosion), for comparison"),
    ("hmaxima_markers", "h-maxima of the distance (dynamic >= separation.h x r): one marker per RBC"),
    ("skiz", "geodesic SKIZ of the markers inside the RBC mask (zones of influence, bonus)"),
    ("watershed_rbc", "watershed of -distance from the markers inside the RBC mask (one colour per RBC)"),
    ("label_map", "all objects: WBCs, platelets, RBCs (precedence WBC > platelet > RBC)"),
    ("label_contours", "morphological gradient of the label map: contours incl. watershed lines"),
    ("classification", "final classes: RBC red, WBC blue, platelet green, with object ids"),
    ("anomalies", "flags: anomaly (yellow fill), clump (orange), border / not eligible (grey)"),
]


def pipeline_step_images(res: Result, cfg: dict) -> dict[str, np.ndarray]:
    """One image per ``PIPELINE_STEPS`` entry (working resolution for the morphology)."""
    from skimage import color

    from .platelets import enclosure, group_components
    from .segmentation import closing_by_disk
    from .separation import skiz
    from .visualization import mask_contours

    a, r = res.analysis, res.analysis.r
    img = a.seg.steps["original"]
    purple = a.seg.purple_mask
    out: dict[str, np.ndarray] = {}
    lobe = disk_radius(cfg["wbc"]["lobe_close_r"], r)
    out["wbc_lobe_grouping"] = closing_by_disk(purple, lobe) | purple if purple.any() else purple
    out["wbc_nuclei"] = color.label2rgb(a.wbc.nuclei, bg_label=0)
    out["wbc_gradient"] = a.wbc.gradient
    n = int(a.wbc.nuclei.max())
    m = a.wbc.markers
    mk = np.zeros(m.shape + (3,))
    mk[(m > 0) & (m <= n)] = (1.0, 0.0, 1.0)
    mk[m == n + 1] = (0.15, 0.15, 0.15)
    mk[m == n + 2] = (1.0, 0.75, 0.75)
    out["wbc_markers"] = mk
    cells = img.copy()
    cells[mask_contours(a.wbc.cells > 0, 2)] = (0, 220, 255)
    cells[mask_contours(a.wbc.nuclei > 0, 2)] = (255, 0, 255)
    out["wbc_cells"] = cells

    cand = purple & (a.wbc.cells == 0)
    groups = group_components(cand, disk_radius(cfg["platelets"]["group_r"], r))
    out["platelet_candidates"] = color.label2rgb(groups, bg_label=0)
    p = cfg["platelets"]
    enc = enclosure(groups, a.seg.cell_mask & ~purple, p["ring_in"] * r, p["ring_out"] * r)
    areas = np.bincount(groups.ravel(), minlength=len(enc))
    unit = np.pi * r**2
    size_ok = (areas >= p["area_min"] * unit) & (areas <= p["area_max"] * unit)
    status = img.copy()
    kept = np.isin(groups, np.unique(groups[a.platelets > 0]))
    status[(groups > 0) & ~size_ok[groups]] = (150, 150, 150)
    status[(groups > 0) & size_ok[groups] & (enc[groups] > p["max_enclosed"])] = (255, 0, 0)
    status[kept & (groups > 0)] = (0, 220, 0)
    out["platelet_enclosure"] = status
    out["platelets"] = a.platelets > 0

    out["rbc_mask"] = a.rbc_mask
    out["distance_transform"] = a.sep.distance
    dist = a.sep.distance / max(float(a.sep.distance.max()), 1e-9)
    ue = morphology_local_maxima(a.sep.distance, a.rbc_mask)
    out["ultimate_erosion"] = _dots(dist, ue, (1.0, 0.2, 0.2))
    out["hmaxima_markers"] = _dots(dist, a.sep.markers > 0, (1.0, 0.85, 0.0))
    zones, lines = skiz(a.sep.markers, mask=a.rbc_mask)
    sk = color.label2rgb(zones, bg_label=0) * 0.6
    sk[lines] = 1.0
    out["skiz"] = sk
    out["watershed_rbc"] = color.label2rgb(a.sep.labels, bg_label=0)
    out["label_map"] = color.label2rgb(a.labels, bg_label=0)
    out["label_contours"] = label_contours(a.labels)

    out["classification"] = res.overlay(show_ids=True)
    f = res.features
    an = res.image.copy()
    lab = res.labels
    lut = np.zeros(int(lab.max()) + 1, np.uint8)  # 0 none, 1 grey, 2 clump, 3 anomaly
    lut[f["label"].to_numpy()[~f["eligible"].to_numpy(bool) & (f["cls"] == "RBC").to_numpy()]] = 1
    lut[f["label"].to_numpy()[f["clump"].to_numpy(bool)]] = 2
    lut[f["label"].to_numpy()[(f["anomaly"] != "").to_numpy()]] = 3
    code = lut[lab]
    for k, colour in ((1, (160, 160, 160)), (2, (255, 140, 0)), (3, (255, 230, 0))):
        sel = code == k
        an[sel] = (0.5 * an[sel] + 0.5 * np.array(colour)).astype(np.uint8)
    out["anomalies"] = class_overlay(an, lab, res.classes)
    return out


def morphology_local_maxima(dist: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Regional maxima of the distance inside the mask (ultimate erosion)."""
    from skimage import morphology

    return morphology.local_maxima(dist, connectivity=2) & mask.astype(bool)


def _dots(gray: np.ndarray, points: np.ndarray, colour) -> np.ndarray:
    """Grey image with ``points`` (dilated by 2 px for visibility) painted in ``colour``."""
    rgb = np.dstack([gray] * 3)
    rgb[ndi.binary_dilation(points, iterations=2)] = colour
    return rgb
