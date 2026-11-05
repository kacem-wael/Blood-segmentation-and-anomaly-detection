"""Per-object features and training labels (project.md, Phase 7).

Every object of the label map gets shape, colour and stain features. Sizes are
divided by the RBC radius ``r`` (areas by pi r^2), so the features are
scale-free and the classifier transfers across magnifications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, measure

from .evaluate import box_iou, match_boxes
from .preprocessing import to_float
from .wbc import nc_ratio

# Inputs of the classifier (project.md 9.2): scale-free, no branch-specific output.
ML_FEATURES = [
    "area_rel",
    "circularity",
    "eccentricity",
    "solidity",
    "extent",
    "equivalent_diameter_rel",
    "L_mean",
    "a_mean",
    "b_mean",
    "L_std",
    "a_std",
    "b_std",
    "purple_fraction",
    "purple_mean",
]

COLUMNS = [
    "label",
    "branch",
    "centroid_x",
    "centroid_y",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "area",
    "perimeter",
    *ML_FEATURES,
    "nucleus_area",
    "cytoplasm_area",
    "nc_ratio",
    "cytoplasm_uncertain",
    "touches_border",
]


def border_labels(labels: np.ndarray) -> np.ndarray:
    """Boolean array indexed by label: True when the object touches the image border."""
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    out = np.zeros(int(labels.max()) + 1, bool)
    out[np.unique(edge)] = True
    out[0] = False
    return out


def object_features(
    img: np.ndarray,
    labels: np.ndarray,
    branch: dict[int, str],
    r: float,
    purple_mask: np.ndarray,
    purple: np.ndarray,
    nuclei: np.ndarray | None = None,
    min_cytoplasm: float = 0.1,
) -> pd.DataFrame:
    """One row per object (columns ``COLUMNS``).

    ``purple_mask`` / ``purple`` are the stain mask and the normalized stain
    channel; ``nuclei`` is the WBC nucleus label map, with the same labels as
    the WBC objects of ``labels`` *before* ``pipeline.combine``: here the
    nucleus of an object is simply ``nuclei`` pixels inside it.
    Boxes use the BCCD convention (VOC, 1-based inclusive).
    """
    n = int(labels.max())
    if n == 0:
        return pd.DataFrame(columns=COLUMNS)
    lab = color.rgb2lab(to_float(img[..., :3]))
    props = measure.regionprops_table(
        labels,
        intensity_image=lab,
        properties=(
            "label",
            "area",
            "centroid",
            "bbox",
            "perimeter_crofton",
            "eccentricity",
            "solidity",
            "extent",
            "intensity_mean",
            "intensity_std",
        ),
    )
    df = pd.DataFrame(props)
    ids = df["label"].to_numpy()
    unit = np.pi * r**2
    area = df["area"].to_numpy(float)
    perimeter = np.maximum(df["perimeter_crofton"].to_numpy(float), 1.0)
    out = pd.DataFrame(
        {
            "label": ids,
            "branch": [branch.get(int(k), "") for k in ids],
            "centroid_x": df["centroid-1"],
            "centroid_y": df["centroid-0"],
            "xmin": df["bbox-1"] + 1,
            "ymin": df["bbox-0"] + 1,
            "xmax": df["bbox-3"],
            "ymax": df["bbox-2"],
            "area": area.astype(int),
            "perimeter": perimeter,
            "area_rel": area / unit,
            "circularity": np.clip(4 * np.pi * area / perimeter**2, 0, 1),
            "eccentricity": df["eccentricity"],
            "solidity": df["solidity"],
            "extent": df["extent"],
            "equivalent_diameter_rel": np.sqrt(area / np.pi) / r,
            "L_mean": df["intensity_mean-0"],
            "a_mean": df["intensity_mean-1"],
            "b_mean": df["intensity_mean-2"],
            "L_std": df["intensity_std-0"],
            "a_std": df["intensity_std-1"],
            "b_std": df["intensity_std-2"],
        }
    )
    out["purple_fraction"] = ndi.mean(purple_mask.astype(float), labels, ids)
    out["purple_mean"] = ndi.mean(purple, labels, ids)

    nucleus = np.full(len(ids), np.nan)
    if nuclei is not None and nuclei.any():
        counts = np.bincount(labels[nuclei > 0], minlength=n + 1)
        is_wbc = out["branch"].to_numpy() == "WBC"
        nucleus[is_wbc] = counts[ids[is_wbc]]
    out["nucleus_area"] = nucleus
    out["cytoplasm_area"] = area - nucleus
    ratios = [
        nc_ratio(nu, a, min_cytoplasm) if np.isfinite(nu) else (np.nan, False) for nu, a in zip(nucleus, area)
    ]
    out["nc_ratio"] = [x[0] for x in ratios]
    out["cytoplasm_uncertain"] = [x[1] for x in ratios]
    out["touches_border"] = border_labels(labels)[ids]
    return out[COLUMNS].reset_index(drop=True)


# ---- Training labels (project.md, Phase 7) --------------------------------


def assign_labels(
    feats: pd.DataFrame,
    gt: pd.DataFrame,
    pad: dict[str, float] | None = None,
    iou_min: float = 0.5,
    exhaustive: bool = False,
    iou_fallback: float = 0.2,
) -> pd.Series:
    """Ground-truth class of every detected object.

    1. One-to-one Hungarian matching of object boxes and GT boxes (all classes
       together, IoU >= ``iou_min``): the object takes the class of its box.
       Object boxes are padded per *branch* (``pad``, as in evaluation).
    2. Fallback for the remaining objects: centroid inside a still unmatched
       GT box whose IoU with the object box is at least ``iou_fallback`` (best
       IoU first). The IoU condition keeps a platelet lying on an unlabelled
       part of an RBC box from becoming an "RBC".
    3. The rest is ``Artefact`` when the annotation is exhaustive (manual
       masks: every cell is annotated) and ``Unlabelled`` otherwise (BCCD
       misses ~30 % of the RBCs, so an unmatched object is often a real RBC);
       ``Unlabelled`` objects are excluded from training.
    """
    out = pd.Series("Artefact" if exhaustive else "Unlabelled", index=feats.index, dtype=object)
    if len(feats) == 0 or len(gt) == 0:
        return out
    boxes = feats[["xmin", "ymin", "xmax", "ymax"]].astype(float).reset_index(drop=True)
    for cls, d in (pad or {}).items():
        sel = (feats["branch"] == cls).to_numpy()
        boxes.loc[sel, ["xmin", "ymin"]] -= d
        boxes.loc[sel, ["xmax", "ymax"]] += d
    gt = gt.reset_index(drop=True)
    cls = gt["cls"].to_numpy()
    matched = np.zeros(len(feats), bool)
    free = np.ones(len(gt), bool)
    for i, j, _ in match_boxes(boxes, gt, iou_min):
        out.iloc[i] = cls[j]
        matched[i], free[j] = True, False
    iou = box_iou(boxes, gt)
    cx = feats["centroid_x"].to_numpy() + 1  # VOC coordinates
    cy = feats["centroid_y"].to_numpy() + 1
    g = gt[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)
    inside = (g[None, :, 0] <= cx[:, None]) & (cx[:, None] <= g[None, :, 2])
    inside &= (g[None, :, 1] <= cy[:, None]) & (cy[:, None] <= g[None, :, 3])
    cand = np.where(inside & (iou >= iou_fallback) & ~matched[:, None] & free[None, :], iou, -1.0)
    # Greedy, best IoU first: each GT box labels at most one more object.
    for flat in np.argsort(-cand, axis=None):
        i, j = np.unravel_index(flat, cand.shape)
        if cand[i, j] < 0:
            break
        if matched[i] or not free[j]:
            continue
        out.iloc[i] = cls[j]
        matched[i], free[j] = True, False
    return out
