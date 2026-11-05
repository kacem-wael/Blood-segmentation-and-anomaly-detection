"""Metrics (project.md, section 10).

BCCD has boxes, not pixel masks, so Phase 4 is scored with box *proxies* until
the manual masks exist. They compare variants; they are not absolute scores
(RBC boxes are incomplete, so nothing here measures false positives on RBCs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """2 |A & B| / (|A| + |B|); 1 when both are empty."""
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


def _centres(boxes: pd.DataFrame, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    cy = ((boxes["ymin"] + boxes["ymax"]) // 2).clip(0, shape[0] - 1).to_numpy(int)
    cx = ((boxes["xmin"] + boxes["xmax"]) // 2).clip(0, shape[1] - 1).to_numpy(int)
    return cy, cx


def box_proxies(cell_mask: np.ndarray, purple_mask: np.ndarray, boxes: pd.DataFrame, r: float) -> dict:
    """Box-based proxies of the cell and purple masks for one image.

    rbc_centre      share of complete RBC boxes whose centre is in the cell mask.
                    The centre of an RBC is its pale centre: this measures hole filling.
    rbc_fill        mean share of the RBC box inscribed ellipse covered by the cell mask.
    rbc_shared      share of RBC centres in a component that holds >= 2 RBC centres
                    (touching cells: higher = more merging; watershed splits them later).
    wbc_found       share of WBC boxes holding >= 0.3 pi r^2 of purple mask (a nucleus).
    platelet_found  share of platelet boxes overlapped by the purple mask.
    purple_outside  purple area outside every WBC / platelet box, in units of pi r^2
                    (false stain detections; some are unlabelled platelets).
    """
    out: dict[str, float] = {}
    rbc = boxes[(boxes["cls"] == "RBC") & ~boxes["truncated"]]
    if len(rbc):
        cy, cx = _centres(rbc, cell_mask.shape)
        inside = cell_mask[cy, cx]
        out["rbc_centre"] = float(inside.mean())
        fills = []
        yy, xx = np.ogrid[: cell_mask.shape[0], : cell_mask.shape[1]]
        for b in rbc.itertuples():
            ay, ax = max((b.ymax - b.ymin) / 2, 1), max((b.xmax - b.xmin) / 2, 1)
            sl = (slice(b.ymin, b.ymax + 1), slice(b.xmin, b.xmax + 1))
            ell = (
                ((yy[sl[0]] - (b.ymin + b.ymax) / 2) / ay) ** 2
                + ((xx[:, sl[1]] - (b.xmin + b.xmax) / 2) / ax) ** 2
            ) <= 1
            fills.append(cell_mask[sl][ell].mean() if ell.any() else np.nan)
        out["rbc_fill"] = float(np.nanmean(fills))
        labels, _ = ndi.label(cell_mask)
        lab = labels[cy, cx]
        counts = np.bincount(lab[lab > 0], minlength=labels.max() + 1)
        out["rbc_shared"] = float(np.mean((lab > 0) & (counts[lab] >= 2)))
    unit = np.pi * r**2
    wbc = boxes[boxes["cls"] == "WBC"]
    if len(wbc):
        found = [
            purple_mask[b.ymin : b.ymax + 1, b.xmin : b.xmax + 1].sum() >= 0.3 * unit
            for b in wbc.itertuples()
        ]
        out["wbc_found"] = float(np.mean(found))
    plt_ = boxes[boxes["cls"] == "Platelet"]
    if len(plt_):
        found = [purple_mask[b.ymin : b.ymax + 1, b.xmin : b.xmax + 1].any() for b in plt_.itertuples()]
        out["platelet_found"] = float(np.mean(found))
    covered = np.zeros_like(purple_mask, dtype=bool)
    for b in boxes[boxes["cls"].isin(["WBC", "Platelet"])].itertuples():
        covered[b.ymin : b.ymax + 1, b.xmin : b.xmax + 1] = True
    out["purple_outside"] = float((purple_mask & ~covered).sum() / unit)
    return out


# ---- Detection (project.md, section 10.1) --------------------------------


def object_boxes(
    labels: np.ndarray,
    classes: dict[int, str] | None = None,
    pad: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Bounding box of every labelled object, in the BCCD box convention.

    BCCD boxes are VOC (1-based, inclusive); pixel (row, col) of the image is
    VOC (col + 1, row + 1), so the box of pixels [r0, r1] x [c0, c1] is
    xmin = c0 + 1, xmax = c1 + 1 (same for y). ``pad`` (class -> pixels)
    enlarges the boxes of a class, clipped to the image (platelets: the
    stained core is segmented, the annotators boxed the whole platelet).
    """
    rows = []
    h, w = labels.shape
    for k, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        cls = (classes or {}).get(k, "")
        d = int(round((pad or {}).get(cls, 0)))
        rows.append(
            {
                "label": k,
                "cls": cls,
                "xmin": max(1, sl[1].start + 1 - d),
                "ymin": max(1, sl[0].start + 1 - d),
                "xmax": min(w, sl[1].stop + d),
                "ymax": min(h, sl[0].stop + d),
            }
        )
    return pd.DataFrame(rows, columns=["label", "cls", "xmin", "ymin", "xmax", "ymax"])


def box_padding(cfg: dict, r: float) -> dict[str, float]:
    """Per-class box padding in pixels used when matching predictions to BCCD boxes."""
    return {"Platelet": cfg["platelets"]["box_pad"] * r}


def box_iou(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """IoU matrix (len(a), len(b)) of two box tables (xmin, ymin, xmax, ymax, inclusive)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    A = a[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)[:, None, :]
    B = b[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)[None, :, :]
    iw = np.clip(np.minimum(A[..., 2], B[..., 2]) - np.maximum(A[..., 0], B[..., 0]) + 1, 0, None)
    ih = np.clip(np.minimum(A[..., 3], B[..., 3]) - np.maximum(A[..., 1], B[..., 1]) + 1, 0, None)
    inter = iw * ih
    area_a = (A[..., 2] - A[..., 0] + 1) * (A[..., 3] - A[..., 1] + 1)
    area_b = (B[..., 2] - B[..., 0] + 1) * (B[..., 3] - B[..., 1] + 1)
    return inter / (area_a + area_b - inter)


def match_boxes(pred: pd.DataFrame, gt: pd.DataFrame, iou_min: float = 0.5) -> list[tuple[int, int, float]]:
    """One-to-one matching (Hungarian, maximising total IoU) of pred and gt rows with IoU >= iou_min.

    Returns (pred position, gt position, IoU) for each matched pair.
    """
    from scipy.optimize import linear_sum_assignment

    iou = box_iou(pred, gt)
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(-iou)
    return [(int(i), int(j), float(iou[i, j])) for i, j in zip(rows, cols) if iou[i, j] >= iou_min]


def detection_counts(pred: pd.DataFrame, gt: pd.DataFrame, iou_min: float = 0.5) -> dict[str, int]:
    """True positives, false positives, false negatives for one class of one image."""
    tp = len(match_boxes(pred, gt, iou_min))
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp}


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision, recall, F1 (NaN when undefined)."""
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if (p + r) and np.isfinite(p) and np.isfinite(r) else np.nan
    return p, r, f


# ---- Classification (project.md 9.2) --------------------------------------


def per_class_report(y_true, y_pred, classes) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, plus a macro-average row."""
    y_true, y_pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    rows = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p, r, f = prf(tp, fp, fn)
        rows.append({"cls": c, "precision": p, "recall": r, "f1": f, "support": tp + fn})
    df = pd.DataFrame(rows)
    present = df[df["support"] > 0]
    macro = {"cls": "macro", "support": int(df["support"].sum())}
    macro.update({k: float(present[k].fillna(0).mean()) for k in ("precision", "recall", "f1")})
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def macro_f1(y_true, y_pred, classes) -> float:
    return float(per_class_report(y_true, y_pred, classes).iloc[-1]["f1"])


def bootstrap_ci(
    y_true, y_pred, groups, classes, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    """95 % CI of the macro-F1 by resampling *images* (groups) with replacement.

    Objects of one image are correlated (same stain, same field), so images,
    not objects, are the independent units.
    """
    y_true, y_pred, groups = (np.asarray(x, dtype=object) for x in (y_true, y_pred, groups))
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        rows = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        scores.append(macro_f1(y_true[rows], y_pred[rows], classes))
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def confusion(y_true, y_pred, classes) -> pd.DataFrame:
    """Confusion matrix (rows = truth, columns = prediction)."""
    y_true, y_pred = pd.Series(y_true, dtype=object), pd.Series(y_pred, dtype=object)
    cm = pd.crosstab(y_true, y_pred).reindex(index=list(classes), columns=list(classes), fill_value=0)
    cm.index.name, cm.columns.name = "truth", "predicted"
    return cm


# ---- Detection tables shared by the evaluation scripts ---------------------

DETECTION_IOUS = (0.5, 0.3)


def detection_rows(pred: pd.DataFrame, gt: pd.DataFrame, classes, skip: tuple[str, ...] = ()) -> list[dict]:
    """Per-class counts (n_pred, n_gt, tp/fp/fn at each IoU of ``DETECTION_IOUS``) for one image.

    ``pred`` and ``gt`` are box tables with a ``cls`` column; classes in
    ``skip`` are left out (RBCs of images without any RBC box).
    """
    rows = []
    for cls in classes:
        if cls in skip:
            continue
        p, g = pred[pred["cls"] == cls], gt[gt["cls"] == cls]
        row = {"cls": cls, "n_pred": len(p), "n_gt": len(g)}
        for iou in DETECTION_IOUS:
            row.update({f"{k}@{iou}": v for k, v in detection_counts(p, g, iou).items()})
        rows.append(row)
    return rows


def summarize_detection(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Micro-averaged precision / recall / F1 per group, count MAE and predicted / GT ratio."""
    out = []
    for keys, g in table.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        for iou in DETECTION_IOUS:
            p, r, f = prf(*(g[f"{k}@{iou}"].sum() for k in ("tp", "fp", "fn")))
            row.update({f"P@{iou}": p, f"R@{iou}": r, f"F1@{iou}": f})
        row["count_mae"] = float((g["n_pred"] - g["n_gt"]).abs().mean())
        row["pred/gt"] = g["n_pred"].sum() / max(g["n_gt"].sum(), 1)
        if "seconds" in g:
            row["s/img"] = float(g["seconds"].mean())
        out.append(row)
    return pd.DataFrame(out)


# ---- Pixel masks (manual annotations, project.md 5.1) ----------------------


def label_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU matrix (n_pred, n_gt) of two label maps, from their contingency table (one pass)."""
    n_p, n_g = int(pred.max()), int(gt.max())
    if n_p == 0 or n_g == 0:
        return np.zeros((n_p, n_g))
    joint = np.bincount(
        pred.ravel().astype(np.int64) * (n_g + 1) + gt.ravel(), minlength=(n_p + 1) * (n_g + 1)
    )
    joint = joint.reshape(n_p + 1, n_g + 1)
    area_p, area_g = joint.sum(axis=1)[1:], joint.sum(axis=0)[1:]
    inter = joint[1:, 1:]
    return inter / (area_p[:, None] + area_g[None, :] - inter)


def mask_scores(pred: np.ndarray, gt: np.ndarray, iou_min: float = 0.5) -> dict[str, float]:
    """Image Dice / IoU of the foregrounds, and per-cell matching of two label maps.

    Cells are matched one-to-one (Hungarian on the mask IoU, IoU >= ``iou_min``);
    ``cell_dice`` is the mean Dice of the matched pairs.
    """
    from scipy.optimize import linear_sum_assignment

    fp, fg = pred > 0, gt > 0
    inter, union = (fp & fg).sum(), (fp | fg).sum()
    out = {"dice": dice(fp, fg), "iou": float(inter / union) if union else 1.0}
    iou = label_iou(pred, gt)
    pairs = []
    if iou.size:
        rows, cols = linear_sum_assignment(-iou)
        pairs = [(i, j) for i, j in zip(rows, cols) if iou[i, j] >= iou_min]
    tp = len(pairs)
    p, r, f = prf(tp, iou.shape[0] - tp, iou.shape[1] - tp)
    ious = np.array([iou[i, j] for i, j in pairs])
    out.update(
        {
            "n_pred": iou.shape[0],
            "n_gt": iou.shape[1],
            "cell_precision": p,
            "cell_recall": r,
            "cell_f1": f,
            "cell_dice": float(np.mean(2 * ious / (1 + ious))) if tp else np.nan,
        }
    )
    return out
