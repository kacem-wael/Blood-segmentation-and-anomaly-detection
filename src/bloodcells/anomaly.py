"""Anomaly flags and RBC statistics (project.md, section 9.3 and 8.3).

Segmentation errors must not be reported as anomalies, so only *eligible*
RBCs are tested: not touching the border, not a clump, solid, not a tiny
fragment. On them a robust z-score (median / MAD) of the size and of the
circularity flags outliers. No dataset has RBC anomaly labels: this is a
qualitative screening aid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826  # MAD of a normal distribution = 0.6745 sigma


def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 MAD); zeros when the MAD is 0."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = MAD_TO_SIGMA * np.nanmedian(np.abs(x - med))
    return np.zeros_like(x) if not np.isfinite(mad) or mad == 0 else (x - med) / mad


def flag_anomalies(feats: pd.DataFrame, cfg: dict, cls_col: str = "cls") -> pd.DataFrame:
    """Add ``clump``, ``clump_cells``, ``eligible``, ``z_area``, ``z_circularity``, ``anomaly``.

    clump        RBC object larger than ``clump_min_area`` x pi r^2 with a
                 solidity below ``clump_max_solidity`` (touching cells the
                 watershed could not split, project.md 8.3).
    clump_cells  estimated number of cells in a clump: area / median area of
                 the single, eligible RBCs (1 for other RBCs, 0 otherwise).
    anomaly      '' or one of 'large', 'small', 'irregular' (|z| > z_max), only
                 for eligible RBCs and only when at least ``min_cells`` of them exist.
    """
    a = cfg["anomaly"]
    out = feats.copy()
    is_rbc = out[cls_col].to_numpy() == "RBC"
    clump = (
        is_rbc
        & (out["area_rel"].to_numpy() > a["clump_min_area"])
        & (out["solidity"] < a["clump_max_solidity"])
    )
    eligible = (
        is_rbc
        & ~clump
        & ~out["touches_border"].to_numpy(bool)
        & (out["solidity"].to_numpy() >= a["min_solidity"])
        & (out["area_rel"].to_numpy() >= a["min_area"])
    )
    out["clump"] = clump
    out["eligible"] = eligible
    out["z_area"] = np.nan
    out["z_circularity"] = np.nan
    out["anomaly"] = ""
    single_area = out.loc[eligible, "area"]
    median_area = float(single_area.median()) if len(single_area) else np.nan
    cells = np.where(is_rbc, 1.0, 0.0)
    if np.isfinite(median_area) and median_area > 0:
        cells[clump] = np.maximum(1, np.round(out.loc[clump, "area"].to_numpy() / median_area))
    out["clump_cells"] = cells.astype(int)
    if eligible.sum() >= a["min_cells"]:
        za = robust_z(out.loc[eligible, "area_rel"].to_numpy())
        zc = robust_z(out.loc[eligible, "circularity"].to_numpy())
        out.loc[eligible, "z_area"] = za
        out.loc[eligible, "z_circularity"] = zc
        flag = np.full(len(za), "", dtype=object)
        flag[zc < -a["z_max"]] = "irregular"
        flag[za < -a["z_max"]] = "small"
        flag[za > a["z_max"]] = "large"
        out.loc[eligible, "anomaly"] = flag
    return out


def rbc_statistics(feats: pd.DataFrame) -> dict[str, float]:
    """Image-level RBC statistics on eligible RBCs (needs ``flag_anomalies`` columns).

    area_cv is the coefficient of variation of the RBC area, a simple
    anisocytosis indicator analogous to the RDW of a blood counter.
    """
    e = feats[feats["eligible"].astype(bool)]
    area = e["area"].to_numpy(float)
    return {
        "rbc_eligible": int(len(e)),
        "rbc_area_mean_rel": float(e["area_rel"].mean()) if len(e) else np.nan,
        "rbc_area_cv": float(area.std(ddof=1) / area.mean()) if len(e) >= 2 else np.nan,
        "rbc_circularity_mean": float(e["circularity"].mean()) if len(e) else np.nan,
        "rbc_flagged": int((e["anomaly"] != "").sum()),
        "rbc_in_clumps": int(feats.loc[feats["clump"].astype(bool), "clump_cells"].sum()),
    }
