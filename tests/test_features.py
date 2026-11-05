import numpy as np
import pandas as pd
from skimage.draw import disk

from bloodcells.anomaly import flag_anomalies, rbc_statistics, robust_z
from bloodcells.config import load_config
from bloodcells.features import ML_FEATURES, assign_labels, object_features

CFG = load_config()


def one_disc(radius=30, shape=(100, 100)):
    labels = np.zeros(shape, np.int32)
    labels[disk((50, 50), radius, shape=shape)] = 1
    img = np.full(shape + (3,), 200, np.uint8)
    return img, labels


def test_disc_features():
    img, labels = one_disc()
    f = object_features(img, labels, {1: "RBC"}, r=30, purple_mask=labels > 0, purple=np.zeros(labels.shape))
    row = f.iloc[0]
    assert row["area"] == (labels == 1).sum()
    assert 0.95 <= row["circularity"] <= 1.0
    assert abs(row["area_rel"] - 1) < 0.05
    assert abs(row["equivalent_diameter_rel"] - 1) < 0.03
    assert row["purple_fraction"] == 1.0
    assert not row["touches_border"]
    assert np.isnan(row["nc_ratio"])  # not a WBC
    assert np.isfinite(f[ML_FEATURES].to_numpy(float)).all()


def test_wbc_nc_ratio():
    img, labels = one_disc()
    nuclei = np.zeros_like(labels)
    nuclei[disk((50, 50), 15, shape=labels.shape)] = 1
    f = object_features(img, labels, {1: "WBC"}, 30, labels > 0, np.zeros(labels.shape), nuclei=nuclei)
    row = f.iloc[0]
    assert row["nucleus_area"] == nuclei.sum()
    assert abs(row["nc_ratio"] - nuclei.sum() / (row["area"] - nuclei.sum())) < 1e-9
    assert not row["cytoplasm_uncertain"]


def test_empty_label_map():
    img, labels = one_disc()
    f = object_features(img, np.zeros_like(labels), {}, 30, labels > 0, np.zeros(labels.shape))
    assert len(f) == 0


def box(cls, x0, y0, x1, y1):
    return {"cls": cls, "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1}


def test_assign_labels_matching_and_fallback():
    feats = pd.DataFrame(
        [
            {
                "branch": "RBC",
                "xmin": 10,
                "ymin": 10,
                "xmax": 50,
                "ymax": 50,
                "centroid_x": 29,
                "centroid_y": 29,
            },
            {
                "branch": "Platelet",
                "xmin": 70,
                "ymin": 70,
                "xmax": 74,
                "ymax": 74,
                "centroid_x": 71,
                "centroid_y": 71,
            },
            {
                "branch": "RBC",
                "xmin": 200,
                "ymin": 200,
                "xmax": 240,
                "ymax": 240,
                "centroid_x": 219,
                "centroid_y": 219,
            },
            # Small speck inside an unmatched RBC box: IoU too low for the fallback.
            {
                "branch": "Platelet",
                "xmin": 120,
                "ymin": 120,
                "xmax": 124,
                "ymax": 124,
                "centroid_x": 121,
                "centroid_y": 121,
            },
        ]
    )
    gt = pd.DataFrame(
        [box("RBC", 11, 11, 51, 51), box("Platelet", 66, 66, 80, 80), box("RBC", 100, 100, 150, 150)]
    )
    labels = assign_labels(feats, gt, pad={"Platelet": 5})
    assert list(labels) == ["RBC", "Platelet", "Unlabelled", "Unlabelled"]
    assert list(assign_labels(feats, gt, pad={"Platelet": 5}, exhaustive=True))[2] == "Artefact"


def test_robust_z_ignores_outliers():
    x = np.r_[np.random.default_rng(0).normal(1, 0.05, 50), 3.0]
    z = robust_z(x)
    assert z[-1] > 10 and np.abs(z[:-1]).max() < 4


def test_anomaly_flags_large_rbc_and_clump():
    rng = np.random.default_rng(0)
    n = 30
    f = pd.DataFrame(
        {
            "cls": ["RBC"] * n,
            "area_rel": rng.normal(0.8, 0.03, n),
            "area": 1000,
            "circularity": rng.normal(0.85, 0.02, n),
            "solidity": 0.97,
            "touches_border": False,
        }
    )
    f.loc[0, "area_rel"] = 1.6  # large, still solid: anomaly
    f.loc[1, ["area_rel", "solidity", "area"]] = (2.6, 0.75, 3000)  # clump of ~3 cells
    out = flag_anomalies(f, CFG)
    assert out.loc[0, "anomaly"] == "large"
    assert out.loc[1, "clump"] and out.loc[1, "clump_cells"] == 3 and out.loc[1, "anomaly"] == ""
    stats = rbc_statistics(out)
    assert stats["rbc_in_clumps"] == 3 and stats["rbc_flagged"] >= 1
