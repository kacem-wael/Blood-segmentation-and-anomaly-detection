"""Integration tests on a synthetic smear with a known content."""

import numpy as np
import pandas as pd
import pytest
from synthetic import smear

from bloodcells.classifier import ml_predict, rule_predict
from bloodcells.config import load_config, with_overrides
from bloodcells.evaluate import label_iou, mask_scores, per_class_report
from bloodcells.io_utils import ensure_rgb
from bloodcells.pipeline import analyze, run
from bloodcells.platelets import enclosure, find_platelets

CFG = load_config()
R = 30


@pytest.fixture(scope="module")
def result():
    return run(smear(r=R), CFG, r=R)


def test_counts_on_synthetic_smear(result):
    # 4 single RBCs + 1 touching pair, 1 WBC, 2 platelets.
    assert result.counts["WBC"] == 1
    assert result.counts["RBC"] == 6
    assert result.counts["Platelet"] == 2


def test_result_keys_and_features(result):
    f = result.features
    assert {"cls", "branch", "area_rel", "anomaly", "clump"} <= set(f.columns)
    assert len(f) == result.labels.max()
    wbc = f[f["cls"] == "WBC"].iloc[0]
    assert wbc["nc_ratio"] > 0 and np.isfinite(wbc["nc_ratio"])
    s = result.summary()
    assert s["n_RBC"] == 6 and s["r"] == R
    assert result.overlay().shape == result.image.shape


def test_blank_image_gives_zero_counts():
    blank = np.full((200, 200, 3), 230, np.uint8)
    res = run(blank, CFG, r=R)
    assert res.counts["RBC"] == res.counts["WBC"] == res.counts["Platelet"] == 0


def test_estimated_scale_on_synthetic_smear():
    res = run(smear(r=R), CFG)
    assert 0.7 * R < res.r < 1.5 * R


@pytest.mark.parametrize(
    "switch",
    [
        ("pipeline", "watershed"),
        ("pipeline", "wbc_first"),
        ("pipeline", "morphology"),
        ("pipeline", "preprocessing"),
    ],
)
def test_ablation_switches_run(switch):
    a = analyze(smear(r=R), with_overrides(CFG, {switch: False}), r=R)
    assert a.labels.max() >= 1
    if switch == ("pipeline", "watershed"):
        assert a.counts["RBC"] == 5  # the touching pair stays merged


def test_enclosed_speck_is_not_a_platelet():
    shape = (120, 120)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cell = (yy - 60) ** 2 + (xx - 60) ** 2 <= 30**2
    purple = np.zeros(shape, bool)
    purple[58:63, 58:63] = True  # speck in the middle of an RBC
    purple[5:10, 5:10] = True  # platelet on the background
    plt = find_platelets(purple, np.zeros(shape, np.int32), cell | purple, R, CFG)
    assert plt.max() == 1 and plt[7, 7] == 1 and plt[60, 60] == 0
    assert enclosure(np.zeros(shape, np.int32), cell, 1, 2).shape == (1,)


def test_ensure_rgb_conversions_and_errors():
    assert ensure_rgb(np.zeros((40, 40), np.uint8)).shape == (40, 40, 3)
    assert ensure_rgb(np.zeros((40, 40, 4), np.uint8)).shape == (40, 40, 3)
    assert ensure_rgb(np.ones((40, 40, 3))).max() == 255
    assert ensure_rgb(np.full((40, 40, 3), 65535, np.uint16)).max() == 255
    with pytest.raises(ValueError):
        ensure_rgb(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError):
        ensure_rgb(np.zeros((40, 40, 5), np.uint8))


def test_mask_scores_perfect_and_split():
    gt = np.zeros((20, 40), np.int32)
    gt[5:15, 2:18], gt[5:15, 22:38] = 1, 2
    s = mask_scores(gt, gt)
    assert s["dice"] == 1 and s["cell_f1"] == 1
    merged = (gt > 0).astype(np.int32)
    s = mask_scores(merged, gt)  # one blob covering both cells matches at most one of them
    assert s["n_pred"] == 1 and s["cell_recall"] <= 0.5 and s["dice"] == 1
    assert label_iou(gt, gt)[0, 0] == 1


def test_rules_and_reports():
    f = pd.DataFrame(
        {"purple_fraction": [0.9, 0.9, 0.0], "area_rel": [2.0, 0.05, 0.8], "branch": ["WBC"] * 3}
    )
    pred = rule_predict(f, CFG)
    assert list(pred) == ["WBC", "Platelet", "RBC"]
    rep = per_class_report(["WBC", "Platelet", "RBC"], pred, ["RBC", "WBC", "Platelet"])
    assert rep.iloc[-1]["f1"] == 1.0

    # A bundle predicting a constant class.
    class Const:
        def predict(self, X):
            return np.array(["RBC"] * len(X))

    f2 = f.assign(
        **{
            c: 0.0
            for c in [
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
                "purple_mean",
            ]
        }
    )
    from bloodcells.features import ML_FEATURES

    assert list(ml_predict({"model": Const(), "features": ML_FEATURES}, f2)) == ["RBC"] * 3


@pytest.mark.parametrize("zoom", [0.5, 2.0])
def test_counts_do_not_depend_on_the_zoom(zoom):
    def z(v):
        return int(round(v * zoom))

    img = smear(
        shape=(z(360), z(480)),
        rbcs=tuple((z(a), z(b)) for a, b in ((90, 90), (90, 300), (250, 120), (250, 380))),
        r=z(R),
        wbc=(z(180), z(240)),
        platelets=tuple((z(a), z(b)) for a, b in ((40, 430), (320, 40))),
        touching=((z(300), z(250)),),
    )
    res = run(img, CFG)
    assert res.labels.shape == img.shape[:2]
    assert res.counts["RBC"] == 6 and res.counts["WBC"] == 1 and res.counts["Platelet"] == 2
