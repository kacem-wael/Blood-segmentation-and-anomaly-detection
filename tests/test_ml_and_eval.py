"""Classifier training / selection / saving, metrics and annotation loading."""

import json

import numpy as np
import pandas as pd
import pytest
from synthetic import smear

from bloodcells.classifier import (
    classify,
    cross_val_predict_grouped,
    fit,
    load_model,
    make_models,
    ml_predict,
    save_model,
)
from bloodcells.config import load_config, with_overrides
from bloodcells.evaluate import (
    bootstrap_ci,
    box_iou,
    confusion,
    detection_counts,
    match_boxes,
    object_boxes,
)
from bloodcells.features import ML_FEATURES
from bloodcells.io_utils import load_labelme
from bloodcells.pipeline import PIPELINE_STEPS, pipeline_step_images, run

CFG = load_config()


def toy_features(n_per_class=40, seed=0) -> pd.DataFrame:
    """Separable toy objects: WBC large and stained, platelet tiny and stained, RBC unstained."""
    rng = np.random.default_rng(seed)
    rows = []
    for cls, area, purple in (("RBC", 0.8, 0.02), ("WBC", 2.5, 0.7), ("Platelet", 0.03, 0.9)):
        for _ in range(n_per_class):
            row = {f: rng.normal(0.5, 0.05) for f in ML_FEATURES}
            row.update(area_rel=rng.normal(area, 0.05 * area), purple_fraction=rng.normal(purple, 0.03))
            row.update(gt=cls, branch=cls, group=f"img{rng.integers(0, 10)}")
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name", ["logreg", "svm", "random_forest"])
def test_models_learn_a_separable_problem(name):
    f = toy_features()
    model = make_models(CFG["seed"])[name]
    X, y = f[ML_FEATURES].to_numpy(float), f["gt"].to_numpy()
    pred = cross_val_predict_grouped(model, X, y, f["group"].to_numpy(), n_splits=5)
    assert (pred == y).mean() > 0.95


def test_unlabelled_rows_are_not_used_for_training():
    f = toy_features()
    f.loc[:5, "gt"] = "Unlabelled"
    model = fit(make_models()["random_forest"], f)
    assert set(model.classes_) == {"RBC", "WBC", "Platelet"}


def test_save_load_and_confidence_gate(tmp_path):
    f = toy_features()
    model = fit(make_models()["random_forest"], f)
    path = tmp_path / "m.joblib"
    save_model(model, path, "random_forest")
    bundle = load_model(path)
    assert bundle["features"] == ML_FEATURES and load_model(tmp_path / "missing.joblib") is None
    # An ambiguous object labelled RBC by the branch: the gate keeps the branch.
    amb = f.iloc[[0]].copy()
    amb["area_rel"], amb["purple_fraction"], amb["branch"] = 0.4, 0.45, "RBC"
    assert ml_predict(bundle, amb, min_proba=1.01)[0] == "RBC"
    cfg = with_overrides(CFG, {("classifier", "method"): "rules"})
    assert list(classify(f.head(3), cfg)) == ["RBC"] * 3


def test_box_metrics():
    a = pd.DataFrame([{"xmin": 1, "ymin": 1, "xmax": 10, "ymax": 10}])
    b = pd.DataFrame(
        [{"xmin": 1, "ymin": 1, "xmax": 10, "ymax": 10}, {"xmin": 6, "ymin": 1, "xmax": 15, "ymax": 10}]
    )
    iou = box_iou(a, b)
    assert iou[0, 0] == 1 and abs(iou[0, 1] - 50 / 150) < 1e-9
    assert match_boxes(a, b) == [(0, 0, 1.0)]
    assert detection_counts(a, b) == {"tp": 1, "fp": 0, "fn": 1}
    labels = np.zeros((20, 20), np.int32)
    labels[2:5, 3:8] = 1
    boxes = object_boxes(labels, {1: "Platelet"}, pad={"Platelet": 2})
    assert boxes.iloc[0][["xmin", "ymin", "xmax", "ymax"]].tolist() == [2, 1, 10, 7]
    cm = confusion(["RBC", "WBC"], ["RBC", "RBC"], ["RBC", "WBC"])
    assert cm.loc["WBC", "RBC"] == 1
    lo, hi = bootstrap_ci(["RBC"] * 10, ["RBC"] * 10, list(range(10)), ["RBC"], n_boot=50)
    assert lo == hi == 1.0


def test_load_labelme(tmp_path):
    shapes = [
        {"label": "RBC", "points": [[2, 2], [12, 2], [12, 12], [2, 12]], "shape_type": "polygon"},
        {"label": "wbc", "points": [[20, 20], [35, 20], [35, 35], [20, 35]], "shape_type": "polygon"},
        {"label": "nucleus", "points": [[24, 24], [30, 24], [30, 30], [24, 30]], "shape_type": "polygon"},
    ]
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"imageHeight": 40, "imageWidth": 40, "shapes": shapes}))
    m = load_labelme(path)
    assert m.classes == {1: "RBC", 2: "WBC"} and m.labels[5, 5] == 1 and m.nuclei[27, 27]
    shapes[0]["label"] = "rbcc"
    path.write_text(json.dumps({"imageHeight": 40, "imageWidth": 40, "shapes": shapes}))
    with pytest.raises(ValueError):
        load_labelme(path)


def test_every_pipeline_step_image_is_produced():
    res = run(smear(r=30), CFG, r=30)
    images = pipeline_step_images(res, CFG)
    assert [n for n, _ in PIPELINE_STEPS] == list(images)
    for name, img in images.items():
        assert img.ndim in (2, 3) and img.size > 0, name


def test_only_reviewed_labelme_files_are_ground_truth(tmp_path):
    from bloodcells.datasets import Bccd
    from bloodcells.io_utils import labelme_reviewed

    base = {"imageHeight": 40, "imageWidth": 40, "shapes": []}
    (tmp_path / "BloodImage_00001.json").write_text(json.dumps({**base, "flags": {"reviewed": False}}))
    (tmp_path / "BloodImage_00002.json").write_text(json.dumps({**base, "flags": {"reviewed": True}}))
    (tmp_path / "BloodImage_00003.json").write_text(json.dumps(base))  # drawn from scratch: no flag
    assert not labelme_reviewed(tmp_path / "BloodImage_00001.json")
    images = pd.DataFrame(index=pd.Index([f"BloodImage_0000{i}" for i in (1, 2, 3)], name="image_id"))
    boxes = pd.DataFrame(columns=["image_id", "cls", "xmin", "ymin", "xmax", "ymax"])
    data = Bccd(images, boxes, {}, tmp_path)
    assert data.manual_masks() == ["BloodImage_00002", "BloodImage_00003"]
