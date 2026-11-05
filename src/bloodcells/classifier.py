"""Cell-type classification: rule baseline and ML models (project.md, Phase 8 and section 9).

Morphology produces the objects; this module only decides what each object is.
The ML models use scale-free features (``features.ML_FEATURES``), never the
branch that produced the object, so they are not a copy of the rules.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import ML_FEATURES

CLASSES = ("RBC", "WBC", "Platelet", "Artefact")
TRAINABLE = ("RBC", "WBC", "Platelet", "Artefact")  # "Unlabelled" objects are never used


def rule_predict(feats: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Rule baseline on two features (project.md, Phase 8).

    Stained (``purple_fraction`` >= purple_min) and large -> WBC; stained and
    small -> Platelet; everything else -> RBC.
    """
    c = cfg["classifier"]["rules"]
    stained = feats["purple_fraction"].to_numpy() >= c["purple_min"]
    area = feats["area_rel"].to_numpy()
    out = np.full(len(feats), "RBC", dtype=object)
    out[stained & (area >= c["wbc_min_area"])] = "WBC"
    out[stained & (area < c["platelet_max_area"])] = "Platelet"
    return out


def make_models(seed: int = 42) -> dict:
    """Candidate models; all handle the class imbalance with balanced class weights."""
    return {
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def training_rows(feats: pd.DataFrame) -> pd.Series:
    """Rows usable for training: a known class and finite features."""
    finite = np.isfinite(feats[ML_FEATURES].to_numpy(float)).all(axis=1)
    return feats["gt"].isin(TRAINABLE) & finite


def cross_val_predict_grouped(
    model, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5
) -> np.ndarray:
    """Out-of-fold predictions with ``GroupKFold`` (no image / field shared by train and test folds)."""
    pred = np.empty(len(y), dtype=object)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = clone(model).fit(X[train], y[train])
        pred[test] = m.predict(X[test])
    return pred


def fit(model, feats: pd.DataFrame):
    """Fit ``model`` on the trainable rows of a feature table (column ``gt``)."""
    rows = training_rows(feats)
    return clone(model).fit(feats.loc[rows, ML_FEATURES].to_numpy(float), feats.loc[rows, "gt"].to_numpy())


def save_model(model, path: str | Path, name: str, metrics: dict | None = None) -> None:
    """Save the model with the feature list it expects (the bundle is what ``load_model`` returns)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "name": name, "features": list(ML_FEATURES), "classes": list(model.classes_)}
    bundle["metrics"] = metrics or {}
    joblib.dump(bundle, path, compress=3)


def load_model(path: str | Path) -> dict | None:
    """Model bundle, or None when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    bundle = joblib.load(path)
    # Prediction is per image (a few dozen objects): one thread, so that parallel
    # image workers do not each start a thread pool.
    model = bundle["model"]
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "n_jobs"):
        est.n_jobs = 1
    return bundle


def ml_predict(bundle: dict, feats: pd.DataFrame, min_proba: float = 0.0) -> np.ndarray:
    """Predicted class per row; rows with non-finite features keep their branch.

    With ``min_proba`` > 0 the model overrides the branch only when it gives
    the new class at least that probability. The model is trained on matched
    objects only (BCCD has no "artefact" label), so on an unmatched object such
    as a small RBC fragment it is out of distribution and its changes are mostly
    wrong (val: 7 of the 9 changes); confident changes are kept.
    """
    out = feats["branch"].to_numpy(dtype=object).copy()
    if len(feats) == 0:
        return out
    X = feats[bundle["features"]].to_numpy(float)
    ok = np.flatnonzero(np.isfinite(X).all(axis=1))
    if not ok.size:
        return out
    model = bundle["model"]
    if min_proba > 0 and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X[ok])
        pred = np.asarray(model.classes_, dtype=object)[proba.argmax(axis=1)]
        confident = (pred == out[ok]) | (proba.max(axis=1) >= min_proba)
        out[ok[confident]] = pred[confident]
    else:
        out[ok] = model.predict(X[ok])
    return out


def feature_importance(bundle: dict) -> pd.Series:
    """Impurity importance of a random forest (empty for other models)."""
    model = bundle["model"]
    est = model[-1] if hasattr(model, "steps") else model
    if not hasattr(est, "feature_importances_"):
        return pd.Series(dtype=float)
    return pd.Series(est.feature_importances_, index=bundle["features"]).sort_values(ascending=False)


def classify(feats: pd.DataFrame, cfg: dict, bundle: dict | None = None) -> np.ndarray:
    """Final class of each object according to ``classifier.method``.

    ``branch``: the pipeline branch (WBC / platelet / RBC watershed);
    ``rules``: ``rule_predict``; ``ml``: the saved model, which overrides the
    branch when its probability is >= ``override_min_proba`` (the branch is
    used when no model is available).
    """
    method = cfg["classifier"]["method"]
    if method == "branch" or (method == "ml" and bundle is None):
        return feats["branch"].to_numpy(dtype=object)
    if method == "rules":
        return rule_predict(feats, cfg)
    if method == "ml":
        return ml_predict(bundle, feats, cfg["classifier"].get("override_min_proba", 0.0))
    raise ValueError(f"unknown classifier method '{method}'")
