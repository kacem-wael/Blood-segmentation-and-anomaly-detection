"""Train and compare the cell-type classifiers (Phase 8, project.md section 9.2).

    python scripts/build_features.py --splits train val   # first
    python scripts/train_classifier.py

Compared: the pipeline branch, the rule baseline, logistic regression, SVM
(RBF) and random forest, all on scale-free features (``features.ML_FEATURES``),
with balanced class weights. Only matched objects are used (``Unlabelled``
objects are excluded: BCCD misses many RBC boxes).

1. Model selection: out-of-fold predictions of a GroupKFold on train + val,
   grouped by microscope field (no image or shifted frame of one field in
   two folds). Macro-F1 with a bootstrap 95 % CI over images.
2. Hold-out report: every model fitted on train, scored on val.
3. The best model (CV macro-F1) is refitted on train + val and saved to
   models/classifier.joblib with its feature list.

Writes results/tables/classification_{cv,val}.csv, confusion_val_<model>.csv,
misclassified_val.csv, and figures in results/figures/classification/.
The test set is not read.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from bloodcells.classifier import (
    cross_val_predict_grouped,
    feature_importance,
    fit,
    make_models,
    ml_predict,
    rule_predict,
    save_model,
    training_rows,
)
from bloodcells.config import PROJECT_ROOT, load_config
from bloodcells.datasets import load_bccd
from bloodcells.evaluate import bootstrap_ci, confusion, per_class_report
from bloodcells.features import ML_FEATURES
from bloodcells.io_utils import load_image


def load(cfg: dict, split: str) -> pd.DataFrame:
    path = cfg["paths"]["data"] / "features" / f"features_{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run scripts/build_features.py first")
    f = pd.read_csv(path)
    return f[training_rows(f)].reset_index(drop=True)


def summary_row(name: str, y, pred, groups, classes, cfg) -> dict:
    rep = per_class_report(y, pred, classes).set_index("cls")
    lo, hi = bootstrap_ci(y, pred, groups, classes, n_boot=500, seed=cfg["seed"])
    row = {"model": name, "macro_f1": rep.loc["macro", "f1"], "ci_low": lo, "ci_high": hi}
    row.update({f"f1_{c}": rep.loc[c, "f1"] for c in classes})
    row.update({f"recall_{c}": rep.loc[c, "recall"] for c in classes})
    row["accuracy"] = float(np.mean(np.asarray(y) == np.asarray(pred)))
    return row


def confusion_figure(cm: pd.DataFrame, title: str) -> Figure:
    fig = Figure(figsize=(4.2, 3.6))
    ax = fig.subplots()
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = norm.iat[i, j]
            ax.text(
                j,
                i,
                f"{cm.iat[i, j]}\n{v:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.6 else "black",
            )
    ax.set_xticks(range(cm.shape[1]), cm.columns, fontsize=8)
    ax.set_yticks(range(cm.shape[0]), cm.index, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("truth (BCCD box)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def importance_figure(imp: pd.Series) -> Figure:
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    imp = imp.sort_values()
    ax.barh(imp.index, imp.to_numpy(), color="tab:blue")
    ax.set_xlabel("impurity importance")
    ax.set_title("Random forest feature importance", fontsize=9)
    fig.tight_layout()
    return fig


def misclassified_gallery(rows: pd.DataFrame, data, pred_col: str, n: int = 10) -> Figure:
    rows = rows.head(n)
    cols = 5
    fig = Figure(figsize=(2.2 * cols, 2.5 * max(1, int(np.ceil(len(rows) / cols)))))
    axes = fig.subplots(max(1, int(np.ceil(len(rows) / cols))), cols, squeeze=False).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, row in zip(axes, rows.itertuples()):
        img = load_image(data.images.loc[row.image_id, "image_path"])
        m = 15
        y0, y1 = max(0, int(row.ymin) - 1 - m), min(img.shape[0], int(row.ymax) + m)
        x0, x1 = max(0, int(row.xmin) - 1 - m), min(img.shape[1], int(row.xmax) + m)
        ax.imshow(img[y0:y1, x0:x1])
        ax.set_title(
            f"{row.image_id[-5:]} #{row.label}\ntrue {row.gt}, pred {getattr(row, pred_col)}", fontsize=7
        )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tables = cfg["paths"]["results"] / "tables"
    figures = cfg["paths"]["results"] / "figures" / "classification"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    train, val = load(cfg, "train"), load(cfg, "val")
    both = pd.concat([train, val], ignore_index=True)
    classes = [c for c in ("RBC", "WBC", "Platelet", "Artefact") if (both["gt"] == c).any()]
    print(
        "Training objects:", both.groupby(["split", "gt"]).size().unstack(fill_value=0).to_string(), sep="\n"
    )
    models = make_models(cfg["seed"])

    # 1. Grouped cross-validation on train + val (model selection).
    X, y, groups = both[ML_FEATURES].to_numpy(float), both["gt"].to_numpy(), both["group"].to_numpy()
    cv_rows = [
        summary_row("branch", y, both["branch"].to_numpy(), groups, classes, cfg),
        summary_row("rules", y, rule_predict(both, cfg), groups, classes, cfg),
    ]
    for name, model in models.items():
        pred = cross_val_predict_grouped(model, X, y, groups, cfg["classifier"]["cv_folds"])
        cv_rows.append(summary_row(name, y, pred, groups, classes, cfg))
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(tables / "classification_cv.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nGroupKFold on train + val (out-of-fold):")
    print(cv.round(3).to_string(index=False))

    # 2. Hold-out: fit on train, score on val.
    val_rows = [
        summary_row("branch", val["gt"], val["branch"], val["group"], classes, cfg),
        summary_row("rules", val["gt"], rule_predict(val, cfg), val["group"], classes, cfg),
    ]
    preds = {"branch": val["branch"].to_numpy(), "rules": rule_predict(val, cfg)}
    for name, model in models.items():
        bundle = {"model": fit(model, train), "features": ML_FEATURES}
        preds[name] = ml_predict(bundle, val)
        val_rows.append(summary_row(name, val["gt"], preds[name], val["group"], classes, cfg))
    for name, pred in preds.items():
        cm = confusion(val["gt"], pred, classes)
        cm.to_csv(tables / f"confusion_val_{name}.csv")
        confusion_figure(cm, f"{name} (val)").savefig(figures / f"confusion_val_{name}.png", dpi=130)
    holdout = pd.DataFrame(val_rows)
    holdout.to_csv(tables / "classification_val.csv", index=False)
    print("\nFit on train, scored on val:")
    print(holdout.round(3).to_string(index=False))

    # 3. Best ML model by CV macro-F1, refitted on train + val.
    ml = cv[cv["model"].isin(list(models))]
    best = ml.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    final = fit(models[best], both)
    metrics = {
        "cv": cv[cv["model"] == best].iloc[0].to_dict(),
        "val": holdout[holdout["model"] == best].iloc[0].to_dict(),
    }
    model_path = PROJECT_ROOT / cfg["classifier"]["model_file"]
    save_model(final, model_path, best, metrics)
    print(f"\nBest model: {best}, saved to {model_path}")

    rf = {"model": fit(models["random_forest"], both), "features": ML_FEATURES}
    imp = feature_importance(rf)
    imp.rename("importance").to_csv(tables / "feature_importance.csv")
    importance_figure(imp).savefig(figures / "feature_importance.png", dpi=130)

    data = load_bccd(cfg)
    wrong = val.assign(pred=preds[best])
    wrong = wrong[wrong["pred"] != wrong["gt"]]
    # The rare classes are the interesting errors: show them first.
    wrong = (
        wrong.assign(_rare=wrong["gt"] != "RBC").sort_values("_rare", ascending=False).drop(columns="_rare")
    )
    cols = [
        "image_id",
        "label",
        "branch",
        "gt",
        "pred",
        "area_rel",
        "purple_fraction",
        "circularity",
        "solidity",
    ]
    wrong[cols + ["xmin", "ymin", "xmax", "ymax"]].to_csv(tables / "misclassified_val.csv", index=False)
    if len(wrong):
        misclassified_gallery(wrong, data, "pred").savefig(figures / "misclassified_val.png", dpi=110)
    print(f"{len(wrong)} misclassified val objects; figures in {figures}")


if __name__ == "__main__":
    main()
