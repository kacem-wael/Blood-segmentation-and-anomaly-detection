"""Reproduce every reported metric (Phase 9, project.md section 10).

    python scripts/run_evaluation.py --split val                # tuning-time report
    python scripts/run_evaluation.py --split val --ablation     # + ablation, sweeps, scale robustness
    python scripts/run_evaluation.py --split test --ablation    # final report: run ONCE, never tune after

Main report (full pipeline, saved classifier):
  * detection P / R / F1 per class, one-to-one box matching at IoU 0.5 and 0.3
    (platelet boxes padded by ``platelets.box_pad`` x r: the core is segmented,
    BCCD boxes the whole platelet). RBC precision is a lower bound (~30 % of
    RBCs have no box); images without any RBC box are left out of RBC metrics;
  * counting error per class and image (RBC: against an incomplete reference);
  * classification of the matched objects (macro-F1, bootstrap 95 % CI over images);
  * Dice / IoU and per-cell F1 on the manual masks of the split, when they exist;
  * seconds per image (one process; the parallel run is slower per image).

``--ablation`` adds the ablation table A-F (project.md 10.2; classes from the
pipeline branch so that only morphology changes), plain vs shape-filtered hole
fill, parameter sweeps (opening radius, h) and the scale-robustness test
(images rescaled x0.5 / x2, r estimated vs fixed-pixel parameters).

Tables go to results/tables/eval_<split>_*.csv, figures to results/figures/evaluation/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from skimage import transform

from bloodcells.classifier import load_model
from bloodcells.config import PROJECT_ROOT, load_config, with_overrides
from bloodcells.datasets import load_bccd
from bloodcells.evaluate import (
    bootstrap_ci,
    box_padding,
    confusion,
    detection_rows,
    mask_scores,
    object_boxes,
    per_class_report,
    summarize_detection,
)
from bloodcells.features import assign_labels
from bloodcells.io_utils import CLASSES, load_image, load_labelme
from bloodcells.pipeline import run
from bloodcells.scale import estimate_scale

BRANCH = {("classifier", "method"): "branch"}
OFF = {
    ("pipeline", "preprocessing"): False,
    ("segmentation", "hole_fill"): "none",
    ("pipeline", "morphology"): False,
    ("pipeline", "wbc_first"): False,
    ("pipeline", "watershed"): False,
}
# Ablation rows (project.md 10.2): each adds one stage to the previous one.
ABLATION = {
    "A Otsu + components": {**BRANCH, **OFF},
    "B + preprocessing": {**BRANCH, **OFF, ("pipeline", "preprocessing"): True},
    "C + shape hole fill": {
        **BRANCH,
        **OFF,
        ("pipeline", "preprocessing"): True,
        ("segmentation", "hole_fill"): "shape",
    },
    "D + opening/closing": {
        **BRANCH,
        **OFF,
        ("pipeline", "preprocessing"): True,
        ("segmentation", "hole_fill"): "shape",
        ("pipeline", "morphology"): True,
    },
    "E + watershed": {**BRANCH, ("pipeline", "wbc_first"): False},
    "F full (+ WBC first)": {**BRANCH},
    "F with plain fill_holes": {**BRANCH, ("segmentation", "hole_fill"): "plain"},
}
FLAGS = ["preprocessing", "hole_fill", "morphology", "wbc_first", "watershed"]
# End-to-end detection with each classifier (the matched-object scores ignore false detections).
CLASSIFIERS = {
    "cls branch": {("classifier", "method"): "branch"},
    "cls rules": {("classifier", "method"): "rules"},
    "cls ml (no gate)": {("classifier", "override_min_proba"): 0.0},
    "full": {},
}
SWEEPS = {
    ("segmentation", "open_r"): [0.05, 0.1, 0.15, 0.2, 0.3],
    ("separation", "h"): [0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
}


def predicted_boxes(res, cfg) -> pd.DataFrame:
    """Boxes of the final objects (artefacts dropped), platelets padded as in the BCCD convention."""
    boxes = object_boxes(res.labels, res.classes, box_padding(cfg, res.r))
    return boxes[boxes["cls"].isin(CLASSES)]


def evaluate_image(image_id, path, gt, cfg, variants, model, no_rbc, mask_path, keep_objects) -> dict:
    """All variants on one image; r is estimated once (it does not depend on the variant)."""
    img = load_image(path)
    r = estimate_scale(img, cfg).r
    manual = load_labelme(mask_path, img.shape[:2]) if mask_path else None
    det, masks, objects = [], [], []
    for name, overrides in variants.items():
        vcfg = with_overrides(cfg, overrides)
        t0 = time.perf_counter()
        res = run(img, vcfg, r=r, model=model if vcfg["classifier"]["method"] == "ml" else None)
        seconds = time.perf_counter() - t0
        for row in detection_rows(predicted_boxes(res, vcfg), gt, CLASSES, skip=("RBC",) if no_rbc else ()):
            det.append({"image_id": image_id, "variant": name, "seconds": seconds, "r": r, **row})
        if manual is not None:
            keep = np.isin(res.labels, [k for k, c in res.classes.items() if c in CLASSES])
            scores = mask_scores(np.where(keep, res.labels, 0), manual.labels)
            masks.append({"image_id": image_id, "variant": name, **scores})
        if name in keep_objects:
            f = res.features.copy()
            f.insert(0, "image_id", image_id)
            # Manual masks annotate every cell: their boxes replace the incomplete BCCD ones.
            label_gt = object_boxes(manual.labels, manual.classes) if manual is not None else gt
            f["gt"] = assign_labels(
                f,
                label_gt,
                box_padding(vcfg, r),
                cfg["features"]["iou_match"],
                manual is not None,
                cfg["features"]["iou_fallback"],
            )
            objects.append(f)
    return {"det": det, "masks": masks, "objects": objects}


def run_variants(data, ids, cfg, variants, model, jobs, keep_objects=()) -> dict[str, pd.DataFrame]:
    mask_dir = cfg["paths"]["masks_manual"]
    manual = set(data.manual_masks())
    results = Parallel(n_jobs=jobs)(
        delayed(evaluate_image)(
            i,
            data.images.loc[i, "image_path"],
            data.boxes_of(i),
            cfg,
            variants,
            model,
            bool(data.images.loc[i, "no_rbc_labels"]),
            mask_dir / f"{i}.json" if i in manual and (mask_dir / f"{i}.json").exists() else None,
            keep_objects,
        )
        for i in ids
    )
    out = {}
    for key in ("det", "masks", "objects"):
        parts = [x for res in results for x in res[key]]
        if key == "objects":
            out[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        else:
            out[key] = pd.DataFrame(parts)
    return out


def classification_report(objects: pd.DataFrame, data, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classification of the detected objects that have a ground-truth class."""
    known = objects[objects["gt"].isin(["RBC", "WBC", "Platelet", "Artefact"])]
    classes = [c for c in ("RBC", "WBC", "Platelet", "Artefact") if (known["gt"] == c).any()]
    groups = known["image_id"].map(data.group_of)
    rows = []
    for name, pred in (("branch", known["branch"]), ("final", known["cls"])):
        rep = per_class_report(known["gt"], pred, classes)
        lo, hi = bootstrap_ci(known["gt"], pred, groups, classes, n_boot=1000, seed=cfg["seed"])
        rep["classifier"] = name
        rep["macro_ci"] = ""
        rep.loc[rep["cls"] == "macro", "macro_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(rep)
    return pd.concat(rows, ignore_index=True), confusion(known["gt"], known["cls"], classes)


def ablation_table(det: pd.DataFrame, masks: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for name, overrides in ABLATION.items():
        vcfg = with_overrides(cfg, overrides)
        flags = {
            "preprocessing": vcfg["pipeline"]["preprocessing"],
            "hole_fill": vcfg["segmentation"]["hole_fill"],
            "morphology": vcfg["pipeline"]["morphology"],
            "wbc_first": vcfg["pipeline"]["wbc_first"],
            "watershed": vcfg["pipeline"]["watershed"],
        }
        d = det[det["variant"] == name]
        s = summarize_detection(d, ["cls"]).set_index("cls")
        row = {"config": name, **flags}
        row["RBC count MAE"] = s.loc["RBC", "count_mae"]
        row["RBC pred/gt"] = s.loc["RBC", "pred/gt"]
        row["RBC R@0.5"] = s.loc["RBC", "R@0.5"]
        row["RBC F1@0.5"] = s.loc["RBC", "F1@0.5"]
        row["WBC F1@0.5"] = s.loc["WBC", "F1@0.5"]
        row["Platelet F1@0.5"] = s.loc["Platelet", "F1@0.5"]
        row["mean F1@0.5"] = float(s["F1@0.5"].fillna(0).mean())
        if len(masks):
            m = masks[masks["variant"] == name]
            row["Dice"] = m["dice"].mean()
            row["cell F1"] = m["cell_f1"].mean()
        row["s/img"] = float(d.groupby("image_id")["seconds"].first().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_figure(sweep: pd.DataFrame) -> Figure:
    params = list(sweep["param"].unique())
    fig = Figure(figsize=(5 * len(params), 3.6))
    axes = fig.subplots(1, len(params), squeeze=False)[0]
    for ax, param in zip(axes, params):
        s = sweep[sweep["param"] == param].sort_values("value")
        for col, style in (("RBC R@0.5", "o-"), ("RBC F1@0.5", "s-"), ("Platelet F1@0.5", "^-")):
            ax.plot(s["value"], s[col], style, label=col)
        ax2 = ax.twinx()
        ax2.plot(s["value"], s["RBC pred/gt"], "k--", alpha=0.5, label="RBC pred/gt")
        ax2.set_ylabel("RBC pred / gt (dashed)")
        ax.set_xlabel(f"{param} (x r)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def scale_robustness(data, ids, cfg, model, jobs) -> pd.DataFrame:
    """Rescale images and boxes by 0.5 / 1 / 2; r estimated vs fixed at ``scale.r_default`` px."""

    def one(i, factor):
        img = load_image(data.images.loc[i, "image_path"])
        if factor != 1:
            img = (transform.rescale(img, factor, channel_axis=-1, anti_aliasing=factor < 1) * 255).astype(
                np.uint8
            )
        gt = data.boxes_of(i).copy()
        for c in ("xmin", "ymin", "xmax", "ymax"):
            gt[c] = np.round((gt[c] - 1) * factor + 1).astype(int)
        no_rbc = ("RBC",) if data.images.loc[i, "no_rbc_labels"] else ()
        rows = []
        for mode, r in (("estimated r", None), ("fixed pixels", float(cfg["scale"]["r_default"]))):
            res = run(img, cfg, r=r, model=model)
            for row in detection_rows(predicted_boxes(res, cfg), gt, CLASSES, skip=no_rbc):
                rows.append({"image_id": i, "factor": factor, "mode": mode, "r": res.r, **row})
        return rows

    rows = Parallel(n_jobs=jobs)(delayed(one)(i, f) for i in ids for f in (0.5, 1.0, 2.0))
    table = pd.DataFrame([x for r in rows for x in r])
    summary = summarize_detection(table, ["mode", "factor", "cls"])
    r_ratio = table.groupby(["mode", "factor"])["r"].median().rename("median r").reset_index()
    return summary.merge(r_ratio, on=["mode", "factor"])


def failure_gallery(data, det: pd.DataFrame, cfg, model, n: int = 10) -> Figure:
    """The n images with the most RBC + WBC + platelet errors at IoU 0.5, overlay + GT boxes."""
    err = det.assign(err=det["fp@0.5"] + det["fn@0.5"]).groupby("image_id")["err"].sum()
    worst = err.sort_values(ascending=False).head(n).index
    cols = 5
    fig = Figure(figsize=(4 * cols, 3.2 * int(np.ceil(len(worst) / cols))))
    axes = fig.subplots(int(np.ceil(len(worst) / cols)), cols, squeeze=False).ravel()
    colors = {"RBC": "red", "WBC": "blue", "Platelet": "lime"}
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, worst):
        img = load_image(data.images.loc[i, "image_path"])
        res = run(img, cfg, model=model)
        ax.imshow(res.overlay())
        for b in data.boxes_of(i).itertuples():
            ax.add_patch(
                Rectangle(
                    (b.xmin - 1, b.ymin - 1),
                    b.xmax - b.xmin,
                    b.ymax - b.ymin,
                    fill=False,
                    ec=colors.get(b.cls, "w"),
                    lw=1,
                    ls="--",
                )
            )
        c = res.counts
        ax.set_title(
            f"{i}: {int(err[i])} errors | RBC {c['RBC']} WBC {c['WBC']} Plt {c['Platelet']}", fontsize=8
        )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--ablation", action="store_true", help="ablation, sweeps and scale robustness")
    parser.add_argument("--scale-n", type=int, default=20, help="images used for the scale-robustness test")
    parser.add_argument("--force", action="store_true", help="re-run on test although it was already run")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = load_bccd(cfg)
    ids = data.ids(args.split)
    tables = cfg["paths"]["results"] / "tables"
    figures = cfg["paths"]["results"] / "figures" / "evaluation"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tag = f"eval_{args.split}"

    if args.split == "test":
        stamp = tables / "TEST_SET_EVALUATED.txt"
        if stamp.exists() and not args.force:
            raise SystemExit(
                f"The test set was already evaluated ({stamp.read_text().strip()}). Use --force."
            )
        stamp.write_text(f"{dt.datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")

    model = load_model(PROJECT_ROOT / cfg["classifier"]["model_file"])
    if cfg["classifier"]["method"] == "ml" and model is None:
        print("No model file: classes come from the pipeline branch (run scripts/train_classifier.py).")
    pd.set_option("display.width", 220)

    variants = {"full": {}}
    if args.ablation:
        variants.update(ABLATION)
        variants.update({k: v for k, v in CLASSIFIERS.items() if k != "full"})
        for (section, key), values in SWEEPS.items():
            variants.update({f"sweep {key}={v}": {**BRANCH, (section, key): v} for v in values})
    print(f"{len(variants)} configurations x {len(ids)} {args.split} images ...", flush=True)
    out = run_variants(data, ids, cfg, variants, model, args.jobs, keep_objects=("full",))
    det, masks, objects = out["det"], out["masks"], out["objects"]
    det.to_csv(tables / f"{tag}_detection.csv", index=False)

    full = det[det["variant"] == "full"]
    summary = summarize_detection(full, ["cls"])
    summary.to_csv(tables / f"{tag}_summary.csv", index=False)
    print("\nDetection (full pipeline):")
    print(summary.round(3).to_string(index=False))

    # Seconds per image in a single process (the parallel figure above is inflated by contention).
    t_ids = ids[:5]
    t0 = time.perf_counter()
    for i in t_ids:
        run(load_image(data.images.loc[i, "image_path"]), cfg, model=model)
    per_image = (time.perf_counter() - t0) / len(t_ids)
    print(f"Runtime: {per_image:.2f} s / image (single process, {len(t_ids)} images)")

    report, cm = classification_report(objects, data, cfg)
    report.to_csv(tables / f"{tag}_classification.csv", index=False)
    cm.to_csv(tables / f"{tag}_confusion.csv")
    print("\nClassification of matched objects:")
    print(report.round(3).to_string(index=False))
    objects.to_csv(tables / f"{tag}_objects.csv", index=False)

    if len(masks):
        masks.to_csv(tables / f"{tag}_masks.csv", index=False)
        print("\nManual masks (full pipeline):")
        print(masks[masks["variant"] == "full"].drop(columns="variant").describe().round(3).to_string())
    else:
        print("\nNo manual masks for this split: Dice / IoU not computed (data/masks_manual/<id>.json).")

    failure_gallery(data, full, cfg, model).savefig(figures / f"{tag}_failures.png", dpi=80)

    if args.ablation:
        abl = ablation_table(det, masks, cfg)
        abl.to_csv(tables / f"{tag}_ablation.csv", index=False)
        print("\nAblation (classes from the pipeline branch):")
        print(abl.round(3).to_string(index=False))
        cls_table = summarize_detection(det[det["variant"].isin(list(CLASSIFIERS))], ["variant", "cls"])
        cls_table["variant"] = cls_table["variant"].replace({"full": "cls ml (gated, final)"})
        cls_table.to_csv(tables / f"{tag}_classifiers.csv", index=False)
        print("\nEnd-to-end detection per classifier:")
        cols = ["variant", "cls", "P@0.5", "R@0.5", "F1@0.5", "pred/gt"]
        print(cls_table[cols].round(3).to_string(index=False))
        rows = []
        for (section, key), values in SWEEPS.items():
            for v in values:
                s = summarize_detection(det[det["variant"] == f"sweep {key}={v}"], ["cls"]).set_index("cls")
                rows.append(
                    {
                        "param": key,
                        "value": v,
                        "RBC R@0.5": s.loc["RBC", "R@0.5"],
                        "RBC F1@0.5": s.loc["RBC", "F1@0.5"],
                        "RBC pred/gt": s.loc["RBC", "pred/gt"],
                        "Platelet F1@0.5": s.loc["Platelet", "F1@0.5"],
                        "WBC F1@0.5": s.loc["WBC", "F1@0.5"],
                    }
                )
        sweep = pd.DataFrame(rows)
        sweep.to_csv(tables / f"{tag}_sweeps.csv", index=False)
        sweep_figure(sweep).savefig(figures / f"{tag}_sweeps.png", dpi=120)
        print("\nSweeps:")
        print(sweep.round(3).to_string(index=False))

        sub = sorted(
            np.random.default_rng(cfg["seed"]).choice(ids, min(args.scale_n, len(ids)), replace=False)
        )
        print(f"\nScale robustness on {len(sub)} images ...", flush=True)
        sr = scale_robustness(data, sub, cfg, model, args.jobs)
        sr.to_csv(tables / f"{tag}_scale_robustness.csv", index=False)
        cols = ["mode", "factor", "cls", "median r", "R@0.5", "F1@0.5", "pred/gt"]
        print(sr[cols].round(3).to_string(index=False))
    print(f"\nTables: {tables}\\{tag}_*.csv   Figures: {figures}")


if __name__ == "__main__":
    main()
