"""WBC detection check on Raabin-WBC crops (project.md Phase 6, without masks).

    python scripts/eval_raabin.py --n 100          # n crops per (split, class), seed from config.yaml

Every Raabin-WBC crop is centred on exactly one WBC, from another lab,
microscope and magnification than BCCD: a test of the scale estimate and of
the WBC branch on unseen data. For each crop we report whether the pipeline
finds exactly one WBC, whether a WBC covers the crop centre, the estimated r,
the N/C ratio, and the per-class rates (neutrophil, lymphocyte, ...).
The Kaggle version has no segmentation masks, so no Dice is computed here.

Writes results/tables/raabin_detection.csv and raabin_detection_summary.csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bloodcells.classifier import load_model
from bloodcells.config import PROJECT_ROOT, load_config
from bloodcells.io_utils import load_image
from bloodcells.pipeline import run


def one(row, cfg, model) -> dict:
    out = {"path": row.path, "split": row.split, "cls": row.cls}
    try:
        res = run(load_image(row.path), cfg, model=model)
    except Exception as exc:  # report, do not stop the batch
        return {**out, "error": f"{type(exc).__name__}: {exc}"}
    f = res.features
    wbc = f[f["cls"] == "WBC"]
    h, w = res.labels.shape
    centre = int(res.labels[h // 2, w // 2])
    out.update(
        {
            "r": res.r,
            "n_wbc": len(wbc),
            "exactly_one": len(wbc) == 1,
            "centre_is_wbc": res.classes.get(centre, "") == "WBC",
            "wbc_area_rel": float(wbc["area_rel"].max()) if len(wbc) else np.nan,
            "nc_ratio": float(wbc["nc_ratio"].iloc[0]) if len(wbc) == 1 else np.nan,
            "cytoplasm_uncertain": bool(wbc["cytoplasm_uncertain"].iloc[0]) if len(wbc) == 1 else False,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--n", type=int, default=100, help="crops per (split, class)")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    index = cfg["paths"]["index"] / "raabin_files.csv"
    if not index.exists():
        raise SystemExit(f"{index} is missing: run scripts/audit_extra_datasets.py first")
    files = pd.read_csv(index)
    # Seeded shuffle, then the first n of each (split, class): a stratified random sample.
    sample = files.sample(frac=1, random_state=cfg["seed"]).groupby(["split", "cls"]).head(args.n)
    model = load_model(PROJECT_ROOT / cfg["classifier"]["model_file"])
    print(f"{len(sample)} Raabin-WBC crops ...", flush=True)
    rows = Parallel(n_jobs=args.jobs)(delayed(one)(row, cfg, model) for row in sample.itertuples())
    table = pd.DataFrame(rows)
    out = cfg["paths"]["results"] / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "raabin_detection.csv", index=False)
    ok = table[table.get("error", pd.Series(np.nan, index=table.index)).isna()]
    summary = (
        ok.groupby(["split", "cls"])
        .agg(
            n=("exactly_one", "size"),
            exactly_one=("exactly_one", "mean"),
            centre_is_wbc=("centre_is_wbc", "mean"),
            median_r=("r", "median"),
            nc_median=("nc_ratio", "median"),
            cytoplasm_uncertain=("cytoplasm_uncertain", "mean"),
        )
        .reset_index()
    )
    total = {
        "split": "all",
        "cls": "all",
        "n": len(ok),
        "exactly_one": ok["exactly_one"].mean(),
        "centre_is_wbc": ok["centre_is_wbc"].mean(),
        "median_r": ok["r"].median(),
        "nc_median": ok["nc_ratio"].median(),
        "cytoplasm_uncertain": ok["cytoplasm_uncertain"].mean(),
    }
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    summary.to_csv(out / "raabin_detection_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    if len(ok) < len(table):
        print(f"{len(table) - len(ok)} crops failed (see the error column)")


if __name__ == "__main__":
    main()
