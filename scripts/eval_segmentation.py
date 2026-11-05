"""Compare Phase 4 segmentation variants on train/val with BCCD box proxies.

Each variant overrides some ``config.yaml['segmentation']`` keys; r is
estimated once per image and shared by all variants. Proxies are defined in
``bloodcells.evaluate.box_proxies``. The test set is not used.

Writes results/tables/segmentation_proxies.csv (one row per image and variant).
"""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bloodcells.config import load_config
from bloodcells.evaluate import box_proxies
from bloodcells.io_utils import load_image
from bloodcells.scale import estimate_scale
from bloodcells.segmentation import segment

VARIANTS: dict[str, dict] = {
    "config": {},
    "gap_close 0.05": {"gap_close_r": 0.05},
    "gap_close 0.1": {"gap_close_r": 0.1},
    "gap_close 0.15": {"gap_close_r": 0.15},
    "gap_close 0.2": {"gap_close_r": 0.2},
    "hole_fill plain": {"hole_fill": "plain"},
    "hole_fill none": {"hole_fill": "none"},
    "open 0.05": {"open_r": 0.05},
    "open 0.2": {"open_r": 0.2},
    "local threshold": {"cell_threshold": "local"},
    "purple multiotsu": {"purple_threshold": "multiotsu"},
}


def with_overrides(cfg: dict, overrides: dict) -> dict:
    out = copy.deepcopy(cfg)
    out["segmentation"].update(overrides)
    return out


def evaluate_image(image_id: str, path: str, boxes: pd.DataFrame, cfg: dict, variants: dict) -> list[dict]:
    img = load_image(path)
    r = estimate_scale(img, cfg).r
    rows = []
    for name, overrides in variants.items():
        seg = segment(img, with_overrides(cfg, overrides), r=r)
        rows.append(
            {
                "image_id": image_id,
                "variant": name,
                "r": r,
                **box_proxies(seg.cell_mask, seg.purple_mask, boxes, r),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--variants", nargs="*", default=None, help=f"subset of {list(VARIANTS)}")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    variants = {k: VARIANTS[k] for k in (args.variants or VARIANTS)}

    images = pd.read_csv(paths["index"] / "bccd_images.csv").set_index("image_id")
    boxes = pd.read_csv(paths["index"] / "bccd_boxes.csv")
    boxes = boxes[~boxes["duplicate"]]
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    split_of = {i: s for s in ("train", "val") for i in splits[s]}
    boxes_of = dict(tuple(boxes.groupby("image_id")))
    ids = [i for i in split_of if i in boxes_of]

    print(f"{len(variants)} variants x {len(ids)} train/val images ...")
    rows = Parallel(n_jobs=args.jobs)(
        delayed(evaluate_image)(i, images.loc[i, "image_path"], boxes_of[i], cfg, variants) for i in ids
    )
    table = pd.DataFrame([row for per_image in rows for row in per_image])
    table["split"] = table["image_id"].map(split_of)
    # Images without RBC boxes have no RBC proxies (NaN): they are skipped by the means.
    no_rbc = set(images.index[images["no_rbc_labels"]])
    table.loc[table["image_id"].isin(no_rbc), ["rbc_centre", "rbc_fill", "rbc_shared"]] = np.nan
    out = paths["results"] / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "segmentation_proxies.csv", index=False)

    metrics = ["rbc_centre", "rbc_fill", "rbc_shared", "wbc_found", "platelet_found", "purple_outside"]
    summary = table.groupby("variant")[metrics].mean().reindex(list(variants))
    pd.set_option("display.width", 160)
    print("\nMean over train/val images (higher = better, except rbc_shared and purple_outside):")
    print(summary.round(3).to_string())
    print(f"\nWritten {out / 'segmentation_proxies.csv'}")


if __name__ == "__main__":
    main()
