"""Compare end-to-end pipeline variants on train/val with BCCD boxes (Phases 5-6 tuning).

Every object of the label map gets the class of the branch that produced it
(WBC / platelet / RBC watershed) and its bounding box is matched one-to-one
with the BCCD boxes of that class (Hungarian, IoU >= 0.5 and >= 0.3).

RBC precision is a lower bound (about 30 % of BCCD RBCs have no box), RBC
recall is not biased by the missing boxes, so it is the main signal for the
watershed: over-splitting and merging both lower it. Images without any RBC
box are left out of the RBC metrics. The test set is never used here.

    python scripts/eval_detection.py                                  # config only
    python scripts/eval_detection.py --variant "h=0.1:separation.h=0.1" \\
                                     --variant "gap=0.15:segmentation.gap_close_r=0.15"

Writes results/tables/detection_<name>.csv (one row per image, variant, class).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bloodcells.config import load_config, parse_overrides, with_overrides
from bloodcells.datasets import load_bccd
from bloodcells.evaluate import box_padding, detection_rows, object_boxes, summarize_detection
from bloodcells.io_utils import CLASSES, load_image
from bloodcells.pipeline import analyze
from bloodcells.scale import estimate_scale


def parse_variant(text: str) -> tuple[str, dict]:
    """'name:section.key=value;section.key=value' -> (name, {(section, key): value})."""
    name, _, spec = text.partition(":")
    return name.strip(), parse_overrides(spec)


def evaluate_image(image_id, path, boxes, cfg, variants, no_rbc) -> list[dict]:
    img = load_image(path)
    r = estimate_scale(img, cfg).r  # shared by all variants (scale is not tuned here)
    rows = []
    for name, overrides in variants.items():
        vcfg = with_overrides(cfg, overrides)
        t0 = time.perf_counter()
        a = analyze(img, vcfg, r=r)
        seconds = time.perf_counter() - t0
        pred = object_boxes(a.labels, a.branch, box_padding(vcfg, r))
        for row in detection_rows(pred, boxes, CLASSES, skip=("RBC",) if no_rbc else ()):
            rows.append({"image_id": image_id, "variant": name, "r": r, "seconds": seconds, **row})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--variant", action="append", default=[], help="name:section.key=value;...")
    parser.add_argument("--split", choices=["train", "val", "trainval"], default="trainval")
    parser.add_argument("--n", type=int, default=0, help="random subset of n images (0 = all)")
    parser.add_argument("--name", default="variants", help="output table suffix")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    variants = dict([("config", {})] + [parse_variant(v) for v in args.variant])
    data = load_bccd(cfg)
    ids = data.ids(args.split)
    if args.n:
        ids = sorted(np.random.default_rng(cfg["seed"]).choice(ids, args.n, replace=False))

    print(f"{len(variants)} variants x {len(ids)} {args.split} images ...")
    rows = Parallel(n_jobs=args.jobs)(
        delayed(evaluate_image)(
            i,
            data.images.loc[i, "image_path"],
            data.boxes_of(i),
            cfg,
            variants,
            bool(data.images.loc[i, "no_rbc_labels"]),
        )
        for i in ids
    )
    table = pd.DataFrame([row for per_image in rows for row in per_image])
    table["split"] = table["image_id"].map(data.split_of)
    out = cfg["paths"]["results"] / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"detection_{args.name}.csv", index=False)
    summary = summarize_detection(table, ["cls", "variant"])
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(f"\nWritten {out / f'detection_{args.name}.csv'}")


if __name__ == "__main__":
    main()
