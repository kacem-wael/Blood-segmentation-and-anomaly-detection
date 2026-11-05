"""Run the morphological pipeline on BCCD and write one feature table per split (Phase 7).

    python scripts/build_features.py                 # train, val and test
    python scripts/build_features.py --splits train val

Each object gets its features (``bloodcells.features``), the branch that
produced it and its ground-truth class ``gt`` from the BCCD boxes
(``features.assign_labels``): RBC / WBC / Platelet when matched, ``Unlabelled``
otherwise (excluded from training: BCCD misses ~30 % of the RBCs), or
``Artefact`` on manual-mask images (exhaustive annotation). On images without
any RBC box, objects not matched to a WBC / platelet box are ``Unlabelled``.

Writes data/features/features_<split>.csv and data/features/images_<split>.csv
(one row per image: r, counts, seconds). Building test features is allowed:
nothing is tuned on them, they are read only by ``run_evaluation.py --split test``.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from bloodcells.config import load_config
from bloodcells.datasets import load_bccd
from bloodcells.evaluate import box_padding, object_boxes
from bloodcells.features import assign_labels
from bloodcells.io_utils import load_image, load_labelme
from bloodcells.pipeline import run


def process(image_id: str, path: str, gt: pd.DataFrame, cfg: dict, no_rbc: bool, mask_path: Path | None):
    t0 = time.perf_counter()
    img = load_image(path)
    res = run(img, cfg)
    exhaustive = mask_path is not None
    if exhaustive:
        # Every cell is annotated: label from the polygons, unmatched objects are Artefact.
        manual = load_labelme(mask_path, img.shape[:2])
        gt = object_boxes(manual.labels, manual.classes)
    seconds = time.perf_counter() - t0
    f = res.features
    f.insert(0, "image_id", image_id)
    fc = cfg["features"]
    f["gt"] = assign_labels(f, gt, box_padding(cfg, res.r), fc["iou_match"], exhaustive, fc["iou_fallback"])
    f["no_rbc_labels"] = no_rbc  # RBC objects of such images are all Unlabelled; RBC metrics skip them
    info = {"image_id": image_id, "r": res.r, "seconds": seconds, **res.counts, "n_gt": len(gt)}
    return f, info


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    cfg["classifier"]["method"] = "branch"  # features only: the classifier is trained on them
    data = load_bccd(cfg)
    manual = set(data.manual_masks())
    mask_dir = cfg["paths"]["masks_manual"]
    if manual:
        print(f"{len(manual)} images with manual masks: labels from the polygons")
    out = cfg["paths"]["data"] / "features"
    out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        ids = data.ids(split)
        print(f"{split}: {len(ids)} images ...", flush=True)
        results = Parallel(n_jobs=args.jobs)(
            delayed(process)(
                i,
                data.images.loc[i, "image_path"],
                data.boxes_of(i),
                cfg,
                bool(data.images.loc[i, "no_rbc_labels"]),
                mask_dir / f"{i}.json" if i in manual else None,
            )
            for i in ids
        )
        feats = pd.concat([f for f, _ in results], ignore_index=True)
        feats.insert(1, "split", split)
        feats.insert(2, "group", feats["image_id"].map(data.group_of))
        info = pd.DataFrame([i for _, i in results])
        feats.to_csv(out / f"features_{split}.csv", index=False)
        info.to_csv(out / f"images_{split}.csv", index=False)
        print(feats.groupby(["branch", "gt"]).size().unstack(fill_value=0).to_string())
        print(f"  {info['seconds'].mean():.2f} s/image (parallel), written {out / f'features_{split}.csv'}\n")


if __name__ == "__main__":
    main()
