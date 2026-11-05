"""Write data/splits.json from the official BCCD split, without same-field leakage.

BCCD contains shifted frames of the same microscope field, some of them in
different official splits. The official test set is kept unchanged (results
stay comparable with other work); train/val images sharing a field with a test
image are excluded, and val images sharing a field with train move to train.
"field_groups" is stored for GroupKFold.

Also proposes the images to annotate by hand (project.md, section 5.1):
10 from train/val and 8 from test, spread over difficulty levels
(number of RBC boxes is used as a crowding proxy). The proposal is
stored as candidates; move an image to "manual_masks" once its mask exists.

Run scripts/build_index.py first.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from bloodcells.config import load_config
from bloodcells.datasets import resolve_field_leakage


def pick_spread(images: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    """Pick n images spread over RBC-count quantiles (one per bin, random within bin).

    Images without RBC boxes are skipped: their RBC count says nothing about crowding.
    """
    candidates = images[(images["n_WBC"] >= 1) & ~images["no_rbc_labels"]].sort_values("n_RBC")
    bins = np.array_split(candidates["image_id"].to_numpy(), n)
    return sorted(str(rng.choice(b)) for b in bins if len(b))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    images = pd.read_csv(cfg["paths"]["index"] / "bccd_images.csv")
    usable = images[images["has_image"] & images["has_annotation"]].copy()
    excluded = sorted(set(images["image_id"]) - set(usable["image_id"]))

    new_split, changes = resolve_field_leakage(
        usable.set_index("image_id")["split"], usable.set_index("image_id")["field_group"]
    )
    usable["split"] = usable["image_id"].map(new_split)
    leaked = sorted(usable.loc[usable["split"] == "excluded_leakage", "image_id"])
    usable = usable[usable["split"] != "excluded_leakage"]

    rng = np.random.default_rng(cfg["seed"])
    trainval = usable[usable["split"].isin(["train", "val"])]
    test = usable[usable["split"] == "test"]

    splits = {
        "source": "BCCD official split (BCCD/ImageSets/Main), same-field leakage removed",
        "seed": cfg["seed"],
        "train": sorted(usable.loc[usable["split"] == "train", "image_id"]),
        "val": sorted(usable.loc[usable["split"] == "val", "image_id"]),
        "test": sorted(test["image_id"]),
        "excluded": excluded,
        "excluded_leakage": leaked,
        "moved_val_to_train": sorted(changes.loc[changes["to"] == "train", "image_id"]),
        "field_groups": usable.set_index("image_id")["field_group"].to_dict(),
        "manual_mask_candidates": {
            "trainval": pick_spread(trainval, cfg["splits"]["manual_masks_trainval"], rng),
            "test": pick_spread(test, cfg["splits"]["manual_masks_test"], rng),
        },
        "manual_masks": {"trainval": [], "test": []},
    }

    path = cfg["paths"]["splits"]
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(
        f"train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])} "
        f"| excluded {len(excluded)} | excluded for leakage {len(leaked)}"
    )
    if len(changes):
        print(changes.to_string(index=False))
    print("Manual-mask candidates (train/val):", ", ".join(splits["manual_mask_candidates"]["trainval"]))
    print("Manual-mask candidates (test):     ", ", ".join(splits["manual_mask_candidates"]["test"]))
    print(f"Written {path}")


if __name__ == "__main__":
    main()
