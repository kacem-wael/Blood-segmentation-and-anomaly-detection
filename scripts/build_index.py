"""Parse all annotations into CSV indexes and print dataset statistics.

Writes to data/index/:
    bccd_images.csv   one row per image (split, size, box counts per class,
                      field group, no_rbc_labels flag)
    bccd_boxes.csv    one row per valid box (duplicate flag for repeated boxes)
    bccd_issues.csv   parsing problems
    bccd_same_field_pairs.csv   pairs of images showing the same microscope field
(Raabin-WBC and ALL-IDB are indexed by scripts/audit_extra_datasets.py.)
"""

from __future__ import annotations

import argparse

from bloodcells.config import load_config
from bloodcells.datasets import (
    bccd_index,
    field_groups,
    flag_duplicate_boxes,
    rbc_radius_from_boxes,
    same_field_pairs,
)
from bloodcells.io_utils import CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = cfg["paths"]["index"]
    out.mkdir(parents=True, exist_ok=True)

    images, boxes, issues = bccd_index(cfg["paths"]["bccd"])

    # Repeated boxes are flagged and left out of the per-image counts.
    boxes["duplicate"] = flag_duplicate_boxes(boxes)
    kept = boxes[~boxes["duplicate"]]
    for c in CLASSES:
        counts = kept[kept["cls"] == c].groupby("image_id").size()
        images[f"n_{c}"] = images["image_id"].map(counts).fillna(0).astype(int)
    # Images whose RBCs were not annotated at all (RBC counts unusable there).
    images["no_rbc_labels"] = images["n_RBC"] == 0

    print("Registering image pairs to find shifted frames of the same field (~1 min) ...")
    usable = images[images["has_image"]]
    pairs = same_field_pairs(list(usable["image_id"]), list(usable["image_path"]))
    group = field_groups(list(images["image_id"]), pairs)
    images["field_group"] = images["image_id"].map(group)
    split_of = images.set_index("image_id")["split"]
    pairs["split_a"] = pairs["image_a"].map(split_of)
    pairs["split_b"] = pairs["image_b"].map(split_of)
    pairs.to_csv(out / "bccd_same_field_pairs.csv", index=False)

    images.to_csv(out / "bccd_images.csv", index=False)
    boxes.to_csv(out / "bccd_boxes.csv", index=False)
    issues.to_csv(out / "bccd_issues.csv", index=False)

    print(f"BCCD: {len(images)} images, {len(boxes)} boxes, {len(issues)} issues")
    print("\nImages per split:")
    print(images["split"].value_counts().to_string())
    print("\nBoxes per class and split:")
    print(
        kept.pivot_table(
            index="cls", columns="split", values="image_id", aggfunc="count", fill_value=0
        ).to_string()
    )
    print("\nImages by number of WBC boxes:")
    print(images["n_WBC"].value_counts().sort_index().to_string())
    print("\nBox size (px), complete boxes only:")
    complete = kept[~kept["truncated"]]
    print(complete.groupby("cls")[["width", "height"]].median().round(1).to_string())
    print(f"\nReference RBC radius from boxes: r = {rbc_radius_from_boxes(kept):.1f} px")
    print(
        f"\nDuplicate boxes (IoU >= 0.7, same class): {boxes['duplicate'].sum()} "
        f"in {boxes.loc[boxes['duplicate'], 'image_id'].nunique()} images"
    )
    print(f"Images with no RBC box: {images['no_rbc_labels'].sum()}")
    cross = pairs[pairs["split_a"] != pairs["split_b"]]
    multi = images.groupby("field_group")["split"].nunique()
    print(
        f"Same-field pairs: {len(pairs)} ({len(cross)} across splits); "
        f"{(multi > 1).sum()} field groups span several splits"
    )
    if len(issues):
        print("\nIssues:")
        print(issues.to_string(index=False))

    print(f"\nIndexes written to {out}")
    print("Raabin-WBC / ALL-IDB indexes: scripts/audit_extra_datasets.py")


if __name__ == "__main__":
    main()
