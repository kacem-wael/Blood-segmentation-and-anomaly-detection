"""Draft LabelMe annotations for the 18 manual-mask images (to correct by hand, NOT ground truth).

    python scripts/prepare_masks.py               # data/masks_manual/drafts/<image_id>.json
    python scripts/prepare_masks.py --overwrite

A draft gives the annotator a starting polygon for every *boxed* cell, so the
work is reshaping, not drawing everything:

* one ellipse inscribed in each BCCD box, with the box class (RBC / WBC / Platelet);
* one ``nucleus`` polygon per WBC: Otsu of the -b* stain channel inside the WBC box.

The drafts come from the BCCD annotation, not from our morphological pipeline,
so correcting them does not bias the masks (and the Dice) towards our method.
BCCD misses ~30 % of the RBCs: those must be added by hand.

Every draft carries the image flag ``reviewed: false``. It is ignored by the
evaluation until the annotator ticks "reviewed" in LabelMe (``scripts/annotate.py``).
"""

from __future__ import annotations

# Run with the project environment even when another one (e.g. .venv-labelme) is active.
try:
    import bloodcells  # noqa: F401
except ModuleNotFoundError:
    import subprocess
    import sys
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[1]
    for _py in (_root / ".venv" / "Scripts" / "python.exe", _root / ".venv" / "bin" / "python"):
        if _py.exists():
            sys.exit(subprocess.run([str(_py), __file__, *sys.argv[1:]]).returncode)
    sys.exit("The bloodcells package is not installed: run 'make setup' (or pip install -e .) first")

import argparse
import json
import os
from pathlib import Path

import numpy as np
from skimage import color, filters, measure

from bloodcells.config import load_config
from bloodcells.datasets import load_bccd
from bloodcells.io_utils import load_image

LABELME_VERSION = "6.3.1"
LABELS = ("RBC", "WBC", "Platelet", "nucleus")


def polygons(mask: np.ndarray, tolerance: float = 1.0) -> list[list[list[float]]]:
    """Outer contours of a binary mask as LabelMe [[x, y], ...] polygons (one per component)."""
    out = []
    for region in measure.regionprops(measure.label(mask)):
        contours = measure.find_contours(np.pad(region.image, 1).astype(float), 0.5)
        if not contours:
            continue
        c = measure.approximate_polygon(max(contours, key=len), tolerance=tolerance)[:-1]
        if len(c) < 3:
            continue
        r0, c0 = region.bbox[0] - 1, region.bbox[1] - 1
        out.append([[round(float(x + c0), 1), round(float(y + r0), 1)] for y, x in c])
    return out


def shape(label: str, points: list[list[float]]) -> dict:
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": "draft",
        "shape_type": "polygon",
        "flags": {},
    }


def ellipse(b, shape_hw: tuple[int, int]) -> np.ndarray:
    """Ellipse inscribed in a VOC box (1-based, inclusive)."""
    h, w = shape_hw
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (b.ymin + b.ymax) / 2 - 1, (b.xmin + b.xmax) / 2 - 1
    ry, rx = max((b.ymax - b.ymin) / 2, 2.0), max((b.xmax - b.xmin) / 2, 2.0)
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def draft(image_id: str, data, cfg) -> dict:
    image_path = Path(data.images.loc[image_id, "image_path"])
    img = load_image(image_path)
    h, w = img.shape[:2]
    stain = -color.rgb2lab(img)[..., 2]
    shapes = []
    # Large cells first, platelets last: in LabelMe the later polygon wins overlaps.
    order = {"RBC": 0, "WBC": 1, "Platelet": 2}
    boxes = data.boxes_of(image_id).sort_values("cls", key=lambda c: c.map(order))
    for b in boxes.itertuples():
        ell = ellipse(b, (h, w))
        shapes += [shape(b.cls, p) for p in polygons(ell)]
        if b.cls == "WBC":
            inner = stain[ell]
            nucleus = ell & (stain > filters.threshold_otsu(inner)) if inner.std() > 0 else ell
            lab = measure.label(nucleus)
            areas = np.bincount(lab.ravel())[1:]
            if areas.size:
                nucleus = np.isin(lab, 1 + np.flatnonzero(areas >= 0.05 * areas.max()))
            shapes += [shape("nucleus", p) for p in polygons(nucleus, tolerance=1.5)]
    return {
        "version": LABELME_VERSION,
        "flags": {"reviewed": False},
        "shapes": shapes,
        # Relative to data/masks_manual/, where annotate.py copies the draft before opening it.
        "imagePath": os.path.relpath(image_path, cfg["paths"]["masks_manual"]).replace("\\", "/"),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    data = load_bccd(cfg)
    out = cfg["paths"]["masks_manual"] / "drafts"
    out.mkdir(parents=True, exist_ok=True)
    for i in [i for v in data.splits["manual_mask_candidates"].values() for i in v]:
        target = out / f"{i}.json"
        if target.exists() and not args.overwrite:
            print(f"{i}: draft exists (use --overwrite)")
            continue
        d = draft(i, data, cfg)
        target.write_text(json.dumps(d, indent=1), encoding="utf-8")
        counts = {k: sum(s["label"] == k for s in d["shapes"]) for k in LABELS}
        print(f"{i}: {counts}")
    print(f"\nDrafts in {out}\nStart annotating with: python scripts/annotate.py next")


if __name__ == "__main__":
    main()
