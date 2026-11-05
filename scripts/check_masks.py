"""Check the manual masks (LabelMe JSON) before using them for the Dice evaluation.

    python scripts/check_masks.py                  # every data/masks_manual/*.json
    python scripts/check_masks.py BloodImage_00109

For each annotation: number of cells per class, WBCs without a nucleus
polygon, nucleus polygons outside every WBC, tiny or huge polygons (probable
mistakes), comparison with the BCCD boxes, and a preview image
outputs/masks_preview/<image_id>.png (contours: RBC red, WBC blue, platelet
green, nucleus magenta). Also says which of the 18 planned images are still missing.
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

import numpy as np

from bloodcells.config import PROJECT_ROOT, load_config
from bloodcells.datasets import load_bccd
from bloodcells.io_utils import labelme_reviewed, load_image, load_labelme
from bloodcells.visualization import CLASS_COLORS, class_overlay, mask_contours, save_image


def check(image_id: str, data, cfg) -> list[str]:
    path = cfg["paths"]["masks_manual"] / f"{image_id}.json"
    img = load_image(data.images.loc[image_id, "image_path"])
    m = load_labelme(path, img.shape[:2])
    problems = []
    areas = np.bincount(m.labels.ravel())[1:]
    counts = {c: sum(v == c for v in m.classes.values()) for c in CLASS_COLORS if c != "Artefact"}
    boxes = data.boxes_of(image_id)["cls"].value_counts().to_dict()
    state = "reviewed" if labelme_reviewed(path) else "NOT reviewed (ignored by the evaluation)"
    print(f"{image_id} [{state}]: {counts}   BCCD boxes {boxes}")
    drafts = sum(
        s.get("description") == "draft" for s in json.loads(path.read_text(encoding="utf-8"))["shapes"]
    )
    if drafts and labelme_reviewed(path):
        problems.append(f"{drafts} polygons were never edited (still 'draft'): check them")
    for k, cls in m.classes.items():
        a = areas[k - 1] if k - 1 < len(areas) else 0
        if a == 0:
            problems.append(f"{cls} #{k} is completely hidden by later polygons")
        elif cls == "RBC" and not 300 <= a <= 25000:
            problems.append(f"RBC #{k} has an unusual area ({a} px): a mislabel or a merged polygon?")
        elif cls == "Platelet" and a > 5000:
            problems.append(f"Platelet #{k} is large ({a} px): an RBC or a WBC?")
    wbc = np.isin(m.labels, [k for k, c in m.classes.items() if c == "WBC"])
    if wbc.any() and not (m.nuclei & wbc).any():
        problems.append("a WBC has no 'nucleus' polygon")
    if (m.nuclei & ~wbc).sum() > 50:
        problems.append("a 'nucleus' polygon lies outside every WBC polygon")
    if counts.get("WBC", 0) != boxes.get("WBC", 0):
        problems.append(f"{counts.get('WBC', 0)} WBC polygons but {boxes.get('WBC', 0)} WBC boxes: check")
    preview = class_overlay(img, m.labels, m.classes, radius=1, show_ids=True)
    preview[mask_contours(m.nuclei, 1)] = (255, 0, 255)
    save_image(PROJECT_ROOT / "outputs" / "masks_preview" / f"{image_id}.png", preview)
    for p in problems:
        print(f"   ! {p}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ids", nargs="*")
    args = parser.parse_args()
    cfg = load_config()
    data = load_bccd(cfg)
    done = data.manual_masks()
    mask_dir = cfg["paths"]["masks_manual"]
    ids = [i if i.startswith("BloodImage_") else f"BloodImage_{i}" for i in args.ids] or sorted(
        p.stem for p in mask_dir.glob("*.json")
    )
    total = 0
    for i in ids:
        total += len(check(i, data, cfg))
    planned = data.splits.get("manual_mask_candidates", {})
    for split, wanted in planned.items():
        missing = [i for i in wanted if i not in done]
        print(
            f"{split}: {len(wanted) - len(missing)} / {len(wanted)} annotated"
            + (f", missing {missing}" if missing else "")
        )
    print(f"{total} problem(s); previews in {PROJECT_ROOT / 'outputs' / 'masks_preview'}")


if __name__ == "__main__":
    main()
