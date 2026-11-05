"""Check the granulometry estimate of the RBC radius r (project.md, Phase 3).

1. Box check (train + val, never test): r estimated by ``estimate_scale`` vs
   the reference radius of each image (median (w + h) / 4 of its complete RBC
   boxes). The calibration factor (box radius / inscribed radius) is fitted on
   train only and checked on val. Target: |error| < 15 % on most images.
   A constant guess is printed as a baseline: BCCD has a single magnification,
   so this check alone cannot show that r follows the scale.
2. Zoom check (val): the whole image is rescaled by 0.5 to 2; r must be
   multiplied by the same factor.

Writes results/tables/scale_check.csv and results/tables/scale_zoom.csv.
Run scripts/build_index.py and scripts/make_splits.py first.
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage import transform

from bloodcells.config import DEFAULT_CONFIG, load_config
from bloodcells.datasets import rbc_radius_per_image
from bloodcells.io_utils import load_image
from bloodcells.scale import estimate_scale

ZOOMS = (0.5, 0.75, 1.5, 2.0)
TOLERANCE = 0.15


def zoom(img: np.ndarray, factor: float) -> np.ndarray:
    """Rescale the whole image: same cells, ``factor`` times bigger in pixels.

    A centre crop enlarged back to the original size would be a more realistic
    magnification, but BCCD images are centred on their WBC, so such a crop is
    mostly WBC: the content would change along with the scale.
    """
    out = transform.rescale(img, factor, channel_axis=-1, order=1, anti_aliasing=factor < 1)
    return (out * 255).round().astype(np.uint8)


def estimate_one(image_id: str, path: str, cfg: dict, zooms: tuple[float, ...]) -> list[dict]:
    img = load_image(path)
    rows = []
    for factor in (1.0, *zooms):
        e = estimate_scale(img if factor == 1 else zoom(img, factor), cfg)
        rows.append(
            {"image_id": image_id, "zoom": factor, "r_inscribed": e.r_inscribed, "r": e.r, "ok": e.ok}
        )
    return rows


def summary(err: pd.Series) -> str:
    return (
        f"n={len(err):3d}  median |err| {err.abs().median():5.1%}  "
        f"within {TOLERANCE:.0%}: {(err.abs() < TOLERANCE).mean():5.1%}  "
        f"> 30 %: {(err.abs() > 0.3).sum()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-zoom", type=int, default=30, help="val images used for the zoom check")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--update-config", action="store_true", help="write the fitted calibration into config.yaml"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]

    images = pd.read_csv(paths["index"] / "bccd_images.csv").set_index("image_id")
    boxes = pd.read_csv(paths["index"] / "bccd_boxes.csv")
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    ref = rbc_radius_per_image(boxes)

    split_of = {i: s for s in ("train", "val") for i in splits[s]}
    ids = [i for i in split_of if i in ref.index]
    rng = np.random.default_rng(cfg["seed"])
    val_ids = [i for i in ids if split_of[i] == "val"]
    zoom_ids = set(rng.choice(val_ids, min(args.n_zoom, len(val_ids)), replace=False))

    print(f"Estimating r on {len(ids)} train/val images ({len(zoom_ids)} also zoomed) ...")
    rows = Parallel(n_jobs=args.jobs)(
        delayed(estimate_one)(i, images.loc[i, "image_path"], cfg, ZOOMS if i in zoom_ids else ())
        for i in ids
    )
    est = pd.DataFrame([row for per_image in rows for row in per_image])

    # 1. Box check
    base = est[est["zoom"] == 1].set_index("image_id").drop(columns="zoom")
    base["split"] = base.index.map(split_of)
    base["r_ref"] = ref.reindex(base.index)
    train = base[(base["split"] == "train") & base["ok"]]
    fitted = float((train["r_ref"] / train["r_inscribed"]).median())
    base["r_fitted"] = base["r_inscribed"] * fitted
    base["rel_err"] = base["r"] / base["r_ref"] - 1
    base.loc[~base["ok"], "r_fitted"] = base["r"]
    tables = paths["results"] / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    base.reset_index().to_csv(tables / "scale_check.csv", index=False)

    calib = cfg["scale"]["calibration"]
    print(f"\nCalibration (box radius / inscribed radius): config {calib:.3f}, fitted on train {fitted:.3f}")
    print(f"Fallback to r_default: {(~base['ok']).sum()} images")
    for split in ("train", "val"):
        part = base[base["split"] == split]
        print(f"  {split:5s} (config calibration): {summary(part['rel_err'])}")
        print(f"  {split:5s} (fitted calibration): {summary(part['r_fitted'] / part['r_ref'] - 1)}")
    constant = float(train["r_ref"].median())
    val = base[base["split"] == "val"]
    print(f"  baseline: constant r = {constant:.1f} px on val: {summary(constant / val['r_ref'] - 1)}")
    print("Worst val images:")
    worst = val.reindex(val["rel_err"].abs().sort_values(ascending=False).index).head(5)
    print(worst[["r", "r_ref", "rel_err"]].round(3).to_string())

    # 2. Zoom check
    zoomed = est[est["image_id"].isin(zoom_ids)]
    r1 = zoomed[zoomed["zoom"] == 1].set_index("image_id")["r"]
    zoomed = zoomed[zoomed["zoom"] != 1].copy()
    zoomed["r_zoom1"] = zoomed["image_id"].map(r1)
    zoomed["ratio"] = zoomed["r"] / (zoomed["zoom"] * zoomed["r_zoom1"])
    zoomed.to_csv(tables / "scale_zoom.csv", index=False)
    print("\nZoom check: r(zoomed) / (factor * r(original)), should be 1")
    for factor, part in zoomed.groupby("zoom"):
        print(f"  x{factor:<4}: median ratio {part['ratio'].median():.3f}  {summary(part['ratio'] - 1)}")

    if args.update_config:
        path = args.config or DEFAULT_CONFIG
        text = open(path, encoding="utf-8").read()
        text, n = re.subn(r"(\n\s*calibration:\s*)[0-9.]+", rf"\g<1>{fitted:.2f}", text, count=1)
        if n:
            open(path, "w", encoding="utf-8").write(text)
            print(f"\ncalibration: {fitted:.2f} written to {path}")
        else:
            print(f"\nNo 'calibration:' entry found in {path}; nothing written")
    print(f"\nTables written to {tables}")


if __name__ == "__main__":
    main()
