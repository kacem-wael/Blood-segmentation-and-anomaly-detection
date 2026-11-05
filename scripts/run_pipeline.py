"""Analyse one image or a whole folder (Phase 9 CLI).

    python scripts/run_pipeline.py path/to/image.jpg
    python scripts/run_pipeline.py path/to/folder --out results/run --ids
    python scripts/run_pipeline.py image.png --r 40          # override the RBC radius (px)

For every image: <out>/<name>_overlay.png (class-coloured contours from the
label map: RBC red, WBC blue, platelet green) and <out>/<name>_cells.csv (one
row per object). For all images: <out>/summary.csv (counts and statistics).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from bloodcells.classifier import load_model
from bloodcells.config import PROJECT_ROOT, load_config
from bloodcells.io_utils import load_image
from bloodcells.pipeline import run
from bloodcells.visualization import save_image

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def process(path: Path, out: Path, cfg: dict, model, r: float | None, ids: bool) -> dict:
    t0 = time.perf_counter()
    try:
        res = run(load_image(path), cfg, r=r, model=model)
    except Exception as exc:  # one bad file must not stop a folder run
        return {"image": path.name, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"}
    save_image(out / f"{path.stem}_overlay.png", res.overlay(show_ids=ids))
    res.features.to_csv(out / f"{path.stem}_cells.csv", index=False)
    return {"image": path.name, **res.summary(), "seconds": time.perf_counter() - t0}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="image file or folder")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "run")
    parser.add_argument("--config", default=None)
    parser.add_argument("--r", type=float, default=None, help="RBC radius in px (default: granulometry)")
    parser.add_argument("--ids", action="store_true", help="write object ids on the overlay")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = load_model(PROJECT_ROOT / cfg["classifier"]["model_file"])

    if args.input.is_dir():
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in SUFFIXES)
    elif args.input.is_file():
        files = [args.input]
    else:
        sys.exit(f"Not found: {args.input}")
    if not files:
        sys.exit(f"No image in {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = Parallel(n_jobs=args.jobs if len(files) > 1 else 1)(
        delayed(process)(p, args.out, cfg, model, args.r, args.ids) for p in files
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    cols = [
        c
        for c in ("image", "r", "n_RBC", "n_WBC", "n_Platelet", "rbc_area_cv", "seconds", "error")
        if c in summary
    ]
    print(summary[cols].round(2).to_string(index=False))
    print(f"\nResults in {args.out}")


if __name__ == "__main__":
    main()
