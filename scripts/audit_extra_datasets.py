"""Audit the Kaggle downloads of Raabin-WBC and ALL-IDB and write clean indexes.

Raabin-WBC (Kaggle raabindata/raabin-wbc): single-WBC crops sorted by class,
no segmentation masks. File names are <slide>_<photo>_<cell>; crops of the same
photo share their background, so the "photo" and "slide" columns are group
keys (several photos and slides appear in both Train and TestA).

ALL-IDB mirror (Kaggle sizlingdhairya1/all-idb-images): the folders are named
L1/L2/L3 (FAB subtypes) but do not match them; many files are re-encoded,
renamed or rotated copies. Exact copies (same pixels at 32x32 under any
rotation/flip) get the same "group"; keep one file per group.

Writes data/index/raabin_files.csv and data/index/all_idb_files.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from bloodcells.config import load_config


def thumbnail(path: Path, size: int = 32) -> np.ndarray:
    """Standardized grayscale thumbnail (zero mean, unit variance)."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size)), float)
    return (a - a.mean()) / (a.std() + 1e-6)


def dihedral(a: np.ndarray) -> list[np.ndarray]:
    """The 8 rotations / flips of a square array."""
    return [v for k in range(4) for v in (np.rot90(a, k), np.rot90(a, k)[:, ::-1])]


def duplicate_groups(paths: list[Path], min_corr: float = 0.9) -> list[int]:
    """Group id per file: files whose thumbnails correlate > min_corr under some rotation/flip."""
    thumbs = [thumbnail(p) for p in paths]
    variants = [np.stack(dihedral(t)).reshape(8, -1) for t in thumbs]
    flat = np.stack([t.ravel() for t in thumbs])
    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for j, v in enumerate(variants):
        corr = (flat[:j] @ v.T).max(axis=1) / flat.shape[1]
        for i in np.flatnonzero(corr > min_corr):
            parent[find(int(i))] = find(j)
    roots = [find(i) for i in range(len(paths))]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots]


def raabin_index(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*.jpg")):
        rel = p.relative_to(root)
        stem = p.stem
        rows.append(
            {
                "path": str(p),
                "split": rel.parts[0],
                "cls": p.parent.name,
                "slide": stem.split("_")[0] if stem.count("_") == 2 else "",
                "photo": stem.rsplit("_", 1)[0] if stem.count("_") == 2 else "",
            }
        )
    df = pd.DataFrame(rows)
    with_size = [Image.open(p).size for p in df["path"]]
    df["width"], df["height"] = zip(*with_size)
    return df


def all_idb_index(root: Path) -> pd.DataFrame:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".tif", ".tiff", ".png"})
    df = pd.DataFrame({"path": [str(p) for p in paths], "folder": [p.parent.name for p in paths]})
    df["name"] = [p.name for p in paths]
    sizes = [Image.open(p).size for p in paths]
    df["width"], df["height"] = zip(*sizes)
    # Original ALL-IDB naming: ImXXX_1 = from a patient with ALL, ImXXX_0 = healthy.
    df["suffix"] = df["name"].str.extract(r"_(\d+)\.")[0]
    df["kind"] = np.where(df[["width", "height"]].max(axis=1) <= 300, "single-cell crop", "full smear")
    df["group"] = duplicate_groups(paths)
    df["keep"] = ~df.duplicated("group")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = cfg["paths"]["index"]
    out.mkdir(parents=True, exist_ok=True)

    root = cfg["paths"]["raabin"]
    if root.is_dir():
        rb = raabin_index(root)
        rb.to_csv(out / "raabin_files.csv", index=False)
        print(f"Raabin-WBC: {len(rb)} crops")
        print(pd.crosstab(rb["cls"], rb["split"], margins=True).to_string())
        print("Sizes:", rb.groupby("split")[["width", "height"]].median().to_dict("index"))
        named = rb[rb["photo"] != ""]
        for key in ("photo", "slide"):
            n_split = named.groupby(key)["split"].nunique()
            shared = n_split[n_split > 1].index
            print(
                f"{key}s in several splits: {len(shared)} of {named[key].nunique()} "
                f"({named[key].isin(shared).sum()} crops)"
            )
        print("Slides per class:", named.groupby("cls")["slide"].nunique().to_dict())
    else:
        print("Raabin-WBC: not downloaded")

    root = cfg["paths"]["all_idb"]
    if root.is_dir():
        ai = all_idb_index(root)
        ai.to_csv(out / "all_idb_files.csv", index=False)
        print(f"\nALL-IDB mirror: {len(ai)} files, {ai['keep'].sum()} distinct images")
        summary = ai.groupby("folder").agg(
            files=("name", "size"),
            distinct=("keep", "sum"),
            size=("width", lambda s: f"{s.mode()[0]} x {ai.loc[s.index, 'height'].mode()[0]}"),
            kind=("kind", "first"),
            suffix_0=("suffix", lambda s: int((s == "0").sum())),
            suffix_1=("suffix", lambda s: int((s == "1").sum())),
        )
        print(summary.to_string())
    else:
        print("\nALL-IDB: not downloaded")
    print(f"\nIndexes written to {out}")


if __name__ == "__main__":
    main()
