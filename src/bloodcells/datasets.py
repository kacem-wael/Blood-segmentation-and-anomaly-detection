"""Dataset indexes: BCCD (main), Raabin-WBC and ALL-IDB (optional)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import color, transform

from .io_utils import CLASSES, labelme_reviewed, load_image, parse_voc, read_id_list

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BCCD_SPLITS = ("train", "val", "test")


def bccd_dirs(bccd_root: str | Path) -> dict[str, Path]:
    """Folders of the BCCD repo (``bccd_root`` is the cloned BCCD_Dataset folder)."""
    base = Path(bccd_root) / "BCCD"
    return {
        "images": base / "JPEGImages",
        "annotations": base / "Annotations",
        "splits": base / "ImageSets" / "Main",
    }


def bccd_official_split(bccd_root: str | Path) -> dict[str, list[str]]:
    """Image ids of the official train / val / test split."""
    split_dir = bccd_dirs(bccd_root)["splits"]
    return {name: read_id_list(split_dir / f"{name}.txt") for name in BCCD_SPLITS}


def bccd_index(bccd_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse every BCCD annotation.

    Returns three DataFrames:
    - images: one row per image (size, split, box counts per class, files present),
    - boxes: one row per valid box,
    - issues: one row per parsing problem (including images missing a file).
    """
    dirs = bccd_dirs(bccd_root)
    split_of = {image_id: name for name, ids in bccd_official_split(bccd_root).items() for image_id in ids}
    image_files = {p.stem: p for p in dirs["images"].iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    xml_files = {p.stem: p for p in dirs["annotations"].glob("*.xml")}

    images, boxes, issues = [], [], []
    for image_id in sorted(set(image_files) | set(xml_files) | set(split_of)):
        row = {
            "image_id": image_id,
            "split": split_of.get(image_id, "none"),
            "image_path": str(image_files[image_id]) if image_id in image_files else "",
            "has_image": image_id in image_files,
            "has_annotation": image_id in xml_files,
            "width": 0,
            "height": 0,
        }
        row.update({f"n_{c}": 0 for c in CLASSES})

        for missing, what in ((not row["has_image"], "image"), (not row["has_annotation"], "annotation")):
            if missing:
                issues.append({"image_id": image_id, "issue": f"missing {what} file"})

        if row["has_annotation"]:
            ann = parse_voc(xml_files[image_id])
            row["width"], row["height"] = ann.width, ann.height
            for b in ann.boxes:
                row[f"n_{b.cls}"] += 1
                boxes.append(
                    {
                        "image_id": image_id,
                        "split": row["split"],
                        "cls": b.cls,
                        "xmin": b.xmin,
                        "ymin": b.ymin,
                        "xmax": b.xmax,
                        "ymax": b.ymax,
                        "width": b.width,
                        "height": b.height,
                        "truncated": b.truncated,
                        "difficult": b.difficult,
                    }
                )
            issues.extend({"image_id": image_id, "issue": msg} for msg in ann.issues)
        images.append(row)

    return (
        pd.DataFrame(images),
        pd.DataFrame(boxes),
        pd.DataFrame(issues, columns=["image_id", "issue"]),
    )


def rbc_radius_from_boxes(boxes: pd.DataFrame) -> float:
    """Reference RBC radius (px): median of (w + h) / 4 over complete RBC boxes.

    Used later to validate the granulometry estimate of ``r`` (project.md, Phase 3).
    """
    rbc = boxes[(boxes["cls"] == "RBC") & ~boxes["truncated"]]
    return float(((rbc["width"] + rbc["height"]) / 4).median())


def rbc_radius_per_image(boxes: pd.DataFrame, min_boxes: int = 3) -> pd.Series:
    """Reference RBC radius (px) of each image, same definition as ``rbc_radius_from_boxes``.

    Duplicate boxes are ignored; images with fewer than ``min_boxes`` complete
    RBC boxes are left out (their median would be too noisy).
    """
    rbc = boxes[(boxes["cls"] == "RBC") & ~boxes["truncated"]]
    if "duplicate" in rbc:
        rbc = rbc[~rbc["duplicate"]]
    radius = (rbc["width"] + rbc["height"]) / 4
    grouped = radius.groupby(rbc["image_id"])
    return grouped.median()[grouped.size() >= min_boxes]


def _overlap(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping parts of a and b when b is shifted by (dy, dx) relative to a."""
    h, w = a.shape
    return (
        a[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)],
        b[max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)],
    )


def same_field_pairs(
    image_ids: list[str],
    paths: list[str],
    scale: int = 4,
    min_overlap: float = 0.3,
    min_ncc: float = 0.8,
) -> pd.DataFrame:
    """Find pairs of images showing the same microscope field (shifted frames).

    Every pair is registered by phase correlation on a downsampled grayscale
    image; the pair is kept when the overlapping areas are large enough and
    strongly correlated (normalized cross-correlation). On BCCD the scores are
    bimodal (same field > 0.9, different fields < 0.6), so the threshold is not
    critical.
    """
    imgs = []
    for p in paths:
        g = transform.downscale_local_mean(color.rgb2gray(load_image(p)), (scale, scale))
        imgs.append((g - g.mean()) / (g.std() + 1e-9))
    imgs = np.array(imgs)
    n, h, w = imgs.shape
    spectra = np.fft.fft2(imgs)

    rows = []
    for i in range(n - 1):
        cross = spectra[i][None] * np.conj(spectra[i + 1 :])
        corr = np.real(np.fft.ifft2(cross / (np.abs(cross) + 1e-9)))
        for k, c in enumerate(corr):
            j = i + 1 + k
            dy, dx = np.unravel_index(int(c.argmax()), c.shape)
            dy, dx = (dy - h if dy > h // 2 else dy), (dx - w if dx > w // 2 else dx)
            a, b = _overlap(imgs[i], imgs[j], dy, dx)
            overlap = a.size / (h * w)
            if overlap < min_overlap:
                continue
            ncc = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            if ncc >= min_ncc:
                rows.append(
                    {
                        "image_a": image_ids[i],
                        "image_b": image_ids[j],
                        "dy": int(dy * scale),
                        "dx": int(dx * scale),
                        "overlap": round(overlap, 3),
                        "ncc": round(ncc, 4),
                    }
                )
    return pd.DataFrame(rows, columns=["image_a", "image_b", "dy", "dx", "overlap", "ncc"])


def field_groups(image_ids: list[str], pairs: pd.DataFrame) -> dict[str, str]:
    """Group images linked by same-field pairs (connected components).

    Returns image_id -> group id; the group id is the first image id of the group,
    so an image with no duplicate is its own group.
    """
    parent = {i: i for i in image_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["image_a"], pairs["image_b"]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in image_ids}


def flag_duplicate_boxes(boxes: pd.DataFrame, min_iou: float = 0.7) -> pd.Series:
    """True for boxes that repeat an earlier box of the same class and image (IoU >= min_iou).

    BCCD contains stacks of near-identical boxes on some images (annotation
    errors); the first box of each stack is kept.
    """
    flags = pd.Series(False, index=boxes.index)
    for _, b in boxes.groupby(["image_id", "cls"]):
        if len(b) < 2:
            continue
        x1 = np.maximum(b["xmin"].values[:, None], b["xmin"].values[None])
        y1 = np.maximum(b["ymin"].values[:, None], b["ymin"].values[None])
        x2 = np.minimum(b["xmax"].values[:, None], b["xmax"].values[None])
        y2 = np.minimum(b["ymax"].values[:, None], b["ymax"].values[None])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (b["width"] * b["height"]).values
        iou = inter / (area[:, None] + area[None] - inter)
        dup = (np.tril(iou, -1) >= min_iou).any(axis=1)
        flags.loc[b.index[dup]] = True
    return flags


def resolve_field_leakage(split: pd.Series, group: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Remove same-field leakage between splits while keeping the official test set intact.

    ``split`` and ``group`` are indexed by image id. Rules, per field group:
    - the group contains a test image: its train/val images become ``excluded_leakage``;
    - the group contains train and val images only: its val images move to train.

    Returns the new split Series and a log of the changes.
    """
    new = split.copy()
    log = []
    for g, members in split.groupby(group):
        kinds = set(members)
        if len(kinds) < 2:
            continue
        for image_id, s in members.items():
            if "test" in kinds and s != "test":
                new[image_id] = "excluded_leakage"
            elif "test" not in kinds and s == "val" and "train" in kinds:
                new[image_id] = "train"
            if new[image_id] != s:
                log.append({"image_id": image_id, "field_group": g, "from": s, "to": new[image_id]})
    return new, pd.DataFrame(log, columns=["image_id", "field_group", "from", "to"])


# ---- Loading the prepared BCCD index (used by the evaluation scripts) ------


@dataclass
class Bccd:
    """BCCD index, boxes and splits as written by ``build_index.py`` / ``make_splits.py``."""

    images: pd.DataFrame  # indexed by image_id: image_path, no_rbc_labels, ...
    boxes: pd.DataFrame  # one row per valid, non-duplicate box
    splits: dict

    def ids(self, split: str) -> list[str]:
        """Image ids of ``train``, ``val``, ``test`` or ``trainval``."""
        names = ["train", "val"] if split == "trainval" else [split]
        return sorted(i for s in names for i in self.splits[s])

    def split_of(self, image_id: str) -> str:
        return next((s for s in ("train", "val", "test") if image_id in self.splits[s]), "")

    def group_of(self, image_id: str) -> str:
        """Field group (shifted frames of one microscope field share it): the GroupKFold key."""
        return self.splits.get("field_groups", {}).get(image_id, image_id)

    def boxes_of(self, image_id: str) -> pd.DataFrame:
        return self._by_image.get(image_id, self.boxes.iloc[:0])

    mask_dir: Path | None = None  # data/masks_manual: <image_id>.json files are found automatically

    def manual_masks(self) -> list[str]:
        """Images with a finished manual mask: a reviewed LabelMe file ``mask_dir/<image_id>.json``.

        Drafts (``reviewed`` flag unticked, see ``io_utils.labelme_reviewed``) are
        ignored, so an uncorrected draft never becomes ground truth.
        """
        if self.mask_dir is None or not Path(self.mask_dir).exists():
            return []
        return sorted(
            p.stem
            for p in Path(self.mask_dir).glob("*.json")
            if p.stem in self.images.index and labelme_reviewed(p)
        )

    def __post_init__(self) -> None:
        self._by_image = {k: g for k, g in self.boxes.groupby("image_id")}


def load_bccd(cfg: dict) -> Bccd:
    """Read data/index/bccd_images.csv, bccd_boxes.csv (duplicates dropped) and data/splits.json."""
    paths = cfg["paths"]
    images = pd.read_csv(paths["index"] / "bccd_images.csv").set_index("image_id")
    boxes = pd.read_csv(paths["index"] / "bccd_boxes.csv")
    boxes = boxes[~boxes["duplicate"]].reset_index(drop=True)
    splits = json.loads(Path(paths["splits"]).read_text(encoding="utf-8"))
    return Bccd(images, boxes, splits, Path(paths["masks_manual"]))
