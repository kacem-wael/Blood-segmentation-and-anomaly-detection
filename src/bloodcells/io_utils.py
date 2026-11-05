"""Image loading and Pascal VOC annotation parsing."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from skimage import io

# Canonical class names; keys are lower-cased raw names found in annotation files.
CLASS_ALIASES = {
    "rbc": "RBC",
    "wbc": "WBC",
    "platelet": "Platelet",
    "platelets": "Platelet",
}
CLASSES = ("RBC", "WBC", "Platelet")


@dataclass
class Box:
    """One annotated object. Coordinates are VOC pixels (inclusive, 1-based)."""

    cls: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    truncated: bool = False
    difficult: bool = False

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin


@dataclass
class VocAnnotation:
    filename: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _int(node: ET.Element | None, tag: str, default: int = 0) -> int:
    """Read an integer child; VOC files sometimes store floats like '12.0'."""
    child = node.find(tag) if node is not None else None
    if child is None or child.text is None or not child.text.strip():
        return default
    return int(round(float(child.text.strip())))


def parse_voc(path: str | Path) -> VocAnnotation:
    """Parse a Pascal VOC XML file and validate every box.

    Invalid boxes (unknown class, empty or inverted) are dropped; boxes that
    exceed the image are clipped. Every correction is recorded in ``issues``.
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    size = root.find("size")
    ann = VocAnnotation(
        filename=(root.findtext("filename") or f"{path.stem}.jpg").strip(),
        width=_int(size, "width"),
        height=_int(size, "height"),
    )

    for i, obj in enumerate(root.iter("object")):
        raw_name = (obj.findtext("name") or "").strip()
        cls = CLASS_ALIASES.get(raw_name.lower())
        if cls is None:
            ann.issues.append(f"object {i}: unknown class '{raw_name}', skipped")
            continue

        bb = obj.find("bndbox")
        if bb is None:
            ann.issues.append(f"object {i}: no bndbox, skipped")
            continue
        xmin, ymin = _int(bb, "xmin"), _int(bb, "ymin")
        xmax, ymax = _int(bb, "xmax"), _int(bb, "ymax")

        if ann.width and ann.height:
            clipped = (
                max(xmin, 0),
                max(ymin, 0),
                min(xmax, ann.width),
                min(ymax, ann.height),
            )
            if clipped != (xmin, ymin, xmax, ymax):
                ann.issues.append(f"object {i}: box clipped to image")
                xmin, ymin, xmax, ymax = clipped

        if xmax <= xmin or ymax <= ymin:
            ann.issues.append(f"object {i}: empty or inverted box, skipped")
            continue

        ann.boxes.append(
            Box(
                cls=cls,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                truncated=_int(obj, "truncated") == 1,
                difficult=_int(obj, "difficult") == 1,
            )
        )
    return ann


MIN_IMAGE_SIDE = 32  # px; smaller inputs cannot hold a single cell


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """RGB uint8 view of any image array (project.md 8.3: non-RGB, RGBA or tiny images).

    Grayscale is expanded to 3 channels, alpha is dropped, uint16 is divided
    by 257, floats in [0, 1] are scaled by 255 (other floats are min-max
    stretched). Raises ``ValueError`` for arrays that are not images or are
    smaller than ``MIN_IMAGE_SIDE`` pixels.
    """
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise ValueError(f"expected an RGB image, got an array of shape {img.shape}")
    img = img[..., :3]
    if min(img.shape[:2]) < MIN_IMAGE_SIDE:
        raise ValueError(f"image too small ({img.shape[1]}x{img.shape[0]} px, minimum {MIN_IMAGE_SIDE})")
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img // 257).astype(np.uint8)
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    x = img.astype(float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not (lo >= 0 and hi <= 1):
        x = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    return np.round(np.nan_to_num(x) * 255).astype(np.uint8)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as RGB uint8 (see ``ensure_rgb``)."""
    return ensure_rgb(io.imread(str(path)))


def read_id_list(path: str | Path) -> list[str]:
    """Read a VOC ImageSets file (one image id per line, CRLF tolerated)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


# ---- Manual masks (LabelMe JSON, project.md 5.1) ---------------------------

NUCLEUS_LABELS = {"nucleus", "wbc_nucleus", "noyau"}


@dataclass
class ManualMask:
    labels: np.ndarray  # one integer per annotated cell, 0 = background
    classes: dict[int, str]  # label -> RBC / WBC / Platelet
    nuclei: np.ndarray  # boolean mask of the WBC nucleus polygons


def labelme_reviewed(path: str | Path) -> bool:
    """True when a LabelMe file is a finished annotation.

    Drafts made by ``scripts/prepare_masks.py`` carry the image flag
    ``reviewed: false``; the annotator ticks it in LabelMe when the image is
    done. Files without that flag (drawn from scratch) count as finished.
    """
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return bool(data.get("flags", {}).get("reviewed", True))


def load_labelme(path: str | Path, shape: tuple[int, int] | None = None) -> ManualMask:
    """Rasterize a LabelMe annotation: one polygon per cell, ``nucleus`` polygons for WBC nuclei.

    Cell labels are matched case-insensitively with ``CLASS_ALIASES``; unknown
    labels raise ``ValueError`` (a typo would silently remove a cell).
    Polygons are drawn in file order, so where two cells overlap the later one wins.
    """
    import json

    from skimage.draw import polygon

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if shape is None:
        shape = (int(data["imageHeight"]), int(data["imageWidth"]))
    labels = np.zeros(shape, np.int32)
    nuclei = np.zeros(shape, bool)
    classes: dict[int, str] = {}
    for item in data.get("shapes", []):
        name = item["label"].strip().lower()
        pts = np.asarray(item["points"], float)
        if item.get("shape_type", "polygon") != "polygon" or len(pts) < 3:
            continue
        rr, cc = polygon(pts[:, 1], pts[:, 0], shape=shape)
        if name in NUCLEUS_LABELS:
            nuclei[rr, cc] = True
            continue
        if name not in CLASS_ALIASES:
            raise ValueError(f"{path}: unknown label '{item['label']}'")
        k = len(classes) + 1
        labels[rr, cc] = k
        classes[k] = CLASS_ALIASES[name]
    return ManualMask(labels, classes, nuclei)
