from pathlib import Path

import numpy as np
from skimage import io

from bloodcells.datasets import bccd_index, rbc_radius_from_boxes
from bloodcells.io_utils import load_image, parse_voc, read_id_list

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_voc_validates_boxes():
    ann = parse_voc(FIXTURES / "sample.xml")
    assert (ann.width, ann.height) == (640, 480)
    assert [b.cls for b in ann.boxes] == ["WBC", "RBC", "Platelet", "RBC"]

    wbc, rbc, platelet, clipped = ann.boxes
    assert (wbc.xmin, wbc.ymin, wbc.xmax, wbc.ymax) == (260, 177, 491, 376)
    assert rbc.truncated and not wbc.truncated
    assert (platelet.width, platelet.height) == (30, 30)  # " Platelets " name normalized
    assert (clipped.xmax, clipped.ymax) == (640, 480)

    joined = " ".join(ann.issues)
    assert "unknown class 'Monocyte'" in joined
    assert "empty or inverted" in joined
    assert "clipped" in joined


def test_read_id_list_handles_crlf(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"BloodImage_00007\r\nBloodImage_00011\r\n\r\n")
    assert read_id_list(f) == ["BloodImage_00007", "BloodImage_00011"]


def test_load_image_returns_rgb_uint8(tmp_path):
    gray = (np.arange(64 * 64) % 256).astype(np.uint8).reshape(64, 64)
    io.imsave(tmp_path / "g.png", gray, check_contrast=False)
    rgba = np.dstack([gray, gray, gray, np.full_like(gray, 255)])
    io.imsave(tmp_path / "a.png", rgba, check_contrast=False)
    for name in ("g.png", "a.png"):
        img = load_image(tmp_path / name)
        assert img.shape == (64, 64, 3) and img.dtype == np.uint8


def _mini_bccd(root: Path) -> None:
    base = root / "BCCD"
    for sub in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (base / sub).mkdir(parents=True)
    xml = (FIXTURES / "sample.xml").read_text(encoding="utf-8")
    for image_id in ("img_a", "img_b"):
        io.imsave(
            base / "JPEGImages" / f"{image_id}.jpg", np.zeros((8, 8, 3), np.uint8), check_contrast=False
        )
        (base / "Annotations" / f"{image_id}.xml").write_text(xml, encoding="utf-8")
    # img_c is listed in the split but has no files
    (base / "ImageSets/Main/train.txt").write_text("img_a\r\n", encoding="utf-8")
    (base / "ImageSets/Main/val.txt").write_text("img_c\r\n", encoding="utf-8")
    (base / "ImageSets/Main/test.txt").write_text("img_b\r\n", encoding="utf-8")


def test_bccd_index(tmp_path):
    _mini_bccd(tmp_path)
    images, boxes, issues = bccd_index(tmp_path)

    assert list(images["image_id"]) == ["img_a", "img_b", "img_c"]
    assert list(images["split"]) == ["train", "test", "val"]
    a = images.set_index("image_id").loc["img_a"]
    assert (a["n_RBC"], a["n_WBC"], a["n_Platelet"]) == (2, 1, 1)
    assert len(boxes) == 8
    assert {"missing image file", "missing annotation file"} <= set(issues["issue"])
    # only the non-truncated RBC box counts; it was clipped to 80 x 80: (80 + 80) / 4
    assert rbc_radius_from_boxes(boxes) == 40.0


def test_resolve_field_leakage():
    import pandas as pd

    from bloodcells.datasets import field_groups, resolve_field_leakage

    ids = ["a", "b", "c", "d", "e", "f"]
    split = pd.Series(["train", "test", "val", "train", "val", "val"], index=ids)
    pairs = pd.DataFrame({"image_a": ["a", "c", "e"], "image_b": ["b", "d", "f"]})
    group = pd.Series(field_groups(ids, pairs))
    assert group["b"] == "a" and group["d"] == "c"

    new, log = resolve_field_leakage(split, group)
    assert new["a"] == "excluded_leakage"  # shares a field with test image b
    assert new["b"] == "test"  # test set never changes
    assert new["c"] == "train"  # val image sharing a field with train
    assert (new["e"], new["f"]) == ("val", "val")  # same split: nothing to do
    assert len(log) == 2


def test_flag_duplicate_boxes():
    import pandas as pd

    from bloodcells.datasets import flag_duplicate_boxes

    b = pd.DataFrame(
        {
            "image_id": ["x"] * 4,
            "cls": ["RBC", "RBC", "RBC", "WBC"],
            "xmin": [0, 2, 200, 0],
            "ymin": [0, 1, 200, 0],
            "xmax": [100, 101, 300, 100],
            "ymax": [100, 100, 300, 100],
        }
    )
    b["width"], b["height"] = b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]
    assert list(flag_duplicate_boxes(b)) == [False, True, False, False]


def test_safe_extract_blocks_path_traversal(tmp_path):
    import importlib.util
    import zipfile

    import pytest

    spec = importlib.util.spec_from_file_location(
        "download_data", Path(__file__).parents[1] / "scripts" / "download_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("Train/neutrophil/a.jpg", b"x")
    assert mod.safe_extract(good, tmp_path / "out") == 1
    assert (tmp_path / "out" / "Train" / "neutrophil" / "a.jpg").is_file()

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", b"x")
    with pytest.raises(SystemExit):
        mod.safe_extract(bad, tmp_path / "out2")
    assert not (tmp_path / "evil.txt").exists()
