"""Regression test (project.md section 12): counts on 5 fixed BCCD images stay within tolerance.

Expected counts are in tests/fixtures/regression_counts.json (written by
``python tests/test_regression.py --update`` after an intended change).
Skipped when the BCCD data is not downloaded (e.g. in CI).
"""

import json
import sys
from pathlib import Path

import pytest

from bloodcells.config import load_config
from bloodcells.io_utils import load_image
from bloodcells.pipeline import run

CFG = load_config()
FIXTURE = Path(__file__).parent / "fixtures" / "regression_counts.json"
IMAGES = ["BloodImage_00014", "BloodImage_00035", "BloodImage_00053", "BloodImage_00099", "BloodImage_00152"]
JPEG = CFG["paths"]["bccd"] / "BCCD" / "JPEGImages"


def counts(image_id: str) -> dict:
    res = run(load_image(JPEG / f"{image_id}.jpg"), CFG)
    return {"r": round(res.r, 1), **res.counts}


@pytest.mark.skipif(not JPEG.exists() or not FIXTURE.exists(), reason="BCCD data or fixture missing")
@pytest.mark.parametrize("image_id", IMAGES)
def test_counts_are_stable(image_id):
    expected = json.loads(FIXTURE.read_text())[image_id]
    got = counts(image_id)
    assert abs(got["r"] - expected["r"]) <= 0.05 * expected["r"]
    assert got["WBC"] == expected["WBC"]
    assert abs(got["Platelet"] - expected["Platelet"]) <= 1
    assert abs(got["RBC"] - expected["RBC"]) <= max(1, round(0.1 * expected["RBC"]))


if __name__ == "__main__" and "--update" in sys.argv:
    FIXTURE.write_text(json.dumps({i: counts(i) for i in IMAGES}, indent=2) + "\n")
    print(f"written {FIXTURE}")
