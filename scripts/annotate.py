"""Open the next manual-mask image in LabelMe, then check it (one command per image).

    python scripts/annotate.py                  # = next: first image not yet reviewed
    python scripts/annotate.py next --who wael  # first of your 9 images
    python scripts/annotate.py BloodImage_00109
    python scripts/annotate.py status           # what is done / left

On first use of an image the draft (``prepare_masks.py``) is copied to
data/masks_manual/<image_id>.json, then LabelMe opens that file with only the
four allowed labels and the image flag "reviewed". Correct the polygons, tick
**reviewed** (Flags panel, top right) when the image is finished, save, close
LabelMe: the annotation is then checked (``check_masks.py``) and from then on
used by every evaluation script.
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
import shutil
import subprocess
import sys

from bloodcells.config import PROJECT_ROOT, load_config
from bloodcells.datasets import load_bccd
from bloodcells.io_utils import labelme_reviewed

LABELS = "RBC,WBC,Platelet,nucleus"
# Who annotates what (docs/manual_tasks.md): 5 train/val + 4 test images each.
ASSIGNMENT = {
    "wael": [
        "BloodImage_00109",
        "BloodImage_00168",
        "BloodImage_00195",
        "BloodImage_00200",
        "BloodImage_00206",
        "BloodImage_00021",
        "BloodImage_00205",
        "BloodImage_00233",
        "BloodImage_00293",
    ],
    "talel": [
        "BloodImage_00207",
        "BloodImage_00214",
        "BloodImage_00235",
        "BloodImage_00264",
        "BloodImage_00272",
        "BloodImage_00320",
        "BloodImage_00341",
        "BloodImage_00366",
        "BloodImage_00407",
    ],
}


def labelme_exe() -> str:
    """LabelMe of the project environment .venv-labelme, else the one on the PATH."""
    for exe in (
        PROJECT_ROOT / ".venv-labelme" / "Scripts" / "labelme.exe",
        PROJECT_ROOT / ".venv-labelme" / "bin" / "labelme",
    ):
        if exe.exists():
            return str(exe)
    found = shutil.which("labelme")
    if found:
        return found
    sys.exit("LabelMe not found: create .venv-labelme (see docs/manual_tasks.md, step 1.2)")


def status(data, cfg) -> list[str]:
    """Print the state of the 18 images; return the ids not reviewed yet (train/val first)."""
    mask_dir = cfg["paths"]["masks_manual"]
    todo = []
    for split, ids in data.splits["manual_mask_candidates"].items():
        for i in ids:
            f = mask_dir / f"{i}.json"
            state = (
                "reviewed"
                if f.exists() and labelme_reviewed(f)
                else ("in progress" if f.exists() else "to do")
            )
            who = next(k for k, v in ASSIGNMENT.items() if i in v)
            print(f"  {split:8s} {i}  {who:6s} {state}")
            if state != "reviewed":
                todo.append(i)
    print(f"{18 - len(todo)} / 18 reviewed")
    return todo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", nargs="?", default="next", help="image id, 'next' (default) or 'status'")
    parser.add_argument("--who", choices=list(ASSIGNMENT), help="with 'next': only your images")
    args = parser.parse_args()
    cfg = load_config()
    data = load_bccd(cfg)
    todo = status(data, cfg)
    if args.image == "status":
        return
    if args.image == "next":
        todo = [i for i in todo if args.who is None or i in ASSIGNMENT[args.who]]
        if not todo:
            print("Nothing left to annotate.")
            return
        image_id = todo[0]
    else:
        image_id = args.image if args.image.startswith("BloodImage_") else f"BloodImage_{args.image}"
    mask_dir = cfg["paths"]["masks_manual"]
    target = mask_dir / f"{image_id}.json"
    draft = mask_dir / "drafts" / f"{image_id}.json"
    if not target.exists():
        if not draft.exists():
            sys.exit(f"No draft for {image_id}: run python scripts/prepare_masks.py first")
        shutil.copy(draft, target)
    print(f"\nOpening {image_id} in LabelMe. Tick 'reviewed' when the image is finished, save, close.")
    subprocess.run(
        [
            labelme_exe(),
            str(target),
            "--labels",
            LABELS,
            "--validate-label",
            "exact",
            "--flags",
            "reviewed",
            "--output",
            str(mask_dir),
            "--nosortlabels",
        ],
        check=False,
    )
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_masks.py"), image_id], check=False)
    if not labelme_reviewed(target):
        print(f"\n{image_id} is not marked 'reviewed' yet: it is not used by the evaluation.")


if __name__ == "__main__":
    main()
