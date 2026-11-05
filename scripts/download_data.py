"""Download the datasets into data/raw/.

BCCD (main dataset) is cloned from GitHub and needs no account.
Raabin-WBC and the ALL-IDB mirror come from Kaggle, in one of two ways:

1. Kaggle API token, looked up in this order:
   - kaggle.json in the project folder (ignored by git),
   - ~/.kaggle/kaggle.json (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json),
   - environment variables KAGGLE_USERNAME + KAGGLE_KEY, or KAGGLE_API_TOKEN.
2. A zip downloaded by hand from the dataset page ("Download" button):
   put it in data/downloads/ named raabin.zip or all_idb.zip, or pass --zip.

Usage:
    python scripts/download_data.py                       # all datasets
    python scripts/download_data.py --only bccd
    python scripts/download_data.py --only raabin --zip C:/Users/me/Downloads/archive.zip
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from bloodcells.config import PROJECT_ROOT, load_config

KAGGLE_DATASETS = {
    # name: (config key for the slug, config key for the target folder, note)
    "raabin": ("raabin_kaggle", "raabin", ""),
    "all_idb": (
        "all_idb_kaggle",
        "all_idb",
        "Third-party mirror. ALL-IDB images must not be redistributed; cite Labati et al., ICIP 2011.",
    ),
}
DOWNLOADS = PROJECT_ROOT / "data" / "downloads"


def download_bccd(cfg: dict) -> None:
    target = cfg["paths"]["bccd"]
    if (target / "BCCD" / "Annotations").is_dir():
        print(f"[bccd] already present: {target}")
        return
    if shutil.which("git") is None:
        raise SystemExit("[bccd] git is not installed; install it or download the repo as a zip.")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bccd] cloning {cfg['datasets']['bccd_repo']} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", cfg["datasets"]["bccd_repo"], str(target)],
        check=True,
    )
    print(f"[bccd] done: {target}")


def kaggle_env() -> dict | None:
    """Environment for the kaggle CLI, or None when no credentials are found."""
    env = dict(os.environ)
    if (PROJECT_ROOT / "kaggle.json").is_file():
        env["KAGGLE_CONFIG_DIR"] = str(PROJECT_ROOT)
        return env
    if env.get("KAGGLE_API_TOKEN") or (env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY")):
        return env
    config_dir = Path(env.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    if (config_dir / "kaggle.json").is_file() or (config_dir / "access_token").is_file():
        return env
    return None


def safe_extract(archive: Path, target: Path) -> int:
    """Extract a zip into target, refusing members that would land outside it."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for m in members:
            dest = (target / m.filename).resolve()
            if dest != target and target not in dest.parents:
                raise SystemExit(f"Unsafe path in {archive.name}: {m.filename}")
        target.mkdir(parents=True, exist_ok=True)
        zf.extractall(target)
    return len(members)


def import_zip(name: str, archive: Path, target: Path) -> None:
    if not archive.is_file():
        raise SystemExit(f"[{name}] zip not found: {archive}")
    print(f"[{name}] extracting {archive} -> {target} ...")
    n = safe_extract(archive, target)
    print(f"[{name}] done: {n} files")


def download_kaggle(name: str, cfg: dict, zip_path: Path | None = None) -> None:
    slug_key, target_key, note = KAGGLE_DATASETS[name]
    slug, target = cfg["datasets"][slug_key], cfg["paths"][target_key]
    if note:
        print(f"[{name}] {note}")

    # 1. explicit or conventional zip
    zip_path = zip_path or (DOWNLOADS / f"{target_key}.zip")
    if zip_path.is_file():
        import_zip(name, zip_path, target)
        return
    if target.is_dir() and any(target.iterdir()):
        print(f"[{name}] already present: {target}")
        return

    # 2. Kaggle API
    env = kaggle_env()
    if shutil.which("kaggle") is None or env is None:
        print(
            f"[{name}] skipped: no Kaggle credentials and no zip.\n"
            f"         Option A: create an API token at https://www.kaggle.com/settings and save\n"
            f"                   kaggle.json in {PROJECT_ROOT} (or in {Path.home() / '.kaggle'}).\n"
            f"         Option B: download https://www.kaggle.com/datasets/{slug} by hand and save it as\n"
            f"                   {DOWNLOADS / (target_key + '.zip')}\n"
            f"         Then re-run: make download-{name.replace('_', '')}"
        )
        return
    target.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] downloading {slug} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", slug, "-p", str(target), "--unzip"], check=True, env=env
    )
    print(f"[{name}] done: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", choices=["bccd", "raabin", "all_idb"], help="download one dataset")
    parser.add_argument("--zip", type=Path, default=None, help="manually downloaded zip (with --only)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()
    if args.zip and args.only not in KAGGLE_DATASETS:
        parser.error("--zip needs --only raabin or --only all_idb")
    cfg = load_config(args.config)

    if args.only in (None, "bccd"):
        download_bccd(cfg)
    for name in KAGGLE_DATASETS:
        if args.only in (None, name):
            download_kaggle(name, cfg, args.zip)


if __name__ == "__main__":
    main()
