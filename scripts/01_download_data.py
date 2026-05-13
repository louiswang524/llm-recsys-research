#!/usr/bin/env python3
"""Download raw datasets (MovieLens, Amazon Reviews 2023)."""

import zipfile
import argparse
import requests
from pathlib import Path
from tqdm import tqdm


def download(url: str, dest: Path, desc: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  already exists: {dest}")
        return
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_movielens(data_dir: Path, version: str = "ml-1m") -> None:
    url = f"https://files.grouplens.org/datasets/movielens/{version}.zip"
    name = "movielens_1m" if "1m" in version else "movielens_20m"
    zip_path = data_dir / f"{version}.zip"
    download(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir / name)
    print(f"MovieLens {version} → {data_dir / name}")


def download_amazon(data_dir: Path, category: str) -> None:
    # Amazon Reviews 2023 (McAuley Lab, UCSD)
    base = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw"
    dest = data_dir / f"amazon_{category.lower()}"
    dest.mkdir(parents=True, exist_ok=True)
    for fname, subdir in [
        (f"{category}.jsonl.gz", "review_categories"),
        (f"meta_{category}.jsonl.gz", "meta_categories"),
    ]:
        download(f"{base}/{subdir}/{fname}", dest / fname)
    print(f"Amazon {category} → {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--movielens", choices=["ml-1m", "ml-20m", "none"], default="ml-1m")
    parser.add_argument("--amazon-category", default="",
                        help="e.g. All_Beauty, Video_Games, Books (empty = skip)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.movielens != "none":
        download_movielens(data_dir, args.movielens)
    if args.amazon_category:
        download_amazon(data_dir, args.amazon_category)


if __name__ == "__main__":
    main()
