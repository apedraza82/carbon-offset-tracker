"""Download and update registry data from Berkeley Carbon Trading Project.

The Berkeley VCMP provides bulk downloads of offset registry data.
This module downloads new data, diffs against previous versions,
and returns only new retirements for processing.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.utils import load_config

# Berkeley data URLs (updated periodically)
BERKELEY_DOWNLOAD_URLS = {
    "verra": "https://gspp.berkeley.edu/assets/uploads/research/pdfs/verra-offsets-database.xlsx",
    "gold": "https://gspp.berkeley.edu/assets/uploads/research/pdfs/gold-standard-offsets-database.xlsx",
    "acr": "https://gspp.berkeley.edu/assets/uploads/research/pdfs/acr-offsets-database.xlsx",
    "car": "https://gspp.berkeley.edu/assets/uploads/research/pdfs/car-offsets-database.xlsx",
}

RAW_DIR = Path("data/raw")
PREV_DIR = Path("data/previous")


def download_file(url: str, dest: Path, timeout: int = 300) -> bool:
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(dest, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except Exception as e:
        print(f"  Download failed for {url}: {e}")
        return False


def file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all_registries(urls: dict | None = None) -> dict[str, Path]:
    """Download all registry files from Berkeley."""
    if urls is None:
        urls = BERKELEY_DOWNLOAD_URLS

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = {}

    for registry, url in urls.items():
        dest = RAW_DIR / f"{registry}_latest.xlsx"
        print(f"Downloading {registry}...")
        if download_file(url, dest):
            downloaded[registry] = dest

    return downloaded


def diff_retirements(new_file: Path, prev_file: Path, registry: str) -> pd.DataFrame:
    """Find new retirements by comparing with previous version.

    Uses serial numbers as unique identifiers for retirements.
    """
    serial_col_map = {
        "verra": "Serial Number",
        "gold": "Serial Number",
        "acr": "Credit Serial Numbers",
        "car": "Offset Credit Serial Numbers",
    }

    serial_col = serial_col_map.get(registry, "Serial Number")

    new_df = pd.read_excel(new_file)
    if not prev_file.exists():
        print(f"  No previous file for {registry} — all {len(new_df):,} rows are new")
        return new_df

    prev_df = pd.read_excel(prev_file)

    # Find rows with serial numbers not in previous
    prev_serials = set(prev_df[serial_col].dropna().astype(str))
    new_mask = ~new_df[serial_col].astype(str).isin(prev_serials)
    new_rows = new_df[new_mask]

    print(f"  [{registry.upper()}] {len(new_rows):,} new rows (of {len(new_df):,} total)")
    return new_rows


def run_download_pipeline(config: dict | None = None) -> dict:
    """Full download pipeline: download, diff, return new retirements."""
    if config is None:
        config = load_config()

    print("=== Downloading registry data ===")
    downloaded = download_all_registries()

    print("\n=== Identifying new retirements ===")
    new_retirements = {}
    for registry, new_path in downloaded.items():
        prev_path = PREV_DIR / f"{registry}_previous.xlsx"
        new_rows = diff_retirements(new_path, prev_path, registry)
        if len(new_rows) > 0:
            new_retirements[registry] = new_rows

    # Archive current as previous for next run
    PREV_DIR.mkdir(parents=True, exist_ok=True)
    for registry, new_path in downloaded.items():
        prev_path = PREV_DIR / f"{registry}_previous.xlsx"
        import shutil
        shutil.copy2(new_path, prev_path)

    # Save download metadata
    meta = {
        "download_date": datetime.now().isoformat(),
        "registries": {
            reg: {"rows": len(df), "file": str(downloaded.get(reg, ""))}
            for reg, df in new_retirements.items()
        },
    }
    meta_path = RAW_DIR / "download_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_new = sum(len(df) for df in new_retirements.values())
    print(f"\n  Total new retirements: {total_new:,}")

    return new_retirements


if __name__ == "__main__":
    run_download_pipeline()
