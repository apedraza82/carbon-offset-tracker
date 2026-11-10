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

# Berkeley VROD raw registry file (single file with all registries as separate sheets)
# URL pattern: VROD-registry-files--YYYY-MM.xlsx
# The page at https://gspp.berkeley.edu/berkeley-carbon-trading-project/offsets-database
# lists the latest version. Update the date suffix for newer releases.
BERKELEY_RAW_REGISTRY_URL = (
    "https://gspp.berkeley.edu/assets/uploads/page/VROD-registry-files--{version}.xlsx"
)
BERKELEY_LATEST_VERSION = "2026-08"

# Verra is pinned to an older VROD because Berkeley dropped the rich
# 'Retirement Details' free-text field in VROD 2026-08. Without that field
# the Verra beneficiary parse rate drops from ~73% to ~55% of rows. The
# 2026-04 Verra file still has the old structure and parses richly.
# Reverse once Berkeley restores a usable fallback field.
BERKELEY_PINNED_VERSIONS = {
    "verra": "2026-04",
}

# Sheet-name map for the LATEST VROD (2026-08) for the non-pinned registries.
# Verra sheet name is handled in _verra_sheet_name() because it depends on version.
BERKELEY_SHEETS = {
    "Gold Retirements": "gold",
    "ACR Retirements":  "acr",
    "CAR Retirements":  "car",
}


def _verra_sheet_name(version: str) -> str:
    """Return the Verra retirements sheet name for the given VROD version.

    Berkeley renamed 'Verra VCUS' to 'Verra VCS Retirements' starting in 2026-08.
    """
    try:
        y, m = (int(x) for x in version.split("-"))
    except ValueError:
        return "Verra VCUS"
    return "Verra VCS Retirements" if (y, m) >= (2026, 8) else "Verra VCUS"

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


def _download_vrod_if_needed(version: str) -> Path:
    """Download the VROD xlsx for a given version if not already on disk; return path."""
    url = BERKELEY_RAW_REGISTRY_URL.format(version=version)
    dest = RAW_DIR / f"VROD-registry-files--{version}.xlsx"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading Berkeley VROD v{version}...")
        if not download_file(url, dest):
            raise RuntimeError(f"Failed to download {url}")
    else:
        print(f"Using cached VROD v{version}: {dest.name}")
    return dest


def download_berkeley_registry(version: str | None = None) -> dict[str, pd.DataFrame]:
    """Download Berkeley VROD raw registry file(s) and extract retirement sheets.

    Non-pinned registries (gold, acr, car) are loaded from the latest VROD
    (`version` or BERKELEY_LATEST_VERSION). Pinned registries (see
    BERKELEY_PINNED_VERSIONS) are loaded from their pinned version instead.
    """
    if version is None:
        version = BERKELEY_LATEST_VERSION

    # Pre-fetch all VROD versions we need (dedup to avoid re-downloading)
    needed_versions = {version} | set(BERKELEY_PINNED_VERSIONS.values())
    paths = {v: _download_vrod_if_needed(v) for v in needed_versions}

    print("Extracting retirement sheets...")
    registry_data = {}

    # Latest-version registries
    for sheet_name, registry_key in BERKELEY_SHEETS.items():
        try:
            df = pd.read_excel(paths[version], sheet_name=sheet_name)
            registry_data[registry_key] = df
            print(f"  [{registry_key.upper()}] {len(df):,} rows (VROD {version}, '{sheet_name}')")
        except Exception as e:
            print(f"  [{registry_key.upper()}] Error: {e}")

    # Verra — pinned version
    v_ver = BERKELEY_PINNED_VERSIONS.get("verra", version)
    v_sheet = _verra_sheet_name(v_ver)
    try:
        df = pd.read_excel(paths[v_ver], sheet_name=v_sheet)
        # Filter to retirements only if it's a mixed-transaction sheet
        if "Retirement/Cancellation Date" in df.columns:
            df = df[df["Retirement/Cancellation Date"].notna()]
        registry_data["verra"] = df
        print(f"  [VERRA] {len(df):,} rows (VROD {v_ver} pinned, '{v_sheet}')")
    except Exception as e:
        print(f"  [VERRA] Error: {e}")

    return registry_data


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


def run_download_pipeline(config: dict | None = None, version: str | None = None) -> dict:
    """Full download pipeline: download Berkeley VROD, return all retirements."""
    if config is None:
        config = load_config()

    print("=== Downloading registry data from Berkeley ===")
    registry_data = download_berkeley_registry(version)

    # Save download metadata
    meta = {
        "download_date": datetime.now().isoformat(),
        "version": version or BERKELEY_LATEST_VERSION,
        "registries": {
            reg: {"rows": len(df)}
            for reg, df in registry_data.items()
        },
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RAW_DIR / "download_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(len(df) for df in registry_data.values())
    print(f"\n  Total retirements: {total:,}")

    return registry_data


if __name__ == "__main__":
    run_download_pipeline()
