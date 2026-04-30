"""Download retirement/cancellation data from BioCarbon Registry (Global CarbonTrace).

BioCarbon Registry (biocarbonregistry.com) has rebranded to Global CarbonTrace
(globalcarbontrace.io). The platform is a Vue.js SPA backed by a Laravel API at
api.globalcarbontrace.io. Public data is accessible via a set of read-only API
endpoints authenticated with a static x-api-key header.

Discovered endpoints (all under https://api.globalcarbontrace.io):
  - /api/public/retreats          -- GHG retirements (paginated, ~10k records)
  - /api/public/initiatives       -- GHG projects (paginated, ~100 records)
  - /api/public/carbon-credits    -- GHG credit issuances (paginated)
  - /api/public/transferences     -- GHG credit transfers (paginated)
  - /api/public/retreats/relations -- filter options (holders, projects, to_names)
  - /api/biodiversity/retreats    -- Biodiversity retirements (currently 0)
  - /api/water/retreats           -- Water retirements (currently 0)
  - /api/ghg/projects/{id}        -- Project detail
  - /api/ghg/retreats/project/{id} -- Retirements by project
  - /api/impact-stats/get-stats   -- Aggregate statistics

Fields returned per retirement record:
  reason, to_name, nit_to_name, final_user, final_user_nit, destination,
  serial, created_at, initial_serial, final_serial, sold (= volume),
  market, passive_user, nit_passive_user, initiative_id, project_name,
  holder, holder_nit, data_visibility
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
API_BASE = "https://api.globalcarbontrace.io"
API_KEY = "SboCiHaHxtC2xRM92hpBjy1S2Y5La7IwjeB76z"

HEADERS = {
    "Accept": "application/json",
    "x-api-key": API_KEY,
}

# Max records the API returns per request (tested; 500 works)
MAX_PER_PAGE = 500

# Output directory (relative to repo root)
RAW_DIR = Path("data/raw/biocarbon")

# Endpoints by program and data category
PROGRAMS = {
    "gei": {
        "retreats":      "/api/public/retreats",
        "initiatives":   "/api/public/initiatives",
        "carbon_credits": "/api/public/carbon-credits",
        "transferences": "/api/public/transferences",
        "relations":     "/api/public/retreats/relations",
    },
    "biodiversity": {
        "retreats":      "/api/biodiversity/retreats",
        "initiatives":   "/api/biodiversity/projects",
        "carbon_credits": "/api/biodiversity/carbon-credits",
        "transferences": "/api/biodiversity/transferences",
        "relations":     "/api/biodiversity/retreats/relations",
    },
    "water": {
        "retreats":      "/api/water/retreats",
        "initiatives":   "/api/water/projects",
        "carbon_credits": "/api/water/carbon-credits",
        "transferences": "/api/water/transferences",
        "relations":     "/api/water/retreats/relations",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_get(endpoint: str, params: dict | None = None,
            timeout: int = 60) -> dict:
    """Send GET request to Global CarbonTrace API and return JSON."""
    url = f"{API_BASE}{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_paginated(endpoint: str, per_page: int = MAX_PER_PAGE,
                    max_pages: int | None = None,
                    delay: float = 0.5) -> list[dict]:
    """Fetch all pages from a paginated Laravel API endpoint.

    Parameters
    ----------
    endpoint : str
        API path (e.g. "/api/public/retreats").
    per_page : int
        Records per page (max 500 tested).
    max_pages : int or None
        If set, stop after this many pages (for testing).
    delay : float
        Seconds to wait between requests to be polite.

    Returns
    -------
    list of dict
        All records concatenated from every page.
    """
    # First request to learn total
    first = api_get(endpoint, params={"per_page": per_page, "page": 1})
    total = first.get("total", 0)
    last_page = first.get("last_page", 1)
    records = first.get("data", [])

    if total == 0:
        return records

    if max_pages is not None:
        last_page = min(last_page, max_pages)

    desc = endpoint.split("/")[-1]
    pbar = tqdm(total=min(total, last_page * per_page),
                initial=len(records), unit="rec", desc=desc)

    for page in range(2, last_page + 1):
        time.sleep(delay)
        data = api_get(endpoint, params={"per_page": per_page, "page": page})
        batch = data.get("data", [])
        records.extend(batch)
        pbar.update(len(batch))

    pbar.close()
    return records


def clean_retirement_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and types for the retirements dataset."""
    rename_map = {
        "reason": "retirement_reason",
        "to_name": "beneficiary",
        "nit_to_name": "beneficiary_id",
        "final_user": "final_user",
        "final_user_nit": "final_user_id",
        "destination": "destination",
        "serial": "serial",
        "created_at": "retirement_date",
        "initial_serial": "serial_start",
        "final_serial": "serial_end",
        "sold": "volume",
        "market": "market_type",
        "passive_user": "passive_user",
        "nit_passive_user": "passive_user_id",
        "initiative_id": "project_id",
        "project_name": "project_name",
        "holder": "project_holder",
        "holder_nit": "project_holder_id",
        "data_visibility": "data_visibility",
    }
    df = df.rename(columns=rename_map)

    # Parse volume as numeric
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Parse retirement date
    if "retirement_date" in df.columns:
        df["retirement_date"] = pd.to_datetime(
            df["retirement_date"], format="%d/%m/%Y", errors="coerce"
        )

    # Parse serial range as int
    for col in ["serial_start", "serial_end"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Main download functions
# ---------------------------------------------------------------------------

def download_retirements(program: str = "gei",
                         per_page: int = MAX_PER_PAGE,
                         max_pages: int | None = None,
                         output_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download all retirement records for a given program.

    Parameters
    ----------
    program : str
        One of 'gei', 'biodiversity', 'water'.
    per_page : int
        Records per API page.
    max_pages : int or None
        Limit pages for testing.
    output_dir : Path
        Directory to save CSV output.

    Returns
    -------
    pd.DataFrame
    """
    endpoint = PROGRAMS[program]["retreats"]
    print(f"Downloading {program} retirements from {API_BASE}{endpoint} ...")

    records = fetch_paginated(endpoint, per_page=per_page, max_pages=max_pages)
    if not records:
        print(f"  No {program} retirements found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = clean_retirement_df(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"biocarbon_{program}_retirements.csv"
    df.to_csv(outfile, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(df)} records to {outfile}")
    return df


def download_projects(program: str = "gei",
                      per_page: int = MAX_PER_PAGE,
                      output_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download all project/initiative records for a given program."""
    endpoint = PROGRAMS[program]["initiatives"]
    print(f"Downloading {program} projects from {API_BASE}{endpoint} ...")

    records = fetch_paginated(endpoint, per_page=per_page)
    if not records:
        print(f"  No {program} projects found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"biocarbon_{program}_projects.csv"

    # Flatten nested dicts for CSV compatibility
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, (dict, list)) else x
            )

    df.to_csv(outfile, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(df)} records to {outfile}")
    return df


def download_carbon_credits(program: str = "gei",
                            per_page: int = MAX_PER_PAGE,
                            output_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download all carbon credit issuance records for a given program."""
    endpoint = PROGRAMS[program]["carbon_credits"]
    print(f"Downloading {program} carbon credits from {API_BASE}{endpoint} ...")

    records = fetch_paginated(endpoint, per_page=per_page)
    if not records:
        print(f"  No {program} carbon credits found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"biocarbon_{program}_credits.csv"

    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, (dict, list)) else x
            )

    df.to_csv(outfile, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(df)} records to {outfile}")
    return df


def download_transferences(program: str = "gei",
                           per_page: int = MAX_PER_PAGE,
                           output_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download all credit transfer records for a given program."""
    endpoint = PROGRAMS[program]["transferences"]
    print(f"Downloading {program} transferences from {API_BASE}{endpoint} ...")

    records = fetch_paginated(endpoint, per_page=per_page)
    if not records:
        print(f"  No {program} transferences found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"biocarbon_{program}_transferences.csv"

    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, (dict, list)) else x
            )

    df.to_csv(outfile, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(df)} records to {outfile}")
    return df


def download_all(programs: list[str] | None = None,
                 output_dir: Path = RAW_DIR,
                 max_pages: int | None = None) -> dict[str, pd.DataFrame]:
    """Download all data categories for all programs.

    Parameters
    ----------
    programs : list of str or None
        Programs to download. Default: all three (gei, biodiversity, water).
    output_dir : Path
        Output directory.
    max_pages : int or None
        Limit pages per endpoint (for testing).

    Returns
    -------
    dict mapping 'program_category' to DataFrame.
    """
    if programs is None:
        programs = ["gei", "biodiversity", "water"]

    results = {}
    for prog in programs:
        print(f"\n{'='*60}")
        print(f"Program: {prog}")
        print(f"{'='*60}")

        results[f"{prog}_retirements"] = download_retirements(
            prog, output_dir=output_dir, max_pages=max_pages
        )
        results[f"{prog}_projects"] = download_projects(
            prog, output_dir=output_dir
        )
        results[f"{prog}_credits"] = download_carbon_credits(
            prog, output_dir=output_dir
        )
        results[f"{prog}_transferences"] = download_transferences(
            prog, output_dir=output_dir
        )

    # Print summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    for key, df in results.items():
        print(f"  {key}: {len(df)} records")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download BioCarbon Registry (Global CarbonTrace) data."
    )
    parser.add_argument(
        "--program", "-p",
        choices=["gei", "biodiversity", "water", "all"],
        default="gei",
        help="Program to download (default: gei). Use 'all' for everything.",
    )
    parser.add_argument(
        "--category", "-c",
        choices=["retirements", "projects", "credits", "transferences", "all"],
        default="retirements",
        help="Data category to download (default: retirements).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=RAW_DIR,
        help=f"Output directory (default: {RAW_DIR}).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit number of pages to fetch (for testing).",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=MAX_PER_PAGE,
        help=f"Records per API page (default: {MAX_PER_PAGE}).",
    )

    args = parser.parse_args()

    programs = (
        ["gei", "biodiversity", "water"] if args.program == "all"
        else [args.program]
    )

    if args.category == "all":
        download_all(programs, output_dir=args.output_dir,
                     max_pages=args.max_pages)
    else:
        for prog in programs:
            if args.category == "retirements":
                download_retirements(prog, per_page=args.per_page,
                                     max_pages=args.max_pages,
                                     output_dir=args.output_dir)
            elif args.category == "projects":
                download_projects(prog, per_page=args.per_page,
                                  output_dir=args.output_dir)
            elif args.category == "credits":
                download_carbon_credits(prog, per_page=args.per_page,
                                        output_dir=args.output_dir)
            elif args.category == "transferences":
                download_transferences(prog, per_page=args.per_page,
                                       output_dir=args.output_dir)


if __name__ == "__main__":
    main()
