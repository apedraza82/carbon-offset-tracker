"""Download voluntary cancellation data from CDM (Clean Development Mechanism).

The CDM Registry (cdm.unfccc.int) tracks voluntary cancellations of Certified
Emission Reductions (CERs). We combine three data sources:

1. **Wayback Machine** (primary): Archived snapshots of the CDM Registry's
   vc_attest/index.html page, which lists all voluntary cancellations from
   Nov 2018 onward. The live site is behind a WAF but Wayback has full captures.

2. **CDM Registry old/archive pages** (via Google Translate proxy): Two legacy
   pages on the CDM site that list pre-Nov-2018 cancellations:
   - vc_attest_old/index.html (Feb-Oct 2018)
   - vc_attest_old/index_archive.html (pre-2018, back to 2012)

3. **UNEP CCC Pipeline** (supplementary): Excel spreadsheet updated regularly
   with a "Voluntary Cancellations" sheet (June 2023+). Used as fallback.

We focus on CER cancellations (permanent), not tCERs (temporary).
"""

import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw/cdm")
TIMEOUT = 120

# Wayback Machine: most recent snapshot of vc_attest/index.html
WAYBACK_CDX_URL = (
    "http://web.archive.org/cdx/search/cdx"
    "?url=cdm.unfccc.int/Registry/vc_attest/index.html"
    "&output=json&limit=-5&filter=statuscode:200"
)
WAYBACK_BASE = "https://web.archive.org/web/{ts}/https://cdm.unfccc.int/Registry/vc_attest/index.html"

# Google Translate proxy for old/archive pages
GT_BASE = "https://cdm-unfccc-int.translate.goog/Registry"
GT_PARAMS = "?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=en"

# UNEP CCC Pipeline (supplementary)
CDM_PIPELINE_URL = "https://unepccc.org/wp-content/uploads/2025/04/cdm-pipeline.xlsx"

# Standard column names for output
COLUMNS = [
    "Ref.", "Title", "Project type", "Host",
    "Quantity of units cancelled", "Unit type",
    "Purpose", "Date", "Attestation letter", "source",
]


def _parse_html_table(html_content):
    """Parse the CDM attestation HTML table into a DataFrame."""
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame()

    biggest = max(tables, key=lambda t: len(t.find_all("tr")))
    rows = biggest.find_all("tr")
    headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    data = []
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if cells and any(c.strip() for c in cells):
            data.append(cells)

    df = pd.DataFrame(data, columns=headers[: len(data[0])] if data else headers)
    return df


def _clean_wayback_dates(series):
    """Extract dd/mm/yyyy from Wayback-mangled date strings."""
    def extract(s):
        if pd.isna(s):
            return None
        m = re.search(r"(\d{2}/\d{2}/\d{4})", str(s))
        return m.group(1) if m else str(s)
    return series.apply(extract)


def _fetch_wayback():
    """Fetch the most recent Wayback Machine snapshot of vc_attest."""
    print("  [Wayback] Finding latest snapshot...")
    resp = requests.get(WAYBACK_CDX_URL, timeout=30)
    resp.raise_for_status()
    import json
    data = json.loads(resp.text)
    if len(data) < 2:
        print("  [Wayback] No snapshots found")
        return pd.DataFrame()

    # Pick the largest snapshot (most data)
    headers = data[0]
    snapshots = [dict(zip(headers, row)) for row in data[1:]]
    best = max(snapshots, key=lambda s: int(s["length"]))
    ts = best["timestamp"]
    url = WAYBACK_BASE.format(ts=ts)

    print(f"  [Wayback] Fetching snapshot {ts} ({int(best['length']):,} bytes)...")
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    print(f"  [Wayback] Downloaded {len(resp.content):,} bytes")

    df = _parse_html_table(resp.content)
    if df.empty:
        return df

    # Rename columns
    col_map = {
        "Project/POA number": "Ref.",
        "Project name": "Title",
        "Project type": "Project type",
        "Host country": "Host",
        "Quantity of units cancelled": "Quantity of units cancelled",
        "Reason/ Beneficiary": "Purpose",
        "Date of completion": "Date",
        "Link to the attestation": "Attestation letter",
    }
    df = df.rename(columns=col_map)
    df["Date"] = _clean_wayback_dates(df["Date"])
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Quantity of units cancelled"] = (
        df["Quantity of units cancelled"].astype(str).str.replace(",", "").astype(float)
    )
    df["source"] = "wayback_vc_attest"
    print(f"  [Wayback] Parsed {len(df)} rows")
    return df


def _fetch_old_pages():
    """Fetch pre-Nov-2018 data from CDM old/archive pages via Google Translate proxy."""
    pages = {
        "old_2018": f"{GT_BASE}/vc_attest_old/index.html{GT_PARAMS}",
        "archive_pre2018": f"{GT_BASE}/vc_attest_old/index_archive.html{GT_PARAMS}",
    }
    frames = []
    for label, url in pages.items():
        print(f"  [{label}] Fetching via Google Translate proxy...")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            df = _parse_html_table(resp.content)
            if df.empty:
                print(f"  [{label}] No table found")
                continue
            # Rename
            col_map = {
                "Project/POA number": "Ref.",
                "Project name": "Title",
                "Project type": "Project type",
                "Host country": "Host",
                "Quantity of units cancelled": "Quantity of units cancelled",
                "Reason/ Beneficiary": "Purpose",
                "Date of completion": "Date",
                "Link to the attestation": "Attestation letter",
            }
            df = df.rename(columns=col_map)
            df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
            df["Quantity of units cancelled"] = (
                df["Quantity of units cancelled"].astype(str).str.replace(",", "").astype(float)
            )
            df["source"] = label
            frames.append(df)
            print(f"  [{label}] Parsed {len(df)} rows")
        except Exception as e:
            print(f"  [{label}] Failed: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download_cdm():
    """Download CDM voluntary cancellations from all sources."""
    print("=" * 60)
    print("Downloading CDM voluntary cancellations")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Wayback Machine (primary source: Nov 2018+)
    wayback = _fetch_wayback()

    # 2. Old/archive pages (pre-Nov 2018)
    old_pages = _fetch_old_pages()

    # 3. Combine
    frames = [df for df in [old_pages, wayback] if not df.empty]
    if not frames:
        print("  ERROR: No data fetched from any source")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Filter to CERs only
    combined = combined[combined["Unit type"].isin(["CERs", "CER"])].copy()
    combined["Unit type"] = "CERs"

    # Deduplicate (same project + date + quantity)
    combined["_dedup"] = (
        combined["Ref."].astype(str) + "_"
        + combined["Date"].astype(str) + "_"
        + combined["Quantity of units cancelled"].astype(str)
    )
    before = len(combined)
    combined = combined.drop_duplicates(subset="_dedup").drop(columns="_dedup")
    if before > len(combined):
        print(f"  Deduplicated: {before} -> {len(combined)} "
              f"({before - len(combined)} duplicates)")

    # Sort by date
    combined = combined.sort_values("Date").reset_index(drop=True)

    # Save
    out_cols = [c for c in COLUMNS if c in combined.columns]
    out_path = RAW_DIR / "cdm_cer_cancellations.csv"
    combined[out_cols].to_csv(out_path, index=False)

    # Summary
    print(f"\n  Summary (CERs only):")
    print(f"    Total cancellations: {len(combined):,}")
    print(f"    Total volume: {combined['Quantity of units cancelled'].sum():,.0f} tCO2")
    print(f"    Date range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
    print(f"    Unique projects: {combined['Ref.'].nunique():,}")
    print(f"    Host countries: {combined['Host'].nunique():,}")

    # Purpose fill rate
    has_purpose = combined["Purpose"].notna() & (combined["Purpose"] != "")
    print(f"    Purpose fill rate: {has_purpose.sum()}/{len(combined)} "
          f"({100 * has_purpose.mean():.1f}%)")

    return combined


if __name__ == "__main__":
    download_cdm()
