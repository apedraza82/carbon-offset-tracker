"""Download voluntary cancellation data from CDM (Clean Development Mechanism).

The CDM Registry (cdm.unfccc.int) tracks voluntary cancellations of Certified
Emission Reductions (CERs). The data is available via the UNEP Copenhagen Climate
Centre's CDM Pipeline Excel spreadsheet, which is updated regularly.

Source: https://unepccc.org/cdm-ji-pipeline/
File: https://unepccc.org/wp-content/uploads/2025/04/cdm-pipeline.xlsx

The "Voluntary Cancellations" sheet contains all cancellation records with:
  - Ref. (CDM project reference number)
  - Title (project name)
  - Project type
  - Host (country)
  - Quantity of units cancelled
  - Unit type (CERs or tCERs)
  - Purpose (free text — contains beneficiary/entity info)
  - Date
  - Type/Subtype (project category)

We focus on CER cancellations (permanent), not tCERs (temporary, mostly small Brazilian).
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CDM_PIPELINE_URL = "https://unepccc.org/wp-content/uploads/2025/04/cdm-pipeline.xlsx"
RAW_DIR = Path("data/raw/cdm")
TIMEOUT = 120


def download_cdm():
    """Download CDM voluntary cancellations from UNEP CCC pipeline spreadsheet."""
    print("=" * 60)
    print("Downloading CDM voluntary cancellations (UNEP CCC Pipeline)")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download the Excel file
    print(f"  Fetching {CDM_PIPELINE_URL} ...")
    resp = requests.get(CDM_PIPELINE_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    print(f"  Downloaded {len(resp.content) / 1e6:.1f} MB")

    # Save raw Excel
    xlsx_path = RAW_DIR / "cdm-pipeline.xlsx"
    with open(xlsx_path, "wb") as f:
        f.write(resp.content)

    # Parse the Voluntary Cancellations sheet (header is on row 5)
    df = pd.read_excel(
        BytesIO(resp.content),
        sheet_name="Voluntary Cancellations",
        header=5,
    )

    # Drop empty/summary rows
    df = df.dropna(subset=["Ref."])
    df["Ref."] = df["Ref."].astype(int)

    # Save full dataset
    full_path = RAW_DIR / "cdm_voluntary_cancellations.csv"
    df.to_csv(full_path, index=False)
    print(f"  Saved all cancellations: {len(df):,} rows to {full_path}")

    # Filter to CERs only (permanent cancellations)
    cers = df[df["Unit type"] == "CERs"].copy()
    cers_path = RAW_DIR / "cdm_cer_cancellations.csv"
    cers.to_csv(cers_path, index=False)
    print(f"  Saved CER cancellations: {len(cers):,} rows to {cers_path}")

    # Summary
    print(f"\n  Summary (CERs only):")
    print(f"    Total cancellations: {len(cers):,}")
    print(f"    Total volume: {cers['Quantity of units cancelled'].sum():,.0f} tCO2")
    print(f"    Date range: {cers['Date'].min()} to {cers['Date'].max()}")
    print(f"    Unique projects: {cers['Ref.'].nunique():,}")
    print(f"    Host countries: {cers['Host'].nunique():,}")

    # Purpose fill rate
    has_purpose = cers["Purpose"].notna() & (cers["Purpose"] != "")
    print(f"    Purpose fill rate: {has_purpose.sum()}/{len(cers)} "
          f"({100 * has_purpose.mean():.1f}%)")

    print(f"\n  tCERs (excluded): {len(df) - len(cers):,} rows, "
          f"{(df[df['Unit type'] == 'tCERs']['Quantity of units cancelled'].sum()):,.0f} tCO2")

    return cers


if __name__ == "__main__":
    download_cdm()
