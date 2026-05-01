"""Download retirement data from Puro.earth Registry.

Puro.earth (puro.earth) is a carbon removal credit registry that issues
CO2 Removal Certificates (CORCs). Each CORC = 1 tonne CO2 removed.

The registry (registry.puro.earth) is a Next.js application that embeds all
retirement transaction data as server-side rendered JSON within the HTML page.
All 1,443 retirements are loaded in a single page with client-side pagination.

Fields per retirement record:
  id, type, accountHolderName, completedOn, volume, retirementId,
  retirementDetails: {usageType, beneficiaryName, beneficiaryType,
    retirementPurpose, publicStatementUrl, beneficiaryLocation,
    countryOfConsumption, consumptionPeriodStartDate, consumptionPeriodEndDate,
    beneficiaryHiddenUntil},
  bundles: [{id, certificates, volume, methodologyCode, projectName,
    methodologyName, issuanceDate}],
  labels: [{id, name}]
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGISTRY_URL = "https://registry.puro.earth/retirements"
RAW_DIR = Path("data/raw/puro")
TIMEOUT = 120


def fetch_retirements_html() -> str:
    """Fetch the retirements page HTML (contains all records as embedded JSON)."""
    print(f"  Fetching {REGISTRY_URL} ...")
    resp = requests.get(REGISTRY_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    print(f"  Page size: {len(resp.text):,} bytes")
    return resp.text


def extract_retirements(html: str) -> list[dict]:
    """Extract retirement records from the embedded Next.js RSC payload.

    The data is embedded as double-escaped JSON within a <script> tag containing
    self.__next_f.push() call. The retirement array is under a "data" key.
    """
    # Find the start of the data array (double-escaped JSON)
    marker = '\\"data\\":['
    start = html.find(marker)
    if start == -1:
        raise ValueError("Could not find retirement data array in HTML")

    array_start = start + len(marker) - 1  # position of '['

    # Find the matching ']' — track bracket depth (escaped brackets)
    # Since quotes are escaped as \\", we can safely count brackets
    depth = 0
    i = array_start
    while i < len(html):
        ch = html[i]
        if ch == '\\' and i + 1 < len(html) and html[i + 1] == '\\':
            # Double backslash escape, skip next char pair
            i += 2
            if i < len(html) and html[i] == '"':
                i += 1  # Skip escaped quote
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                break
        i += 1

    array_end = i + 1
    raw_array = html[array_start:array_end]
    print(f"  Raw data array: {len(raw_array):,} chars")

    # Unescape: \\" -> ", \\\\ -> \\
    # The JSON is double-escaped (inside a JS string literal)
    unescaped = raw_array.replace('\\"', '"').replace('\\\\', '\\')

    # Parse JSON
    try:
        records = json.loads(unescaped)
    except json.JSONDecodeError as e:
        # Try alternative unescaping
        print(f"  JSON parse failed: {e}")
        print(f"  Trying alternative unescape...")
        # Sometimes there's triple escaping for backslashes in URLs
        unescaped2 = raw_array.replace('\\\\"', '"').replace('\\\\', '\\')
        records = json.loads(unescaped2)

    print(f"  Extracted {len(records):,} retirement records")
    return records


def flatten_records(records: list[dict]) -> pd.DataFrame:
    """Flatten nested retirement records into a tabular DataFrame."""
    rows = []
    for rec in records:
        details = rec.get("retirementDetails", {}) or {}
        bundles = rec.get("bundles", []) or []
        labels = rec.get("labels", []) or []

        # Aggregate bundle info (a retirement can span multiple issuances)
        project_names = []
        facility_names = []
        methodologies = []
        vintages = []
        countries = set()
        for b in bundles:
            # Project name is nested: bundle['project']['name']
            project = b.get("project")
            if isinstance(project, dict):
                pname = project.get("name", "")
            else:
                pname = b.get("projectName", "")
            if pname and pname not in project_names:
                project_names.append(pname)

            fname = b.get("productionFacilityName", "")
            if fname and fname not in facility_names:
                facility_names.append(fname)

            mname = b.get("methodologyName", "")
            if mname and mname not in methodologies:
                methodologies.append(mname)

            vintage = b.get("vintage")
            if vintage and vintage not in vintages:
                vintages.append(vintage)

            # Extract country from certificate code: PURO_PR_CORC_XX_...
            cert = b.get("certificates", "")
            if cert:
                parts = cert.split("_")
                if len(parts) >= 4:
                    country_code = parts[3]
                    if len(country_code) == 2 and country_code.isalpha():
                        countries.add(country_code)

        row = {
            "retirement_id": rec.get("retirementId") or rec.get("id"),
            "account_holder": rec.get("accountHolderName", ""),
            "completed_on": rec.get("completedOn", ""),
            "volume": rec.get("volume", 0),
            "usage_type": details.get("usageType", ""),
            "beneficiary_name": details.get("beneficiaryName", ""),
            "beneficiary_type": details.get("beneficiaryType", ""),
            "beneficiary_location": details.get("beneficiaryLocation", ""),
            "country_of_consumption": details.get("countryOfConsumption", ""),
            "retirement_purpose": details.get("retirementPurpose", ""),
            "consumption_period_start": details.get("consumptionPeriodStartDate", ""),
            "consumption_period_end": details.get("consumptionPeriodEndDate", ""),
            "beneficiary_hidden_until": details.get("beneficiaryHiddenUntil", ""),
            "public_statement_url": details.get("publicStatementUrl", ""),
            "project_name": "; ".join(project_names),
            "facility_name": "; ".join(facility_names),
            "methodology": "; ".join(methodologies),
            "vintage": vintages[0] if len(vintages) == 1 else ("; ".join(str(v) for v in vintages) if vintages else ""),
            "project_country": "; ".join(sorted(countries)) if countries else "",
            "label": "; ".join(l.get("name", "") for l in labels),
            "num_bundles": len(bundles),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def download_puro():
    """Main download function."""
    print("=" * 60)
    print("Downloading Puro.earth Registry retirements")
    print("=" * 60)

    # Create output directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch HTML
    html = fetch_retirements_html()

    # Save raw HTML for debugging
    raw_html_path = RAW_DIR / "retirements_page.html"
    raw_html_path.write_text(html, encoding="utf-8")
    print(f"  Saved raw HTML to {raw_html_path}")

    # Extract retirement records
    records = extract_retirements(html)

    # Save raw JSON
    raw_json_path = RAW_DIR / "retirements_raw.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Saved raw JSON to {raw_json_path}")

    # Flatten to DataFrame
    df = flatten_records(records)

    # Save as CSV
    csv_path = RAW_DIR / "puro_retirements.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df):,} retirements to {csv_path}")

    # Print summary
    print(f"\n  Summary:")
    print(f"    Total retirements: {len(df):,}")
    print(f"    Total volume (tCO2): {df['volume'].sum():,.0f}")
    print(f"    Date range: {df['completed_on'].min()[:10]} to {df['completed_on'].max()[:10]}")
    print(f"    Unique beneficiaries: {df['beneficiary_name'].nunique():,}")
    print(f"    Unique account holders: {df['account_holder'].nunique():,}")
    print(f"    Unique projects: {df['project_name'].nunique():,}")
    print(f"    Unique methodologies: {df['methodology'].nunique():,}")

    # Show beneficiary fill rate
    has_beneficiary = df["beneficiary_name"].notna() & (df["beneficiary_name"] != "")
    print(f"    Beneficiary fill rate: {has_beneficiary.sum()}/{len(df)} "
          f"({100*has_beneficiary.mean():.1f}%)")

    return df


if __name__ == "__main__":
    download_puro()
