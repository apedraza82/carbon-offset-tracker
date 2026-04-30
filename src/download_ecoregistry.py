"""Download retirement and project data from EcoRegistry (Cercarbono standard).

EcoRegistry is the registry platform for the Cercarbono carbon standard.
The site (ecoregistry.io) is a JavaScript SPA that loads data from an
internal REST API at api-front.ecoregistry.io/platform.

Two public endpoints provide all the data we need without authentication:
  - /analytics/get-retirements  -> all retirement transactions
  - /analytics/projects         -> all registered projects with serials

The API requires two custom headers:
  - Platform: ecoregistry
  - Lng: en  (language; 'en' for English descriptions)

Discovered via reverse-engineering the JS bundles:
  - app.ecoregistry.io/static/js/main.44843c91.js
  - www.ecoregistry.io/assets/index-050ee4eb.js
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

API_BASE = "https://api-front.ecoregistry.io/platform"

HEADERS = {
    "Platform": "ecoregistry",
    "Lng": "en",
}

# Where to save output
RAW_DIR = Path("data/raw/ecoregistry")


def _get(endpoint: str, timeout: int = 120) -> dict:
    """GET request to the EcoRegistry API."""
    url = f"{API_BASE}{endpoint}"
    print(f"  Fetching {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Retirements
# ---------------------------------------------------------------------------

def download_retirements() -> pd.DataFrame:
    """Download all retirement/cancellation transactions.

    Returns a DataFrame with one row per retirement, including:
      - id: unique retirement ID
      - project_id: EcoRegistry project ID
      - serial: credit serial number
      - quantity: number of credits retired (tonnes CO2e unless is_kg=1)
      - date: retirement datetime (ISO 8601)
      - vintage: vintage date
      - is_kg: 1 if quantity is in kg rather than tonnes
      - final_user: beneficiary / entity that retired the credits
      - country_final_user: country of the retiring entity
      - reason_using: reason for retirement (e.g. 'Colombian Carbon Tax')
      - methodology: methodology name
      - sector: sector name (e.g. 'Land use (AFOLU)')
    """
    data = _get("/analytics/get-retirements")

    if data.get("status") != 1:
        raise RuntimeError(f"API returned unexpected status: {data.get('status')}")

    retirements = data["retirements"]
    print(f"  Retrieved {len(retirements):,} retirement records")

    # Flatten nested fields
    rows = []
    for r in retirements:
        row = {
            "id": r["id"],
            "project_id": r["project_id"],
            "serial": r["serial"],
            "quantity": r["quantity"],
            "date": r["date"],
            "vintage": r["vintage"],
            "is_kg": r["is_kg"],
            "final_user": r["final_user"],
            "country_final_user": r["country_final_user"],
            "reason_using": r.get("reason_using", {}).get("description", ""),
        }

        # Flatten methodologies (join multiple with '; ')
        methodologies = r.get("methodologies", [])
        row["methodology"] = "; ".join(
            m.get("description", "") for m in methodologies
        )

        # Flatten sectors
        sectors = r.get("sectors", [])
        row["sector"] = "; ".join(
            s.get("description", "") for s in sectors
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["vintage"] = pd.to_datetime(df["vintage"], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def download_projects() -> pd.DataFrame:
    """Download all registered projects.

    Returns a DataFrame with one row per project, including:
      - id: EcoRegistry project ID
      - name: project name
      - standard: standard name (e.g. 'Cercarbono')
      - stage: project stage (e.g. 'Certified')
      - owner: project owner
      - developer: project developer
      - validator, verifier: VVB names
      - credits_type: type of credits
      - evaluation_criteria: protocol used
      - quantification_method: methodology
      - sector: sector(s)
      - country, region, city: project location
      - latitude, longitude: coordinates
      - total_issued: sum of all issued quantities across serials
      - serials_count: number of serial numbers / issuances
    """
    data = _get("/analytics/projects")

    if not data.get("status"):
        raise RuntimeError(f"API returned unexpected status: {data.get('status')}")

    projects = data["project"]
    print(f"  Retrieved {len(projects):,} project records")

    rows = []
    for p in projects:
        # Primary location (first entry)
        locations = p.get("locations", [])
        loc = locations[0] if locations else {}
        data_map = loc.get("data_map", {})
        if isinstance(data_map, str):
            data_map = {}

        # Sectors
        sectors = p.get("sector", [])
        sector_str = "; ".join(s.get("description", "") for s in sectors)

        # Serials summary
        serials = p.get("serials", [])
        total_issued = sum(s.get("issued_quantity", 0) for s in serials)

        row = {
            "id": p["id"],
            "name": p.get("name", ""),
            "standard": p.get("standard", ""),
            "stage": p.get("stage", ""),
            "is_public": p.get("is_public", None),
            "owner": p.get("owner", ""),
            "developer": p.get("developer", ""),
            "validator": p.get("validator", ""),
            "verifier": p.get("verifier", ""),
            "credits_type": p.get("credits_type", ""),
            "evaluation_criteria": p.get("evaluation_criteria", ""),
            "quantification_method": p.get("quantification_method", ""),
            "sector": sector_str,
            "country": loc.get("country", ""),
            "region": loc.get("region", ""),
            "city": loc.get("city", ""),
            "latitude": data_map.get("latitude"),
            "longitude": data_map.get("longitude"),
            "total_issued": total_issued,
            "serials_count": len(serials),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Serial-level issuance data
# ---------------------------------------------------------------------------

def download_serials() -> pd.DataFrame:
    """Download serial-level issuance data from the projects endpoint.

    Returns a DataFrame with one row per serial/issuance, including:
      - project_id: EcoRegistry project ID
      - project_name: project name
      - serial: serial number
      - issued_quantity: number of credits issued
      - issuance_date: date of issuance
      - vintage_year: vintage year
      - vintage_of_credits: full vintage period string
      - is_buffer: whether this is a buffer pool allocation
      - verification: verification number
      - eligible: eligible uses (e.g. 'Colombian Carbon Tax', 'Voluntary compensation')
    """
    data = _get("/analytics/projects")
    projects = data["project"]

    rows = []
    for p in projects:
        for s in p.get("serials", []):
            eligible = "; ".join(
                e.get("description", "") for e in s.get("elegible", [])
            )
            row = {
                "project_id": p["id"],
                "project_name": p.get("name", ""),
                "standard": p.get("standard", ""),
                "serial": s.get("serial", ""),
                "issued_quantity": s.get("issued_quantity", 0),
                "issuance_date": s.get("issuance_date", ""),
                "vintage_year": s.get("year"),
                "vintage_of_credits": s.get("vintage_of_credits", ""),
                "is_buffer": s.get("is_buffer", False),
                "verification": s.get("verification"),
                "eligible": eligible,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["issuance_date"] = pd.to_datetime(df["issuance_date"], errors="coerce")

    print(f"  Extracted {len(df):,} serial-level issuance records")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_download_pipeline() -> dict[str, pd.DataFrame]:
    """Download all EcoRegistry data and save to CSV.

    Returns dict with keys 'retirements', 'projects', 'serials'.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading EcoRegistry (Cercarbono) data")
    print("=" * 60)

    # 1. Retirements
    print("\n[1/3] Retirements...")
    retirements = download_retirements()
    ret_path = RAW_DIR / "ecoregistry_retirements.csv"
    retirements.to_csv(ret_path, index=False, encoding="utf-8-sig")
    print(f"  Saved to {ret_path}")

    # 2. Projects
    print("\n[2/3] Projects...")
    projects = download_projects()
    proj_path = RAW_DIR / "ecoregistry_projects.csv"
    projects.to_csv(proj_path, index=False, encoding="utf-8-sig")
    print(f"  Saved to {proj_path}")

    # 3. Serials
    print("\n[3/3] Serials (issuance-level)...")
    serials = download_serials()
    ser_path = RAW_DIR / "ecoregistry_serials.csv"
    serials.to_csv(ser_path, index=False, encoding="utf-8-sig")
    print(f"  Saved to {ser_path}")

    # Save metadata
    meta = {
        "download_date": datetime.now().isoformat(),
        "source": "EcoRegistry API (api-front.ecoregistry.io)",
        "standard": "Cercarbono",
        "retirements_count": len(retirements),
        "projects_count": len(projects),
        "serials_count": len(serials),
    }
    meta_path = RAW_DIR / "download_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Retirements: {len(retirements):,}")
    print(f"  Projects:    {len(projects):,}")
    print(f"  Serials:     {len(serials):,}")
    print(f"  Output dir:  {RAW_DIR}")
    print(f"{'=' * 60}")

    return {
        "retirements": retirements,
        "projects": projects,
        "serials": serials,
    }


if __name__ == "__main__":
    run_download_pipeline()
