"""End-to-end pipeline: download → parse → match → output.

Can be run manually or via GitHub Actions.

Usage:
    python -m src.pipeline [--skip-download] [--skip-llm] [--config CONFIG]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.download import run_download_pipeline
from src.match_firms import FirmMatcher
from src.parse_beneficiary import parse_retirements
from src.utils import load_config


def build_summary_stats(matched: pd.DataFrame, output_path: str) -> dict:
    """Generate summary statistics JSON for the landing page."""
    qty_col = "quantity" if "quantity" in matched.columns else None
    has_qty = qty_col is not None
    matched_mask = matched["factset_entity_id"].notna() & (matched["factset_entity_id"] != "") & (matched["factset_entity_id"] != "None")

    total_qty = matched[qty_col].sum() if has_qty else 0
    matched_qty = matched.loc[matched_mask, qty_col].sum() if has_qty else 0

    stats = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "total_retirements": int(len(matched)),
        "matched_retirements": int(matched_mask.sum()),
        "unique_firms": int(matched.loc[matched_mask, "factset_entity_id"].nunique()),
        "total_mtco2": round(total_qty / 1e6, 1) if has_qty else 0,
        "matched_mtco2": round(matched_qty / 1e6, 1) if has_qty else 0,
        "registries": {},
        "years": {},
        "match_methods": {},
    }

    # By registry
    for reg in matched["registry"].unique():
        sub = matched[matched["registry"] == reg]
        sub_matched = sub[matched_mask.reindex(sub.index)]
        stats["registries"][reg] = {
            "total": int(len(sub)),
            "matched": int(len(sub_matched)),
            "total_mtco2": round(sub[qty_col].sum() / 1e6, 1) if has_qty else 0,
            "matched_mtco2": round(sub_matched[qty_col].sum() / 1e6, 1) if has_qty else 0,
        }

    # By year (if retirement date available)
    if "retirement_year" in matched.columns:
        for yr in sorted(matched["retirement_year"].dropna().unique()):
            sub = matched[matched["retirement_year"] == yr]
            sub_matched = sub[matched_mask.reindex(sub.index)]
            stats["years"][str(int(yr))] = {
                "total": int(len(sub)),
                "matched": int(len(sub_matched)),
                "total_mtco2": round(sub[qty_col].sum() / 1e6, 1) if has_qty else 0,
                "matched_mtco2": round(sub_matched[qty_col].sum() / 1e6, 1) if has_qty else 0,
            }

    # Project countries count (matched only)
    listed = matched[matched_mask]
    if "country" in listed.columns:
        stats["project_countries"] = int(listed["country"].nunique())
    if "hq_country" in listed.columns:
        stats["hq_countries"] = int(listed["hq_country"].replace("", pd.NA).dropna().nunique())

    # By match method
    if "match_method" in matched.columns:
        for method in matched["match_method"].unique():
            stats["match_methods"][method] = int((matched["match_method"] == method).sum())

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Copy to docs/ for GitHub Pages
    docs_path = Path("docs/data") / Path(output_path).name
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Summary stats saved to {output_path} + {docs_path}")
    return stats


# Country name → ISO3 lookup for project map (Berkeley uses full country names)
_COUNTRY_TO_ISO3 = {
    # Short codes appearing in data
    "US": "USA", "BR": "BRA", "CA": "CAN", "FR": "FRA", "MX": "MEX", "TH": "THA",
    "Bolivia Plurinational State of": "BOL", "New Caledonia": "NCL",
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Angola": "AGO", "Argentina": "ARG",
    "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE",
    "Bangladesh": "BGD", "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ",
    "Benin": "BEN", "Bhutan": "BTN", "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH",
    "Botswana": "BWA", "Brazil": "BRA", "Brunei": "BRN", "Bulgaria": "BGR",
    "Burkina Faso": "BFA", "Burundi": "BDI", "Cambodia": "KHM", "Cameroon": "CMR",
    "Canada": "CAN", "Central African Republic": "CAF", "Chad": "TCD", "Chile": "CHL",
    "China": "CHN", "Colombia": "COL", "Comoros": "COM", "Congo": "COG",
    "Costa Rica": "CRI", "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP",
    "Czech Republic": "CZE", "Czechia": "CZE",
    "Democratic Republic of the Congo": "COD", "Dem. Rep. Congo": "COD",
    "Congo, The Democratic Republic of The": "COD", "DR Congo": "COD",
    "Denmark": "DNK", "Djibouti": "DJI", "Dominican Republic": "DOM",
    "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV", "Equatorial Guinea": "GNQ",
    "Eritrea": "ERI", "Estonia": "EST", "Eswatini": "SWZ", "Ethiopia": "ETH",
    "Fiji": "FJI", "Finland": "FIN", "France": "FRA", "Gabon": "GAB", "Gambia": "GMB",
    "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC",
    "Guatemala": "GTM", "Guinea": "GIN", "Guinea-Bissau": "GNB", "Guyana": "GUY",
    "Haiti": "HTI", "Honduras": "HND", "Hungary": "HUN", "Iceland": "ISL",
    "India": "IND", "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ", "Ireland": "IRL",
    "Israel": "ISR", "Italy": "ITA", "Ivory Coast": "CIV", "Cote d'Ivoire": "CIV",
    "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Korea, South": "KOR", "South Korea": "KOR",
    "Korea, Republic of": "KOR", "Kuwait": "KWT",
    "Kyrgyzstan": "KGZ", "Laos": "LAO", "Lao People's Democratic Republic": "LAO",
    "Latvia": "LVA", "Lebanon": "LBN", "Lesotho": "LSO", "Liberia": "LBR",
    "Libya": "LBY", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Madagascar": "MDG", "Malawi": "MWI", "Malaysia": "MYS", "Maldives": "MDV",
    "Mali": "MLI", "Malta": "MLT", "Mauritania": "MRT", "Mauritius": "MUS",
    "Mexico": "MEX", "Moldova": "MDA", "Mongolia": "MNG", "Montenegro": "MNE",
    "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR", "Namibia": "NAM",
    "Nepal": "NPL", "Netherlands": "NLD", "New Zealand": "NZL", "Nicaragua": "NIC",
    "Niger": "NER", "Nigeria": "NGA", "North Macedonia": "MKD", "Norway": "NOR",
    "Oman": "OMN", "Pakistan": "PAK", "Palestine": "PSE", "Panama": "PAN",
    "Papua New Guinea": "PNG", "Paraguay": "PRY", "Peru": "PER", "Philippines": "PHL",
    "Poland": "POL", "Portugal": "PRT", "Qatar": "QAT", "Romania": "ROU",
    "Russia": "RUS", "Russian Federation": "RUS", "Rwanda": "RWA",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB", "Sierra Leone": "SLE",
    "Singapore": "SGP", "Slovakia": "SVK", "Slovenia": "SVN", "Solomon Islands": "SLB",
    "Somalia": "SOM", "South Africa": "ZAF", "South Sudan": "SSD", "Spain": "ESP",
    "Sri Lanka": "LKA", "Sudan": "SDN", "Suriname": "SUR", "Sweden": "SWE",
    "Switzerland": "CHE", "Syria": "SYR", "Taiwan": "TWN",
    "Tajikistan": "TJK", "Tanzania": "TZA", "United Republic of Tanzania": "TZA",
    "Thailand": "THA", "Timor-Leste": "TLS", "Togo": "TGO",
    "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR", "Turkiye": "TUR",
    "Turkmenistan": "TKM", "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "United Kingdom": "GBR",
    "United States": "USA", "United States of America": "USA",
    "Uruguay": "URY", "Uzbekistan": "UZB", "Vanuatu": "VUT",
    "Venezuela": "VEN", "Vietnam": "VNM", "Viet Nam": "VNM",
    "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
    # Spanish names from BioCarbon Registry
    "Brasil": "BRA", "Estados Unidos": "USA", "Kenia": "KEN",
    "Malasia": "MYS", "Panamá": "PAN", "Perú": "PER",
    "Turquía": "TUR", "México": "MEX",
    # Other
    "Cayman Islands": "CYM", "Aruba": "ABW",
}

# ISO2 → ISO3 lookup for HQ map
_ISO2TO3 = {
    "US": "USA", "BR": "BRA", "DE": "DEU", "GB": "GBR", "AU": "AUS", "JP": "JPN",
    "FR": "FRA", "CH": "CHE", "CN": "CHN", "CO": "COL", "KR": "KOR", "IN": "IND",
    "IT": "ITA", "CA": "CAN", "ES": "ESP", "NL": "NLD", "SE": "SWE", "NO": "NOR",
    "DK": "DNK", "FI": "FIN", "AT": "AUT", "BE": "BEL", "IE": "IRL", "PT": "PRT",
    "MX": "MEX", "CL": "CHL", "ZA": "ZAF", "NZ": "NZL", "SG": "SGP", "HK": "HKG",
    "TW": "TWN", "TH": "THA", "MY": "MYS", "ID": "IDN", "PH": "PHL", "TR": "TUR",
    "PL": "POL", "CZ": "CZE", "HU": "HUN", "RO": "ROU", "GR": "GRC", "IL": "ISR",
    "AE": "ARE", "SA": "SAU", "RU": "RUS", "UA": "UKR", "AR": "ARG", "PE": "PER",
    "KE": "KEN", "NG": "NGA", "EG": "EGY", "PK": "PAK", "BD": "BGD", "VN": "VNM",
    "LU": "LUX", "PA": "PAN", "CR": "CRI", "GT": "GTM", "BM": "BMU", "JE": "JEY",
    "KY": "CYM", "LR": "LBR", "MU": "MUS",
    "BH": "BHR", "KW": "KWT", "LK": "LKA", "MO": "MAC", "OM": "OMN",
    "GG": "GGY", "MT": "MLT", "QA": "QAT", "IS": "ISL",
}


def build_map_data(matched: pd.DataFrame, public_firms: pd.DataFrame):
    """Generate map_data.json for the landing page choropleths."""
    listed = matched[matched["factset_entity_id"].notna() & (matched["factset_entity_id"] != "") & (matched["factset_entity_id"] != "None")]

    qty_col = "quantity" if "quantity" in listed.columns else None
    if qty_col is None:
        print("Warning: no quantity column found, skipping map data")
        return

    # Project country map
    # Try isocode (ISO3) first, then country (full name -> ISO3)
    has_isocode = ("isocode" in listed.columns
                   and listed["isocode"].str.strip().replace("", pd.NA).notna().any())
    if has_isocode:
        proj = listed.groupby("isocode")[qty_col].sum().reset_index()
        proj.columns = ["iso3", "tonnes"]
        proj = proj[proj["iso3"].str.strip() != ""]
        proj = proj[proj["tonnes"] > 0]
    elif "country" in listed.columns:
        proj = listed.groupby("country")[qty_col].sum().reset_index()
        proj.columns = ["country_name", "tonnes"]
        proj["iso3"] = proj["country_name"].map(_COUNTRY_TO_ISO3)
        proj = proj.dropna(subset=["iso3"])
        proj = proj[proj["tonnes"] > 0][["iso3", "tonnes"]]
    else:
        proj = pd.DataFrame(columns=["iso3", "tonnes"])

    # HQ country map — use hq_country column (already populated with fallback)
    if "hq_country" in listed.columns:
        hq_col = listed["hq_country"].astype(str).str.strip()
        hq_valid = listed[hq_col != ""].copy()
        hq_valid["_hq"] = hq_col[hq_col != ""]
        hq = hq_valid.groupby("_hq")[qty_col].sum().reset_index()
        hq.columns = ["iso2", "tonnes"]
        hq["iso3"] = hq["iso2"].map(_ISO2TO3)
        hq = hq.dropna(subset=["iso3"])
        hq = hq[hq["tonnes"] > 0]
    else:
        hq = pd.DataFrame(columns=["iso3", "tonnes"])

    map_data = {
        "project_countries": {
            "iso3": proj["iso3"].tolist(),
            "tonnes": proj["tonnes"].astype(int).tolist(),
        },
        "hq_countries": {
            "iso3": hq["iso3"].tolist(),
            "tonnes": hq["tonnes"].astype(int).tolist(),
        },
    }

    out_path = Path("docs/data/map_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(map_data, f)

    print(f"Map data saved: {len(proj)} project countries, {len(hq)} HQ countries")


# Serial number column per registry (Berkeley column names)
_SERIAL_COL = {
    "verra": "Serial Number",
    "gold": "Serial Number",
    "acr": "Credit Serial Numbers",
    "car": "Offset Credit Serial Numbers",
    "ecoregistry": "serial",
    "biocarbon": "serial",
    "puro": "retirement_id",
    "cdm": "Attestation letter",
}

# Unified output columns
_OUTPUT_COLS = [
    "raw_beneficiary", "matched_name", "factset_entity_id", "hq_country", "registry",
    "retirement_year", "country", "quantity", "match_confidence", "match_method",
    "projectname", "projecttype", "vintage", "isocode", "serialnumber",
    "retirement_reason", "market_type",
]


def _load_base_dataset(config: dict) -> pd.DataFrame:
    """Load the existing matched retirements as the base dataset."""
    source_path = Path(config["sources"]["matched_retirements"])
    if not source_path.exists():
        print("  No base dataset found, starting from scratch")
        return pd.DataFrame()

    print(f"  Loading base dataset from {source_path}")
    base = pd.read_parquet(source_path)
    print(f"  Base: {len(base):,} rows, {(base['factset_entity_id'] != '').sum():,} matched")

    # Harmonize column names from original dataset
    renames = {
        "company": "raw_beneficiary",
        "official_name": "matched_name",
        "retirementdate": "retirement_date_str",
    }
    base = base.rename(columns={k: v for k, v in renames.items() if k in base.columns})

    # Parse retirement year from ret_date
    if "ret_date" in base.columns and "retirement_year" not in base.columns:
        base["retirement_date"] = pd.to_datetime(base["ret_date"], errors="coerce")
        base["retirement_year"] = base["retirement_date"].dt.year
    elif "year" in base.columns and "retirement_year" not in base.columns:
        base["retirement_year"] = base["year"]

    # Normalize registry names (base has "Acr", "Car"; Berkeley uses "ACR", "CAR")
    _REG_NORM = {"Acr": "ACR", "Car": "CAR", "gold": "Gold", "verra": "Verra"}
    if "registry" in base.columns:
        base["registry"] = base["registry"].replace(_REG_NORM)

    # Add match fields for existing data
    if "match_confidence" not in base.columns:
        base["match_confidence"] = base["factset_entity_id"].apply(
            lambda x: "high" if x and x != "" else "none"
        )
    if "match_method" not in base.columns:
        base["match_method"] = base["factset_entity_id"].apply(
            lambda x: "original" if x and x != "" else "unmatched"
        )

    return base


def _find_new_retirements(
    registry_data: dict[str, pd.DataFrame],
    base: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Find retirements in Berkeley data not present in the base dataset.

    Uses serial numbers for deduplication.
    """
    # Collect all serial numbers from base dataset
    base_serials = set()
    if "serialnumber" in base.columns:
        base_serials = set(base["serialnumber"].dropna().astype(str))
    elif "serialnumber_fixed" in base.columns:
        base_serials = set(base["serialnumber_fixed"].dropna().astype(str))
    print(f"  Base serial numbers: {len(base_serials):,}")

    new_data = {}
    for registry, df in registry_data.items():
        serial_col = _SERIAL_COL.get(registry, "Serial Number")
        if serial_col not in df.columns:
            print(f"  [{registry.upper()}] No serial column '{serial_col}', taking all rows")
            new_data[registry] = df
            continue

        # Find rows with serial numbers not in base
        df_serials = df[serial_col].astype(str)
        new_mask = ~df_serials.isin(base_serials)
        new_rows = df[new_mask]
        print(f"  [{registry.upper()}] {len(new_rows):,} new rows (of {len(df):,} total)")
        if len(new_rows) > 0:
            new_data[registry] = new_rows

    return new_data


def _load_additional_registries() -> dict[str, pd.DataFrame]:
    """Load EcoRegistry and BioCarbon data from data/raw/ if available."""
    additional = {}

    eco_path = Path("data/raw/ecoregistry/ecoregistry_retirements.csv")
    if eco_path.exists():
        df = pd.read_csv(eco_path, encoding="utf-8-sig")
        print(f"  Loaded EcoRegistry: {len(df):,} retirements")
        additional["ecoregistry"] = df

    bio_path = Path("data/raw/biocarbon/biocarbon_gei_retirements.csv")
    if bio_path.exists():
        df = pd.read_csv(bio_path, encoding="utf-8-sig")
        print(f"  Loaded BioCarbon: {len(df):,} retirements")

        # Enrich with project country from projects file
        proj_path = Path("data/raw/biocarbon/biocarbon_gei_projects.csv")
        if proj_path.exists():
            proj = pd.read_csv(proj_path, encoding="utf-8-sig")
            proj_country = proj[["id", "country_iso", "country", "type_project_name"]].rename(
                columns={"id": "project_id", "country_iso": "project_country_iso",
                          "country": "project_country", "type_project_name": "project_type"}
            )
            before = len(df)
            df = df.merge(proj_country, on="project_id", how="left")
            print(f"  BioCarbon: enriched {df['project_country_iso'].notna().sum()}/{before} rows with project country")
        else:
            print(f"  Warning: {proj_path} not found — run download_biocarbon with --category projects")

        additional["biocarbon"] = df

    # Puro.earth Registry
    puro_path = Path("data/raw/puro/puro_retirements.csv")
    if puro_path.exists():
        df = pd.read_csv(puro_path, encoding="utf-8-sig")
        print(f"  Loaded Puro: {len(df):,} retirements")
        additional["puro"] = df

    # CDM (UNFCCC Clean Development Mechanism) — CER cancellations only
    cdm_path = Path("data/raw/cdm/cdm_cer_cancellations.csv")
    if cdm_path.exists():
        df = pd.read_csv(cdm_path, encoding="utf-8-sig")
        print(f"  Loaded CDM: {len(df):,} CER cancellations")
        additional["cdm"] = df

    return additional


def run_pipeline(config: dict, skip_download: bool = False, skip_llm: bool = False):
    """Run the incremental pipeline.

    Loads the existing matched retirements as a base, downloads new data from
    Berkeley, finds new retirements not in the base, parses and matches them,
    then combines everything.
    """

    # Step 0: Load base dataset
    print("=== Loading base dataset ===")
    base = _load_base_dataset(config)

    # Step 1: Download new data
    if skip_download:
        print("\n=== Skipping download (using local registry files) ===")
        # Prefer VROD file in data/raw/ over individual Dropbox files
        from src.download import BERKELEY_SHEETS
        vrod_version = config.get("sources", {}).get("berkeley_version", "2026-02")
        vrod_path = Path("data/raw") / f"VROD-registry-files--{vrod_version}.xlsx"
        registry_data = {}
        if vrod_path.exists():
            print(f"  Using VROD file: {vrod_path}")
            for sheet_name, registry_key in BERKELEY_SHEETS.items():
                try:
                    df = pd.read_excel(vrod_path, sheet_name=sheet_name)
                    if registry_key == "verra" and "Retirement/Cancellation Date" in df.columns:
                        df = df[df["Retirement/Cancellation Date"].notna()]
                    registry_data[registry_key] = df
                    print(f"  [{registry_key.upper()}] {len(df):,} rows")
                except Exception as e:
                    print(f"  [{registry_key.upper()}] Error: {e}")
        else:
            # Fallback to individual files from config
            registry_dir = Path(config["sources"]["registry_dir"])
            for reg, fname in config["sources"]["registry_files"].items():
                fpath = registry_dir / fname
                if fpath.exists():
                    print(f"  Loading {reg} from {fpath}")
                    registry_data[reg] = pd.read_excel(fpath)
                else:
                    print(f"  Warning: {fpath} not found")
    else:
        version = config.get("sources", {}).get("berkeley_version")
        registry_data = run_download_pipeline(config, version=version)

    # Load additional registries (EcoRegistry, BioCarbon)
    print("\n=== Loading additional registries ===")
    additional = _load_additional_registries()
    registry_data.update(additional)

    if not registry_data:
        print("No registry data to process.")
        return

    # Step 2: Find new retirements not in base
    print("\n=== Finding new retirements ===")
    new_retirements = _find_new_retirements(registry_data, base)

    if not new_retirements:
        print("No new retirements found. Base dataset is up to date.")
        # Still regenerate outputs from base
        combined = base
    else:
        # Step 3: Parse beneficiary names from new retirements
        print("\n=== Parsing new beneficiary names ===")
        all_parsed = []
        for registry, df in new_retirements.items():
            parsed = parse_retirements(df, registry)
            all_parsed.append(parsed)

        new_parsed = pd.concat(all_parsed, ignore_index=True)
        print(f"  Total new parsed: {len(new_parsed):,}")

        # Step 4: Match new names to firms
        print("\n=== Matching new names to firms ===")
        matcher = FirmMatcher.from_files(config)

        unique_names = new_parsed["raw_beneficiary"].unique().tolist()
        print(f"  Unique new beneficiary names: {len(unique_names):,}")

        cache_results, unmatched_names = matcher.match_batch_cache(unique_names)
        print(f"  Cache hits: {len(cache_results):,}")
        print(f"  Unmatched: {len(unmatched_names):,}")

        # LLM matching
        llm_results = []
        if not skip_llm and unmatched_names:
            print(f"\n  Sending {len(unmatched_names):,} names to LLM...")
            llm_results = matcher.match_batch_llm(unmatched_names)
            matcher.update_cache(llm_results)
            matched_by_llm = sum(1 for r in llm_results if r.factset_entity_id)
            print(f"  LLM matched: {matched_by_llm:,} / {len(unmatched_names):,}")

        # Build name->result lookup
        all_results = cache_results + llm_results
        result_map = {r.raw_name: r for r in all_results if r.factset_entity_id}

        # Apply matches to new data
        new_parsed["factset_entity_id"] = new_parsed["raw_beneficiary"].map(
            lambda x: result_map[x].factset_entity_id if x in result_map else ""
        )
        new_parsed["matched_name"] = new_parsed["raw_beneficiary"].map(
            lambda x: result_map[x].matched_name if x in result_map else ""
        )
        new_parsed["match_confidence"] = new_parsed["raw_beneficiary"].map(
            lambda x: result_map[x].confidence if x in result_map else "none"
        )
        new_parsed["match_method"] = new_parsed["raw_beneficiary"].map(
            lambda x: result_map[x].match_method if x in result_map else "unmatched"
        )

        new_matched = new_parsed["factset_entity_id"].apply(lambda x: x != "").sum()
        print(f"  New retirements matched: {new_matched:,}")

        # Step 5: Combine base + new
        print("\n=== Combining base + new retirements ===")
        # Ensure both have the same output columns
        for col in _OUTPUT_COLS:
            if col not in base.columns:
                base[col] = ""
            if col not in new_parsed.columns:
                new_parsed[col] = ""

        combined = pd.concat(
            [base[_OUTPUT_COLS], new_parsed[_OUTPUT_COLS]],
            ignore_index=True,
        )
        print(f"  Base: {len(base):,} + New: {len(new_parsed):,} = Combined: {len(combined):,}")

    # Step 6: Add HQ country from public_firms + populate isocode
    pf_path = Path(config["output"]["public_firms"])
    public_firms = pd.read_parquet(pf_path) if pf_path.exists() else pd.DataFrame()
    if not public_firms.empty:
        hq_map = public_firms.drop_duplicates("factset_entity_id").set_index("factset_entity_id")["iso_country"].to_dict()
        combined["hq_country"] = combined["factset_entity_id"].map(hq_map).fillna("")
    else:
        combined["hq_country"] = ""

    # Fill missing HQ countries for firms not in public_firms (stale IDs from base dataset)
    _MANUAL_HQ = {
        "05HDYH-E": "DE",  # Audi AG
        "002TQJ-E": "AU",  # Telstra Corporation Limited
        "0H62NM-E": "CO",  # Avianca Group International Limited
        "05HM77-E": "SE",  # Vattenfall AB
        "05L79F-E": "FR",  # ENGIE SA
        "001GCM-E": "US",  # Delta Air Lines Inc.
        "003MPQ-E": "AU",  # Australia and New Zealand Banking Group Limited
        "001YJY-E": "CH",  # Credit Suisse Group AG
        "0D0DNB-E": "CA",  # Frontera Energy Corporation
        "000N7V-E": "US",  # Jacobs Engineering Group Inc.
        "007TJ9-E": "SE",  # ICA Gruppen AB
        "06X4CQ-E": "CA",  # Karora Resources Inc.
        "05LN3R-E": "IT",  # Enel S.p.A.
        "002HJD-E": "US",  # BlackRock, Inc.
        "000Y86-E": "US",  # Marathon Oil Corporation
        "000BVL-E": "US",  # Hess Corporation
        "009WT1-E": "GB",  # Good Energy Group PLC
        "0791K2-E": "NZ",  # Z Energy Limited
        "00G1G3-E": "GB",  # Atlantica Sustainable Infrastructure plc
        "001NZM-E": "GB",  # Barclays Bank PLC
        "002NKS-E": "GB",  # EBRD
        "05WCVC-E": "NZ",  # Precinct Properties New Zealand Limited
        "05HDYG-E": "IT",  # Atlantia S.p.A.
        "08D377-E": "ES",  # Ferrovial, S.A.
        "05KZ90-E": "CH",  # Holcim Ltd.
        "06GTV0-E": "FR",  # Natixis S.A.
        "0FZ1HS-E": "US",  # Avangrid, Inc.
        "0013FT-E": "US",  # Atlas Air Worldwide Holdings, Inc.
        "05LGXS-E": "CL",  # S.A.C.I. Falabella
        "05J0M3-E": "FR",  # Rothschild & Co
        "05W4YC-E": "BR",  # Cielo S.A.
        "05LG9D-E": "CL",  # SAAM S.A.
        "05KS1C-E": "MX",  # CEMEX S.A.B. de C.V.
        "05HWY4-E": "AU",  # CIMIC Group Limited
        "088LTZ-E": "US",  # KKR & Co. Inc.
        "000RGN-E": "US",  # NorthWestern Corporation
        "0B7V0T-E": "PA",  # Copa Holdings S.A.
        "0KRF06-E": "NO",  # Adevinta ASA
        "05W7VF-E": "BR",  # NotreDame Intermedica
        "05HNPP-E": "JP",  # Toshiba Corporation
        "05J0Y0-E": "FR",  # Manutan International
        "00FJY1-E": "IE",  # Keywords Studios plc
        "05JG8G-E": "BR",  # Ultrapar Participacoes SA
        "09CV3L-E": "MX",  # Grupo Aeromexico
        "05QBT3-E": "JP",  # Nikko Asset Management Co., Ltd.
        "002CM9-E": "AR",  # Banco Santander Rio S.A.
        "0CN05R-E": "BR",  # Azul S.A.
        "0B84DW-E": "FI",  # Rovio Entertainment Corporation
        "0HDSVK-E": "ES",  # Greenalia, S.A.
        "05HF52-E": "CH",  # Syngenta Group Co. Ltd.
        "0MKL64-E": "US",  # ThoughtWorks Holding, Inc.
        "06FLN6-E": "BR",  # TIM S.A.
        "0CYKFN-E": "NO",  # Magseis Fairfield ASA
        "05FWKH-E": "GB",  # Santander UK plc.
        "0DT7Z3-E": "AU",  # BWX Limited
        "0074TY-E": "GB",  # Blancco Technology Group plc
        "0Q5N23-E": "US",  # ChampionX Corp.
        "05GXPR-E": "NL",  # de Volksbank N.V.
        "05LGFR-E": "CL",  # Echeverria Izquierdo S.A.
        "000GRQ-E": "US",  # Flowserve Corp.
        "05L0DP-E": "MX",  # Grupo Bimbo S.A.B. de C.V.
        "0GVFTL-E": "SE",  # Essity AB
        "0HBX5Z-E": "GB",  # Alpha Financial Markets Consulting plc
        "05J06W-E": "GB",  # Wincanton plc
        "05FWM1-E": "CH",  # Zurcher Kantonalbank
        "0648TF-E": "CL",  # Sigdo Koppers S.A.
        "070WNP-E": "SE",  # Klarna Bank AB
        "002JGK-E": "DE",  # Bertelsmann SE & Co. KGaA
        "003JJX-E": "NL",  # Hunter Douglas N.V.
        "05WKCF-E": "CA",  # Gran Tierra Energy Inc.
        "003LD3-E": "ES",  # Siemens Gamesa Renewable Energy S.A.
        "095B5H-E": "GB",  # Capital & Counties Properties plc
        "05ZTX1-E": "MT",  # Kindred Group plc
        "05HYXM-E": "DE",  # Stada Arzneimittel AG
        "066NHL-E": "US",  # IsoRay Inc.
        "0H5F9C-E": "HK",  # Razer Inc.
        "0NL3PH-E": "BR",  # AES Brasil Energia S.A.
        "05RP6T-E": "ES",  # Mediaset Espana Comunicacion, S.A.
        "0FB2CF-E": "GB",  # Kinovo plc
        "05L0C8-E": "GB",  # Hargreaves Lansdown plc
        "05J63N-E": "AU",  # Blackmores Limited
        "060QKN-E": "DK",  # Nykredit Realkredit A/S
        "05JFP5-E": "DE",  # Beta Systems Software AG
        "0BWHLY-E": "GB",  # IQGeo Group plc
        "00FD8L-E": "GB",  # Liberty Global plc
        "07N5YW-E": "US",  # Zendesk, Inc.
        "0FS8HP-E": "NL",  # Intertrust N.V.
        "071212-E": "AU",  # Pendal Group Limited
        "06CKF3-E": "AU",  # Crown Resorts Limited
        "0DXZ08-E": "GB",  # Sureserve Group plc
        "004NV8-E": "JP",  # Toyota Motor Corp.
        "06K972-E": "ID",  # Silkroad Nickel Ltd
        "00D7D5-E": "IE",  # Aptiv PLC
        "05J0W7-E": "SE",  # Haldex AB
        "008SGW-E": "SE",  # Radisson Hospitality AB
        "061F6G-E": "US",  # Discover Financial Services
        "001TJJ-E": "CA",  # Teck Resources Ltd.
        "0K9V1T-E": "BM",  # Noble Group Holdings Limited
        "05HHCW-E": "FR",  # Electricite de France S.A.
        "05J0M6-E": "GB",  # John Menzies plc
        "05JSCJ-E": "GB",  # Alliance Pharma plc
        "009R8Q-E": "GG",  # HarbourVest Global Private Equity Limited
        "00353K-E": "GB",  # RIT Capital Partners plc
        "06LWW0-E": "FR",  # La Banque Postale S.A.
        "0GQKH2-E": "US",  # Momentive Global Inc.
        "05HYWS-E": "FR",  # Somfy SA
        "003B1C-E": "US",  # Bunge Limited
        "05J03V-E": "JP",  # Kintetsu World Express, Inc.
        "0GR7BF-E": "SE",  # Urb-it AB
        "0KBSXG-E": "US",  # WestRock Company
        "060J6T-E": "US",  # Iteris, Inc.
        "05MR64-E": "SE",  # Ahlsell AB
        "063WPG-E": "US",  # Splunk Inc.
        "05HX13-E": "TW",  # Shin Kong Financial Holding Co., Ltd.
        "000RG6-E": "US",  # NW Natural
        "0BZ4L0-E": "BM",  # GasLog Ltd.
        "06QZYR-E": "DE",  # Blue Cap AG
        "079SG6-E": "US",  # Activision Blizzard, Inc.
        "05YH2X-E": "IN",  # Sintex Industries Limited
        "0012KJ-E": "US",  # Tupperware Brands Corporation
        "05VCZW-E": "IN",  # Tata Coffee Limited
        "0015CS-E": "US",  # IBRD (World Bank)
        "06J41J-E": "BR",  # JBS S.A.
        "0JVPW7-E": "IT",  # Guala Closures S.p.A.
        "09CVYQ-E": "LU",  # L'Occitane International S.A.
        "05HS8B-E": "DE",  # Leoni AG
        "001K4D-E": "US",  # World Fuel Services Corp.
    }
    needs_hq = (combined["hq_country"].isna()) | (combined["hq_country"].astype(str).str.strip() == "")
    if needs_hq.any():
        manual_fill = combined.loc[needs_hq, "factset_entity_id"].map(_MANUAL_HQ)
        combined.loc[needs_hq, "hq_country"] = manual_fill.fillna("")
        filled = manual_fill.notna().sum()
        still_missing = needs_hq.sum() - filled
        print(f"  HQ country: filled {filled} rows from manual mapping, {still_missing} still missing")

    # Populate isocode from country name if missing/empty
    if "country" in combined.columns:
        needs_iso = combined["isocode"].isna() | (combined["isocode"].astype(str).str.strip() == "")
        if needs_iso.any():
            combined.loc[needs_iso, "isocode"] = combined.loc[needs_iso, "country"].map(_COUNTRY_TO_ISO3).fillna("")

    # Normalize country names (merge ISO2 codes, variants, and Spanish names)
    _COUNTRY_NORMALIZE = {
        "BR": "Brazil", "CA": "Canada", "FR": "France", "MX": "Mexico",
        "TH": "Thailand", "US": "United States",
        "Viet Nam": "Vietnam", "Russian Federation": "Russia",
        "Congo, The Democratic Republic of The": "DR Congo",
        "Congo, the Democratic Republic of the": "DR Congo",
        "Lao People's Democratic Republic": "Laos",
        "Korea, Republic of": "South Korea",
        "Tanzania, United Republic of": "Tanzania",
        "Bolivia Plurinational State of": "Bolivia",
        "United States of America": "United States",
        # Spanish names from BioCarbon
        "Brasil": "Brazil", "Estados Unidos": "United States",
        "Kenia": "Kenya", "Malasia": "Malaysia",
        "Panamá": "Panama", "Perú": "Peru",
        "Turquía": "Turkey", "México": "Mexico",
    }
    if "country" in combined.columns:
        combined["country"] = combined["country"].replace(_COUNTRY_NORMALIZE)

    # Step 7: Save output
    print("\n=== Saving outputs ===")
    out_path = Path(config["output"]["matched_retirements"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure quantity is numeric
    if "quantity" in combined.columns:
        combined["quantity"] = pd.to_numeric(combined["quantity"], errors="coerce").fillna(0)
    if "retirement_year" in combined.columns:
        combined["retirement_year"] = pd.to_numeric(combined["retirement_year"], errors="coerce")

    # Convert mixed-type object columns to string to avoid parquet errors
    for col in combined.select_dtypes(include=["object"]).columns:
        combined[col] = combined[col].fillna("").astype(str).replace("nan", "").replace("None", "")

    combined.to_parquet(out_path, index=False)
    print(f"  Matched retirements: {out_path} ({len(combined):,} rows)")

    # Save CSV for download (matched only, key columns)
    csv_path = out_path.with_suffix(".csv")
    key_cols = [c for c in _OUTPUT_COLS if c in combined.columns]
    matched_only = combined[combined["factset_entity_id"].notna() & (combined["factset_entity_id"] != "")]
    matched_only[key_cols].to_csv(csv_path, index=False)
    print(f"  CSV (matched only): {csv_path} ({len(matched_only):,} rows)")

    # Summary stats
    stats = build_summary_stats(combined, config["output"]["summary_stats"])

    # Map data (public_firms already loaded above)
    build_map_data(combined, public_firms)

    print(f"\n=== Pipeline complete ===")
    print(f"  Total retirements: {stats['total_retirements']:,}")
    print(f"  Matched to firms: {stats['matched_retirements']:,}")
    print(f"  Unique firms: {stats['unique_firms']:,}")
    print(f"  Total MtCO2: {stats['total_mtco2']}")
    print(f"  Matched MtCO2: {stats['matched_mtco2']}")


def main():
    parser = argparse.ArgumentParser(description="Run carbon offset matching pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Use local registry files instead of downloading")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM matching (cache only)")
    parser.add_argument("--version", type=str, default=None, help="Berkeley VROD version (e.g. 2026-02)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.version:
        config.setdefault("sources", {})["berkeley_version"] = args.version
    run_pipeline(config, skip_download=args.skip_download, skip_llm=args.skip_llm)


if __name__ == "__main__":
    main()
