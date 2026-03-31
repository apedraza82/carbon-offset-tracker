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
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Argentina": "ARG",
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
    if "isocode" in listed.columns:
        proj = listed.groupby("isocode")[qty_col].sum().reset_index()
        proj.columns = ["iso3", "tonnes"]
        proj = proj[proj["tonnes"] > 0]
    elif "country" in listed.columns:
        proj = listed.groupby("country")[qty_col].sum().reset_index()
        proj.columns = ["country_name", "tonnes"]
        proj["iso3"] = proj["country_name"].map(_COUNTRY_TO_ISO3)
        proj = proj.dropna(subset=["iso3"])
        proj = proj[proj["tonnes"] > 0][["iso3", "tonnes"]]
    else:
        proj = pd.DataFrame(columns=["iso3", "tonnes"])

    # HQ country map (merge with public_firms for iso_country)
    if not public_firms.empty:
        merged = listed.merge(
            public_firms[["factset_entity_id", "iso_country"]],
            on="factset_entity_id", how="left",
        )
        hq = merged.groupby("iso_country")[qty_col].sum().reset_index()
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


def run_pipeline(config: dict, skip_download: bool = False, skip_llm: bool = False):
    """Run the full pipeline."""

    # Step 1: Download new data
    if skip_download:
        print("=== Skipping download (using local registry files) ===")
        # Load from local registry files
        registry_dir = Path(config["sources"]["registry_dir"])
        new_retirements = {}
        for reg, fname in config["sources"]["registry_files"].items():
            fpath = registry_dir / fname
            if fpath.exists():
                print(f"  Loading {reg} from {fpath}")
                new_retirements[reg] = pd.read_excel(fpath)
            else:
                print(f"  Warning: {fpath} not found")
    else:
        version = config.get("sources", {}).get("berkeley_version")
        new_retirements = run_download_pipeline(config, version=version)

    if not new_retirements:
        print("No new retirements to process.")
        return

    # Step 2: Parse beneficiary names
    print("\n=== Parsing beneficiary names ===")
    all_parsed = []
    for registry, df in new_retirements.items():
        parsed = parse_retirements(df, registry)
        all_parsed.append(parsed)

    combined = pd.concat(all_parsed, ignore_index=True)
    print(f"  Total parsed: {len(combined):,}")

    # Step 3: Match to firms
    print("\n=== Matching to firms ===")
    matcher = FirmMatcher.from_files(config)

    unique_names = combined["raw_beneficiary"].unique().tolist()
    print(f"  Unique beneficiary names: {len(unique_names):,}")

    # Cache matching
    cache_results, unmatched_names = matcher.match_batch_cache(unique_names)
    print(f"  Cache hits: {len(cache_results):,}")
    print(f"  Unmatched: {len(unmatched_names):,}")

    # LLM matching for remaining
    llm_results = []
    if not skip_llm and unmatched_names:
        print(f"\n  Sending {len(unmatched_names):,} names to LLM...")
        llm_results = matcher.match_batch_llm(unmatched_names)

        # Update cache with new matches
        matcher.update_cache(llm_results)

        matched_by_llm = sum(1 for r in llm_results if r.factset_entity_id)
        print(f"  LLM matched: {matched_by_llm:,} / {len(unmatched_names):,}")

    # Build name→result lookup
    all_results = cache_results + llm_results
    result_map = {}
    for r in all_results:
        if r.factset_entity_id:
            result_map[r.raw_name] = r

    # Merge matches back to retirement data
    combined["factset_entity_id"] = combined["raw_beneficiary"].map(
        lambda x: result_map[x].factset_entity_id if x in result_map else None
    )
    combined["matched_name"] = combined["raw_beneficiary"].map(
        lambda x: result_map[x].matched_name if x in result_map else None
    )
    combined["match_confidence"] = combined["raw_beneficiary"].map(
        lambda x: result_map[x].confidence if x in result_map else "none"
    )
    combined["match_method"] = combined["raw_beneficiary"].map(
        lambda x: result_map[x].match_method if x in result_map else "unmatched"
    )

    # Step 4: Save output
    print("\n=== Saving outputs ===")
    out_path = Path(config["output"]["matched_retirements"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert mixed-type object columns to string to avoid parquet errors
    # Skip columns that should stay as-is (datetime, numeric)
    for col in combined.select_dtypes(include=["object"]).columns:
        combined[col] = combined[col].fillna("").astype(str).replace("nan", "").replace("None", "")

    combined.to_parquet(out_path, index=False)
    print(f"  Matched retirements: {out_path} ({len(combined):,} rows)")

    # Also save as CSV for easy download (matched only, key columns)
    csv_path = out_path.with_suffix(".csv")
    key_cols = [c for c in [
        "raw_beneficiary", "matched_name", "factset_entity_id", "registry",
        "retirement_year", "country", "quantity", "match_confidence", "match_method",
        "projectname", "projecttype", "vintage",
    ] if c in combined.columns]
    matched_only = combined[combined["factset_entity_id"].notna() & (combined["factset_entity_id"] != "")]
    matched_only[key_cols].to_csv(csv_path, index=False)

    # Summary stats
    stats = build_summary_stats(combined, config["output"]["summary_stats"])

    # Map data
    pf_path = Path(config["output"]["public_firms"])
    public_firms = pd.read_parquet(pf_path) if pf_path.exists() else pd.DataFrame()
    build_map_data(combined, public_firms)

    print(f"\n=== Pipeline complete ===")
    print(f"  Total retirements: {stats['total_retirements']:,}")
    print(f"  Matched to firms: {stats['matched_retirements']:,}")
    print(f"  Unique firms: {stats['unique_firms']:,}")


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
