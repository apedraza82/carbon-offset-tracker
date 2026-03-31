"""Build lookup tables from existing data sources.

Usage:
    python -m src.build_lookup [--known-matches] [--public-firms] [--all]

This is a one-time script to build:
1. data/known_matches.csv — cached name→FactSet mappings from existing matched data
2. data/public_firms.parquet — FactSet public company universe
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import load_config, normalize_name


def build_known_matches(config: dict) -> pd.DataFrame:
    """Extract known company→FactSet matches from existing retirements data."""
    src = config["sources"]["matched_retirements"]
    print(f"Reading matched retirements from: {src}")

    df = pd.read_parquet(src)
    matched = df[df["factset_entity_id"] != ""].copy()
    print(f"  {len(matched):,} matched retirement rows ({matched['factset_entity_id'].nunique()} unique firms)")

    # Extract unique (company, official_name, factset_entity_id) tuples
    lookup = (
        matched.groupby("factset_entity_id")
        .agg(
            official_name=("official_name", "first"),
            company_names=("company", lambda x: list(x.unique())),
            n_retirements=("company", "size"),
        )
        .reset_index()
    )

    # Explode company_names to get one row per (company_name, factset_entity_id)
    rows = []
    suffixes = config["normalization"]["legal_suffixes"]
    for _, row in tqdm(lookup.iterrows(), total=len(lookup), desc="Building known matches"):
        fid = row["factset_entity_id"]
        official = row["official_name"]
        for name in row["company_names"]:
            if not name or not isinstance(name, str) or name.strip() == "":
                continue
            rows.append({
                "company_raw": name.strip(),
                "company_normalized": normalize_name(name, suffixes),
                "official_name": official,
                "factset_entity_id": fid,
            })

    result = pd.DataFrame(rows).drop_duplicates(subset=["company_normalized", "factset_entity_id"])
    print(f"  {len(result):,} unique name-to-firm mappings")

    # Save
    out_path = Path(config["output"]["known_matches"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return result


def build_public_firms(config: dict) -> pd.DataFrame:
    """Extract public company universe from FactSet entity coverage."""
    src = config["sources"]["factset_entity_coverage"]
    print(f"Reading FactSet entity coverage from: {src}")
    print("  (5.9 GB file — reading in chunks...)")

    # Read in chunks, filter for PUB entities
    chunks = []
    reader = pd.read_stata(src, chunksize=100_000)
    total_rows = 0
    pub_rows = 0

    for chunk in tqdm(reader, desc="Reading entity coverage"):
        total_rows += len(chunk)
        pub = chunk[chunk["entity_type"] == "PUB"].copy()
        if len(pub) > 0:
            chunks.append(pub)
            pub_rows += len(pub)

    firms = pd.concat(chunks, ignore_index=True)
    print(f"  {total_rows:,} total entities, {pub_rows:,} public")

    # Select columns
    keep_cols = [
        "factset_entity_id", "entity_name", "entity_proper_name",
        "iso_country", "sector_code", "industry_code", "primary_sic_code",
    ]
    firms = firms[[c for c in keep_cols if c in firms.columns]].copy()

    # Add normalized name for matching
    suffixes = config["normalization"]["legal_suffixes"]
    firms["entity_name_normalized"] = firms["entity_proper_name"].apply(
        lambda x: normalize_name(x, suffixes) if isinstance(x, str) else ""
    )

    # Add identifiers (ISIN, LEI)
    id_src = config["sources"]["factset_entity_identifiers"]
    print(f"Reading identifiers from: {id_src}")

    pub_ids = set(firms["factset_entity_id"])
    id_chunks = []
    reader2 = pd.read_stata(id_src, chunksize=100_000)

    for chunk in tqdm(reader2, desc="Reading identifiers"):
        filtered = chunk[
            (chunk["factset_entity_id"].isin(pub_ids))
            & (chunk["entity_id_type"].isin(["ISIN", "LEI"]))
        ]
        if len(filtered) > 0:
            id_chunks.append(filtered)

    if id_chunks:
        ids = pd.concat(id_chunks, ignore_index=True)
        # Pivot: one row per entity, columns for each id type
        ids_pivot = ids.pivot_table(
            index="factset_entity_id",
            columns="entity_id_type",
            values="entity_id_value",
            aggfunc="first",
        ).reset_index()
        ids_pivot.columns.name = None

        # Rename columns to lowercase
        rename_map = {c: c.lower() for c in ids_pivot.columns if c != "factset_entity_id"}
        ids_pivot = ids_pivot.rename(columns=rename_map)
        firms = firms.merge(ids_pivot, on="factset_entity_id", how="left")
        id_cols_added = [c for c in rename_map.values() if c in firms.columns]
        for c in id_cols_added:
            print(f"  Added {c.upper()} for {firms[c].notna().sum():,} firms")

    # Save
    out_path = Path(config["output"]["public_firms"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    firms.to_parquet(out_path, index=False)
    print(f"  Saved {len(firms):,} public firms to {out_path}")

    return firms


def main():
    parser = argparse.ArgumentParser(description="Build lookup tables for carbon offset tracker")
    parser.add_argument("--known-matches", action="store_true", help="Build known_matches.csv")
    parser.add_argument("--public-firms", action="store_true", help="Build public_firms.parquet")
    parser.add_argument("--all", action="store_true", help="Build all lookup tables")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    if not any([args.known_matches, args.public_firms, args.all]):
        args.all = True

    config = load_config(args.config)

    if args.all or args.known_matches:
        build_known_matches(config)

    if args.all or args.public_firms:
        build_public_firms(config)

    print("\nDone!")


if __name__ == "__main__":
    main()
