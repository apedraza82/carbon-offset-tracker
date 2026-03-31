"""Registry-specific beneficiary name extraction.

Each registry stores the retirement beneficiary in different columns with different
formats. This module provides a unified interface to extract the beneficiary name.
"""

import re
from dataclasses import dataclass

import pandas as pd

from src.utils import extract_on_behalf_of


@dataclass
class BeneficiaryResult:
    """Result of beneficiary extraction for a single retirement."""
    raw_name: str
    source_field: str  # which column the name came from
    registry: str


def clean_raw_name(name: str) -> str:
    """Basic cleaning of raw beneficiary string."""
    if not name or not isinstance(name, str):
        return ""

    # Strip whitespace
    name = name.strip()

    # Normalize unicode whitespace
    name = re.sub(r"\s+", " ", name)

    # Remove Brazilian CNPJ (XX.XXX.XXX/XXXX-XX)
    name = re.sub(r"\s*\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\s*", " ", name)

    # Remove generic tax ID patterns at end
    name = re.sub(r"\s*[-–]\s*\d{5,}$", "", name)

    # Remove email addresses
    name = re.sub(r"\S+@\S+\.\S+", "", name)

    # Remove URLs
    name = re.sub(r"https?://\S+", "", name)

    # Collapse whitespace again
    name = re.sub(r"\s+", " ", name).strip()

    return name


def parse_verra(row: pd.Series) -> BeneficiaryResult | None:
    """Extract beneficiary from Verra retirement row.

    Primary: 'Retirement Beneficiary'
    Fallback: 'Retirement Details' (parse "on behalf of X")
    """
    # Primary field
    beneficiary = row.get("Retirement Beneficiary", "")
    if isinstance(beneficiary, str) and beneficiary.strip():
        cleaned = clean_raw_name(beneficiary)
        if cleaned:
            return BeneficiaryResult(raw_name=cleaned, source_field="Retirement Beneficiary", registry="Verra")

    # Fallback: parse from Retirement Details
    details = row.get("Retirement Details", "")
    if isinstance(details, str) and details.strip():
        name = extract_on_behalf_of(details)
        if name:
            return BeneficiaryResult(raw_name=clean_raw_name(name), source_field="Retirement Details", registry="Verra")

    return None


def parse_gold(row: pd.Series) -> BeneficiaryResult | None:
    """Extract beneficiary from Gold Standard retirement row.

    Primary: '* Using Entity'
    Fallback: 'Note' (parse "on behalf of X" / "for X")
    """
    using = row.get("* Using Entity", "") or row.get("Using Entity", "")
    if isinstance(using, str) and using.strip():
        cleaned = clean_raw_name(using)
        if cleaned:
            return BeneficiaryResult(raw_name=cleaned, source_field="* Using Entity", registry="Gold")

    note = row.get("Note", "")
    if isinstance(note, str) and note.strip():
        name = extract_on_behalf_of(note)
        if name:
            return BeneficiaryResult(raw_name=clean_raw_name(name), source_field="Note", registry="Gold")

    return None


def parse_acr(row: pd.Series) -> BeneficiaryResult | None:
    """Extract beneficiary from ACR retirement row.

    Primary: 'Retired on Behalf of'
    Fallback: 'Account Holder'
    """
    behalf = row.get("Retired on Behalf of", "")
    if isinstance(behalf, str) and behalf.strip():
        cleaned = clean_raw_name(behalf)
        if cleaned:
            return BeneficiaryResult(raw_name=cleaned, source_field="Retired on Behalf of", registry="ACR")

    holder = row.get("Account Holder", "")
    if isinstance(holder, str) and holder.strip():
        cleaned = clean_raw_name(holder)
        if cleaned:
            return BeneficiaryResult(raw_name=cleaned, source_field="Account Holder", registry="ACR")

    return None


def parse_car(row: pd.Series) -> BeneficiaryResult | None:
    """Extract beneficiary from CAR retirement row.

    Primary: 'Account Holder'
    Fallback: 'Retirement Reason Details' (parse for company names)
    """
    holder = row.get("Account Holder", "")
    if isinstance(holder, str) and holder.strip():
        cleaned = clean_raw_name(holder)
        if cleaned:
            return BeneficiaryResult(raw_name=cleaned, source_field="Account Holder", registry="CAR")

    details = row.get("Retirement Reason Details", "")
    if isinstance(details, str) and details.strip():
        name = extract_on_behalf_of(details)
        if name:
            return BeneficiaryResult(raw_name=clean_raw_name(name), source_field="Retirement Reason Details", registry="CAR")

    return None


# Registry parser dispatch
PARSERS = {
    "verra": parse_verra,
    "gold": parse_gold,
    "acr": parse_acr,
    "car": parse_car,
}


# Column name harmonization: registry-specific -> unified names
_COLUMN_RENAMES = {
    "verra": {
        "Quantity Issued": "quantity",
        "Country/Area": "country",
        "Retirement/Cancellation Date": "retirement_date",
        "Name": "projectname",
        "Project Type": "projecttype",
        "ID": "project_id",
    },
    "gold": {
        "Quantity": "quantity",
        "Country": "country",
        "Retirement Date": "retirement_date",
        "Project Name": "projectname",
        "Project Type": "projecttype",
        "GSID": "project_id",
    },
    "acr": {
        "Quantity of Credits": "quantity",
        "Project Site Country": "country",
        "Status Effective (GMT)": "retirement_date",
        "Project Name": "projectname",
        "Project Type": "projecttype",
        "Project ID": "project_id",
    },
    "car": {
        "Quantity of Offset Credits": "quantity",
        "Project Site Country": "country",
        "Status Effective": "retirement_date",
        "Project Name": "projectname",
        "Project Type": "projecttype",
        "Project ID": "project_id",
    },
}


def parse_retirements(df: pd.DataFrame, registry: str) -> pd.DataFrame:
    """Parse beneficiary names from a registry DataFrame.

    Args:
        df: Raw registry data
        registry: One of 'verra', 'gold', 'acr', 'car'

    Returns:
        DataFrame with columns: raw_beneficiary, source_field, registry,
        plus harmonized columns (quantity, country, retirement_date, etc.).
    """
    parser = PARSERS.get(registry.lower())
    if parser is None:
        raise ValueError(f"Unknown registry: {registry}. Must be one of {list(PARSERS.keys())}")

    results = []
    for idx, row in df.iterrows():
        result = parser(row)
        if result:
            results.append({
                "original_index": idx,
                "raw_beneficiary": result.raw_name,
                "source_field": result.source_field,
                "registry": result.registry,
            })

    parsed = pd.DataFrame(results)
    if len(parsed) == 0:
        return pd.DataFrame(columns=["raw_beneficiary", "source_field", "registry"])

    # Merge back original columns
    parsed = parsed.set_index("original_index")
    out = df.join(parsed, how="inner")

    # Harmonize column names (rename registry-specific -> unified)
    renames = _COLUMN_RENAMES.get(registry.lower(), {})
    # Only rename columns that exist and whose target doesn't already exist
    actual_renames = {k: v for k, v in renames.items() if k in out.columns and v not in out.columns}
    out = out.rename(columns=actual_renames)

    # Ensure quantity is numeric
    if "quantity" in out.columns:
        out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").fillna(0)

    # Parse retirement year
    if "retirement_date" in out.columns:
        out["retirement_date"] = pd.to_datetime(out["retirement_date"], errors="coerce")
        out["retirement_year"] = out["retirement_date"].dt.year

    # Parse vintage
    if "Vintage Start" in out.columns and "vintage" not in out.columns:
        out["vintage"] = pd.to_datetime(out["Vintage Start"], errors="coerce").dt.year
    elif "Vintage" in out.columns and "vintage" not in out.columns:
        out.rename(columns={"Vintage": "vintage"}, inplace=True)

    print(f"  [{registry.upper()}] Parsed {len(out):,} / {len(df):,} rows "
          f"({100*len(out)/len(df):.1f}%)")

    return out
