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

    # Remove Colombian NIT (e.g., NIT 890.903.938-4 or NIT: 890903938-4)
    name = re.sub(r"\s*\bNIT[:\s]*[\d.\-]+\s*", " ", name, flags=re.IGNORECASE)

    # Remove standalone NIT-like patterns (9-10 digits with optional check digit)
    name = re.sub(r"\s+\d{3}\.?\d{3}\.?\d{3}[\-.]?\d\b", "", name)

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


def parse_ecoregistry(row: pd.Series) -> BeneficiaryResult | None:
    """Extract beneficiary from EcoRegistry (Cercarbono) retirement row.

    Primary: 'final_user'
    """
    name = row.get("final_user", "")
    if isinstance(name, str) and name.strip():
        cleaned = clean_raw_name(name)
        if cleaned:
            return BeneficiaryResult(raw_name=cleaned, source_field="final_user", registry="EcoRegistry")
    return None


def parse_biocarbon(row: pd.Series) -> BeneficiaryResult | None:
    """Extract beneficiary from BioCarbon (Global CarbonTrace) retirement row.

    Primary: 'beneficiary' (to_name — the entity that retired)
    Fallback: 'final_user' (end user, sometimes different from beneficiary)
    """
    name = row.get("beneficiary", "")
    if isinstance(name, str) and name.strip():
        cleaned = clean_raw_name(name)
        if cleaned:
            return BeneficiaryResult(raw_name=cleaned, source_field="beneficiary", registry="BioCarbon")

    name = row.get("final_user", "")
    if isinstance(name, str) and name.strip():
        cleaned = clean_raw_name(name)
        if cleaned:
            return BeneficiaryResult(raw_name=cleaned, source_field="final_user", registry="BioCarbon")

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


_CDM_JUNK = re.compile(
    r"(?:identificado|this cancellation|of this|GP-\d+|by Planton|"
    r"Program Managed by|'s cus$|'s client$)",
    re.IGNORECASE,
)


def _cdm_clean(name: str) -> str:
    """Extra cleaning for CDM beneficiary names."""
    if not name:
        return ""
    # Strip trailing "identificado con..." (Colombian NIT pattern)
    name = re.sub(r"\s+identificado.*$", "", name, flags=re.IGNORECASE)
    # Strip trailing "in order..." or "in 202X"
    name = re.sub(r"\s+in\s+(?:order|20\d\d).*$", "", name, flags=re.IGNORECASE)
    # "Lenovo Clients 1 May..." -> "Lenovo"
    name = re.sub(r"\s+Clients?\s+\d.*$", "", name)
    # Strip quotes
    name = name.strip('"\'')
    return name.strip()


def _cdm_result(name: str) -> BeneficiaryResult | None:
    """Validate and return a CDM beneficiary result, or None if junk."""
    name = _cdm_clean(name)
    cleaned = clean_raw_name(name)
    if not cleaned or len(cleaned) <= 2:
        return None
    if _CDM_JUNK.search(cleaned):
        return None
    return BeneficiaryResult(raw_name=cleaned, source_field="Purpose", registry="CDM")


def parse_cdm(row: pd.Series) -> BeneficiaryResult | None:
    """Extract beneficiary from CDM voluntary cancellation row.

    The 'Purpose' field contains free-form text describing who cancelled and why.
    """
    purpose = row.get("Purpose", "")
    if not isinstance(purpose, str) or not purpose.strip():
        return None

    # Strip leading "Purpose:" prefix and quotes
    purpose = re.sub(r'^"?\s*Purpose:\s*', '', purpose).strip().strip('"')

    # Try standard "on behalf of" extraction first
    name = extract_on_behalf_of(purpose)
    if name:
        result = _cdm_result(name)
        if result:
            return result

    # Try Spanish patterns: "a favor de X" / "a nombre de X"
    m = re.search(r"(?:a favor de|a nombre de)\s+(.+?)(?:\s+para\s|\s+con\s|$)", purpose, re.IGNORECASE)
    if m:
        result = _cdm_result(m.group(1).split(",")[0])
        if result:
            return result

    # "Cancelled to neutralize... of/by X" or "To offset... of X"
    m = re.search(
        r"(?:neutralize|offset|compensate)\s+(?:the\s+)?(?:carbon\s+)?(?:emissions?\s+)?(?:generated\s+)?(?:by|of|from)\s+(.+?)(?:\s+in\s+\d|\s+from\s|\s+for\s|$)",
        purpose, re.IGNORECASE
    )
    if m:
        candidate = re.split(r"\s+(?:in|from|for|during)\s+\d", m.group(1).strip().rstrip("."))[0]
        result = _cdm_result(candidate)
        if result:
            return result

    # "X purchase(d) N CERs/tons/credits"
    m = re.match(r"^(.+?)\s+purchas(?:e[ds]?)\s+\d", purpose, re.IGNORECASE)
    if m:
        result = _cdm_result(m.group(1))
        if result:
            return result

    # "Sponsored by X"
    m = re.search(r"[Ss]ponsored by\s+(.+?)(?:,|\s+to\s)", purpose)
    if m:
        result = _cdm_result(m.group(1))
        if result:
            return result

    # Brazilian pattern: "COMPANY_NAME S.A./LTDA" at start
    m = re.match(r"^([A-Z][A-Z\s&.]+(?:S\.?A\.?|LTDA|S/A))", purpose)
    if m:
        result = _cdm_result(m.group(1))
        if result:
            return result

    # "Cancelled by X"
    m = re.search(r"[Cc]ancelled by\s+(.+?)(?:\s+to\s|\s+for\s|$)", purpose)
    if m:
        result = _cdm_result(m.group(1))
        if result:
            return result

    # "carbon neutrality/footprint of X" or "GHG emissions of/from X"
    m = re.search(r"(?:carbon neutrality|carbon footprint|GHG emissions?)\s+(?:of|from)\s+(.+?)(?:\s+in\s+\d|\s+from\s+\d|\s+for\s|$)", purpose, re.IGNORECASE)
    if m:
        result = _cdm_result(m.group(1).rstrip("."))
        if result:
            return result

    # "COMPANY, CNPJ..." at start (Brazilian with CNPJ)
    m = re.match(r"^(.+?)(?:,\s*CNPJ|,\s*inscrita|,\s*localizada|\s+neutraliz|\s+offset)", purpose, re.IGNORECASE)
    if m and len(m.group(1)) > 3:
        result = _cdm_result(m.group(1))
        if result:
            return result

    # "X retires/retired the CERs"
    m = re.match(r"^(.+?)\s+retire[sd]?\s+(?:the\s+)?(?:CERs|credits)", purpose, re.IGNORECASE)
    if m:
        result = _cdm_result(m.group(1))
        if result:
            return result

    # German: "Klimaneutrales Unternehmen X, City"
    m = re.search(r"Klimaneutrales? Unternehmen\s+(.+?)(?:,\s*\w+,\s*(?:für|Deutschland|Germany|Frankreich)|\s+für)", purpose)
    if m:
        result = _cdm_result(m.group(1).rstrip(","))
        if result:
            return result

    # Spanish: "Cancelación voluntaria realizada por X a favor de Y"
    m = re.search(r"realizada por\s+\w+\s+(?:\w+\s+)?a favor de[l]?\s+(.+?)(?:\s+para\s|\s+con\s|$)", purpose, re.IGNORECASE)
    if m:
        result = _cdm_result(m.group(1).split(",")[0])
        if result:
            return result

    # "Voluntary compensation from X to offset"
    m = re.search(r"[Vv]oluntary (?:compensation|cancellation) from\s+(.+?)\s+to\s+(?:offset|compensate)", purpose)
    if m:
        result = _cdm_result(m.group(1))
        if result:
            return result

    # Portuguese: "Neutralização/Compensação das emissões... da/do X"
    m = re.search(r"(?:Neutraliza[çc][aã]o|Compensa[çc][aã]o)\s+.{0,80}?\s+d[aeo]\s+(.+?)(?:\s+entre\s|\s+no\s|\s+em\s|\s+referente|,\s*(?:localizada|CNPJ|entre|no|em))", purpose, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().rstrip(",.")
        if len(candidate) > 3 and not candidate[0].islower():
            result = _cdm_result(candidate)
            if result:
                return result

    # "Beneficiary: X" or "Beneficiário: X"
    m = re.search(r"[Bb]enefi[cá]i[aá]r[yo][:.]?\s+(.+?)(?:\s+CNPJ|\s+End:|\s+RUT|$)", purpose)
    if m:
        result = _cdm_result(m.group(1).rstrip(","))
        if result:
            return result

    return None


_PURO_SKIP = {
    "patch buyer", "patch customer", "hidden", "anonymous", "anonymous buyer",
    "cnaught customers",
}

_PURO_GENERIC_PATTERNS = re.compile(
    r"^(patch retiring on behalf of client|"
    r"on behalf of patch|"
    r"on behalf of cloverly|"
    r"\d{8}-\d+ - customer of|"
    r"climatefi.*on behalf of beneficiary|"
    r"rvcmc on behalf of participant|"
    r"2050 on behalf of customer)",
    re.IGNORECASE,
)


def parse_puro(row: pd.Series) -> BeneficiaryResult | None:
    """Extract beneficiary from Puro.earth retirement row.

    Primary: 'beneficiary_name'
    Fallback: 'account_holder'

    Handles "Retired on behalf of X" patterns to extract actual beneficiary.
    Skips generic placeholders (Patch buyer, Anonymous, etc.).
    """
    name = row.get("beneficiary_name", "")
    if isinstance(name, str) and name.strip():
        lower = name.strip().lower()
        # Skip known placeholders
        if lower in _PURO_SKIP:
            pass
        elif _PURO_GENERIC_PATTERNS.match(lower):
            pass
        else:
            # Extract "Retired on behalf of X" / "On behalf of X"
            obo = extract_on_behalf_of(name)
            if obo:
                cleaned = clean_raw_name(obo)
                if cleaned:
                    return BeneficiaryResult(raw_name=cleaned, source_field="beneficiary_name", registry="Puro")
            else:
                cleaned = clean_raw_name(name)
                if cleaned:
                    return BeneficiaryResult(raw_name=cleaned, source_field="beneficiary_name", registry="Puro")

    holder = row.get("account_holder", "")
    if isinstance(holder, str) and holder.strip():
        cleaned = clean_raw_name(holder)
        if cleaned:
            return BeneficiaryResult(raw_name=cleaned, source_field="account_holder", registry="Puro")

    return None


# Registry parser dispatch
PARSERS = {
    "verra": parse_verra,
    "gold": parse_gold,
    "acr": parse_acr,
    "car": parse_car,
    "ecoregistry": parse_ecoregistry,
    "biocarbon": parse_biocarbon,
    "puro": parse_puro,
    "cdm": parse_cdm,
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
    "ecoregistry": {
        "quantity": "quantity",
        "country_final_user": "country",
        "date": "retirement_date",
        "reason_using": "retirement_reason",
        "serial": "serialnumber",
        "project_id": "project_id",
        "methodology": "projecttype",
        "sector": "sector",
    },
    "biocarbon": {
        "volume": "quantity",
        "destination": "market_type",
        "project_country": "country",
        "project_type": "projecttype",
        "retirement_date": "retirement_date",
        "retirement_reason": "retirement_reason",
        "serial": "serialnumber",
        "project_id": "project_id",
        "project_name": "projectname",
    },
    "puro": {
        "volume": "quantity",
        "project_country": "country",
        "completed_on": "retirement_date",
        "project_name": "projectname",
        "methodology": "projecttype",
        "retirement_purpose": "retirement_reason",
        "retirement_id": "project_id",
        "vintage": "vintage",
    },
    "cdm": {
        "Quantity of units cancelled": "quantity",
        "Host": "country",
        "Date": "retirement_date",
        "Title": "projectname",
        "Project type": "projecttype",
        "Ref.": "project_id",
        "Purpose": "retirement_reason",
        "Type": "sector",
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
