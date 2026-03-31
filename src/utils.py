"""Shared utilities for carbon offset tracker."""

import re
import unicodedata
from pathlib import Path

import yaml


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_name(name: str, legal_suffixes: list[str] | None = None) -> str:
    """Normalize a company name for matching.

    Steps:
    1. Unicode normalize (NFKD) and strip accents
    2. Lowercase
    3. Remove punctuation (keep spaces and alphanumeric)
    4. Remove legal suffixes
    5. Collapse whitespace and strip
    """
    if not name or not isinstance(name, str):
        return ""

    # Unicode normalize
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Lowercase
    name = name.lower()

    # Remove common tax IDs (Brazilian CNPJ, etc.)
    name = re.sub(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", "", name)  # CNPJ
    name = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "", name)  # SSN-like
    name = re.sub(r"\bein[:\s]*\d+-\d+\b", "", name)  # EIN

    # Remove punctuation (keep alphanumeric and spaces)
    name = re.sub(r"[^\w\s]", " ", name)

    # Remove legal suffixes
    if legal_suffixes is None:
        legal_suffixes = [
            "inc", "incorporated", "corp", "corporation", "ltd", "limited",
            "llc", "llp", "lp", "plc", "sa", "se", "ag", "gmbh", "co",
            "company", "group", "holdings", "holding", "nv", "bv", "pty",
            "pte", "srl", "spa", "ab", "asa", "as", "oyj", "tbk", "bhd",
            "sdn", "sarl", "kk", "kabushiki kaisha",
        ]

    # Build pattern: match suffixes at word boundaries
    suffix_pattern = r"\b(" + "|".join(re.escape(s) for s in legal_suffixes) + r")\b"
    name = re.sub(suffix_pattern, "", name)

    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name


def extract_on_behalf_of(text: str) -> str | None:
    """Extract entity name from 'on behalf of X' patterns."""
    if not text or not isinstance(text, str):
        return None

    patterns = [
        r"(?:on behalf of|for the benefit of|representing)\s+(.+?)(?:\s*(?:for|to|regarding|in respect|$))",
        r"(?:retired for|retired by)\s+(.+?)(?:\s*(?:for|to|$))",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = match.group(1).strip().rstrip(".,;:")
            if len(result) > 3:  # skip very short matches
                return result

    return None
