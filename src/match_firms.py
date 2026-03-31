"""Firm matching module: cache lookup + Claude API for unmatched names.

Matching pipeline:
1. Exact match on normalized name → known_matches cache
2. Fuzzy match on normalized name (optional, for near-misses)
3. Claude API (Haiku) for remaining unmatched names
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils import load_config, normalize_name

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class MatchResult:
    """Result of firm matching for a single beneficiary name."""
    raw_name: str
    matched_name: str | None = None
    factset_entity_id: str | None = None
    confidence: str = "none"  # high, medium, low, none
    match_method: str = "unmatched"  # cache_exact, cache_normalized, llm, unmatched
    reasoning: str = ""


@dataclass
class FirmMatcher:
    """Matches beneficiary names to FactSet entity IDs."""

    known_matches: pd.DataFrame = field(default_factory=pd.DataFrame)
    public_firms: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict = field(default_factory=dict)

    # Internal lookup dicts (built on load)
    _exact_lookup: dict = field(default_factory=dict, repr=False)
    _normalized_lookup: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._build_lookups()

    def _build_lookups(self):
        """Build fast lookup dictionaries from known_matches."""
        if self.known_matches.empty:
            return

        # Exact lookup: raw company name → (official_name, factset_entity_id)
        for _, row in self.known_matches.iterrows():
            raw = row.get("company_raw", "")
            if raw:
                self._exact_lookup[raw.lower().strip()] = (
                    row["official_name"],
                    row["factset_entity_id"],
                )

        # Normalized lookup: normalized name → (official_name, factset_entity_id)
        for _, row in self.known_matches.iterrows():
            norm = row.get("company_normalized", "")
            if norm:
                self._normalized_lookup[norm] = (
                    row["official_name"],
                    row["factset_entity_id"],
                )

        print(f"  Loaded {len(self._exact_lookup):,} exact + "
              f"{len(self._normalized_lookup):,} normalized lookup entries")

    @classmethod
    def from_files(cls, config: dict | None = None) -> "FirmMatcher":
        """Load matcher from saved lookup files."""
        if config is None:
            config = load_config()

        km_path = Path(config["output"]["known_matches"])
        pf_path = Path(config["output"]["public_firms"])

        known = pd.read_csv(km_path) if km_path.exists() else pd.DataFrame()
        firms = pd.read_parquet(pf_path) if pf_path.exists() else pd.DataFrame()

        print(f"Loaded known_matches: {len(known):,} rows")
        print(f"Loaded public_firms: {len(firms):,} rows")

        return cls(known_matches=known, public_firms=firms, config=config)

    def match_name(self, raw_name: str) -> MatchResult:
        """Match a single beneficiary name using cache lookups."""
        if not raw_name or not isinstance(raw_name, str):
            return MatchResult(raw_name=raw_name or "")

        # 1. Exact match (case-insensitive)
        key = raw_name.lower().strip()
        if key in self._exact_lookup:
            official, fid = self._exact_lookup[key]
            return MatchResult(
                raw_name=raw_name,
                matched_name=official,
                factset_entity_id=fid,
                confidence="high",
                match_method="cache_exact",
            )

        # 2. Normalized match
        suffixes = self.config.get("normalization", {}).get("legal_suffixes")
        norm = normalize_name(raw_name, suffixes)
        if norm and norm in self._normalized_lookup:
            official, fid = self._normalized_lookup[norm]
            return MatchResult(
                raw_name=raw_name,
                matched_name=official,
                factset_entity_id=fid,
                confidence="high",
                match_method="cache_normalized",
            )

        # 3. No cache match
        return MatchResult(raw_name=raw_name)

    def match_batch_cache(self, names: list[str]) -> tuple[list[MatchResult], list[str]]:
        """Match a batch of names using cache only. Returns (matched, unmatched_names)."""
        matched = []
        unmatched = []

        for name in names:
            result = self.match_name(name)
            if result.factset_entity_id:
                matched.append(result)
            else:
                unmatched.append(name)

        return matched, unmatched

    def match_batch_llm(
        self,
        names: list[str],
        registries: list[str] | None = None,
        countries: list[str] | None = None,
    ) -> list[MatchResult]:
        """Match unmatched names using Claude API.

        Args:
            names: List of raw beneficiary names
            registries: Optional list of registries (same length as names)
            countries: Optional list of retirement countries
        """
        if anthropic is None:
            raise ImportError("anthropic package required for LLM matching. Install with: pip install anthropic")

        if not names:
            return []

        model = self.config.get("matching", {}).get("model", "claude-haiku-4-5-20251001")
        max_tokens = self.config.get("matching", {}).get("max_tokens", 300)
        batch_size = self.config.get("matching", {}).get("batch_size", 20)

        # Build context: sample of public firms for reference
        firm_context = self._build_firm_context(countries)

        client = anthropic.Anthropic()
        all_results = []

        for i in range(0, len(names), batch_size):
            batch = names[i : i + batch_size]
            batch_registries = registries[i : i + batch_size] if registries else [None] * len(batch)
            batch_countries = countries[i : i + batch_size] if countries else [None] * len(batch)

            results = self._call_llm(
                client, model, max_tokens, batch, batch_registries, batch_countries, firm_context
            )
            all_results.extend(results)

            # Rate limiting
            if i + batch_size < len(names):
                time.sleep(0.5)

        return all_results

    def _build_firm_context(self, countries: list[str] | None = None) -> str:
        """Build a compact reference list of public firms for LLM context."""
        if self.public_firms.empty:
            return "No public firm reference available."

        firms = self.public_firms.copy()

        # If countries provided, prioritize those
        if countries:
            unique_countries = set(c for c in countries if c)
            if unique_countries:
                firms_in_country = firms[firms["iso_country"].isin(unique_countries)]
                firms_other = firms[~firms["iso_country"].isin(unique_countries)].sample(
                    min(500, len(firms)), random_state=42
                ) if len(firms) > 500 else firms[~firms["iso_country"].isin(unique_countries)]
                firms = pd.concat([firms_in_country, firms_other]).drop_duplicates("factset_entity_id")

        # Limit to reasonable size for context
        if len(firms) > 2000:
            firms = firms.sample(2000, random_state=42)

        lines = []
        for _, r in firms.iterrows():
            lines.append(f"{r['factset_entity_id']}|{r.get('entity_proper_name', '')}|{r.get('iso_country', '')}")

        return "\n".join(lines)

    def _call_llm(
        self,
        client,
        model: str,
        max_tokens: int,
        names: list[str],
        registries: list[str],
        countries: list[str],
        firm_context: str,
    ) -> list[MatchResult]:
        """Make a single Claude API call for a batch of names."""

        # Build the user message with names to match
        name_entries = []
        for j, (name, reg, ctry) in enumerate(zip(names, registries, countries)):
            entry = f"{j+1}. \"{name}\""
            if reg:
                entry += f" (registry: {reg})"
            if ctry:
                entry += f" (country: {ctry})"
            name_entries.append(entry)

        system_prompt = """You are an entity resolution assistant for carbon offset retirements.
Your task: match retirement beneficiary names to publicly listed parent companies.

Key rules:
- Subsidiaries should map to their publicly listed parent (e.g., "Nespresso" → Nestlé)
- Joint ventures map to the majority-owning listed parent
- Government entities, NGOs, and individuals are NOT matches — return null
- Brokers/intermediaries (e.g., "South Pole", "3Degrees") are NOT listed firms — return null
- Only match to the provided public firms list

For each name, respond with a JSON object:
{"index": N, "parent_name": "...", "factset_entity_id": "...", "confidence": "high|medium|low", "reasoning": "..."}

Confidence levels:
- high: Clear, unambiguous match (exact or near-exact name)
- medium: Likely match but requires subsidiary/brand knowledge
- low: Uncertain, possible match but could be wrong
- If no match found, use: {"index": N, "parent_name": null, "factset_entity_id": null, "confidence": "none", "reasoning": "..."}"""

        user_message = f"""Match these carbon offset retirement beneficiaries to publicly listed firms.

PUBLIC FIRMS REFERENCE (format: factset_id|name|country):
{firm_context[:8000]}

NAMES TO MATCH:
{chr(10).join(name_entries)}

Respond with a JSON array of match results, one per name."""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens * len(names),
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            # Parse response
            text = response.content[0].text
            # Extract JSON array from response
            json_match = _extract_json_array(text)
            if json_match:
                matches = json.loads(json_match)
            else:
                print(f"  Warning: Could not parse LLM response as JSON")
                return [MatchResult(raw_name=n, match_method="llm_error") for n in names]

            # Convert to MatchResults
            results = []
            match_dict = {m.get("index", j + 1): m for j, m in enumerate(matches)}
            for j, name in enumerate(names):
                m = match_dict.get(j + 1, {})
                results.append(MatchResult(
                    raw_name=name,
                    matched_name=m.get("parent_name"),
                    factset_entity_id=m.get("factset_entity_id"),
                    confidence=m.get("confidence", "none"),
                    match_method="llm",
                    reasoning=m.get("reasoning", ""),
                ))

            return results

        except Exception as e:
            print(f"  LLM API error: {e}")
            return [MatchResult(raw_name=n, match_method="llm_error", reasoning=str(e)) for n in names]

    def update_cache(self, results: list[MatchResult], min_confidence: str = "medium"):
        """Add successful LLM matches back to the cache."""
        confidence_order = {"high": 3, "medium": 2, "low": 1, "none": 0}
        threshold = confidence_order.get(min_confidence, 2)

        new_entries = []
        suffixes = self.config.get("normalization", {}).get("legal_suffixes")

        for r in results:
            if (
                r.factset_entity_id
                and confidence_order.get(r.confidence, 0) >= threshold
                and r.match_method == "llm"
            ):
                norm = normalize_name(r.raw_name, suffixes)
                new_entries.append({
                    "company_raw": r.raw_name,
                    "company_normalized": norm,
                    "official_name": r.matched_name or "",
                    "factset_entity_id": r.factset_entity_id,
                })

                # Update in-memory lookups
                self._exact_lookup[r.raw_name.lower().strip()] = (
                    r.matched_name or "",
                    r.factset_entity_id,
                )
                if norm:
                    self._normalized_lookup[norm] = (r.matched_name or "", r.factset_entity_id)

        if new_entries:
            new_df = pd.DataFrame(new_entries)
            self.known_matches = pd.concat([self.known_matches, new_df], ignore_index=True)
            self.known_matches.drop_duplicates(
                subset=["company_normalized", "factset_entity_id"], inplace=True
            )

            # Save updated cache
            out_path = Path(self.config["output"]["known_matches"])
            self.known_matches.to_csv(out_path, index=False)
            print(f"  Updated cache: +{len(new_entries)} entries -> {len(self.known_matches):,} total")


def _extract_json_array(text: str) -> str | None:
    """Extract a JSON array from LLM response text."""
    # Try to find [...] in the text
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None
