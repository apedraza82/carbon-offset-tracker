# Carbon Offset Tracker

A public dataset matching voluntary carbon offset retirements to publicly listed firms, updated monthly from registry data.

## Overview

This project automatically:
1. **Downloads** retirement data from the [Berkeley Carbon Trading Project](https://gspp.berkeley.edu/research/osf-bctp/offsets-database) (Verra, Gold Standard, ACR, CAR)
2. **Parses** retirement beneficiary names from registry-specific formats
3. **Matches** beneficiaries to publicly listed firms using cached lookups + Claude API for entity resolution
4. **Publishes** a matched dataset, interactive dashboard, and landing page

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build lookup tables (one-time, requires FactSet data access)
python -m src.build_lookup --all

# Run pipeline (cache-only matching, no downloads)
python -m src.pipeline --skip-download --skip-llm

# Run pipeline with LLM matching (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.pipeline --skip-download

# Run full pipeline (download + match)
python -m src.pipeline

# Launch interactive dashboard
streamlit run dashboard/app.py
```

## Architecture

```
carbon-offset-tracker/
├── src/
│   ├── build_lookup.py        # One-time: build known_matches + public_firms
│   ├── download.py            # Download Berkeley data
│   ├── parse_beneficiary.py   # Registry-specific beneficiary extraction
│   ├── match_firms.py         # Cache lookup + Claude API matching
│   ├── pipeline.py            # End-to-end orchestration
│   └── utils.py               # Name normalization, config loading
├── dashboard/
│   └── app.py                 # Streamlit interactive dashboard
├── site/
│   └── index.html             # GitHub Pages landing page
├── data/
│   ├── known_matches.csv      # 3,000+ cached name→firm mappings
│   ├── public_firms.parquet   # ~50K public company universe
│   └── matched_retirements.*  # Output dataset
├── .github/workflows/
│   └── monthly_update.yml     # Automated monthly pipeline
├── config.yaml                # Data source paths, thresholds
└── requirements.txt
```

## Matching Pipeline

1. **Registry-specific parsing**: Each registry stores beneficiary names differently
   - Verra: `Retirement Beneficiary` field
   - Gold Standard: `* Using Entity` field
   - ACR: `Retired on Behalf of` field
   - CAR: `Account Holder` field

2. **Cache lookup**: Exact and normalized name matching against 3,000+ known mappings

3. **LLM matching**: For unmatched names, Claude Haiku resolves subsidiaries/brands to parent listed firms (e.g., Nespresso → Nestlé). Cost: ~$0.50-2/month.

4. **Confidence scoring**: HIGH (auto-accept + cache), MEDIUM (auto-accept + flag), LOW (manual review)

## Data

| File | Description |
|------|-------------|
| `matched_retirements.parquet` | All retirement transactions matched to firms |
| `known_matches.csv` | Beneficiary name → FactSet entity ID cache |
| `public_firms.parquet` | Public company universe with identifiers |
| `summary_stats.json` | Aggregate statistics for landing page |

## Citation

If you use this dataset, please cite:

> Pedraza, A., Williams, T., & Zeni, F. (2025). "Local Visibility vs. Global Integrity: Evidence from Corporate Carbon Offset Portfolios." Working Paper.

## License

Data is provided for research purposes. See the paper for details on methodology and limitations.
