# Carbon Offset Tracker

A public dataset matching voluntary carbon offset retirements to publicly listed firms, updated monthly from registry data.

**[Landing Page](https://apedraza82.github.io/carbon-offset-tracker/)** | **[Interactive Dashboard](https://carbon-offset-tracker-dsxjevtd5fj7bife2dnxmr.streamlit.app/)** | **[Research Paper](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099338203162614529)**

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total retirements | 428,464 |
| Matched to listed firms | 38,307 |
| Unique public firms | 1,897 |
| Total volume | 1,057 MtCO2 |
| Matched volume | 336 MtCO2 |
| Registries | 8 |
| Project countries | 98 |
| Coverage | 2004–2026 |

## Data Sources

| Registry | Source | Retirements |
|----------|--------|-------------|
| Verra (VCS) | [Berkeley Carbon Trading Project](https://gspp.berkeley.edu/research/osf-bctp/offsets-database) | 235,563 |
| Gold Standard | [Berkeley Carbon Trading Project](https://gspp.berkeley.edu/research/osf-bctp/offsets-database) | 148,566 |
| Climate Action Reserve (CAR) | [Berkeley Carbon Trading Project](https://gspp.berkeley.edu/research/osf-bctp/offsets-database) | 10,299 |
| American Carbon Registry (ACR) | [Berkeley Carbon Trading Project](https://gspp.berkeley.edu/research/osf-bctp/offsets-database) | 9,590 |
| EcoRegistry (Cercarbono) | [EcoRegistry API](https://www.ecoregistry.io/) | 8,695 |
| BioCarbon Registry | [Global CarbonTrace API](https://globalcarbontrace.io/) | 10,078 |
| Puro.earth | [Puro Registry](https://registry.puro.earth/retirements) | 1,443 |
| CDM (UNFCCC) | [CDM Registry](https://cdm.unfccc.int/Registry/vc_attest/) via Wayback Machine | 7,041 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline (cache-only matching, no downloads)
python -m src.pipeline --skip-download --skip-llm

# Run pipeline with LLM matching (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.pipeline --skip-download

# Run full pipeline (download + match)
python -m src.pipeline

# Download EcoRegistry, BioCarbon, and Puro data
python -m src.download_ecoregistry
python -m src.download_biocarbon
python -m src.download_puro
python -m src.download_cdm

# Launch interactive dashboard
streamlit run dashboard/app.py
```

## Architecture

```
carbon-offset-tracker/
├── src/
│   ├── build_lookup.py          # One-time: build known_matches + public_firms
│   ├── download.py              # Download Berkeley VROD data
│   ├── download_ecoregistry.py  # Download EcoRegistry (Cercarbono) data
│   ├── download_biocarbon.py    # Download BioCarbon (Global CarbonTrace) data
│   ├── download_puro.py         # Download Puro.earth Registry data
│   ├── download_cdm.py          # Download CDM/UNFCCC voluntary cancellations (Wayback + archive)
│   ├── parse_beneficiary.py     # Registry-specific beneficiary extraction
│   ├── match_firms.py           # Cache lookup + Claude API matching
│   ├── pipeline.py              # End-to-end orchestration
│   └── utils.py                 # Name normalization, config loading
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── docs/
│   ├── index.html               # GitHub Pages landing page
│   └── data/                    # JSON data for landing page charts
├── data/
│   ├── known_matches.csv        # 3,000+ cached name→firm mappings
│   ├── manual_matches_latam.csv # Manual matches for Latin American subsidiaries
│   ├── public_firms.parquet     # ~50K public company universe (FactSet)
│   ├── matched_retirements.*    # Output dataset (parquet + CSV)
│   ├── summary_stats.json       # Aggregate statistics
│   └── raw/                     # Raw registry downloads
├── .github/workflows/
│   └── monthly_update.yml       # Automated monthly pipeline
├── config.yaml                  # Data source paths, thresholds
└── requirements.txt
```

## Matching Pipeline

1. **Registry-specific parsing**: Each registry stores beneficiary names differently
   - Verra: `Retirement Beneficiary` field, with fallback to `Retirement Details`
   - Gold Standard: `* Using Entity` field, with fallback to `Note`
   - ACR: `Retired on Behalf of`, fallback to `Account Holder`
   - CAR: `Account Holder` field
   - EcoRegistry: `final_user` field
   - BioCarbon: `beneficiary` field, with fallback to `final_user`
   - Puro.earth: `beneficiary_name` field, with fallback to `account_holder`
   - CDM: Free-text `Purpose`/`Reason/ Beneficiary` field (parsed for entity names via multiple patterns)

2. **Name normalization**: Strip legal suffixes (Inc, SA, SAS, Ltd, etc.), Colombian NIT tax IDs, CNPJ numbers; resolve subsidiary names to parent companies

3. **Cache lookup**: Exact and normalized name matching against 3,000+ known mappings, plus 136 manual matches for Latin American subsidiaries (e.g., Biomax → Ecopetrol, Noel → Grupo Nutresa)

4. **LLM matching**: For unmatched names, Claude Haiku resolves subsidiaries and brands to parent listed firms (e.g., Nespresso → Nestlé). Cost: ~$0.50–2/month.

5. **Confidence scoring**: HIGH (auto-accept + cache), MEDIUM (auto-accept + flag), LOW (manual review)

## Output Files

| File | Description |
|------|-------------|
| `matched_retirements.parquet` | All 422K retirement transactions (matched + unmatched) |
| `matched_retirements.csv` | Matched retirements only (33K rows) — [download](https://github.com/apedraza82/carbon-offset-tracker/raw/master/data/matched_retirements.csv) |
| `known_matches.csv` | Beneficiary name → FactSet entity ID cache |
| `manual_matches_latam.csv` | Curated matches for Latin American firms |
| `public_firms.parquet` | Public company universe with identifiers |
| `summary_stats.json` | Aggregate statistics for landing page |

## Citation

If you use this dataset, please cite:

> Pedraza, Alvaro & Williams, Tomas & Zeni, Federica, 2026. "[Where Firms Retire Carbon Offsets: Operational Footprints and Offset Quality](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099338203162614529)," Policy Research Working Paper Series 11331, The World Bank.

## License

Data is provided for research purposes. See the paper for details on methodology and limitations.
