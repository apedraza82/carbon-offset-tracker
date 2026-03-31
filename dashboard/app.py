"""Streamlit dashboard for Carbon Offset Tracker.

Run locally: streamlit run dashboard/app.py
Deploy: Connect GitHub repo to Streamlit Cloud (free tier)
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page config
st.set_page_config(
    page_title="Carbon Offset Tracker",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data(ttl=3600)
def load_data():
    """Load matched retirements data."""
    csv_path = DATA_DIR / "matched_retirements.csv"
    parquet_path = DATA_DIR / "matched_retirements.parquet"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        return None

    # Parse dates if available
    for col in ["ret_date", "retirementdate", "Retirement/Cancellation Date", "Retirement Date"]:
        if col in df.columns:
            df["retirement_date"] = pd.to_datetime(df[col], errors="coerce")
            df["retirement_year"] = df["retirement_date"].dt.year
            break

    # Filter to matched only
    df = df[df["factset_entity_id"].notna() & (df["factset_entity_id"] != "")].copy()

    # Add grouped project type
    if "projecttype" in df.columns:
        df["project_category"] = df["projecttype"].apply(classify_project_type)

    # Clean up long country names
    _COUNTRY_RENAMES = {
        "Congo, The Democratic Republic of The": "DR Congo",
        "Congo, the Democratic Republic of the": "DR Congo",
        "Korea, Republic of": "South Korea",
        "Tanzania, United Republic of": "Tanzania",
        "Lao People's Democratic Republic": "Laos",
    }
    for col in ["country", "Country/Area", "Country"]:
        if col in df.columns:
            df[col] = df[col].replace(_COUNTRY_RENAMES)

    # Add full HQ country name
    if "hq_country" in df.columns:
        df["hq_country_name"] = df["hq_country"].map(_ISO2_TO_NAME)

    return df


@st.cache_data(ttl=3600)
def load_stats():
    """Load summary statistics."""
    stats_path = DATA_DIR / "summary_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def fmt_tonnes(t):
    """Format tonnes as human-readable string."""
    if t >= 1e6:
        return f"{t/1e6:.1f}M"
    if t >= 1e3:
        return f"{t/1e3:.0f}K"
    return f"{t:,.0f}"


_ISO2_TO_NAME = {
    "AE": "United Arab Emirates", "AR": "Argentina", "AT": "Austria", "AU": "Australia",
    "BD": "Bangladesh", "BE": "Belgium", "BM": "Bermuda", "BR": "Brazil", "CA": "Canada", "CD": "DR Congo",
    "CH": "Switzerland", "CL": "Chile", "CN": "China", "CO": "Colombia", "CR": "Costa Rica",
    "CY": "Cyprus", "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "EG": "Egypt",
    "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "United Kingdom", "GR": "Greece",
    "GT": "Guatemala", "HK": "Hong Kong", "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland",
    "IL": "Israel", "IN": "India", "IT": "Italy", "JE": "Jersey", "JP": "Japan",
    "KE": "Kenya", "KR": "South Korea", "KY": "Cayman Islands", "LR": "Liberia",
    "LU": "Luxembourg", "MU": "Mauritius", "MX": "Mexico", "MY": "Malaysia", "NG": "Nigeria",
    "NL": "Netherlands", "NO": "Norway", "NZ": "New Zealand", "PA": "Panama", "PE": "Peru",
    "PH": "Philippines", "PK": "Pakistan", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "RU": "Russia", "SA": "Saudi Arabia", "SE": "Sweden", "SG": "Singapore", "TH": "Thailand",
    "TR": "Turkey", "TW": "Taiwan", "UA": "Ukraine", "US": "United States", "VN": "Vietnam",
    "ZA": "South Africa",
}


def classify_project_type(ptype):
    """Group granular project types into broad categories."""
    if not isinstance(ptype, str):
        return "Other"
    p = ptype.lower()
    if any(x in p for x in ["forest", "a/r", "redd", "afforestation", "avoided conversion", "avoided grassland"]):
        return "Forestry & Land Use"
    if any(x in p for x in ["agriculture", "soil", "agricultural", "livestock", "manure"]):
        return "Agriculture & Livestock"
    if any(x in p for x in ["renewable", "wind", "solar", "hydro", "geothermal", "biomass", "biogas", "biofuel"]):
        return "Renewable Energy"
    if any(x in p for x in ["energy efficiency", "energy demand", "energy distribution"]):
        return "Energy Efficiency"
    if any(x in p for x in ["landfill", "waste", "composting", "digestion"]):
        return "Waste Management"
    if any(x in p for x in ["industrial", "manufacturing", "mining", "chemical", "adipic", "nitric", "cement"]):
        return "Industrial Processes"
    if any(x in p for x in ["ozone", "halocarbon", "hfc", "refrigerant", "industrial gas"]):
        return "Ozone & Industrial Gases"
    if any(x in p for x in ["transport", "fleet"]):
        return "Transport"
    if any(x in p for x in ["carbon capture", "ccs"]):
        return "Carbon Capture"
    return "Other"


def main():
    st.title("Carbon Offset Tracker")
    st.markdown(
        "**Corporate Carbon Offset Retirements Matched to Listed Firms** "
        "| [Paper](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099338203162614529) | [GitHub](https://github.com/apedraza82/carbon-offset-tracker)"
    )

    df = load_data()
    if df is None:
        st.error("No data found. Run the pipeline first: `python -m src.pipeline --skip-download --skip-llm`")
        return

    stats = load_stats()

    # Quantity column
    qty_col = "quantity" if "quantity" in df.columns else "Quantity" if "Quantity" in df.columns else None

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Year filter
    year_range = None
    if "retirement_year" in df.columns:
        years = sorted(df["retirement_year"].dropna().unique())
        if years:
            year_range = st.sidebar.slider(
                "Retirement Year",
                min_value=int(min(years)),
                max_value=int(max(years)),
                value=(int(min(years)), int(max(years))),
            )

    # HQ country filter (full names)
    selected_hq = []
    if "hq_country_name" in df.columns:
        hq_countries = sorted(df["hq_country_name"].dropna().unique())
        selected_hq = st.sidebar.multiselect("HQ Country (firm)", hq_countries, default=[])

    # Country filter (project)
    country_col = next((c for c in ["country", "Country/Area", "Country"] if c in df.columns), None)
    selected_countries = []
    if country_col:
        countries = sorted(df[country_col].dropna().unique())
        selected_countries = st.sidebar.multiselect("Project Country", countries, default=[])

    # Project category filter
    selected_categories = []
    if "project_category" in df.columns:
        categories = sorted(df["project_category"].dropna().unique())
        selected_categories = st.sidebar.multiselect("Project Type", categories, default=[])

    # Firm search
    firm_search = st.sidebar.text_input("Search firm name")

    # --- Apply Filters ---
    mask = pd.Series(True, index=df.index)

    if year_range and "retirement_year" in df.columns:
        mask &= df["retirement_year"].between(*year_range)

    if selected_hq and "hq_country_name" in df.columns:
        mask &= df["hq_country_name"].isin(selected_hq)

    if country_col and selected_countries:
        mask &= df[country_col].isin(selected_countries)

    if selected_categories and "project_category" in df.columns:
        mask &= df["project_category"].isin(selected_categories)

    if firm_search:
        name_col = "matched_name" if "matched_name" in df.columns else "raw_beneficiary"
        mask &= df[name_col].str.contains(firm_search, case=False, na=False)

    filtered = df[mask]

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    total_qty = filtered[qty_col].sum() if qty_col else 0
    col1.metric("Total Quantity Retired", fmt_tonnes(total_qty) + " tCO2")
    col2.metric("Transactions", f"{len(filtered):,}")
    col3.metric("Unique Firms", f"{filtered['factset_entity_id'].nunique():,}")
    if country_col:
        col4.metric("Countries", f"{filtered[country_col].nunique():,}")

    if stats:
        st.caption(f"Last updated: {stats.get('last_updated', 'unknown')}")

    # --- Charts ---
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    # Quantity retired over time
    with chart_col1:
        st.subheader("Quantity Retired Over Time")
        if "retirement_year" in filtered.columns and qty_col:
            yearly = filtered.groupby("retirement_year")[qty_col].sum().reset_index()
            yearly[qty_col] = yearly[qty_col] / 1e6  # Convert to MtCO2
            fig = px.bar(yearly, x="retirement_year", y=qty_col,
                         labels={"retirement_year": "Year", qty_col: "MtCO2"},
                         color_discrete_sequence=["#2E86AB"])
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # By project category
    with chart_col2:
        st.subheader("Quantity by Project Type")
        if "project_category" in filtered.columns and qty_col:
            cat_qty = filtered.groupby("project_category")[qty_col].sum().reset_index()
            cat_qty.columns = ["Category", "Tonnes"]
            cat_qty = cat_qty.sort_values("Tonnes", ascending=False)
            # Hide text labels for slices below 1%
            total = cat_qty["Tonnes"].sum()
            cat_qty["pct"] = cat_qty["Tonnes"] / total * 100
            text_labels = [f"{row.pct:.1f}%" if row.pct >= 1 else "" for row in cat_qty.itertuples()]
            fig = px.pie(cat_qty, values="Tonnes", names="Category",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(text=text_labels, textinfo="text+label",
                              insidetextorientation="radial")
            # Hide label for tiny slices too
            fig.update_traces(texttemplate=[
                "%{label}<br>%{text}" if pct >= 1 else ""
                for pct in cat_qty["pct"]
            ])
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # Geographic distribution
    if country_col and qty_col and len(filtered) > 0:
        st.subheader("Top 20 Countries by Quantity Retired (tCO2)")
        top_countries = filtered.groupby(country_col)[qty_col].sum().nlargest(20).reset_index()
        top_countries.columns = ["Country", "Tonnes"]
        fig = px.bar(top_countries, x="Country", y="Tonnes",
                     labels={"Tonnes": "tCO2"},
                     color_discrete_sequence=["#2E86AB"])
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Top firms
    if qty_col:
        st.subheader("Top 20 Firms by Quantity Retired (tCO2)")
        name_col = "matched_name" if "matched_name" in filtered.columns else "raw_beneficiary"
        top_firms = filtered.groupby(name_col)[qty_col].sum().nlargest(20).reset_index()
        top_firms.columns = ["Firm", "Tonnes"]
        fig = px.bar(top_firms, x="Tonnes", y="Firm", orientation="h",
                     labels={"Tonnes": "tCO2"},
                     color_discrete_sequence=["#A23B72"])
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # --- Data Table (matched only) ---
    st.markdown("---")
    st.subheader("Data Explorer")

    display_cols = [c for c in [
        "matched_name", "hq_country_name", "registry",
        "retirement_year", "country", "Country/Area", "quantity", "Quantity",
        "projectname", "projecttype", "vintage",
    ] if c in filtered.columns]

    st.dataframe(filtered[display_cols].head(500), use_container_width=True, height=400)

    # --- Download (matched only, no MSCI) ---
    st.markdown("---")
    download_cols = [c for c in display_cols if c not in ("rating_msci", "numrating_msci")]

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_data = filtered[download_cols].to_csv(index=False)
        st.download_button(
            label=f"Download Filtered Data ({len(filtered):,} rows)",
            data=csv_data,
            file_name="carbon_offset_retirements_filtered.csv",
            mime="text/csv",
        )

    with col_dl2:
        full_csv = df[download_cols].to_csv(index=False)
        st.download_button(
            label=f"Download Full Dataset ({len(df):,} rows)",
            data=full_csv,
            file_name="carbon_offset_retirements_full.csv",
            mime="text/csv",
        )

    # --- Citation ---
    st.markdown("---")
    st.subheader("Citation")
    st.info(
        "If you use this dataset in your research, please cite:\n\n"
        "> Pedraza, Alvaro & Williams, Tomas & Zeni, Federica, 2026. "
        "\"[Local Visibility vs. Global Integrity: Evidence from Corporate Carbon Offsetting]"
        "(https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099338203162614529),\" "
        "Policy Research Working Paper Series 11331, The World Bank.\n\n"
        "Data source: [Berkeley Carbon Trading Project](https://gspp.berkeley.edu/research/osf-bctp/offsets-database). "
        "Firm matching via [FactSet](https://www.factset.com/)."
    )


if __name__ == "__main__":
    main()
